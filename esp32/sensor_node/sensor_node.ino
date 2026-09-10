#include <Wire.h>
#include <OneWire.h>
#include <DallasTemperature.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <math.h>

// =====================================================
// PIN DEFINITIONS
// =====================================================

#define TEMP_PIN 4

#define START_BUTTON       5
#define SPEED_UP_BUTTON    6
#define SPEED_DOWN_BUTTON  7
#define STOP_BUTTON        8

#define MOTOR_PWM_PIN      9

// =====================================================
// TEMPERATURE SETTINGS
// =====================================================

#define TEMP_HIGH_LIMIT 36.0

// If DS18B20 gives -127 C or another invalid value,
// use 35 C instead.
#define TEMP_ERROR_VALUE 35.0

// =====================================================
// VIBRATION SETTINGS
// =====================================================

#define VIBRATION_WARNING_LIMIT 20.0
#define VIBRATION_HIGH_LIMIT    100.0

// =====================================================
// MOTOR SETTINGS
// =====================================================

int motorSpeedPercent = 20;

#define SPEED_STEP 10

bool motorRunning = true;

// =====================================================
// SENSOR OBJECTS
// =====================================================

OneWire oneWire(TEMP_PIN);

DallasTemperature temperatureSensor(&oneWire);

Adafruit_MPU6050 mpu;

// =====================================================
// VIBRATION BASELINE
// =====================================================

float vibrationBaseline = 9.80665;

// =====================================================
// SENSOR VALUES
// =====================================================

float temperature = 35.0;
float vibration = 0.0;

// =====================================================
// BUTTON STATES
// =====================================================

bool previousStart = HIGH;
bool previousUp = HIGH;
bool previousDown = HIGH;
bool previousStop = HIGH;

// =====================================================
// TIMING
// =====================================================

unsigned long lastMonitoringTime = 0;

const unsigned long monitoringInterval = 2000;

// =====================================================
// SETUP
// =====================================================

void setup()
{
  Serial.begin(9600);

  // ===================================================
  // BUTTONS
  // ===================================================

  pinMode(START_BUTTON, INPUT_PULLUP);
  pinMode(SPEED_UP_BUTTON, INPUT_PULLUP);
  pinMode(SPEED_DOWN_BUTTON, INPUT_PULLUP);
  pinMode(STOP_BUTTON, INPUT_PULLUP);

  // ===================================================
  // MOTOR (AUTO-START AT 20% SPEED)
  // ===================================================

  pinMode(MOTOR_PWM_PIN, OUTPUT);

  setMotorSpeed();

  // ===================================================
  // DS18B20 TEMPERATURE SENSOR
  // ===================================================

  temperatureSensor.begin();

  // Do not show "NOT FOUND"
  Serial.println();
  Serial.println("DS18B20 : READY");

  // Non-blocking conversion
  temperatureSensor.setWaitForConversion(false);

  // ===================================================
  // MPU6050
  // ===================================================

  Wire.begin();

  if (!mpu.begin())
  {
    Serial.println("MPU6050 : NOT FOUND");
    Serial.println("Continuing without stopping program...");
  }
  else
  {
    Serial.println("MPU6050 : ACTIVE");

    mpu.setAccelerometerRange(MPU6050_RANGE_8_G);

    mpu.setGyroRange(MPU6050_RANGE_500_DEG);

    mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);

    // =================================================
    // VIBRATION CALIBRATION
    // =================================================

    Serial.println("--------------------------------");
    Serial.println("Keep MPU6050 STILL");
    Serial.println("Calibrating vibration...");
    Serial.println("--------------------------------");

    delay(1000);

    float total = 0;

    const int samples = 100;

    for (int i = 0; i < samples; i++)
    {
      sensors_event_t acceleration;
      sensors_event_t gyro;
      sensors_event_t tempMPU;

      mpu.getEvent(
        &acceleration,
        &gyro,
        &tempMPU
      );

      float magnitude = sqrt(
        acceleration.acceleration.x *
        acceleration.acceleration.x +

        acceleration.acceleration.y *
        acceleration.acceleration.y +

        acceleration.acceleration.z *
        acceleration.acceleration.z
      );

      total += magnitude;

      delay(10);
    }

    vibrationBaseline =
      total / samples;

    Serial.print("Vibration baseline: ");
    Serial.print(vibrationBaseline, 2);
    Serial.println(" m/s2");
  }

  // ===================================================
  // SYSTEM INFORMATION
  // ===================================================

  Serial.println("--------------------------------");

  Serial.println("TEMPERATURE:");
  Serial.println("< 36 C  = NORMAL");
  Serial.println(">= 36 C = DANGER");

  Serial.println("--------------------------------");

  Serial.println("VIBRATION:");
  Serial.println("< 50     = NORMAL");
  Serial.println("50-99.99 = WARNING");
  Serial.println(">= 100   = DANGER");

  Serial.println("--------------------------------");

  Serial.println("Temperature sensor error:");
  Serial.println("Invalid reading -> 35 C NORMAL");

  Serial.println("--------------------------------");

  Serial.println("CONTROLS:");
  Serial.println("D5 = START");
  Serial.println("D6 = SPEED +");
  Serial.println("D7 = SPEED -");
  Serial.println("D8 = STOP");

  Serial.println("--------------------------------");

  Serial.print("Initial Motor Speed: ");
  Serial.print(motorSpeedPercent);
  Serial.println("%");

  Serial.println("SYSTEM READY");
  Serial.println("================================");

  // Start monitoring immediately
  lastMonitoringTime = millis();
}

// =====================================================
// MAIN LOOP
// =====================================================

void loop()
{
  // ===================================================
  // CHECK BUTTONS & SERIAL COMMANDS CONTINUOUSLY
  // ===================================================

  checkButtons();
  checkSerialCommands();

  // ===================================================
  // KEEP MOTOR RUNNING
  // ===================================================

  if (motorRunning)
  {
    setMotorSpeed();
  }
  else
  {
    analogWrite(MOTOR_PWM_PIN, 0);
  }

  // ===================================================
  // SENSOR MONITORING EVERY 2 SECONDS
  // ===================================================

  unsigned long currentTime = millis();

  if (
    currentTime - lastMonitoringTime >=
    monitoringInterval
  )
  {
    lastMonitoringTime = currentTime;

    // Read sensors
    readTemperature();

    readVibration();

    // Display readings
    displayReadings();

    // =================================================
    // SAFETY WARNING LOGGING (CONTINUOUS MOTOR RUNNING)
    // =================================================

    bool temperatureDanger =
      temperature >= TEMP_HIGH_LIMIT;

    bool vibrationDanger =
      vibration >= VIBRATION_HIGH_LIMIT;

    if (
      temperatureDanger ||
      vibrationDanger
    )
    {
      Serial.println();
      Serial.println("!!! SENSOR WARNING !!!");

      if (temperatureDanger)
      {
        Serial.println("Temperature DANGER");
      }

      if (vibrationDanger)
      {
        Serial.println("Vibration DANGER");
      }
      // Note: Motor is kept running continuously as requested.
    }
  }
}

// =====================================================
// TEMPERATURE READING
// =====================================================

void readTemperature()
{
  // Request temperature
  temperatureSensor.requestTemperatures();

  // Small time for conversion
  delay(10);

  float newTemperature =
    temperatureSensor.getTempCByIndex(0);

  // ===================================================
  // HANDLE INVALID DS18B20 READING
  // ===================================================

  if (
    newTemperature == DEVICE_DISCONNECTED_C ||
    newTemperature < -50 ||
    newTemperature > 125 ||
    isnan(newTemperature)
  )
  {
    // Use 35 C
    temperature = TEMP_ERROR_VALUE;
  }
  else
  {
    temperature = newTemperature;
  }
}

// =====================================================
// VIBRATION READING
// =====================================================

void readVibration()
{
  sensors_event_t acceleration;
  sensors_event_t gyro;
  sensors_event_t tempMPU;

  // Read MPU6050
  mpu.getEvent(
    &acceleration,
    &gyro,
    &tempMPU
  );

  // Calculate total acceleration
  float magnitude = sqrt(
    acceleration.acceleration.x *
    acceleration.acceleration.x +

    acceleration.acceleration.y *
    acceleration.acceleration.y +

    acceleration.acceleration.z *
    acceleration.acceleration.z
  );

  // Remove gravity/baseline
  vibration =
    fabs(
      magnitude -
      vibrationBaseline
    );

  // Prevent invalid values
  if (
    isnan(vibration) ||
    isinf(vibration)
  )
  {
    vibration = 0.0;
  }
}

// =====================================================
// DISPLAY READINGS
// =====================================================

void displayReadings()
{
  Serial.println();
  Serial.println("================================");

  // ===================================================
  // TEMPERATURE
  // ===================================================

  Serial.print("Temperature : ");
  Serial.print(temperature, 2);
  Serial.print(" C   ");

  if (
    temperature >=
    TEMP_HIGH_LIMIT
  )
  {
    Serial.println("[DANGER]");
  }
  else
  {
    Serial.println("[NORMAL]");
  }

  // ===================================================
  // VIBRATION
  // ===================================================

  Serial.print("Vibration   : ");
  Serial.print(vibration, 2);
  Serial.print(" m/s2   ");

  if (
    vibration >=
    VIBRATION_HIGH_LIMIT
  )
  {
    Serial.println("[DANGER]");
  }
  else if (
    vibration >=
    VIBRATION_WARNING_LIMIT
  )
  {
    Serial.println("[WARNING]");
  }
  else
  {
    Serial.println("[NORMAL]");
  }

  // ===================================================
  // MOTOR SPEED
  // ===================================================

  Serial.print("Motor Speed : ");
  Serial.print(motorSpeedPercent);
  Serial.println("%");

  // ===================================================
  // MOTOR STATE
  // ===================================================

  Serial.print("Motor       : ");

  if (motorRunning)
  {
    Serial.println("RUNNING");
  }
  else
  {
    Serial.println("STOPPED");
  }

  Serial.println("================================");
}

// =====================================================
// BUTTON CONTROL
// =====================================================

void checkButtons()
{
  bool startState =
    digitalRead(START_BUTTON);

  bool upState =
    digitalRead(SPEED_UP_BUTTON);

  bool downState =
    digitalRead(SPEED_DOWN_BUTTON);

  bool stopState =
    digitalRead(STOP_BUTTON);

  // ===================================================
  // START BUTTON
  // ===================================================

  if (
    previousStart == HIGH &&
    startState == LOW
  )
  {
    motorRunning = true;

    Serial.println();
    Serial.println("START BUTTON PRESSED");
    Serial.println("Motor STARTED");

    setMotorSpeed();

    delay(200);
  }

  // ===================================================
  // SPEED UP
  // ===================================================

  if (
    previousUp == HIGH &&
    upState == LOW
  )
  {
    motorSpeedPercent += SPEED_STEP;

    if (motorSpeedPercent > 100)
    {
      motorSpeedPercent = 100;
    }

    Serial.print("Speed increased: ");
    Serial.print(motorSpeedPercent);
    Serial.println("%");

    if (motorRunning)
    {
      setMotorSpeed();
    }

    delay(200);
  }

  // ===================================================
  // SPEED DOWN
  // ===================================================

  if (
    previousDown == HIGH &&
    downState == LOW
  )
  {
    motorSpeedPercent -= SPEED_STEP;

    if (motorSpeedPercent < 0)
    {
      motorSpeedPercent = 0;
    }

    Serial.print("Speed decreased: ");
    Serial.print(motorSpeedPercent);
    Serial.println("%");

    if (motorRunning)
    {
      setMotorSpeed();
    }

    delay(200);
  }

  // ===================================================
  // STOP BUTTON
  // ===================================================

  if (
    previousStop == HIGH &&
    stopState == LOW
  )
  {
    motorRunning = false;

    analogWrite(
      MOTOR_PWM_PIN,
      0
    );

    Serial.println();
    Serial.println("STOP BUTTON PRESSED");
    Serial.println("Motor STOPPED");

    delay(200);
  }

  // ===================================================
  // SAVE BUTTON STATES
  // ===================================================

  previousStart = startState;
  previousUp = upState;
  previousDown = downState;
  previousStop = stopState;
}

// =====================================================
// MOTOR SPEED
// =====================================================

void setMotorSpeed()
{
  int pwmValue =
    map(
      motorSpeedPercent,
      0,
      100,
      0,
      255
    );

  analogWrite(
    MOTOR_PWM_PIN,
    pwmValue
  );
}

// =====================================================
// SERIAL COMMAND LISTENER (NON-BLOCKING)
// =====================================================

void checkSerialCommands()
{
  while (Serial.available() > 0)
  {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    cmd.toUpperCase();

    if (cmd == "STOP")
    {
      motorRunning = false;
      analogWrite(MOTOR_PWM_PIN, 0);

      Serial.println();
      Serial.println("================================");
      Serial.println("COMMAND RECEIVED: STOP");
      Serial.println("Motor STOPPED");
      Serial.println("================================");
    }
    else if (cmd == "RESUME" || cmd == "START")
    {
      motorRunning = true;
      setMotorSpeed();

      Serial.println();
      Serial.println("================================");
      Serial.println("COMMAND RECEIVED: RESUME");
      Serial.println("Motor STARTED");
      Serial.println("================================");
    }
  }
}
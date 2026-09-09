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
#define TEMP_ERROR_VALUE 35.0

// =====================================================
// VIBRATION SETTINGS
// =====================================================

#define VIBRATION_WARNING_LIMIT 50.0
#define VIBRATION_HIGH_LIMIT    175.0

// =====================================================
// MOTOR SETTINGS
// =====================================================

int motorSpeedPercent = 30;

#define SPEED_STEP 10

bool motorRunning = false;

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
// BUTTON STATES
// =====================================================

bool previousStart = HIGH;
bool previousUp = HIGH;
bool previousDown = HIGH;
bool previousStop = HIGH;

// =====================================================
// SETUP
// =====================================================

void setup()
{
  Serial.begin(9600);

  // ---------------------------------------------------
  // BUTTONS
  // ---------------------------------------------------

  pinMode(START_BUTTON, INPUT_PULLUP);
  pinMode(SPEED_UP_BUTTON, INPUT_PULLUP);
  pinMode(SPEED_DOWN_BUTTON, INPUT_PULLUP);
  pinMode(STOP_BUTTON, INPUT_PULLUP);

  // ---------------------------------------------------
  // MOTOR
  // ---------------------------------------------------

  pinMode(MOTOR_PWM_PIN, OUTPUT);

  analogWrite(MOTOR_PWM_PIN, 0);

  // ---------------------------------------------------
  // TEMPERATURE SENSOR
  // ---------------------------------------------------

  temperatureSensor.begin();

  Serial.println();
  Serial.println("================================");
  Serial.println("       JOINTGUARD SYSTEM");
  Serial.println("================================");

  if (temperatureSensor.getDeviceCount() > 0)
  {
    Serial.println("DS18B20 : ACTIVE");
  }
  else
  {
    Serial.println("DS18B20 : NOT FOUND");
  }

  // ---------------------------------------------------
  // MPU6050
  // ---------------------------------------------------

  Wire.begin();

  if (!mpu.begin())
  {
    Serial.println("MPU6050 : NOT FOUND");

    while (1)
    {
      delay(1000);
    }
  }

  Serial.println("MPU6050 : ACTIVE");

  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
  mpu.setGyroRange(MPU6050_RANGE_500_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);

  // ===================================================
  // VIBRATION CALIBRATION
  // ===================================================

  Serial.println("--------------------------------");
  Serial.println("Keep MPU6050 COMPLETELY STILL");
  Serial.println("Calibrating vibration baseline...");
  Serial.println("--------------------------------");

  delay(1000);

  float total = 0;

  const int calibrationSamples = 200;

  for (int i = 0; i < calibrationSamples; i++)
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
    total / calibrationSamples;

  Serial.print("Vibration baseline: ");
  Serial.print(vibrationBaseline, 2);
  Serial.println(" m/s2");

  // ===================================================
  // SYSTEM INFORMATION
  // ===================================================

  Serial.println("--------------------------------");

  Serial.println("TEMPERATURE:");
  Serial.println("< 36 C       = NORMAL");
  Serial.println(">= 36 C      = DANGER");

  Serial.println("--------------------------------");

  Serial.println("VIBRATION:");
  Serial.println("< 50 m/s2    = NORMAL");
  Serial.println("50-99.99     = WARNING");
  Serial.println(">= 100       = DANGER");

  Serial.println("--------------------------------");

  Serial.println("CONTROLS:");
  Serial.println("D5 = START");
  Serial.println("D6 = SPEED +");
  Serial.println("D7 = SPEED -");
  Serial.println("D8 = STOP");

  Serial.println("--------------------------------");

  Serial.print("Initial motor speed: ");
  Serial.print(motorSpeedPercent);
  Serial.println("%");

  Serial.println("System ready.");
  Serial.println("Press START to run motor.");

  Serial.println("================================");
}

// =====================================================
// MAIN LOOP
// =====================================================

void loop()
{
  // ===================================================
  // READ TEMPERATURE
  // ===================================================

  temperatureSensor.requestTemperatures();

  float temperature =
    temperatureSensor.getTempCByIndex(0);

  // If DS18B20 gives -127 C/error,
  // use 35 C instead

  if (temperature == DEVICE_DISCONNECTED_C)
  {
    temperature = TEMP_ERROR_VALUE;
  }

  // ===================================================
  // TEMPERATURE STATUS
  // ===================================================

  bool temperatureHigh =
    temperature >= TEMP_HIGH_LIMIT;

  // ===================================================
  // READ MPU6050
  // ===================================================

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

  // Remove gravity/baseline

  float vibration =
    fabs(magnitude - vibrationBaseline);

  // ===================================================
  // VIBRATION STATUS
  // ===================================================

  bool vibrationWarning =
    vibration >= VIBRATION_WARNING_LIMIT;

  bool vibrationHigh =
    vibration >= VIBRATION_HIGH_LIMIT;

  // ===================================================
  // SAFETY STOP
  // ===================================================

  if (temperatureHigh || vibrationHigh)
  {
    if (motorRunning)
    {
      Serial.println();
      Serial.println("!!! SAFETY STOP !!!");

      if (temperatureHigh)
      {
        Serial.println("Temperature = DANGER");
      }

      if (vibrationHigh)
      {
        Serial.println("Vibration = DANGER");
      }

      Serial.println("Motor STOPPED");
    }

    motorRunning = false;

    analogWrite(MOTOR_PWM_PIN, 0);
  }

  // ===================================================
  // BUTTON CONTROL
  // ===================================================

  checkButtons(
    temperatureHigh,
    vibrationHigh
  );

  // ===================================================
  // IMPORTANT:
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
  // DISPLAY
  // ===================================================

  Serial.println();
  Serial.println("================================");

  // Temperature

  Serial.print("Temperature : ");
  Serial.print(temperature, 2);
  Serial.print(" C   ");

  if (temperatureHigh)
  {
    Serial.println("[DANGER]");
  }
  else
  {
    Serial.println("[NORMAL]");
  }

  // Vibration

  Serial.print("Vibration   : ");
  Serial.print(vibration, 2);
  Serial.print(" m/s2   ");

  if (vibrationHigh)
  {
    Serial.println("[DANGER]");
  }
  else if (vibrationWarning)
  {
    Serial.println("[WARNING]");
  }
  else
  {
    Serial.println("[NORMAL]");
  }

  // Motor speed

  Serial.print("Motor Speed : ");
  Serial.print(motorSpeedPercent);
  Serial.println("%");

  // Motor state

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

  // ===================================================
  // MONITOR EVERY 2 SECONDS
  // ===================================================

  delay(2000);
}

// =====================================================
// BUTTON CONTROL
// =====================================================

void checkButtons(
  bool temperatureHigh,
  bool vibrationHigh
)
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
  // START
  // ===================================================

  if (
    previousStart == HIGH &&
    startState == LOW
  )
  {
    Serial.println();
    Serial.println("START BUTTON PRESSED");

    // Start only if sensors are safe

    if (
      !temperatureHigh &&
      !vibrationHigh
    )
    {
      motorRunning = true;

      Serial.println("Motor STARTED");
      Serial.print("Speed: ");
      Serial.print(motorSpeedPercent);
      Serial.println("%");

      setMotorSpeed();
    }
    else
    {
      Serial.println("Motor CANNOT START");

      if (temperatureHigh)
      {
        Serial.println("Temperature is DANGER");
      }

      if (vibrationHigh)
      {
        Serial.println("Vibration is DANGER");
      }
    }

    delay(250);
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

    Serial.print("Speed increased to ");
    Serial.print(motorSpeedPercent);
    Serial.println("%");

    if (motorRunning)
    {
      setMotorSpeed();
    }

    delay(250);
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

    Serial.print("Speed decreased to ");
    Serial.print(motorSpeedPercent);
    Serial.println("%");

    if (motorRunning)
    {
      setMotorSpeed();
    }

    delay(250);
  }

  // ===================================================
  // STOP
  // ===================================================

  if (
    previousStop == HIGH &&
    stopState == LOW
  )
  {
    motorRunning = false;

    analogWrite(MOTOR_PWM_PIN, 0);

    Serial.println();
    Serial.println("STOP BUTTON PRESSED");
    Serial.println("Motor STOPPED");

    delay(250);
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

/*
 * JointGuard - ESP32 Initial Hardware Verification Test Sketch (Phase 1 Baseline)
 * 
 * OBJECTIVE:
 * - Perform the initial hardware sanity test for the ESP32 development board.
 * - Verify USB communication, firmware flash process, ESP32 boot sequence, and built-in LED control.
 * 
 * HARDWARE REQUIREMENTS FOR THIS TEST:
 * - ESP32 Dev Module (USB connection to computer only).
 * - NO external sensors (MPU6050, DS18B20, A3144, HMC5883L, etc.).
 * - NO external libraries required (only standard ESP32 Arduino Core).
 * 
 * SAFETY NOTICE:
 * - Do NOT connect external sensors or feed 5V directly to any GPIO pin during this basic test.
 */

// Fallback definition for built-in LED pin (GPIO 2 is default on most ESP32 Dev Boards)
#ifndef LED_BUILTIN
#define LED_BUILTIN 2
#endif

// Global counter to verify continuous loop execution without crashes or resets
unsigned long heartbeatCounter = 0;
bool ledState = false;

void setup() {
  // 1. Initialize Serial communication at 115200 baud
  Serial.begin(115200);
  
  // Brief delay to allow Serial Monitor connection to stabilize after reboot
  delay(1000);

  // 2. Configure built-in LED pin as digital output
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, LOW);

  // 3. Print clear startup messages
  Serial.println("=================================================");
  Serial.println("   JointGuard ESP32 Test Starting...            ");
  Serial.println("   ESP32 initialized successfully               ");
  Serial.println("   Board test running...                        ");
  Serial.println("=================================================");
}

void loop() {
  // Increment continuous execution counter
  heartbeatCounter++;

  // Toggle built-in LED state
  ledState = !ledState;
  digitalWrite(LED_BUILTIN, ledState ? HIGH : LOW);

  // Print heartbeat counter and LED status to Serial Monitor
  Serial.print("Heartbeat: ");
  Serial.println(heartbeatCounter);

  if (ledState) {
    Serial.println("LED ON");
  } else {
    Serial.println("LED OFF");
  }

  // Wait 1 second (500ms toggle speed = 1s per complete ON/OFF cycle)
  delay(1000);
}

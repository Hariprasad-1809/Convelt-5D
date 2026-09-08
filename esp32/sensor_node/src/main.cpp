/*
 * JointGuard - Basic Ultrasonic Sensor (HC-SR04) Hardware Test
 * 
 * Hardware:
 * - ESP32 Development Board
 * - HC-SR04 Ultrasonic Distance Sensor
 * 
 * Pin Connections:
 * - HC-SR04 VCC  -> ESP32 5V (or Vin / 5V pin on board)
 * - HC-SR04 GND  -> ESP32 GND
 * - HC-SR04 TRIG -> ESP32 GPIO 5
 * - HC-SR04 ECHO -> Voltage Divider -> ESP32 GPIO 18
 * 
 * Voltage Divider (Crucial for ESP32 3.3V Logic Protection):
 * HC-SR04 ECHO (5V Signal)
 *      |
 *    1 kΩ (R1)
 *      |
 *      +-------> ESP32 GPIO 18 (Receives max ~3.33V)
 *      |
 *    2 kΩ (R2)
 *      |
 *    ESP32 GND
 */

#include <Arduino.h>

// Define GPIO pin mapping for HC-SR04
const int TRIG_PIN = 5;  // Output pin to send trigger pulse
const int ECHO_PIN = 18; // Input pin to read echo pulse duration

// Maximum duration to wait for echo in microseconds (30,000 µs = ~5 meters maximum distance)
// Prevents pulseIn() from hanging indefinitely if no pulse returns.
const unsigned long ECHO_TIMEOUT = 30000;

void setup() {
    // 1. Initialize Serial Communication at 115200 baud
    Serial.begin(115200);

    // Give Serial Monitor 1 second to attach after boot
    delay(1000);

    // 2. Print initial header message
    Serial.println("JointGuard Ultrasonic Sensor Test");

    // 3. Configure GPIO pin modes
    pinMode(TRIG_PIN, OUTPUT);
    pinMode(ECHO_PIN, INPUT);

    // Ensure trigger pin starts LOW
    digitalWrite(TRIG_PIN, LOW);
}

void loop() {
    // Step 1: Ensure a clean LOW pulse on TRIG pin for 2 microseconds
    digitalWrite(TRIG_PIN, LOW);
    delayMicroseconds(2);

    // Step 2: Generate a HIGH pulse on TRIG pin for 10 microseconds
    digitalWrite(TRIG_PIN, HIGH);
    delayMicroseconds(10);
    digitalWrite(TRIG_PIN, LOW);

    // Step 3: Measure the duration of the HIGH pulse received on ECHO pin (in microseconds)
    // pulseIn returns 0 if no pulse is detected within the ECHO_TIMEOUT period.
    unsigned long duration = pulseIn(ECHO_PIN, HIGH, ECHO_TIMEOUT);

    // Step 4: Calculate and output the measured distance
    if (duration == 0) {
        // Safe timeout handling: no echo pulse was returned
        Serial.println("Distance: No echo / out of range");
    } else {
        // Calculate distance in centimeters:
        // Speed of sound in air = 343 m/s = 0.0343 cm/microsecond
        // Distance = (duration * 0.0343 cm/µs) / 2 (round trip travel to object and back)
        float distance = (duration * 0.0343f) / 2.0f;

        // Print distance formatted to 2 decimal places
        Serial.print("Distance: ");
        Serial.print(distance, 2);
        Serial.println(" cm");
    }

    // Step 5: Wait 500 milliseconds before taking the next reading
    delay(500);
}

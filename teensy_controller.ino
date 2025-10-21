/*
 * NAOMI SOL HUB - Teensy 4.1 Firmware
 * ====================================
 * Master controller firmware for dodecahedron robotic chamber
 * 
 * Hardware:
 * - Teensy 4.1 (600MHz ARM Cortex-M7)
 * - 2× PCA9685 servo driver boards (36 servos total)
 * - 3× MPU-9250 IMU sensors via I2C
 * - Serial communication with Python control system
 * 
 * Open-Source Libraries Used:
 * - Adafruit PWM Servo Driver Library
 *   https://github.com/adafruit/Adafruit-PWM-Servo-Driver-Library
 * - Adafruit MPU6050 Library (compatible with MPU-9250)
 *   https://github.com/adafruit/Adafruit_MPU6050
 * - Madgwick AHRS Filter
 *   https://github.com/PaulStoffregen/MadgwickAHRS
 * 
 * Installation via Arduino Library Manager:
 * - Adafruit PWM Servo Driver Library
 * - Adafruit MPU6050
 * - Adafruit BusIO
 * - Adafruit Sensor
 * 
 * Author: Integrated from open-source libraries
 * License: MIT
 */

#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <MadgwickAHRS.h>

// ===== CONFIGURATION =====
#define SERVO_FREQ 60  // Analog servos run at ~60 Hz
#define NUM_BOARDS 2   // Number of PCA9685 boards
#define SERVOS_PER_BOARD 16
#define TOTAL_SERVOS 36  // 12 panels × 3 servos
#define NUM_PANELS 12

// PCA9685 board addresses (set via jumpers)
#define PCA9685_ADDR_1 0x40
#define PCA9685_ADDR_2 0x41

// MPU-9250 addresses (0x68, 0x69, 0x6A via AD0 pin)
#define MPU_ADDR_1 0x68
#define MPU_ADDR_2 0x69  
#define MPU_ADDR_3 0x6A

// MG90S servo pulse width range (microseconds)
#define SERVO_MIN_PULSE 500
#define SERVO_MAX_PULSE 2500

// Control loop timing
#define CONTROL_RATE_HZ 100
#define CONTROL_PERIOD_US (1000000 / CONTROL_RATE_HZ)

// ===== GLOBAL OBJECTS =====
Adafruit_PWMServoDriver pca9685_boards[NUM_BOARDS];
Adafruit_MPU6050 mpu_sensors[3];
Madgwick madgwick_filters[3];

// ===== STATE VARIABLES =====
struct PanelState {
  uint8_t panel_id;
  float servo_angles[3];  // degrees
  float orientation[3];   // roll, pitch, yaw in radians
  unsigned long timestamp;
};

PanelState panel_states[NUM_PANELS];

// Timing
unsigned long last_control_update = 0;
unsigned long last_imu_update = 0;

// Command buffer
const int CMD_BUFFER_SIZE = 128;
char cmd_buffer[CMD_BUFFER_SIZE];
int cmd_index = 0;

// ===== SERVO CONTROL FUNCTIONS =====

// Convert angle (0-180°) to PCA9685 pulse width
uint16_t angleToPulse(float angle) {
  // Clamp angle to valid range
  angle = constrain(angle, 0.0, 180.0);
  
  // Map angle to pulse width
  uint16_t pulse = map(angle * 100, 0, 18000, 
                      SERVO_MIN_PULSE, SERVO_MAX_PULSE);
  
  // Convert microseconds to PCA9685 counts (4096 counts per period)
  // At 60Hz, period = 16.67ms = 16667µs
  // pulse_count = pulse_us * 4096 / 16667
  uint16_t pulse_count = (pulse * 4096) / 16667;
  
  return pulse_count;
}

// Set single servo angle
void setServoAngle(uint8_t servo_num, float angle) {
  if (servo_num >= TOTAL_SERVOS) return;
  
  // Calculate which board and which channel
  uint8_t board_idx = servo_num / SERVOS_PER_BOARD;
  uint8_t channel = servo_num % SERVOS_PER_BOARD;
  
  if (board_idx >= NUM_BOARDS) return;
  
  // Convert angle to pulse
  uint16_t pulse = angleToPulse(angle);
  
  // Set PWM
  pca9685_boards[board_idx].setPWM(channel, 0, pulse);
}

// Set all three servos for a panel
void setPanelServos(uint8_t panel_id, float angles[3]) {
  if (panel_id >= NUM_PANELS) return;
  
  uint8_t base_servo = panel_id * 3;
  
  for (int i = 0; i < 3; i++) {
    setServoAngle(base_servo + i, angles[i]);
    panel_states[panel_id].servo_angles[i] = angles[i];
  }
  
  panel_states[panel_id].timestamp = millis();
}

// Initialize all servos to center position
void centerAllServos() {
  Serial.println("Centering all servos...");
  
  float center_angles[3] = {90.0, 90.0, 90.0};
  
  for (uint8_t panel = 0; panel < NUM_PANELS; panel++) {
    setPanelServos(panel, center_angles);
    delay(50);  // Small delay between panels
  }
  
  Serial.println("All servos centered");
}

// ===== IMU FUNCTIONS =====

void updateIMUData() {
  for (int i = 0; i < 3; i++) {
    sensors_event_t accel, gyro, temp;
    
    if (mpu_sensors[i].getEvent(&accel, &gyro, &temp)) {
      // Update Madgwick filter (expects rad/s for gyro)
      madgwick_filters[i].update(
        gyro.gyro.x, gyro.gyro.y, gyro.gyro.z,
        accel.acceleration.x, accel.acceleration.y, accel.acceleration.z,
        0, 0, 0  // No magnetometer in MPU6050 mode
      );
      
      // Get orientation (roll, pitch, yaw)
      panel_states[i * 4].orientation[0] = madgwick_filters[i].getRoll();
      panel_states[i * 4].orientation[1] = madgwick_filters[i].getPitch();
      panel_states[i * 4].orientation[2] = madgwick_filters[i].getYaw();
    }
  }
}

// ===== COMMAND PARSING =====

void parseCommand(String cmd) {
  cmd.trim();
  
  if (cmd.startsWith("SET_SERVO:")) {
    // Format: SET_SERVO:panel_id,angle1,angle2,angle3
    String params = cmd.substring(10);
    
    int panel_id = params.substring(0, params.indexOf(',')).toInt();
    params = params.substring(params.indexOf(',') + 1);
    
    float angles[3];
    for (int i = 0; i < 3; i++) {
      angles[i] = params.substring(0, params.indexOf(',')).toFloat();
      if (i < 2) params = params.substring(params.indexOf(',') + 1);
    }
    
    setPanelServos(panel_id, angles);
    Serial.print("OK: Set panel ");
    Serial.println(panel_id);
  }
  else if (cmd == "CENTER_ALL") {
    centerAllServos();
    Serial.println("OK: All centered");
  }
  else if (cmd == "STATUS") {
    // Send back panel states
    for (int i = 0; i < NUM_PANELS; i++) {
      Serial.print("PANEL:");
      Serial.print(i);
      Serial.print(",");
      Serial.print(panel_states[i].servo_angles[0]);
      Serial.print(",");
      Serial.print(panel_states[i].servo_angles[1]);
      Serial.print(",");
      Serial.print(panel_states[i].servo_angles[2]);
      Serial.print(",");
      Serial.print(panel_states[i].orientation[0]);
      Serial.print(",");
      Serial.print(panel_states[i].orientation[1]);
      Serial.print(",");
      Serial.println(panel_states[i].orientation[2]);
    }
    Serial.println("OK: Status sent");
  }
  else if (cmd == "PING") {
    Serial.println("PONG");
  }
  else {
    Serial.print("ERROR: Unknown command: ");
    Serial.println(cmd);
  }
}

// ===== SETUP =====

void setup() {
  // Initialize serial communication
  Serial.begin(115200);
  while (!Serial && millis() < 3000); // Wait up to 3 seconds
  
  Serial.println("\n===================================");
  Serial.println("NAOMI SOL HUB - Teensy 4.1");
  Serial.println("===================================");
  Serial.println("Firmware v1.0");
  Serial.println();
  
  // Initialize I2C bus
  Wire.begin();
  Wire.setClock(400000);  // 400kHz fast I2C
  
  // Initialize PCA9685 boards
  Serial.println("Initializing PCA9685 servo drivers...");
  
  pca9685_boards[0] = Adafruit_PWMServoDriver(PCA9685_ADDR_1);
  pca9685_boards[1] = Adafruit_PWMServoDriver(PCA9685_ADDR_2);
  
  for (int i = 0; i < NUM_BOARDS; i++) {
    if (!pca9685_boards[i].begin()) {
      Serial.print("ERROR: PCA9685 board ");
      Serial.print(i);
      Serial.println(" not found!");
    } else {
      pca9685_boards[i].setPWMFreq(SERVO_FREQ);
      Serial.print("  Board ");
      Serial.print(i);
      Serial.println(" initialized");
    }
    delay(10);
  }
  
  // Initialize MPU sensors
  Serial.println("Initializing MPU-9250 sensors...");
  
  uint8_t mpu_addrs[] = {MPU_ADDR_1, MPU_ADDR_2, MPU_ADDR_3};
  
  for (int i = 0; i < 3; i++) {
    mpu_sensors[i] = Adafruit_MPU6050();
    
    if (!mpu_sensors[i].begin(mpu_addrs[i], &Wire)) {
      Serial.print("WARNING: MPU at 0x");
      Serial.print(mpu_addrs[i], HEX);
      Serial.println(" not found");
    } else {
      // Configure MPU
      mpu_sensors[i].setAccelerometerRange(MPU6050_RANGE_8_G);
      mpu_sensors[i].setGyroRange(MPU6050_RANGE_500_DEG);
      mpu_sensors[i].setFilterBandwidth(MPU6050_BAND_21_HZ);
      
      // Initialize Madgwick filter
      madgwick_filters[i].begin(CONTROL_RATE_HZ);
      
      Serial.print("  MPU ");
      Serial.print(i);
      Serial.println(" initialized");
    }
    delay(10);
  }
  
  // Initialize panel states
  for (int i = 0; i < NUM_PANELS; i++) {
    panel_states[i].panel_id = i;
    for (int j = 0; j < 3; j++) {
      panel_states[i].servo_angles[j] = 90.0;
      panel_states[i].orientation[j] = 0.0;
    }
    panel_states[i].timestamp = 0;
  }
  
  // Center all servos
  delay(1000);
  centerAllServos();
  
  Serial.println("\nSystem ready!");
  Serial.println("Commands: SET_SERVO, CENTER_ALL, STATUS, PING");
  Serial.println("===================================\n");
  
  last_control_update = micros();
  last_imu_update = millis();
}

// ===== MAIN LOOP =====

void loop() {
  unsigned long current_micros = micros();
  unsigned long current_millis = millis();
  
  // Control loop at fixed rate
  if (current_micros - last_control_update >= CONTROL_PERIOD_US) {
    last_control_update = current_micros;
    
    // Update IMU data at 100Hz
    if (current_millis - last_imu_update >= 10) {
      updateIMUData();
      last_imu_update = current_millis;
    }
  }
  
  // Process serial commands
  while (Serial.available()) {
    char c = Serial.read();
    
    if (c == '\n' || c == '\r') {
      if (cmd_index > 0) {
        cmd_buffer[cmd_index] = '\0';
        parseCommand(String(cmd_buffer));
        cmd_index = 0;
      }
    }
    else if (cmd_index < CMD_BUFFER_SIZE - 1) {
      cmd_buffer[cmd_index++] = c;
    }
  }
  
  // Optional: LED heartbeat
  static unsigned long last_blink = 0;
  if (current_millis - last_blink > 1000) {
    digitalWrite(LED_BUILTIN, !digitalRead(LED_BUILTIN));
    last_blink = current_millis;
  }
}

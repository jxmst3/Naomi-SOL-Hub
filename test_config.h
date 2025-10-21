// Naomi SOL Configuration
// Auto-generated from system_config.json
// Generated: 2025-10-20T21:40:39.830012

#ifndef NAOMI_CONFIG_H
#define NAOMI_CONFIG_H

// System
#define FIRMWARE_VERSION "3.0"
#define PANEL_COUNT 12
#define SERVO_COUNT 36

// Hardware
#define I2C_SPEED 400000L
#define PWM_FREQUENCY 50
#define BAUD_RATE 115200L

// Sensors
#define IMU_SAMPLE_RATE 100
#define MADGWICK_BETA 0.041f

// Control
#define PID_P 2.0f
#define PID_I 0.5f
#define PID_D 0.1f
#define MAX_TILT_ANGLE 15.0f
#define SERVO_MIN_ANGLE 60.0f
#define SERVO_MAX_ANGLE 120.0f

// Update rates
#define SENSOR_UPDATE_RATE 100
#define CONTROL_UPDATE_RATE 100
#define BLE_UPDATE_RATE 10

// Safety
#define MAX_TEMPERATURE 60.0f
#define MIN_BATTERY_VOLTAGE 10.8f
#define WATCHDOG_TIMEOUT 5000

// Design parameters
#define PANEL_SIDE_LENGTH 160f
#define PANEL_THICKNESS 4.0f
#define MIRROR_DIAMETER 70.0f

#endif // NAOMI_CONFIG_H

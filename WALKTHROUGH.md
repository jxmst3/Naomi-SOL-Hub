# 🚀 NAOMI SOL HUB - COMPLETE WALKTHROUGH

## Setup and Use Guide - Step by Step

This guide will walk you through setting up and using Naomi SOL Hub, from complete beginner to advanced user.

---

## TABLE OF CONTENTS

1. [Prerequisites & System Requirements](#prerequisites)
2. [Installation - Virtual Mode](#installation-virtual)
3. [First Run - Virtual Simulation](#first-run-virtual)
4. [Understanding the System](#understanding-the-system)
5. [CAD Generation](#cad-generation)
6. [Hardware Setup](#hardware-setup)
7. [Physical Assembly](#physical-assembly)
8. [Firmware Upload](#firmware-upload)
9. [Calibration](#calibration)
10. [Full System Operation](#full-operation)
11. [Cloud Integration](#cloud-integration)
12. [Troubleshooting](#troubleshooting)

---

<a name="prerequisites"></a>
## 1. PREREQUISITES & SYSTEM REQUIREMENTS

### Computer Requirements

**Minimum:**
- Windows 10/11, macOS 10.15+, or Linux
- Intel i5 or equivalent (4 cores)
- 8 GB RAM
- 5 GB free disk space
- Integrated graphics (Intel HD 4000+)

**Recommended:**
- Intel i7/AMD Ryzen 5 or better (8 cores)
- 16 GB RAM
- 20 GB free disk space (for CAD files)
- Dedicated GPU (NVIDIA/AMD for physics simulation)

### Software Prerequisites

**Required:**
```
✓ Python 3.10 or higher
✓ Visual Studio Code (or any IDE)
✓ Git (optional, for version control)
```

**Optional:**
```
○ Arduino IDE 2.0+ (for hardware)
○ OpenSCAD (for CAD editing)
○ Anaconda (for heavy packages)
○ Docker (for n8n cloud integration)
```

### Hardware Prerequisites (Optional - Virtual mode doesn't need these)

**For Full Physical Build:**
```
□ 3D Printer (or access to one)
□ Soldering iron + supplies
□ Multimeter
□ Basic hand tools
□ Arduino Portenta H7 or Nano 33 BLE
□ Electronic components (see bill of materials)
```

---

<a name="installation-virtual"></a>
## 2. INSTALLATION - VIRTUAL MODE

### Step 2.1: Get the Files

**Option A: Download ZIP**
1. Download `NaomiSOL_Ultimate_Final.zip`
2. Extract to `C:\Users\YourName\Projects\NaomiSOL`
3. Open folder in File Explorer

**Option B: Git Clone**
```powershell
cd C:\Users\YourName\Projects
git clone https://github.com/yourusername/naomi-sol-hub.git NaomiSOL
cd NaomiSOL
```

### Step 2.2: Open in Visual Studio Code

1. Launch Visual Studio Code
2. File → Open Folder
3. Navigate to `NaomiSOL` folder
4. Click "Select Folder"

You should see:
```
NaomiSOL/
├── main.py
├── requirements.txt
├── README.md
├── sim/
├── ai/
├── visualizer/
├── naomi/
├── hardware/
└── ... (other folders)
```

### Step 2.3: Create Virtual Environment

**In VS Code Terminal (Ctrl + `):**

```powershell
# Create virtual environment
python -m venv .venv

# Activate it (PowerShell)
.\.venv\Scripts\Activate.ps1

# If you get execution policy error:
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned

# Try activating again
.\.venv\Scripts\Activate.ps1
```

**You should see `(.venv)` appear in your terminal prompt:**
```
(.venv) PS C:\Users\YourName\Projects\NaomiSOL>
```

### Step 2.4: Install Python Packages

```powershell
# Upgrade pip first
python -m pip install --upgrade pip

# Install core packages (this takes 5-10 minutes)
pip install pygame PyOpenGL pybullet numpy scipy networkx

# Install AI packages
pip install torch  # This is large, ~2GB

# Install visualization
pip install matplotlib pandas

# Install optional packages (can skip if issues)
pip install rich tqdm python-dotenv
```

**If you get errors**, that's okay! Most packages are optional. Continue to next step.

### Step 2.5: Verify Installation

```powershell
# Run quick test
python main.py --mode test
```

**Expected Output:**
```
╔══════════════════════════════════════════════════════════════════════════════╗
║                    NAOMI SOL HUB - ULTIMATE INTEGRATION                       ║
║                           Version 4.0 - Final Release                         ║
╚══════════════════════════════════════════════════════════════════════════════╝

[INFO] Running system tests...
[INFO] ✓ Shape Logic Simulator initialized
[INFO] ✓ SwarmLords AI Controller initialized
[INFO] ✓ Mock Hardware Controller initialized
[INFO] ✓ Tests complete
```

**If you see this, CONGRATULATIONS! Installation successful!** 🎉

---

<a name="first-run-virtual"></a>
## 3. FIRST RUN - VIRTUAL SIMULATION

### Step 3.1: Launch the System

```powershell
python main.py
```

**What happens:**
1. System initializes all components (~5 seconds)
2. Window opens showing simulation
3. Background threads start (optimization, physics)
4. You see real-time visualization!

### Step 3.2: Understanding the Interface

**Window Layout:**

```
┌─────────────────────────────────────────────────────────────┐
│  NAOMI SOL HUB v4.0                               [_][□][X] │
├─────────────────────────────────────────────────────────────┤
│ ┌──────────────────────┐ ┌──────────────────────┐          │
│ │                      │ │                      │          │
│ │   Shape Logic Grid   │ │   3D Physics View    │          │
│ │                      │ │                      │          │
│ │  (Colorful squares)  │ │  (Rotating cube)     │          │
│ │                      │ │                      │          │
│ └──────────────────────┘ └──────────────────────┘          │
│                                                              │
│ ┌────────────────────────────────────────────────────────┐ │
│ │  Real-Time Graphs:                                      │ │
│ │  - Orientation (roll/pitch)                             │ │
│ │  - Light intensity                                      │ │
│ │  - Anomaly score                                        │ │
│ │  - Frequency spectrum                                   │ │
│ └────────────────────────────────────────────────────────┘ │
│                                                              │
│ Status: Running | FPS: 60 | Panels: 12 | Playbook: 3       │
└─────────────────────────────────────────────────────────────┘
```

### Step 3.3: Interactive Controls

**Keyboard Shortcuts:**

| Key | Action | Description |
|-----|--------|-------------|
| **SPACE** | Pause/Resume | Freeze/unfreeze simulation |
| **R** | Reset | Start over from beginning |
| **I** | Interactive Mode | Manually approve AI proposals |
| **G** | Generate CAD | Create STL files now |
| **S** | Screenshot | Save current view |
| **F** | FPS Toggle | Show/hide frame rate |
| **ESC** | Exit | Close program |

### Step 3.4: Watch the AI Work

In the terminal, you'll see:

```
[INFO] Optimization loop started
[INFO] Proposal from Strength: {'side_length': 151.2, 'thickness': 4.1}
[INFO] Score: 0.847
[INFO] Proposal from Tilt: {'side_length': 149.8, 'thickness': 4.2}
[INFO] Score: 0.851 ← Better!
[INFO] ✓ New best design accepted
[INFO] Generating CAD...
[INFO] ✓ CAD exported: output/cad_models/optimized_panel_1729456789.stl
```

The AI is continuously improving the design!

### Step 3.5: Interactive Mode

Press **I** to enable interactive mode:

```
Interactive Mode: ON

Proposal from Weight: {'side_length': 150.5, 'thickness': 3.9}
Score: 0.863
Accept this proposal? (y/n):
```

Type `y` and press Enter to accept, or `n` to reject.

**This lets you guide the AI!**

### Step 3.6: Generate CAD Files

Press **G** or:

```powershell
python main.py --mode generate-cad
```

Files appear in `output/cad_models/`:
```
Pentagon_Base_Panel.stl
Mirror_Platform.stl
Servo_Mount.stl
... (more files)
```

**These are ready for 3D printing!**

### Step 3.7: Exit

Press **ESC** or Ctrl+C in terminal.

```
[INFO] Shutting down Naomi SOL Hub...
[INFO] ✓ Shutdown complete
```

---

<a name="understanding-the-system"></a>
## 4. UNDERSTANDING THE SYSTEM

### What's Happening Behind the Scenes?

**Thread 1: Shape Logic Simulation**
- Updates grid every frame
- Calculates polarity interactions
- Accumulates residues
- Sends state to visualizer

**Thread 2: SwarmLords Optimization**
- 10 AI agents propose designs
- Each agent has a specialty
- Best design is selected
- ACE system learns from results

**Thread 3: PyBullet Physics**
- Simulates real-world physics
- Tests structural integrity
- Calculates forces and torques
- Validates servo loads

**Thread 4: Hardware Interface** (Mock in virtual mode)
- Receives sensor data
- Sends control commands
- Logs all communications
- Streams to cloud (if enabled)

**Main Thread: Visualization**
- Renders OpenGL graphics
- Processes user input
- Updates displays
- Manages UI

### The AI Learning Process

```
1. GENERATE (10 agents create proposals)
   ↓
2. EVALUATE (Neural network predicts quality)
   ↓
3. SELECT (Best design chosen)
   ↓
4. VALIDATE (Physics simulation confirms)
   ↓
5. REFLECT (ACE analyzes what worked)
   ↓
6. CURATE (Best strategies saved to playbook)
   ↓
7. REPEAT (Now smarter for next iteration!)
```

**The system literally gets smarter over time!**

### Data Flow

```
Sensors → Arduino → Bluetooth → Python → {
    → Visualizer (display)
    → SwarmLords (optimize)
    → Database (store)
    → Cloud (sync)
    → Logs (record)
}
```

---

<a name="cad-generation"></a>
## 5. CAD GENERATION

### Understanding Parametric Design

Your design is defined by parameters:

```python
{
    "side_length": 150.0,      # Pentagon edge in mm
    "thickness": 4.0,          # Panel wall thickness
    "pocket_depth": 24.0,      # Servo pocket depth
    "mirror_diameter": 70.0,   # Mirror platform size
    "servo_spacing": 45.0,     # Distance between servos
    "infill_percentage": 30    # 3D print infill
}
```

**Change these → Different STL file!**

### Generate All Parts

```powershell
python main.py --mode generate-cad
```

**Output:**
```
[INFO] Generating Pentagon_Base_Panel...
[INFO]   ✓ Exported: Pentagon_Base_Panel.stl (4.2 MB)
[INFO] Generating Mirror_Platform...
[INFO]   ✓ Exported: Mirror_Platform.stl (1.8 MB)
[INFO] Generating Servo_Mount...
[INFO]   ✓ Exported: Servo_Mount.stl (0.9 MB)
...
[INFO] ✓ Generated 15 CAD files
```

### Customize Parameters

Edit `main.py`:

```python
# Around line 450
base_design = {
    "side_length": 160.0,  # Make it bigger!
    "thickness": 5.0,      # Make it stronger!
    "pocket_depth": 26.0,  # Deeper pockets
    "mirror_diameter": 80.0  # Larger mirror
}
```

Run again → New STL files with your dimensions!

### View STL Files

**Free STL Viewers:**
- Windows 3D Viewer (built-in)
- [MeshLab](https://www.meshlab.net/)
- [Cura Slicer](https://ultimaker.com/software/ultimaker-cura)
- [PrusaSlicer](https://www.prusa3d.com/page/prusaslicer_424/)

**Open Pentagon_Base_Panel.stl:**
- Right-click → Open with → 3D Viewer
- Rotate with mouse
- Zoom with scroll wheel

---

<a name="hardware-setup"></a>
## 6. HARDWARE SETUP

*This section is for building the physical system. Skip if staying virtual.*

### Bill of Materials

**Electronics (~$300):**

| Part | Qty | Price | Source |
|------|-----|-------|--------|
| Arduino Portenta H7 | 1 | $90 | Arduino Store |
| Arduino Nano 33 BLE | 1 | $25 | Amazon |
| PCA9685 Servo Driver | 3 | $10 ea | Amazon |
| MG90S Metal Servo | 36 | $5 ea | AliExpress |
| MPU-9250 IMU | 12 | $8 ea | Amazon |
| TSL2561 Light Sensor | 12 | $3 ea | AliExpress |
| Piezo Sensor | 12 | $1 ea | Amazon |
| NEMA 17 Stepper | 1 | $15 | Amazon |
| TMC2208 Driver | 1 | $8 | Amazon |
| N20 DC Motor | 1 | $5 | Amazon |
| Slip Ring 12-wire | 1 | $12 | Amazon |
| Power Supply 6V 10A | 1 | $18 | Amazon |
| Misc (wire, connectors) | - | $40 | Various |

**3D Printing (~$150):**
- 5 kg PETG filament (main panels)
- 1 kg PLA filament (small parts)

**Mechanical (~$100):**
- Lazy Susan bearing (6")
- M3 screws/nuts assortment
- Ball joints (108x)
- Bearings
- Kevlar fishing line
- Diamond mirror tiles

**Total: ~$650**

### Ordering Strategy

**Week 1: Order from Amazon (Fast shipping)**
- Arduinos
- Servo drivers
- Power supply
- Tools

**Week 1: Order from AliExpress (Slow but cheap)**
- Servos (36x)
- IMUs (12x)
- Light sensors (12x)

**Week 2: Start 3D printing**
- Print while waiting for parts
- Test print one panel first

**Week 3-4: Parts arrive**
- Check everything
- Test components individually

### Testing Each Component

**Test Servo:**
```cpp
#include <Servo.h>
Servo testServo;

void setup() {
    testServo.attach(9);
}

void loop() {
    testServo.write(0);
    delay(1000);
    testServo.write(90);
    delay(1000);
    testServo.write(180);
    delay(1000);
}
```

**Test IMU:**
```cpp
#include <MPU9250.h>
MPU9250 IMU(Wire, 0x68);

void setup() {
    Serial.begin(115200);
    IMU.begin();
}

void loop() {
    IMU.readSensor();
    Serial.print("Roll: ");
    Serial.println(IMU.getRoll());
    delay(100);
}
```

**Test PCA9685:**
```cpp
#include <Adafruit_PWMServoDriver.h>
Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();

void setup() {
    pwm.begin();
    pwm.setPWMFreq(50);
}

void loop() {
    pwm.setPWM(0, 0, 375);  // Servo to center
    delay(1000);
}
```

---

<a name="physical-assembly"></a>
## 7. PHYSICAL ASSEMBLY

### Assembly Station Setup

**Workspace:**
```
┌─────────────────────────────────────────────────────────────┐
│                    Your Desk/Table                          │
│                                                              │
│  [Soldering Iron]  [Parts Bins]  [Hand Tools]              │
│                                                              │
│  [Anti-static Mat] [Helping Hands] [Multimeter]            │
│                                                              │
│  [Wire Cutters]    [Label Maker]  [Cable Ties]             │
│                                                              │
│  [Computer with Arduino IDE and Documentation]              │
└─────────────────────────────────────────────────────────────┘
```

### Panel Assembly (Repeat 12x)

**Per Panel Assembly Time: ~2 hours**

#### Step 7.1: Servo Installation

1. **Prepare Servo Pockets**
   ```
   - Check fit (servo should slide in snugly)
   - If too tight, sand lightly
   - If too loose, add tape padding
   ```

2. **Install Servos**
   ```
   a. Insert servo into pocket (wire side up)
   b. Align mounting holes
   c. Insert M3×10mm screws
   d. Tighten (don't over-tighten!)
   e. Attach servo horn (center position)
   ```

3. **Repeat for All 3 Servos**
   ```
   Servo positions:
   - 0° (bottom)
   - 120° (left)
   - 240° (right)
   ```

#### Step 7.2: Sensor Installation

1. **IMU Mounting**
   ```
   - Position at panel center
   - Align X-axis forward
   - Secure with M2×8mm screws
   - Ensure level (use bubble level)
   ```

2. **Light Sensor**
   ```
   - Mount near mirror position
   - Point sensor inward
   - Secure with hot glue or screws
   ```

3. **Piezo Sensor**
   ```
   - Attach to panel back
   - Good mechanical coupling
   - Hot glue works well
   ```

#### Step 7.3: Wiring

**Color Code (Recommended):**
```
Power:
- Red: +6V (servo power)
- Black: GND
- Orange: +3.3V (sensors)

Data:
- Yellow: SDA (I2C data)
- Green: SCL (I2C clock)
- Blue: Servo PWM
- White: Analog signals
```

**Wire Routing:**
1. Keep servo wires short (~15cm)
2. Bundle sensor wires together
3. Use spiral wrap or heat shrink
4. Label every connection!

**Example Label:**
```
P1-S1: Panel 1, Servo 1 (0°)
P1-IMU: Panel 1, IMU sensor
P1-LIGHT: Panel 1, Light sensor
```

#### Step 7.4: BaBot Platform Assembly

1. **Connecting Rods**
   ```
   Length: 28mm (adjust for your geometry)
   
   Assembly:
   a. Screw rod into servo horn
   b. Attach ball joint to other end
   c. Check free rotation
   d. Repeat for all 3 rods
   ```

2. **Platform Attachment**
   ```
   a. Position platform above servos
   b. Connect ball joints to platform mounts
   c. Verify all 3 connections secure
   d. Test tilt motion
   ```

3. **Movement Test**
   ```
   - All servos to 90° → Platform level
   - Servo 1 to 75° → Platform tilts
   - All servos coordinated → 2D tilt
   ```

#### Step 7.5: Mirror Installation

1. **Surface Preparation**
   ```
   - Clean platform with isopropyl alcohol
   - Ensure completely dry
   - No dust or fingerprints
   ```

2. **Tile Layout**
   ```
   - Plan 20mm×20mm tile positions
   - Leave 0.5mm gaps
   - Mark positions lightly with pencil
   ```

3. **Adhesive Application**
   ```
   - Use UV-cure optical cement
   - Apply thin layer to platform
   - Place tiles carefully
   - Press gently (no sliding!)
   ```

4. **UV Curing**
   ```
   - Use UV flashlight or lamp
   - 5 minutes per area
   - Check edges are secure
   - Clean excess cement
   ```

5. **Quality Check**
   ```
   - Look for gaps (should be minimal)
   - Check flatness with straight edge
   - Test reflection quality
   ```

### Complete Panel Checklist

□ 3 servos installed and secured  
□ IMU mounted and level  
□ Light sensor positioned  
□ Piezo sensor attached  
□ All wires color-coded and labeled  
□ BaBot platform assembled  
□ Platform movement tested  
□ Mirrors installed and cured  
□ Electrical connections verified  
□ Visual inspection passed  

**Repeat for all 12 panels!**

### Dodecahedron Assembly

**Assembly Time: ~4 hours**

#### Pentagon Connection Pattern

```
        (Top)
          1
       /  |  \
      2   3   4
       \ / \ /
        5   6
       / \ / \
      7   8   9
       \  |  /
         10
       /  |  \
     11  12 (Bottom)
```

#### Assembly Steps

1. **First Pentagon Ring (Panels 2-6)**
   ```
   a. Lay panel 2 flat
   b. Attach panel 3 at 108° angle
   c. Add panel 4
   d. Continue around
   e. Close ring with panel 6 to 2
   ```

2. **Add Top (Panel 1)**
   ```
   a. Position panel 1 above ring
   b. Connect to panels 2,3,4,5,6
   c. Verify angles (use protractor)
   d. Tighten screws incrementally
   ```

3. **Second Ring (Panels 7-11)**
   ```
   a. Attach to bottom of first ring
   b. Offset from top ring
   c. Maintain dihedral angles
   ```

4. **Bottom Cap (Panel 12)**
   ```
   - Make this one hinged!
   - Allows access to interior
   - Use small hinges
   - Add magnetic latch
   ```

#### Alignment Tips

**Use Jig:**
```
Build a simple wooden jig:
- 5 pieces at 108° angles
- Holds panels during assembly
- Ensures consistent geometry
```

**Check Frequently:**
- Measure opposite panel distances
- Should all be equal
- Adjust before final tightening

### Central Suspension

1. **Top Bearing Mount**
   ```
   - 3D print bearing holder
   - Install small bearing (608ZZ works)
   - Secure to dodecahedron top
   ```

2. **Thread the Line**
   ```
   - Use 0.3mm Kevlar fishing line
   - Thread through bearing
   - Attach N20 motor above
   - Crystal attaches below
   ```

3. **Crystal Mounting**
   ```
   - Drill small hole in crystal (carefully!)
   - OR use wire wrap method
   - Ensure balanced (center of mass)
   - Test spin - should be smooth
   ```

### Rotation Base

1. **Lazy Susan Installation**
   ```
   - Attach to base plate (wood/acrylic)
   - Center under dodecahedron
   - Verify smooth rotation
   ```

2. **Stepper Motor**
   ```
   - Mount NEMA 17 to base
   - Align drive gear
   - Connect with belt/gear
   - Test slow rotation
   ```

3. **Slip Ring**
   ```
   - Mount to rotating axis
   - Connect all 12 panel wires
   - Route to stationary side
   - Label every wire!
   ```

---

<a name="firmware-upload"></a>
## 8. FIRMWARE UPLOAD

### Arduino Portenta H7 Setup

#### Step 8.1: Install Board Support

1. Open Arduino IDE 2.0+
2. Tools → Board Manager
3. Search "Portenta"
4. Install "Arduino Mbed OS Portenta Boards"
5. Wait for installation (~5 minutes)

#### Step 8.2: Install Libraries

Tools → Manage Libraries, install:
```
- Adafruit PWM Servo Driver Library
- MPU9250 by Bolder Flight Systems
- MadgwickAHRS
- ArduinoBLE
- PID by Brett Beauregard
- ArduinoJson
```

#### Step 8.3: Open Firmware

File → Open → Navigate to:
```
NaomiSOL_Ultimate_Final/firmware/NaomiSOL_Firmware.ino
```

#### Step 8.4: Configure

Edit these lines if needed:
```cpp
// Line 20-25: Verify your hardware
#define PANEL_COUNT 12
#define SERVOS_PER_PANEL 3

// Line 35-40: I2C addresses (match your hardware)
#define PCA9685_ADDR_1 0x40
#define PCA9685_ADDR_2 0x41
#define PCA9685_ADDR_3 0x42

// Line 50: BLE name
#define BLE_NAME "NaomiSOL"
```

#### Step 8.5: Upload

1. Connect Portenta H7 via USB-C
2. Tools → Board → Arduino Portenta H7 (M7 core)
3. Tools → Port → (select your port)
4. Click ➡️ Upload button
5. Wait ~30 seconds
6. Look for "Done uploading"

#### Step 8.6: Verify

1. Tools → Serial Monitor
2. Set baud rate: 115200
3. Press reset button on Portenta

**You should see:**
```
=============================================
NAOMI SOL FIRMWARE v3.0 - INITIALIZING
=============================================
[OK] I2C initialized at 400kHz
[OK] Servos initialized (36/36)
[OK] IMUs initialized (12/12)
[OK] BLE advertising as: NaomiSOL
[OK] System ready!
```

**If you see errors, check:**
- Wire connections
- I2C addresses
- Power supply

---

<a name="calibration"></a>
## 9. CALIBRATION

### IMU Calibration

```powershell
python tools/calibrate_imus.py --port COM3
```

**Process:**
1. Place system on perfectly level surface
2. Keep absolutely still
3. Script reads 100 samples per sensor
4. Calculates offsets
5. Saves to `calibration/imu_offsets.json`

**Verify:**
```
Panel 1: Roll: 0.2°, Pitch: -0.1° ✓
Panel 2: Roll: -0.3°, Pitch: 0.2° ✓
...
All sensors within ±0.5° tolerance ✓
```

### Servo Calibration

```powershell
python tools/calibrate_servos.py --port COM3
```

**Process:**
1. Moves each servo through full range
2. Detects mechanical limits
3. Tests response time
4. Saves profiles

**Output:**
```
Servo 0: Range 10-170°, Center: 90°, Response: 120ms ✓
Servo 1: Range 12-168°, Center: 91°, Response: 118ms ✓
...
```

### Optical Calibration

```powershell
python tools/calibrate_optics.py --port COM3
```

**Process:**
1. Turn off room lights
2. Activate laser
3. Reads all light sensors
4. Records baseline + range
5. Saves to calibration file

**Expected:**
```
Sensor 0: Baseline 523 lux, Range 0-2500 lux ✓
Sensor 1: Baseline 518 lux, Range 0-2480 lux ✓
...
```

### Full System Test

```powershell
python tools/system_test.py --port COM3
```

**Tests:**
1. All servos move ✓
2. All IMUs report data ✓
3. All light sensors respond ✓
4. BLE connection stable ✓
5. Data rate >95 Hz ✓
6. No I2C errors ✓

---

<a name="full-operation"></a>
## 10. FULL SYSTEM OPERATION

### Starting the Complete System

1. **Power On**
   ```
   a. Plug in 6V power supply
   b. Connect Arduino via USB (for programming)
   c. Wait for firmware to boot (~5 sec)
   ```

2. **Connect Computer**
   ```powershell
   # Activate venv
   .\.venv\Scripts\Activate.ps1
   
   # Connect via BLE (default)
   python main.py
   
   # OR connect via Serial
   python main.py --serial-port COM3
   ```

3. **Verify Connection**
   ```
   [INFO] ✓ Hardware connected
   [INFO] ✓ Receiving sensor data
   [INFO] ✓ All 12 panels online
   ```

4. **System Running!**
   - Visualization shows real sensor data
   - Servos respond to commands
   - Optimization running
   - Data logging to files

### Running Experiments

#### Experiment 1: Baseline Measurement

```
Goal: Establish normal operating parameters

Steps:
1. No laser active
2. Chamber rotating at 1 RPM
3. Crystal rotating at 90 RPM
4. Record for 10 minutes
5. Note baseline values
```

#### Experiment 2: Laser Stimulation

```
Goal: Observe laser-crystal interaction

Steps:
1. Activate 405nm laser (low power)
2. Aim at spinning crystal
3. Observe light sensor changes
4. Look for patterns
5. Increase power gradually
```

#### Experiment 3: Perturbation Response

```
Goal: Test system stability

Steps:
1. Introduce airflow (fans)
2. Vary chamber rotation speed
3. Change crystal speed
4. Monitor anomaly scores
5. Check recovery time
```

### Reading the Data

**Real-Time Display:**
```
Top graphs: Orientation (tilting of panels)
Middle: Light intensity (optical activity)
Bottom: Anomaly score (unusual events)
```

**Interpreting Anomaly Scores:**
```
0.0 - 0.3: Normal operation ✓
0.3 - 0.7: Interesting patterns 🤔
0.7 - 1.0: Highly anomalous! ⚠️
```

**When anomaly detected:**
1. System saves detailed snapshot
2. Sends alert (if cloud enabled)
3. Increases data logging rate
4. Highlights in visualization

### Data Files

**Location:** `data/`

**Session Log:** `session_20241020_143022.json`
```json
{
  "session_id": "session_20241020_143022",
  "start_time": "2024-10-20T14:30:22Z",
  "samples": 12000,
  "anomalies_detected": 3,
  "highest_anomaly_score": 0.847
}
```

**Raw Data:** `raw_data_20241020.csv`
```csv
timestamp,panel_id,roll,pitch,light,anomaly
1729436422.1,0,0.2,-0.1,523,0.12
1729436422.2,1,-0.3,0.2,518,0.14
...
```

---

<a name="cloud-integration"></a>
## 11. CLOUD INTEGRATION

### Setup n8n

1. **Install Node.js**
   - Download from nodejs.org
   - Install (default options)

2. **Install n8n**
   ```bash
   npm install -g n8n
   ```

3. **Start n8n**
   ```bash
   n8n start
   ```

4. **Open Dashboard**
   - Browser: `http://localhost:5678`
   - Create account

5. **Import Workflow**
   - Menu → Import
   - Select `workflows/naomi_sol_complete.json`
   - Click Import

6. **Configure**
   - Edit webhook URL
   - Add credentials (Google, PostgreSQL, etc.)
   - Test webhook

7. **Run with Cloud**
   ```powershell
   python main.py --enable-cloud --webhook-url http://localhost:5678/webhook/naomi-sol
   ```

---

<a name="troubleshooting"></a>
## 12. TROUBLESHOOTING

### Problem: Can't find Arduino

**Check:**
```powershell
# List COM ports
mode

# Windows Device Manager
devmgmt.msc
→ Ports (COM & LPT)
```

**Solution:**
```powershell
# Specify port explicitly
python main.py --serial-port COM5
```

### Problem: BLE not connecting

**Solutions:**
1. Check Bluetooth is on
2. Restart Arduino
3. Restart computer Bluetooth
4. Use Serial instead:
   ```powershell
   python main.py --serial-port COM3
   ```

### Problem: Servo not moving

**Check:**
1. ✓ 6V power connected?
2. ✓ PCA9685 powered?
3. ✓ Servo plugged in correctly?
4. ✓ Channel number correct?

**Test individually:**
```cpp
// Upload this test sketch
pwm.setPWM(0, 0, 375);  // Channel 0, center
```

### Problem: CAD generation fails

**Likely cause:** CadQuery not installed

**Solutions:**
1. Install via conda:
   ```bash
   conda install -c conda-forge cadquery
   ```

2. Or use alternative STL generation:
   ```powershell
   python tools/generate_stl_basic.py
   ```

### Problem: High CPU usage

**Solutions:**
1. Run headless:
   ```powershell
   python main.py --headless
   ```

2. Reduce rates in `main.py`:
   ```python
   SENSOR_UPDATE_RATE = 50  # Was 100
   CONTROL_UPDATE_RATE = 50  # Was 100
   ```

3. Disable physics:
   ```python
   # Comment out in main.py:
   # self.start_physics_simulation()
   ```

---

## CONGRATULATIONS! 🎉

You've completed the Naomi SOL Hub walkthrough!

**What you've learned:**
✓ Virtual simulation
✓ CAD generation
✓ Hardware assembly
✓ Firmware programming
✓ System calibration
✓ Full operation
✓ Cloud integration
✓ Troubleshooting

**Next steps:**
- Run experiments
- Collect data
- Analyze patterns
- Share discoveries
- Improve designs
- Join community

**Your Naomi SOL Hub is ready!** 🚀

---

**Need more help?**
- Read README.md
- Check docs/ folder
- View examples/
- GitHub issues
- Community forum

**Happy experimenting!** 🌟

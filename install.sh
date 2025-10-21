#!/bin/bash
#
# Naomi SOL Hub - Automated Installation Script
# ==============================================
# This script sets up the complete Naomi SOL Hub environment
# including all dependencies from open-source libraries
#
# Usage: ./install.sh
#

set -e  # Exit on error

echo "========================================"
echo "Naomi SOL Hub Installation"
echo "========================================"
echo ""

# Detect OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
    echo "Detected OS: Linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="mac"
    echo "Detected OS: macOS"
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
    OS="windows"
    echo "Detected OS: Windows (Git Bash/Cygwin)"
else
    echo "Unknown OS: $OSTYPE"
    echo "Manual installation may be required"
    OS="unknown"
fi

echo ""

# Check Python version
echo "Checking Python installation..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    echo "Python $PYTHON_VERSION found"
    
    # Check if version is >= 3.8
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
    
    if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
        echo "ERROR: Python 3.8 or higher required"
        exit 1
    fi
else
    echo "ERROR: Python 3 not found"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

echo ""

# Install system dependencies
echo "Installing system dependencies..."
if [ "$OS" == "linux" ]; then
    echo "Updating package list..."
    sudo apt update
    
    echo "Installing I2C tools..."
    sudo apt install -y i2c-tools python3-smbus
    
    echo "Installing OpenSCAD (optional, for CAD generation)..."
    read -p "Install OpenSCAD? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        sudo apt install -y openscad
    fi
    
    echo "Installing build tools..."
    sudo apt install -y build-essential cmake git
    
elif [ "$OS" == "mac" ]; then
    if ! command -v brew &> /dev/null; then
        echo "Homebrew not found. Please install from https://brew.sh"
        exit 1
    fi
    
    echo "Installing OpenSCAD (optional, for CAD generation)..."
    read -p "Install OpenSCAD? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        brew install openscad
    fi
    
    echo "Installing I2C tools..."
    brew install i2c-tools
fi

echo ""

# Create virtual environment
echo "Creating Python virtual environment..."
if [ -d "venv" ]; then
    echo "Virtual environment already exists"
    read -p "Recreate? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf venv
        python3 -m venv venv
    fi
else
    python3 -m venv venv
fi

echo ""

# Activate virtual environment
echo "Activating virtual environment..."
if [ "$OS" == "windows" ]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

echo ""

# Install Python dependencies
echo "Installing Python dependencies..."
echo "This may take several minutes..."
pip install -r requirements.txt

echo ""

# Create necessary directories
echo "Creating directory structure..."
mkdir -p logs
mkdir -p recorded_data
mkdir -p models
mkdir -p cad_output
mkdir -p calibration

echo ""

# Download additional resources
echo "Downloading additional open-source resources..."

# Clone Stewart_Py if needed
if [ ! -d "external/Stewart_Py" ]; then
    echo "Cloning Stewart_Py..."
    mkdir -p external
    git clone https://github.com/Yeok-c/Stewart_Py.git external/Stewart_Py
fi

# Clone BOSL2 for OpenSCAD
if [ ! -d "external/BOSL2" ]; then
    echo "Cloning BOSL2..."
    mkdir -p external
    git clone https://github.com/revarbat/BOSL2.git external/BOSL2
fi

echo ""

# Test installation
echo "Testing installation..."
python3 -c "
import numpy
import cv2
import serial
print('✓ NumPy:', numpy.__version__)
print('✓ OpenCV:', cv2.__version__)
print('✓ PySerial:', serial.__version__)

try:
    import pybullet
    print('✓ PyBullet:', pybullet.getVersionString())
except:
    print('⚠ PyBullet: Not installed (optional)')

try:
    import stable_baselines3
    print('✓ Stable-Baselines3:', stable_baselines3.__version__)
except:
    print('⚠ Stable-Baselines3: Not installed (optional)')
"

echo ""
echo "========================================"
echo "Installation Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Upload firmware to Teensy 4.1:"
echo "   - Open Arduino IDE"
echo "   - Install libraries via Library Manager:"
echo "     • Adafruit PWM Servo Driver"
echo "     • Adafruit MPU6050"
echo "     • Adafruit BusIO"
echo "     • Adafruit Sensor"
echo "   - Open firmware/teensy_controller/teensy_controller.ino"
echo "   - Select Tools > Board > Teensy 4.1"
echo "   - Click Upload"
echo ""
echo "2. Configure hardware:"
echo "   - Edit config.yaml with your settings"
echo "   - Set correct serial ports"
echo "   - Configure I2C addresses"
echo ""
echo "3. Test in simulation mode:"
echo "   source venv/bin/activate  # or venv\\Scripts\\activate on Windows"
echo "   python naomi_hub.py --mode simulation"
echo ""
echo "4. Generate CAD files:"
echo "   python naomi_hub.py --generate-cad"
echo ""
echo "5. Run with hardware:"
echo "   python naomi_hub.py --mode hardware"
echo ""
echo "For more information, see README.md"
echo ""
echo "Happy building! 🚀"

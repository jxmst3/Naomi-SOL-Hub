#!/bin/bash
#
# Design Agent Launcher Script
# Provides automated setup, system checks, and service management
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"
VENV_DIR="$PROJECT_DIR/venv"

# ============================================================================
# Helper Functions
# ============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[✓]${NC} $*"
}

log_warning() {
    echo -e "${YELLOW}[!]${NC} $*"
}

log_error() {
    echo -e "${RED}[✗]${NC} $*"
}

# ============================================================================
# System Checks
# ============================================================================

check_python() {
    log_info "Checking Python installation..."
    
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 not found. Install with:"
        echo "  Ubuntu/Debian: sudo apt-get install python3 python3-venv"
        echo "  macOS: brew install python"
        return 1
    fi
    
    PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    log_success "Python $PYTHON_VERSION found"
    
    # Check version >= 3.10
    REQUIRED_VERSION="3.10"
    if (( $(echo "$PYTHON_VERSION < $REQUIRED_VERSION" | bc -l) )); then
        log_error "Python 3.10+ required, found $PYTHON_VERSION"
        return 1
    fi
    
    return 0
}

check_git() {
    log_info "Checking git installation..."
    
    if ! command -v git &> /dev/null; then
        log_warning "Git not found (optional, needed for cloning repos)"
        return 1
    fi
    
    GIT_VERSION=$(git --version | awk '{print $3}')
    log_success "Git $GIT_VERSION found"
    return 0
}

check_ffmpeg() {
    log_info "Checking FFmpeg installation..."
    
    if ! command -v ffmpeg &> /dev/null; then
        log_warning "FFmpeg not found (optional, for video/audio processing)"
        echo "  Ubuntu/Debian: sudo apt-get install ffmpeg"
        echo "  macOS: brew install ffmpeg"
        return 1
    fi
    
    FFMPEG_VERSION=$(ffmpeg -version 2>/dev/null | head -1 | awk '{print $3}')
    log_success "FFmpeg $FFMPEG_VERSION found"
    return 0
}

check_slic3r() {
    log_info "Checking slic3r installation..."
    
    if ! command -v slic3r &> /dev/null; then
        log_warning "slic3r not found (needed for 3D printing slicing)"
        echo "  Ubuntu/Debian: sudo apt-get install slic3r"
        echo "  macOS: brew install slic3r"
        return 1
    fi
    
    SLIC3R_VERSION=$(slic3r --version 2>/dev/null | head -1)
    log_success "slic3r found"
    return 0
}

check_gpu() {
    log_info "Checking GPU availability..."
    
    if command -v nvidia-smi &> /dev/null; then
        GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
        GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
        log_success "NVIDIA GPU found: $GPU_INFO (Count: $GPU_COUNT)"
        return 0
    else
        log_warning "No NVIDIA GPU detected (using CPU may be slow)"
        return 1
    fi
}

check_disk_space() {
    log_info "Checking disk space..."
    
    AVAILABLE=$(df "$PROJECT_DIR" | tail -1 | awk '{print $4}')
    AVAILABLE_GB=$((AVAILABLE / 1024 / 1024))
    
    if [ "$AVAILABLE_GB" -lt 20 ]; then
        log_warning "Low disk space: ${AVAILABLE_GB}GB (20GB+ recommended)"
        return 1
    fi
    
    log_success "Disk space: ${AVAILABLE_GB}GB available"
    return 0
}

check_memory() {
    log_info "Checking available memory..."
    
    AVAILABLE=$(free -m | grep Mem | awk '{print $7}')
    
    if [ "$AVAILABLE" -lt 8000 ]; then
        log_warning "Low memory: ${AVAILABLE}MB (8GB+ recommended)"
        return 1
    fi
    
    log_success "Memory: ${AVAILABLE}MB available"
    return 0
}

check_system() {
    log_info "Starting system check..."
    echo ""
    
    check_python || return 1
    check_git
    check_ffmpeg
    check_slic3r
    check_gpu
    check_disk_space
    check_memory
    
    echo ""
    log_success "System check complete!"
    return 0
}

# ============================================================================
# Virtual Environment Setup
# ============================================================================

setup_venv() {
    log_info "Setting up Python virtual environment..."
    
    if [ -d "$VENV_DIR" ]; then
        log_warning "Virtual environment already exists at $VENV_DIR"
        read -p "Remove and recreate? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$VENV_DIR"
        else
            return 0
        fi
    fi
    
    python3 -m venv "$VENV_DIR"
    log_success "Virtual environment created"
    
    # Activate venv for subsequent commands
    source "$VENV_DIR/bin/activate"
    
    log_info "Upgrading pip, setuptools, wheel..."
    pip install --upgrade pip setuptools wheel
    
    log_success "Virtual environment setup complete"
    return 0
}

# ============================================================================
# Dependency Installation
# ============================================================================

install_dependencies() {
    log_info "Installing Python dependencies..."
    
    if [ ! -f "$PROJECT_DIR/requirements.txt" ]; then
        log_error "requirements.txt not found in $PROJECT_DIR"
        return 1
    fi
    
    if [ ! -d "$VENV_DIR" ]; then
        log_error "Virtual environment not found. Run: $0 --setup-venv"
        return 1
    fi
    
    source "$VENV_DIR/bin/activate"
    
    log_info "Installing from requirements.txt..."
    pip install -r "$PROJECT_DIR/requirements.txt"
    
    log_success "Dependencies installed"
    return 0
}

# ============================================================================
# Application Launch
# ============================================================================

run_gui() {
    log_info "Launching GUI..."
    
    if [ ! -d "$VENV_DIR" ]; then
        log_error "Virtual environment not found. Run: $0 --setup-venv first"
        return 1
    fi
    
    source "$VENV_DIR/bin/activate"
    
    cd "$PROJECT_DIR"
    python3 design_agent_local.py --open_gui
}

run_cli() {
    log_info "Running CLI..."
    
    if [ ! -d "$VENV_DIR" ]; then
        log_error "Virtual environment not found. Run: $0 --setup-venv first"
        return 1
    fi
    
    source "$VENV_DIR/bin/activate"
    
    cd "$PROJECT_DIR"
    python3 design_agent_local.py "$@"
}

run_tests() {
    log_info "Running unit tests..."
    
    if [ ! -d "$VENV_DIR" ]; then
        log_error "Virtual environment not found. Run: $0 --setup-venv first"
        return 1
    fi
    
    source "$VENV_DIR/bin/activate"
    
    cd "$PROJECT_DIR"
    pytest test_design_agent_local.py -v
}

# ============================================================================
# Service Management (systemd)
# ============================================================================

install_service() {
    log_info "Installing systemd service..."
    
    if [ ! -f "$PROJECT_DIR/design-agent.service" ]; then
        log_error "design-agent.service not found"
        return 1
    fi
    
    if [ ! -w /etc/systemd/system ]; then
        log_error "Insufficient permissions. Run with sudo:"
        echo "  sudo $0 --install-service"
        return 1
    fi
    
    cp "$PROJECT_DIR/design-agent.service" /etc/systemd/system/
    systemctl daemon-reload
    
    log_success "Service installed"
    log_info "Start with: sudo systemctl start design-agent.service"
    log_info "Enable on boot: sudo systemctl enable design-agent.service"
    return 0
}

start_service() {
    log_info "Starting design-agent service..."
    
    if ! systemctl is-active --quiet design-agent; then
        sudo systemctl start design-agent.service
        sleep 2
    fi
    
    if systemctl is-active --quiet design-agent; then
        log_success "Service started"
        sudo systemctl status design-agent.service
    else
        log_error "Failed to start service"
        sudo systemctl status design-agent.service
        return 1
    fi
}

stop_service() {
    log_info "Stopping design-agent service..."
    
    sudo systemctl stop design-agent.service
    
    if ! systemctl is-active --quiet design-agent; then
        log_success "Service stopped"
    else
        log_error "Failed to stop service"
        return 1
    fi
}

status_service() {
    log_info "Design-agent service status:"
    echo ""
    sudo systemctl status design-agent.service
}

logs_service() {
    log_info "Design-agent service logs (last 50 lines):"
    echo ""
    sudo journalctl -u design-agent.service -n 50 -f
}

# ============================================================================
# Development Tools
# ============================================================================

code_format() {
    log_info "Formatting code with black..."
    
    if [ ! -d "$VENV_DIR" ]; then
        log_error "Virtual environment not found"
        return 1
    fi
    
    source "$VENV_DIR/bin/activate"
    
    black "$PROJECT_DIR/design_agent_local.py" "$PROJECT_DIR/test_design_agent_local.py"
    
    log_success "Code formatted"
}

code_lint() {
    log_info "Linting code with flake8..."
    
    if [ ! -d "$VENV_DIR" ]; then
        log_error "Virtual environment not found"
        return 1
    fi
    
    source "$VENV_DIR/bin/activate"
    
    flake8 "$PROJECT_DIR/design_agent_local.py" --max-line-length=100 || true
    
    log_success "Linting complete"
}

# ============================================================================
# Usage
# ============================================================================

usage() {
    cat << EOF
Design Agent Launcher

USAGE:
    $0 [COMMAND] [OPTIONS]

SETUP COMMANDS:
    --check-system              Verify system requirements
    --setup-venv                Create and activate virtual environment
    --install-deps              Install Python dependencies
    --full-setup                Run all setup steps

APPLICATION COMMANDS:
    --gui                       Launch graphical interface
    --cli [ARGS]                Run command-line interface
    --tests                     Run unit tests

SERVICE COMMANDS (require sudo):
    --install-service           Install systemd service
    --start-service             Start systemd service
    --stop-service              Stop systemd service
    --status-service            Show service status
    --logs-service              Show service logs

DEVELOPMENT COMMANDS:
    --format                    Format code with black
    --lint                      Lint code with flake8

EXAMPLES:
    # Initial setup
    $0 --full-setup
    
    # Launch GUI
    $0 --gui
    
    # Generate design from command line
    $0 --cli --prompt "robot" --backend shap_e
    
    # Run tests
    $0 --tests
    
    # Install and start service
    sudo $0 --install-service
    sudo $0 --start-service

EOF
}

# ============================================================================
# Main
# ============================================================================

main() {
    case "${1:-}" in
        --check-system)
            check_system
            ;;
        --setup-venv)
            check_python || exit 1
            setup_venv
            ;;
        --install-deps)
            install_dependencies
            ;;
        --full-setup)
            check_python || exit 1
            check_system || log_warning "Some system checks failed"
            setup_venv || exit 1
            install_dependencies || exit 1
            log_success "Full setup complete!"
            ;;
        --gui)
            run_gui
            ;;
        --cli)
            shift
            run_cli "$@"
            ;;
        --tests)
            run_tests
            ;;
        --install-service)
            install_service
            ;;
        --start-service)
            start_service
            ;;
        --stop-service)
            stop_service
            ;;
        --status-service)
            status_service
            ;;
        --logs-service)
            logs_service
            ;;
        --format)
            code_format
            ;;
        --lint)
            code_lint
            ;;
        --help|-h)
            usage
            ;;
        *)
            usage
            exit 1
            ;;
    esac
}

main "$@"

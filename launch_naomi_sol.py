#!/usr/bin/env python3
"""
NAOMI SOL HUB - QUICK LAUNCHER
==============================
Simple launcher with menu for easy access to all features
"""

import sys
import os
import subprocess
from pathlib import Path

# Add project to path
project_dir = Path(__file__).parent / "NaomiSOL_Ultimate_System"
if project_dir.exists():
    sys.path.insert(0, str(project_dir))


def print_banner():
    """Print welcome banner"""
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║                                                          ║
    ║        🌟 NAOMI SOL HUB - ULTIMATE SYSTEM 🌟            ║
    ║                                                          ║
    ║         Virtual & Physical Dodecahedron Control         ║
    ║              With AI, CAD, and Physics                  ║
    ║                                                          ║
    ╚══════════════════════════════════════════════════════════╝
    """)


def print_menu():
    """Print main menu"""
    print("""
    ┌──────────────────────────────────────────────────────────┐
    │                     MAIN MENU                           │
    ├──────────────────────────────────────────────────────────┤
    │                                                          │
    │  1. 🎮  Run Virtual Simulation                          │
    │  2. 🤖  Run with Mock Hardware                          │
    │  3. 🔌  Connect to Real Hardware                        │
    │  4. 🌐  Full Integration Mode                           │
    │  5. 📐  Generate CAD Models (STL)                       │
    │  6. 🧬  Run Design Optimization                         │
    │  7. 🧪  Run Test Suite                                  │
    │  8. 🔧  Build & Package System                          │
    │  9. 📚  View Documentation                              │
    │  0. ❌  Exit                                            │
    │                                                          │
    └──────────────────────────────────────────────────────────┘
    """)


def run_command(cmd, cwd=None):
    """Run a command and handle errors"""
    try:
        print(f"\n🚀 Running: {' '.join(cmd)}\n")
        subprocess.run(cmd, cwd=cwd or project_dir, check=True)
        print("\n✅ Command completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Command failed: {e}")
    except FileNotFoundError:
        print(f"\n❌ Command not found. Make sure Python is in your PATH.")
    
    input("\nPress Enter to continue...")


def main():
    """Main launcher loop"""
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        print_banner()
        print_menu()
        
        choice = input("\n    Enter your choice (0-9): ").strip()
        
        if choice == '1':
            print("\n🎮 Starting Virtual Simulation...")
            print("Controls: ESC=Exit, SPACE=Pause, O=Optimize, G=Generate CAD")
            run_command([sys.executable, "main.py", "--mode", "virtual_only"])
            
        elif choice == '2':
            print("\n🤖 Starting Mock Hardware Mode...")
            print("This simulates hardware without needing physical components.")
            run_command([sys.executable, "main.py", "--mode", "hardware_only", 
                        "--hardware-connection", "Mock"])
            
        elif choice == '3':
            print("\n🔌 Connecting to Real Hardware...")
            port = input("Enter serial port (e.g., COM3, /dev/ttyACM0) or press Enter for auto-detect: ").strip()
            cmd = [sys.executable, "main.py", "--mode", "hardware_only"]
            if port:
                cmd.extend(["--hardware-port", port])
            run_command(cmd)
            
        elif choice == '4':
            print("\n🌐 Starting Full Integration Mode...")
            print("This runs everything: simulation, optimization, physics, and hardware.")
            run_command([sys.executable, "main.py", "--mode", "full_integration"])
            
        elif choice == '5':
            print("\n📐 Generating CAD Models...")
            print("STL files will be saved to: output/cad_models/")
            run_command([sys.executable, "main.py", "--generate-cad"])
            
        elif choice == '6':
            print("\n🧬 Running Design Optimization...")
            iterations = input("Number of iterations (default 100): ").strip()
            iterations = iterations if iterations.isdigit() else "100"
            run_command([sys.executable, "main.py", "--optimize", 
                        "--optimize-iterations", iterations])
            
        elif choice == '7':
            print("\n🧪 Running Test Suite...")
            run_command([sys.executable, "tests/test_naomi_sol.py"])
            
        elif choice == '8':
            print("\n🔧 Building & Packaging System...")
            run_command([sys.executable, "build.py", "--all"])
            
        elif choice == '9':
            print("\n📚 Available Documentation:\n")
            docs = [
                "QUICKSTART.md - Getting Started Guide",
                "README.md - Project Overview",
                "NAOMI_SOL_COMPLETE_OVERVIEW.md - Full System Documentation"
            ]
            for doc in docs:
                print(f"  • {doc}")
            
            print("\nOpen these files in any text editor to read.")
            input("\nPress Enter to continue...")
            
        elif choice == '0':
            print("\n👋 Thank you for using Naomi SOL Hub System!")
            print("   Build your vision! 🚀\n")
            sys.exit(0)
            
        else:
            print(f"\n⚠️ Invalid choice: {choice}")
            input("Press Enter to continue...")


if __name__ == "__main__":
    # Check if we're in the right directory
    if not project_dir.exists():
        print("❌ Error: NaomiSOL_Ultimate_System directory not found!")
        print(f"   Looking for: {project_dir}")
        print("\n   Please run this launcher from the parent directory of the project.")
        sys.exit(1)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8+ required")
        print(f"   Current version: {sys.version}")
        sys.exit(1)
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n✋ Interrupted by user. Goodbye!\n")
        sys.exit(0)

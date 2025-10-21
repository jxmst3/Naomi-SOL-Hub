#!/usr/bin/env python3
"""
Naomi SOL Build and Deployment Script
=====================================
Automates building, testing, and deployment of the Naomi SOL system.
"""

import os
import sys
import subprocess
import shutil
import zipfile
import json
import argparse
from pathlib import Path
from datetime import datetime


class NaomiSOLBuilder:
    """Build and deployment manager for Naomi SOL"""
    
    def __init__(self, project_dir: Path = None):
        """Initialize builder"""
        self.project_dir = project_dir or Path.cwd()
        self.build_dir = self.project_dir / "build"
        self.dist_dir = self.project_dir / "dist"
        self.version = self._get_version()
        
    def _get_version(self) -> str:
        """Get version from configuration"""
        config_file = self.project_dir / "config" / "system_config.json"
        if config_file.exists():
            with open(config_file) as f:
                config = json.load(f)
                return config.get("version", "3.0")
        return "3.0"
    
    def clean(self):
        """Clean build directories"""
        print("🧹 Cleaning build directories...")
        
        for dir_path in [self.build_dir, self.dist_dir]:
            if dir_path.exists():
                shutil.rmtree(dir_path)
                print(f"  ✓ Removed {dir_path}")
        
        # Clean Python cache
        for cache_dir in self.project_dir.rglob("__pycache__"):
            shutil.rmtree(cache_dir)
        
        for pyc_file in self.project_dir.rglob("*.pyc"):
            pyc_file.unlink()
        
        print("  ✓ Cleaned Python cache")
    
    def install_dependencies(self):
        """Install Python dependencies"""
        print("📦 Installing dependencies...")
        
        requirements_file = self.project_dir / "requirements.txt"
        if not requirements_file.exists():
            print("  ✗ requirements.txt not found")
            return False
        
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)],
                check=True
            )
            print("  ✓ Dependencies installed")
            return True
        except subprocess.CalledProcessError as e:
            print(f"  ✗ Failed to install dependencies: {e}")
            return False
    
    def run_tests(self) -> bool:
        """Run test suite"""
        print("🧪 Running tests...")
        
        test_file = self.project_dir / "tests" / "test_naomi_sol.py"
        if not test_file.exists():
            print("  ⚠️ Test file not found")
            return True  # Don't fail if tests don't exist
        
        try:
            result = subprocess.run(
                [sys.executable, str(test_file)],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print("  ✓ All tests passed")
                return True
            else:
                print("  ✗ Tests failed")
                print(result.stdout)
                return False
                
        except Exception as e:
            print(f"  ✗ Error running tests: {e}")
            return False
    
    def generate_cad_models(self):
        """Generate all CAD models"""
        print("📐 Generating CAD models...")
        
        try:
            result = subprocess.run(
                [sys.executable, "main.py", "--generate-cad"],
                cwd=self.project_dir,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print("  ✓ CAD models generated")
                
                # Count STL files
                stl_dir = self.project_dir / "output" / "cad_models"
                if stl_dir.exists():
                    stl_files = list(stl_dir.glob("*.stl"))
                    print(f"  ✓ Generated {len(stl_files)} STL files")
            else:
                print("  ⚠️ CAD generation had issues")
                
        except Exception as e:
            print(f"  ⚠️ Could not generate CAD: {e}")
    
    def compile_arduino_firmware(self):
        """Compile Arduino firmware"""
        print("🔧 Compiling Arduino firmware...")
        
        firmware_dir = self.project_dir / "firmware"
        if not firmware_dir.exists():
            print("  ⚠️ Firmware directory not found")
            return
        
        # Check for Arduino CLI
        try:
            subprocess.run(["arduino-cli", "version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("  ⚠️ Arduino CLI not installed")
            print("     Install from: https://arduino.github.io/arduino-cli/")
            return
        
        # Compile for Portenta H7
        sketch = firmware_dir / "NaomiSOL_Firmware.ino"
        if sketch.exists():
            try:
                subprocess.run([
                    "arduino-cli", "compile",
                    "--fqbn", "arduino:mbed_portenta:envie_m7",
                    str(firmware_dir)
                ], check=True)
                print("  ✓ Firmware compiled for Arduino Portenta H7")
            except subprocess.CalledProcessError:
                print("  ⚠️ Firmware compilation failed")
        else:
            print("  ⚠️ Firmware sketch not found")
    
    def build_documentation(self):
        """Build documentation"""
        print("📚 Building documentation...")
        
        docs_dir = self.build_dir / "docs"
        docs_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy markdown files
        for md_file in self.project_dir.glob("*.md"):
            shutil.copy(md_file, docs_dir)
            print(f"  ✓ Copied {md_file.name}")
        
        # Generate API documentation (if using sphinx or similar)
        # This is a placeholder - implement based on your doc system
        
        print("  ✓ Documentation built")
    
    def create_package(self):
        """Create distributable package"""
        print("📦 Creating distribution package...")
        
        self.dist_dir.mkdir(parents=True, exist_ok=True)
        
        # Package name with version
        package_name = f"NaomiSOL_v{self.version}_{datetime.now().strftime('%Y%m%d')}"
        package_path = self.dist_dir / f"{package_name}.zip"
        
        with zipfile.ZipFile(package_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add Python source
            for py_file in self.project_dir.rglob("*.py"):
                if "test" not in str(py_file) and "__pycache__" not in str(py_file):
                    arcname = py_file.relative_to(self.project_dir)
                    zipf.write(py_file, arcname)
            
            # Add configuration
            for json_file in (self.project_dir / "config").glob("*.json"):
                arcname = json_file.relative_to(self.project_dir)
                zipf.write(json_file, arcname)
            
            # Add firmware
            firmware_dir = self.project_dir / "firmware"
            if firmware_dir.exists():
                for file in firmware_dir.rglob("*"):
                    if file.is_file():
                        arcname = file.relative_to(self.project_dir)
                        zipf.write(file, arcname)
            
            # Add CAD models
            cad_dir = self.project_dir / "output" / "cad_models"
            if cad_dir.exists():
                for stl_file in cad_dir.glob("*.stl"):
                    arcname = Path("cad_models") / stl_file.name
                    zipf.write(stl_file, arcname)
            
            # Add documentation
            for md_file in self.project_dir.glob("*.md"):
                zipf.write(md_file, md_file.name)
            
            # Add requirements
            req_file = self.project_dir / "requirements.txt"
            if req_file.exists():
                zipf.write(req_file, "requirements.txt")
        
        print(f"  ✓ Created package: {package_path}")
        
        # Calculate package size
        size_mb = package_path.stat().st_size / (1024 * 1024)
        print(f"  ✓ Package size: {size_mb:.2f} MB")
        
        return package_path
    
    def create_installer(self):
        """Create installer script"""
        print("🔨 Creating installer...")
        
        installer_path = self.dist_dir / "install_naomi_sol.py"
        
        installer_script = '''#!/usr/bin/env python3
"""
Naomi SOL Installer
===================
Automated installation script for Naomi SOL Hub System
"""

import os
import sys
import subprocess
import zipfile
from pathlib import Path


def install_naomi_sol():
    """Install Naomi SOL system"""
    print("="*60)
    print("NAOMI SOL HUB SYSTEM INSTALLER")
    print("="*60)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    
    # Get installation directory
    default_dir = Path.home() / "NaomiSOL"
    install_dir = input(f"Installation directory [{default_dir}]: ").strip()
    install_dir = Path(install_dir) if install_dir else default_dir
    
    # Create directory
    install_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Installing to {install_dir}")
    
    # Extract package
    package_file = Path(__file__).parent / "NaomiSOL_package.zip"
    if package_file.exists():
        print("📦 Extracting files...")
        with zipfile.ZipFile(package_file, 'r') as zipf:
            zipf.extractall(install_dir)
        print("✓ Files extracted")
    
    # Install dependencies
    print("📚 Installing dependencies...")
    requirements = install_dir / "requirements.txt"
    if requirements.exists():
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", str(requirements)
        ])
    
    # Create desktop shortcut (Windows)
    if sys.platform == "win32":
        create_windows_shortcut(install_dir)
    
    # Create launcher script
    launcher = install_dir / "launch_naomi_sol.py"
    with open(launcher, 'w') as f:
        f.write(f"""#!/usr/bin/env python3
import sys
sys.path.insert(0, r'{install_dir}')
from main import main
if __name__ == "__main__":
    main()
""")
    
    launcher.chmod(0o755)
    
    print()
    print("="*60)
    print("✅ INSTALLATION COMPLETE!")
    print("="*60)
    print(f"Installed to: {install_dir}")
    print(f"To run: python {launcher}")
    print()
    
    return True


def create_windows_shortcut(install_dir):
    """Create Windows desktop shortcut"""
    try:
        import winshell
        from win32com.client import Dispatch
        
        desktop = winshell.desktop()
        shortcut_path = Path(desktop) / "Naomi SOL.lnk"
        
        shell = Dispatch('WScript.Shell')
        shortcut = shell.CreateShortCut(str(shortcut_path))
        shortcut.Targetpath = sys.executable
        shortcut.Arguments = str(install_dir / "launch_naomi_sol.py")
        shortcut.WorkingDirectory = str(install_dir)
        shortcut.IconLocation = sys.executable
        shortcut.save()
        
        print(f"✓ Created desktop shortcut")
    except ImportError:
        print("⚠️ Could not create desktop shortcut (install pywin32)")


if __name__ == "__main__":
    install_naomi_sol()
'''
        
        with open(installer_path, 'w', encoding='utf-8') as f:
            f.write(installer_script)
        
        installer_path.chmod(0o755)
        print(f"  ✓ Created installer: {installer_path}")
    
    def build_all(self):
        """Run complete build process"""
        print("\n" + "="*60)
        print("NAOMI SOL BUILD PROCESS")
        print(f"Version: {self.version}")
        print("="*60 + "\n")
        
        steps = [
            ("Clean", self.clean),
            ("Install Dependencies", self.install_dependencies),
            ("Run Tests", self.run_tests),
            ("Generate CAD Models", self.generate_cad_models),
            ("Compile Firmware", self.compile_arduino_firmware),
            ("Build Documentation", self.build_documentation),
            ("Create Package", self.create_package),
            ("Create Installer", self.create_installer)
        ]
        
        success = True
        for step_name, step_func in steps:
            print(f"\n[{steps.index((step_name, step_func))+1}/{len(steps)}] {step_name}")
            print("-" * 40)
            
            if step_name == "Run Tests":
                if not step_func():
                    success = False
                    print("⚠️ Tests failed but continuing build")
            else:
                step_func()
        
        print("\n" + "="*60)
        if success:
            print("✅ BUILD COMPLETE!")
        else:
            print("⚠️ BUILD COMPLETE WITH WARNINGS")
        print("="*60)
        
        # Print summary
        print(f"\nVersion: {self.version}")
        print(f"Build directory: {self.build_dir}")
        print(f"Distribution directory: {self.dist_dir}")
        
        # List distribution files
        if self.dist_dir.exists():
            dist_files = list(self.dist_dir.glob("*"))
            if dist_files:
                print(f"\nDistribution files:")
                for file in dist_files:
                    size_kb = file.stat().st_size / 1024
                    print(f"  • {file.name} ({size_kb:.1f} KB)")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Naomi SOL Build System")
    parser.add_argument("--clean", action="store_true", help="Clean build directories")
    parser.add_argument("--test", action="store_true", help="Run tests only")
    parser.add_argument("--cad", action="store_true", help="Generate CAD models only")
    parser.add_argument("--package", action="store_true", help="Create package only")
    parser.add_argument("--all", action="store_true", help="Run complete build")
    
    args = parser.parse_args()
    
    builder = NaomiSOLBuilder()
    
    if args.clean:
        builder.clean()
    elif args.test:
        builder.run_tests()
    elif args.cad:
        builder.generate_cad_models()
    elif args.package:
        builder.create_package()
    elif args.all or not any(vars(args).values()):
        builder.build_all()


if __name__ == "__main__":
    main()

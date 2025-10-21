"""Dependency checker for Naomi SOL Hub."""

import sys
import pkg_resources
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

CORE_REQUIREMENTS = {
    'pygame': '2.5.0',
    'PyOpenGL': '3.1.7',
    'numpy': '1.24.0',
    'scipy': '1.10.0',
    'networkx': '3.1'
}

OPTIONAL_REQUIREMENTS = {
    'pybullet': '3.2.5',  # Physics simulation
    'torch': '2.0.0',     # AI features
    'cadquery': '2.3.0',  # CAD generation
    'bleak': '0.21.0',    # BLE communication
    'rich': '13.5.0'      # Enhanced output
}

def check_package(package: str, min_version: str) -> Tuple[bool, str]:
    """Check if a package is installed and meets minimum version."""
    try:
        installed = pkg_resources.get_distribution(package)
        version_ok = pkg_resources.parse_version(installed.version) >= pkg_resources.parse_version(min_version)
        return True, installed.version
    except pkg_resources.DistributionNotFound:
        return False, "not installed"

def check_dependencies(check_optional: bool = True) -> Dict[str, List[Tuple[str, str, bool]]]:
    """
    Check all dependencies and return status.
    
    Args:
        check_optional: Whether to check optional packages
        
    Returns:
        Dict with categories and status of each package
    """
    results = {
        'core': [],
        'optional': []
    }
    
    # Check core requirements
    for package, version in CORE_REQUIREMENTS.items():
        installed, actual_version = check_package(package, version)
        results['core'].append((package, actual_version, installed))
        
        if not installed:
            logger.warning(f"Core package {package} >= {version} is required but not installed")
    
    # Check optional requirements
    if check_optional:
        for package, version in OPTIONAL_REQUIREMENTS.items():
            installed, actual_version = check_package(package, version)
            results['optional'].append((package, actual_version, installed))
            
            if not installed:
                logger.info(f"Optional package {package} >= {version} is not installed")
    
    return results

def print_dependency_status(include_optional: bool = True):
    """Print dependency status in a user-friendly format."""
    results = check_dependencies(include_optional)
    
    print("\nNAOMI SOL HUB - Dependency Check")
    print("=" * 50)
    
    print("\nCore Dependencies:")
    print("-" * 30)
    for pkg, version, installed in results['core']:
        status = "✓" if installed else "✗"
        print(f"{status} {pkg:<20} {version}")
    
    if include_optional:
        print("\nOptional Dependencies:")
        print("-" * 30)
        for pkg, version, installed in results['optional']:
            status = "✓" if installed else "-"
            print(f"{status} {pkg:<20} {version}")
    
    # Print summary
    core_installed = sum(1 for _, _, installed in results['core'] if installed)
    core_total = len(results['core'])
    
    print("\nSummary:")
    print(f"Core Packages: {core_installed}/{core_total} installed")
    
    if include_optional:
        opt_installed = sum(1 for _, _, installed in results['optional'] if installed)
        opt_total = len(results['optional'])
        print(f"Optional Packages: {opt_installed}/{opt_total} installed")
    
    # Print recommendation if needed
    if core_installed < core_total:
        print("\nMissing core dependencies. Please install with:")
        print("pip install -r requirements.txt")
    
    print("\nNote: Some packages may require conda installation.")
    print("See requirements.txt for detailed instructions.")
    print("It's also recommended to ensure that pip, setuptools, and wheel are up-to-date:")
    print("pip install --upgrade pip setuptools wheel")

if __name__ == "__main__":
    print_dependency_status()
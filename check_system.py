#!/usr/bin/env python
"""
System Check Script for Detox Project
Verifies all dependencies and system requirements
"""

import sys
import subprocess
import os

def print_header(text):
    print(f"\n{'='*60}")
    print(f"{text:^60}")
    print(f"{'='*60}\n")

def check_python():
    """Check Python version"""
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    
    if version.major >= 3 and version.minor >= 8:
        print(f"✓ Python {version_str} - OK")
        return True
    else:
        print(f"✗ Python {version_str} - REQUIRED: Python 3.8+")
        return False

def check_java():
    """Check Java installation"""
    try:
        result = subprocess.run(['java', '-version'], 
                              capture_output=True, 
                              text=True, 
                              timeout=5)
        
        # Java version is printed to stderr
        output = result.stderr
        if 'version' in output.lower():
            version_line = output.split('\n')[0]
            print(f"✓ Java installed - {version_line}")
            return True
        else:
            print("✗ Java not found")
            return False
    except Exception as e:
        print(f"✗ Java not found - {str(e)}")
        return False

def check_package(package_name):
    """Check if a Python package is installed"""
    try:
        __import__(package_name)
        print(f"✓ {package_name} - Installed")
        return True
    except ImportError:
        print(f"✗ {package_name} - NOT INSTALLED")
        return False

def check_directories():
    """Check required directories"""
    dirs = ['data', 'output', 'models']
    all_exist = True
    
    for dir_name in dirs:
        if os.path.exists(dir_name):
            print(f"✓ {dir_name}/ - Exists")
        else:
            print(f"✗ {dir_name}/ - NOT FOUND")
            all_exist = False
    
    return all_exist

def check_dataset():
    """Check if dataset exists"""
    dataset_path = 'data/chat_data.csv'
    
    if os.path.exists(dataset_path):
        size = os.path.getsize(dataset_path)
        size_mb = size / (1024 * 1024)
        print(f"✓ Dataset found - {size_mb:.2f} MB")
        return True
    else:
        print(f"✗ Dataset not found at: {dataset_path}")
        print(f"  Please add your dataset to: data/chat_data.csv")
        return False

def check_port(port=4040):
    """Check if port is available"""
    import socket
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('localhost', port))
    sock.close()
    
    if result != 0:
        print(f"✓ Port {port} - Available")
        return True
    else:
        print(f"⚠ Port {port} - In use (Spark UI may use another port)")
        return True

def main():
    print_header("DETOX SYSTEM CHECK")
    
    print("Student: SARAVANA PRIYAN S T")
    print("Reg No: 927623BAD100")
    print("Project: Chat Message Toxicity Detector\n")
    
    results = {}
    
    # Check Python
    print_header("1. Python Environment")
    results['python'] = check_python()
    
    # Check Java
    print_header("2. Java Environment")
    results['java'] = check_java()
    
    # Check Python packages
    print_header("3. Python Packages")
    packages = ['pyspark', 'pandas', 'numpy', 'matplotlib', 'seaborn', 'jupyter']
    package_results = []
    
    for package in packages:
        package_results.append(check_package(package))
    
    results['packages'] = all(package_results)
    
    # Check directories
    print_header("4. Project Directories")
    results['directories'] = check_directories()
    
    # Check dataset
    print_header("5. Dataset")
    results['dataset'] = check_dataset()
    
    # Check port
    print_header("6. Network Ports")
    results['port'] = check_port(4040)
    
    # Summary
    print_header("SUMMARY")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("✅ All checks passed!")
        print("\n🚀 You're ready to run Detox!")
        print("\nNext steps:")
        print("  1. Run main application: python main.py")
        print("  2. Or open notebook: jupyter notebook detox_analysis.ipynb")
        print("  3. Access Spark UI: http://localhost:4040")
    else:
        print("⚠️  Some checks failed. Please fix the issues above.")
        print("\nFailed checks:")
        for check, passed in results.items():
            if not passed:
                print(f"  ✗ {check}")
        
        print("\n📚 For help, see:")
        print("  - README.md for detailed setup")
        print("  - QUICKSTART.md for quick start guide")
        print("  - Run: ./setup.sh for automatic setup")
    
    print(f"\n{'='*60}\n")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Deployment script for QuantLab Professional.
"""
import subprocess
import sys
import os
from pathlib import Path


def run_command(command: str, description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stdout:
            print(f"STDOUT: {e.stdout}")
        if e.stderr:
            print(f"STDERR: {e.stderr}")
        return False


def check_dependencies():
    """Check if required dependencies are installed."""
    print("\n📋 Checking dependencies...")
    
    required_packages = [
        "streamlit", "pandas", "numpy", "plotly", "scikit-learn", 
        "xgboost", "yfinance", "scipy", "pyyaml"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - Missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies satisfied")
    return True


def run_tests():
    """Run the test suite."""
    if not Path("tests").exists():
        print("⚠️  No tests directory found, skipping tests")
        return True
    
    return run_command("python -m pytest tests/ -v", "Running test suite")


def run_linting():
    """Run code quality checks."""
    checks = [
        ("python -m flake8 . --max-line-length=100 --extend-ignore=E203,W503", "Flake8 linting"),
        ("python -m black . --check", "Black formatting check"),
    ]
    
    all_passed = True
    for command, description in checks:
        if not run_command(command, description):
            all_passed = False
    
    return all_passed


def create_logs_directory():
    """Create logs directory if it doesn't exist."""
    logs_dir = Path("logs")
    if not logs_dir.exists():
        logs_dir.mkdir()
        print("✅ Created logs directory")
    return True


def validate_config():
    """Validate configuration files."""
    config_file = Path("config.yaml")
    if not config_file.exists():
        print("⚠️  config.yaml not found, using defaults")
        return True
    
    try:
        import yaml
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        print("✅ Configuration file validated")
        return True
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return False


def main():
    """Main deployment function."""
    print("🚀 QuantLab Professional - Deployment Script")
    print("=" * 50)
    
    # Check current directory
    if not Path("app_professional.py").exists():
        print("❌ app_professional.py not found. Are you in the correct directory?")
        sys.exit(1)
    
    # Deployment steps
    steps = [
        ("Checking dependencies", check_dependencies),
        ("Creating logs directory", create_logs_directory),
        ("Validating configuration", validate_config),
        ("Running code quality checks", run_linting),
        ("Running tests", run_tests),
    ]
    
    failed_steps = []
    for step_name, step_function in steps:
        print(f"\n📋 {step_name}...")
        if not step_function():
            failed_steps.append(step_name)
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 DEPLOYMENT SUMMARY")
    print("=" * 50)
    
    if not failed_steps:
        print("🎉 All deployment checks passed!")
        print("\n🚀 Ready to launch QuantLab Professional")
        print("\nRun: streamlit run app_professional.py")
        
        # Optional: Auto-launch
        launch = input("\n❓ Launch application now? (y/N): ").strip().lower()
        if launch == 'y':
            print("\n🚀 Launching QuantLab Professional...")
            os.system("streamlit run app_professional.py")
    else:
        print(f"❌ Deployment failed. Issues with: {', '.join(failed_steps)}")
        print("\n🔧 Please fix the above issues and run deploy.py again")
        sys.exit(1)


if __name__ == "__main__":
    main()
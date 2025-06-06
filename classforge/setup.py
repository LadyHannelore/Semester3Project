#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Setup script for ClassForge project.
This script helps with setting up the project environment.
"""

import os
import sys
import argparse
import subprocess
import shutil
import webbrowser
from pathlib import Path

def setup_backend(backend_dir):
    """Set up the backend environment"""
    print("\n[1/4] Setting up backend environment...")
    
    # Create virtual environment
    venv_dir = os.path.join(backend_dir, "venv")
    if not os.path.exists(venv_dir):
        print("Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", venv_dir])
    
    # Determine activate script
    if sys.platform == "win32":
        activate_script = os.path.join(venv_dir, "Scripts", "activate")
    else:
        activate_script = os.path.join(venv_dir, "bin", "activate")
    
    # Install requirements
    print("Installing dependencies...")
    requirements_file = os.path.join(backend_dir, "requirements.txt")
    
    if sys.platform == "win32":
        pip_cmd = [os.path.join(venv_dir, "Scripts", "pip")]
    else:
        pip_cmd = [os.path.join(venv_dir, "bin", "pip")]
        
    subprocess.run(pip_cmd + ["install", "-r", requirements_file])
    
    print("Backend setup complete!")

def setup_frontend(frontend_dir):
    """Set up the frontend environment"""
    print("\n[2/4] Setting up frontend environment...")
    index_file = os.path.join(frontend_dir, "index.html")
    
    if os.path.exists(index_file):
        print("Frontend files found!")
    else:
        print("ERROR: Frontend files not found at", frontend_dir)
        return False
        
    print("Frontend setup complete!")
    return True

def start_backend_server(backend_dir):
    """Start the backend server"""
    print("\n[3/4] Starting backend server...")
    
    # Determine Python executable in virtual environment
    if sys.platform == "win32":
        python_exe = os.path.join(backend_dir, "venv", "Scripts", "python.exe")
    else:
        python_exe = os.path.join(backend_dir, "venv", "bin", "python")
    
    # Start backend server
    main_script = os.path.join(backend_dir, "main.py")
    if os.path.exists(main_script):
        subprocess.Popen([python_exe, main_script])
        print("Backend server started at http://localhost:5001")
    else:
        print("ERROR: Backend main script not found at", main_script)
        return False
        
    return True

def start_frontend_server(frontend_dir):
    """Start a simple HTTP server for the frontend"""
    print("\n[4/4] Starting frontend server...")
    
    # Start a simple HTTP server for the frontend
    os.chdir(frontend_dir)
    
    # Use http.server module
    port = 8000
    if sys.platform == "win32":
        # On Windows, use subprocess to run in background
        subprocess.Popen([sys.executable, "-m", "http.server", str(port)])
    else:
        # On Unix-like systems, use subprocess with different approach
        subprocess.Popen([sys.executable, "-m", "http.server", str(port)], 
                         stdout=subprocess.DEVNULL, 
                         stderr=subprocess.DEVNULL)
    
    print(f"Frontend server started at http://localhost:{port}")
    
    # Open browser
    webbrowser.open(f"http://localhost:{port}")
    
    return True

def main():
    """Main entry point for the setup script"""
    parser = argparse.ArgumentParser(description="Set up and run ClassForge")
    parser.add_argument("--backend-only", action="store_true", help="Only set up and run the backend")
    parser.add_argument("--frontend-only", action="store_true", help="Only set up and run the frontend")
    args = parser.parse_args()
    
    # Get the project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.join(script_dir, "backend")
    frontend_dir = os.path.join(script_dir, "frontend")
    
    print("ClassForge Setup")
    print("===============")
    
    if not args.frontend_only:
        setup_backend(backend_dir)
        start_backend_server(backend_dir)
    
    if not args.backend_only:
        setup_frontend(frontend_dir)
        start_frontend_server(frontend_dir)
    
    print("\nSetup completed!")
    print("ClassForge is now running at http://localhost:8000")
    print("API server is running at http://localhost:5001")
    print("\nPress Ctrl+C to stop the servers")
    
    try:
        # Keep the script running
        while True:
            pass
    except KeyboardInterrupt:
        print("\nShutting down...")
        sys.exit(0)

if __name__ == "__main__":
    main()

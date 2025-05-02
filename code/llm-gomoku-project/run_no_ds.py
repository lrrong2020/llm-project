#!/usr/bin/env python
"""
Wrapper script to run the trainer without DeepSpeed.
"""
import os
import sys
import subprocess

# Set environment variables to disable DeepSpeed
os.environ["DISABLE_DEEPSPEED"] = "1"

# Get the command line arguments
args = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else ""

# Build the command to run
command = f"python src/trainer.py {args}"
print(f"Running: {command}")

# Execute the command
result = subprocess.run(command, shell=True, check=False)
sys.exit(result.returncode) 
import os
import subprocess
import sys


def create_venv(venv_dir, python_executable):
    try:
        subprocess.check_output([python_executable, '-m', 'venv', venv_dir])
        print(f"Virtual environment created successfully at {venv_dir}")
    except subprocess.CalledProcessError:
        print("Error occurred while creating the virtual environment.")
        sys.exit(1)


def main():
    # Path to the Python 3.9 interpreter
    python_3_9_executable = "/usr/bin/python3.9"

    # if len(sys.argv) != 2:
    #     print("Usage: python create_venv.py <venv_directory>")
    #     sys.exit(1)

    venv_dir = "/home/myenv"
    create_venv(venv_dir, python_3_9_executable)


if __name__ == "__main__":
    main()

import os
import subprocess
import sys


def create_venv(venv_dir):
    try:
        subprocess.check_output([sys.executable, '-m', 'venv', venv_dir])
        print(f"Virtual environment created successfully at {venv_dir}")
    except subprocess.CalledProcessError:
        print("Error occurred while creating the virtual environment.")
        sys.exit(1)


def main():
    # if len(sys.argv) != 2:
    #     print("Usage: python create_venv.py <venv_directory>")
    #     sys.exit(1)

    venv_dir = "/home/myenv"
    create_venv(venv_dir)


if __name__ == "__main__":
    main()

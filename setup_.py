import subprocess
import sys
import venv


def create_python_env(env_path, python_version):
    """
    Create a new Python environment with the specified Python version using venv.
    
    Args:
    - env_path (str): Path where the environment should be created.
    - python_version (str): Version of Python for the environment.
    """
    venv.create(env_path, system_site_packages=False, clear=False, with_pip=True, prompt=None, upgrade=False,
                symlinks=False, base=None, without_pip=False, implementation=False, app_data=False, symlink=False, progress=None)


def activate_python_env(env_path):
    """
    Activate a Python environment.
    
    Args:
    - env_path (str): Path to the environment to activate.
    """
    if sys.platform == 'win32':
        activate_script = "Scripts/activate.bat"
    else:
        activate_script = "bin/activate"

    activate_path = f"{env_path}/{activate_script}"
    command = f"source {activate_path}"
    subprocess.run(command, shell=True, check=True)


def main():
    env_path = "/home/myenv"
    python_version = "3.9"

    # Create the Python environment
    create_python_env(env_path, python_version)

    # Activate the Python environment
    # activate_python_env(env_path)


if __name__ == "__main__":
    main()

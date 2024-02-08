import subprocess


def create_python_env(env_path, python_version):
    """
    Create a new Python environment with the specified Python version using conda.
    
    Args:
    - env_path (str): Path where the environment should be created.
    - python_version (str): Version of Python for the environment.
    """
    command = f"conda create --prefix {env_path} python={python_version} -y"
    subprocess.run(command, shell=True, check=True)


def activate_python_env(env_path):
    """
    Activate a Python environment using conda.
    
    Args:
    - env_path (str): Path to the environment to activate.
    """
    command = f"conda activate {env_path}"
    subprocess.run(command, shell=True, check=True)


def main():
    env_path = "/home/myenv"
    python_version = "3.9"

    # Create the Python environment
    create_python_env(env_path, python_version)

    # Activate the Python environment
    activate_python_env(env_path)


if __name__ == "__main__":
    main()

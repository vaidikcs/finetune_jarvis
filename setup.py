import subprocess
import os


def create_python_env():
    # Install Miniconda (change URL if needed)
    miniconda_installer = "Miniconda3-latest-Linux-x86_64.sh"
    os.system(
        f"wget https://repo.anaconda.com/miniconda/{miniconda_installer}")
    os.system(f"bash {miniconda_installer} -b -p $HOME/miniconda")
    os.environ["PATH"] += ":$HOME/miniconda/bin"
    os.system("source ~/.bashrc")

    # Create Python 3.9 environment
    os.system("conda create --name python39 python=3.9 -y")


def main():
    create_python_env()


if __name__ == "__main__":
    main()

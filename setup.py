import os, shutil
import subprocess
import urllib.request

def download_anaconda_installer():
    url = "https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh"
    filename = "Anaconda3-2021.05-Linux-x86_64.sh"
    urllib.request.urlretrieve(url, filename)

def install_anaconda():
    os.system("sudo bash Anaconda3-2021.05-Linux-x86_64.sh -b -p /opt/conda")
    os.system("conda init")
    os.system("exec $SHELL")

def create_python_env():
    os.system("conda create --prefix /opt/conda/envs/python39 python=3.9 -y")
    os.system("source /opt/conda/etc/profile.d/conda.sh")
    os.system("conda activate /opt/conda/envs/python39")

def clone_repository():
    os.system("git clone https://github.com/vaidikcs/test.git")

def main():
    if not shutil.which("conda"):
        download_anaconda_installer()
        install_anaconda()

    if not os.path.exists("/opt/conda/envs"):
        os.makedirs("/opt/conda/envs")

    create_python_env()

if __name__ == "__main__":
    main()

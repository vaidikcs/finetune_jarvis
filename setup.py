import os
import subprocess
import urllib.request
import shutil


def download_anaconda_installer():
    url = "https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh"
    filename = "Anaconda3-2021.05-Linux-x86_64.sh"
    urllib.request.urlretrieve(url, filename)


def install_anaconda():
    subprocess.run(
        ["sudo", "bash", "Anaconda3-2021.05-Linux-x86_64.sh", "-b", "-p", "/opt/conda"])
    subprocess.run(["/opt/conda/bin/conda", "init"])
    subprocess.run(["source", "$HOME/.bashrc"])


def create_python_env():
    subprocess.run(["/opt/conda/bin/conda", "create", "--prefix",
                   "/opt/conda/envs/python39", "python=3.9", "-y"])
    subprocess.run(["/opt/conda/bin/conda", "init", "bash"])


def clone_repository():
    subprocess.run(["git", "clone", "https://github.com/vaidikcs/test.git"])


def main():
    if not shutil.which("conda"):
        download_anaconda_installer()
        install_anaconda()

    if not os.path.exists("/opt/conda/envs"):
        os.makedirs("/opt/conda/envs")

    create_python_env()


if __name__ == "__main__":
    main()

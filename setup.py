import os
import subprocess
import urllib.request


def download_anaconda_installer():
    url = "https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh"
    filename = "Anaconda3-2021.05-Linux-x86_64.sh"
    urllib.request.urlretrieve(url, filename)


def install_anaconda():
    subprocess.run(
        ["sudo", "bash", "Anaconda3-2021.05-Linux-x86_64.sh", "-b", "-p", "/opt/conda"])
    subprocess.run(["/opt/conda/bin/conda", "init"])
    subprocess.run(["exec", "$SHELL"])


def create_python_env():
    # Activate the conda base environment
    os.environ["PATH"] = "/opt/conda/bin:" + os.environ["PATH"]

    # Create the python39 environment
    subprocess.run(["conda", "create", "--prefix",
                   "/opt/conda/envs/python39", "python=3.9", "-y"])
    subprocess.run(["conda", "init", "bash"])
    subprocess.run(["exec", "$SHELL"])


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

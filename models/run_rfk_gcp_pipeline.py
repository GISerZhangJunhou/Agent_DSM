from config.paths import GCP_SCRIPT, RFK_SCRIPT
from config.settings import PYTHON_EXECUTABLE
import subprocess
import sys

def run(cmd):
    p = subprocess.run(cmd)
    if p.returncode != 0:
        sys.exit(p.returncode)

if __name__ == "__main__":
    run([PYTHON_EXECUTABLE, str(RFK_SCRIPT)])
    run([PYTHON_EXECUTABLE, str(GCP_SCRIPT)])

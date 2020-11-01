""" Removes temporary files from testing and setup """

import shutil
import os

remove = ".tox .eggs .pytest_cache agentpy.egg-info build dist"

for path in remove.split(' '):
    if path in os.listdir():
        shutil.rmtree(path)
        print(f"Removed '{path}'")

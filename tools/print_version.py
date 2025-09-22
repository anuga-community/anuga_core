"""
This script prints the current version from pyproject.toml.
"""

import re
import os


# ===================================================
# Read VERSION from pyproject.toml file
# ===================================================
with open('pyproject.toml') as infile:
    for line in infile:
        match = re.match(r'version = ', line)
        if match != None:
            VERSION = re.findall(r'\d.\d.\ddev|\d.\d.\d',line)[0]


print(VERSION)


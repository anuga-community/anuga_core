"""
This script prints the current version from pyproject.toml.
"""

import toml

# Read version from pyproject.toml
data = toml.load("pyproject.toml")
version = data["project"]["version"]

print(version)




"""Pytest configuration file.

This file is automatically loaded by pytest and ensures that the project root
is added to sys.path, allowing tests to import from the stable_datasets package
without needing sys.path.insert in each test file.
"""
import sys
from pathlib import Path

# Add the project root to sys.path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

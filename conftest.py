"""
conftest.py — pytest configuration for sparse_attn test suite.
"""
import sys
import os

# Ensure sparse_attn is importable from the repo root
sys.path.insert(0, os.path.dirname(__file__))

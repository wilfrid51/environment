"""
Trace task evaluation package.

This package provides tools for evaluating LLMs on trace tasks - predicting
the exact stdout output of Python programs with injected debug print statements.

Main exports:
    Actor: Main class for trace task evaluation and training interface
"""

from .env import Actor
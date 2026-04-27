"""Shared fork-safety helpers for backend implementations.

This module is intentionally small for now. Lance-backed layouts should keep
opened dataset handles out of parent processes before DataLoader forks, and
video backends should pin OpenCV worker threads inside children.
"""

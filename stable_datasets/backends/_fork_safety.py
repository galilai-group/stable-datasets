"""Shared fork-safety helpers for backend implementations.

Backends use this module for process-local runtime state, including
worker-local library configuration after DataLoader forks.
"""

"""Robustness evaluation of in-domain benchmark checkpoints under distribution shift.

Trains nothing: each evaluator reloads a backbone and its online linear probe
from a ``benchmarks.run`` checkpoint trained on clean data and scores them on
shifted test sets.
"""

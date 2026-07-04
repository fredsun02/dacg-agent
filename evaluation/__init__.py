"""
Evaluation modules for KGSA Agent
"""

from .cochrane_eval import evaluate_agent, EvalResult
from .case_study import run_case_study

__all__ = ["evaluate_agent", "EvalResult", "run_case_study"]

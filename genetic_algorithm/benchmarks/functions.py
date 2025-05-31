import numpy as np
from typing import Dict, Any


class BenchmarkFunctions:
    """Collection of benchmark optimization functions"""

    @staticmethod
    def ackley(x: float, y: float) -> float:
        """
        Ackley function: f(x,y) = -20*exp(-0.2*sqrt(0.5*(x²+y²))) - exp(0.5*(cos(2πx)+cos(2πy))) + e + 20
        Domain: [-5, 5] × [-5, 5]
        Global minimum: f(0, 0) = 0
        """
        return (
            -20 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2)))
            - np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)))
            + np.e
            + 20
        )

    @staticmethod
    def rastrigin(x: float, y: float) -> float:
        """
        Rastrigin function: f(x,y) = 20 + x² + y² - 10*(cos(2πx) + cos(2πy))
        Domain: [-5.12, 5.12] × [-5.12, 5.12]
        Global minimum: f(0, 0) = 0
        """
        return 20 + x**2 + y**2 - 10 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))

    @staticmethod
    def get_function_info() -> Dict[str, Dict[str, Any]]:
        """Return information about the benchmark functions"""
        return {
            "ackley": {
                "function": BenchmarkFunctions.ackley,
                "domain": [[-5, 5], [-5, 5]],
                "global_min": (0, 0, 0),
                "name": "Ackley Function",
            },
            "rastrigin": {
                "function": BenchmarkFunctions.rastrigin,
                "domain": [[-5.12, 5.12], [-5.12, 5.12]],
                "global_min": (0, 0, 0),
                "name": "Rastrigin Function",
            },
        }

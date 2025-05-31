import numpy as np
from genetic_algorithm.core.genetic_algorithm import GeneticAlgorithm
from genetic_algorithm.benchmarks.functions import BenchmarkFunctions
from genetic_algorithm.visualization.plotter import Plotter
from .report_generator import ReportGenerator


class ExperimentRunner:
    """Runs experiments and collects statistics"""

    def __init__(self):
        self.functions = BenchmarkFunctions.get_function_info()
        self.results = {}
        self.report_generator = ReportGenerator()

    def run_experiments(self, num_runs: int = 30) -> None:
        """Run all experimental configurations"""
        configurations = [
            {"representation": "binary", "crossover": "1-point", "mutation_rate": 0.05},
            {"representation": "binary", "crossover": "2-point", "mutation_rate": 0.05},
            {
                "representation": "real",
                "crossover": "arithmetic",
                "mutation_rate": 0.02,
            },
            {"representation": "real", "crossover": "blx-alpha", "mutation_rate": 0.02},
        ]

        print("Running experiments...")

        for func_name, func_info in self.functions.items():
            print(f"\nOptimizing {func_info['name']}...")
            self.results[func_name] = {}

            for config in configurations:
                config_name = f"{config['representation']}_{config['crossover']}"
                print(f"  Configuration: {config_name}")

                run_results = []
                fitness_histories = []
                avg_fitness_histories = []

                for run in range(num_runs):
                    if run % 10 == 0:
                        print(f"    Run {run + 1}/{num_runs}")

                    ga = GeneticAlgorithm(
                        objective_function=func_info["function"],
                        domain=func_info["domain"],
                        representation=config["representation"],
                        crossover_type=config["crossover"],
                        population_size=50,
                        mutation_rate=config["mutation_rate"],
                        crossover_rate=0.8,
                        max_evaluations=10000,
                    )

                    result = ga.run()
                    run_results.append(result["best_fitness"])
                    fitness_histories.append(result["fitness_history"])
                    avg_fitness_histories.append(result["avg_fitness_history"])

                # Calculate mean convergence curves
                mean_fitness_history = np.mean(fitness_histories, axis=0)
                mean_avg_fitness_history = np.mean(avg_fitness_histories, axis=0)

                self.results[func_name][config_name] = {
                    "fitnesses": run_results,
                    "mean": np.mean(run_results),
                    "std": np.std(run_results),
                    "best": np.min(run_results),
                    "worst": np.max(run_results),
                    "fitness_history": mean_fitness_history,
                    "avg_fitness_history": mean_avg_fitness_history,
                }

    def statistical_analysis(self) -> None:
        """Perform comprehensive statistical analysis of results and save to HTML file"""
        self.report_generator.generate_report(self.results, self.functions)

    def visualize_results(self) -> None:
        """Visualize functions and results"""
        print("Generating function visualizations...")
        Plotter.visualize_functions(self.functions)

        print("\nGenerating result plots...")
        Plotter.plot_results(self.results, self.functions)

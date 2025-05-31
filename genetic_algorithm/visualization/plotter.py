import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any


class Plotter:
    """Visualization utilities for genetic algorithm results"""

    @staticmethod
    def visualize_functions(functions_info: Dict[str, Dict[str, Any]]) -> None:
        """Create visualizations of the benchmark functions"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Benchmark Optimization Functions", fontsize=16)

        for idx, (_, func_info) in enumerate(functions_info.items()):
            # 2D Contour plot
            ax_contour = axes[idx, 0]
            domain = func_info["domain"]
            x = np.linspace(domain[0][0], domain[0][1], 100)
            y = np.linspace(domain[1][0], domain[1][1], 100)
            X, Y = np.meshgrid(x, y)
            Z = np.zeros_like(X)

            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    Z[i, j] = func_info["function"](X[i, j], Y[i, j])

            contour = ax_contour.contour(X, Y, Z, levels=20)
            ax_contour.clabel(contour, inline=True, fontsize=8)
            ax_contour.set_title(f'{func_info["name"]} - Contour Plot')
            ax_contour.set_xlabel("x")
            ax_contour.set_ylabel("y")
            ax_contour.plot(
                func_info["global_min"][0],
                func_info["global_min"][1],
                "r*",
                markersize=10,
                label="Global Min",
            )
            ax_contour.legend()

            # 3D Surface plot
            ax_3d = plt.subplot(2, 2, idx * 2 + 2, projection="3d")
            surface = ax_3d.plot_surface(X, Y, Z, cmap="viridis", alpha=0.8)
            ax_3d.set_title(f'{func_info["name"]} - 3D Surface')
            ax_3d.set_xlabel("x")
            ax_3d.set_ylabel("y")
            ax_3d.set_zlabel("f(x,y)")

        plt.tight_layout()
        plt.savefig("results/benchmark_functions.png", dpi=300, bbox_inches="tight")
        plt.show()

    @staticmethod
    def plot_results(
        results: Dict[str, Dict[str, Dict[str, Any]]],
        functions_info: Dict[str, Dict[str, Any]],
    ) -> None:
        """Plot experimental results"""
        # Performance comparison plot
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        for idx, (func_name, func_info) in enumerate(functions_info.items()):
            ax = axes[idx]

            configs = list(results[func_name].keys())
            means = [results[func_name][config]["mean"] for config in configs]
            stds = [results[func_name][config]["std"] for config in configs]

            x_pos = np.arange(len(configs))
            bars = ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7)

            ax.set_xlabel("Configuration")
            ax.set_ylabel("Mean Best Fitness")
            ax.set_title(f'{func_info["name"]} - Performance Comparison')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(configs, rotation=45)

            # Add value labels on bars
            for bar, mean, std in zip(bars, means, stds):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + std,
                    f"{mean:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        plt.tight_layout()
        plt.savefig("results/experimental_results.png", dpi=300, bbox_inches="tight")
        plt.show()

        # Convergence plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Convergence Analysis", fontsize=16)

        for idx, (func_name, func_info) in enumerate(functions_info.items()):
            # Best fitness convergence
            ax_best = axes[idx, 0]
            for config in results[func_name].keys():
                best_fitness = results[func_name][config]["fitness_history"]
                generations = range(len(best_fitness))
                ax_best.plot(generations, best_fitness, label=config, alpha=0.7)

            ax_best.set_xlabel("Generation")
            ax_best.set_ylabel("Best Fitness")
            ax_best.set_title(f'{func_info["name"]} - Best Fitness Convergence')
            ax_best.legend()
            ax_best.grid(True)

            # Average fitness convergence
            ax_avg = axes[idx, 1]
            for config in results[func_name].keys():
                avg_fitness = results[func_name][config]["avg_fitness_history"]
                generations = range(len(avg_fitness))
                ax_avg.plot(generations, avg_fitness, label=config, alpha=0.7)

            ax_avg.set_xlabel("Generation")
            ax_avg.set_ylabel("Average Fitness")
            ax_avg.set_title(f'{func_info["name"]} - Average Fitness Convergence')
            ax_avg.legend()
            ax_avg.grid(True)

        plt.tight_layout()
        plt.savefig("results/convergence_analysis.png", dpi=300, bbox_inches="tight")
        plt.show()

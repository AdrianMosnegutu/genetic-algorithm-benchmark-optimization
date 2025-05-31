from genetic_algorithm.utils.experiment_runner import ExperimentRunner


def main():
    """Main execution function"""
    print("Benchmark Optimization Functions Using Genetic Algorithms")
    print("=" * 60)

    # Create experiment runner
    runner = ExperimentRunner()

    # Run experiments
    runner.run_experiments(num_runs=30)

    # Statistical analysis
    runner.statistical_analysis()

    # Visualize results
    runner.visualize_results()

    print("\nExperiment completed!")
    print("Generated files:")
    print("- benchmark_functions.png: Function visualizations")
    print("- experimental_results.png: Performance comparison")


if __name__ == "__main__":
    main()

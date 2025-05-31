# Genetic Algorithm Benchmark Optimization

This project implements a genetic algorithm to optimize benchmark functions, comparing different genetic algorithm configurations and representations.

## Project Structure

```
genetic_algorithm/
├── benchmarks/
│   ├── __init__.py
│   └── functions.py      # Benchmark optimization functions
├── core/
│   ├── __init__.py
│   ├── individual.py     # Individual class for GA
│   └── genetic_algorithm.py  # Main GA implementation
├── utils/
│   ├── __init__.py
│   └── experiment_runner.py  # Experiment management
├── visualization/
│   ├── __init__.py
│   └── plotter.py        # Visualization utilities
├── __init__.py
└── main.py              # Main entry point
```

## Features

- Multiple benchmark functions (Ackley, Rastrigin)
- Different genetic algorithm representations:
  - Binary encoding
  - Real-valued encoding
- Various crossover operators:
  - 1-point crossover
  - 2-point crossover
  - Arithmetic crossover
  - BLX-α crossover
- Statistical analysis of results
- Visualization of functions and results

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the main script to execute the experiments:
```bash
python -m genetic_algorithm.main
```

This will:
1. Run experiments with different configurations
2. Perform statistical analysis
3. Generate visualizations:
   - `benchmark_functions.png`: Function visualizations
   - `experimental_results.png`: Performance comparison

## Dependencies

- numpy>=1.21.0
- matplotlib>=3.4.0
- scipy>=1.7.0
- pandas>=1.3.0 
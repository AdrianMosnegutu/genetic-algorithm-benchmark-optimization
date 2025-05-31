import random
import numpy as np
from typing import List, Tuple, Callable, Dict, Any
from .individual import Individual


class GeneticAlgorithm:
    """Configurable Genetic Algorithm implementation"""

    def __init__(
        self,
        objective_function: Callable,
        domain: List[List[float]],
        representation: str = "real",
        crossover_type: str = "arithmetic",
        population_size: int = 50,
        mutation_rate: float = None,
        crossover_rate: float = 0.8,
        max_evaluations: int = 10000,
    ):
        self.objective_function = objective_function
        self.domain = domain
        self.representation = representation
        self.crossover_type = crossover_type
        self.population_size = population_size
        # Set default mutation rates based on representation
        self.mutation_rate = (
            mutation_rate
            if mutation_rate is not None
            else (0.05 if representation == "binary" else 0.02)
        )
        self.crossover_rate = crossover_rate
        self.max_evaluations = max_evaluations
        self.evaluation_count = 0

        # Binary encoding parameters
        self.bits_per_variable = 16 if representation == "binary" else None

    def create_individual(self) -> Individual:
        """Create a random individual based on representation"""
        if self.representation == "binary":
            # Binary representation: encode each variable with specified bits
            genes = []
            for _ in range(2):  # 2D functions
                bits = [random.randint(0, 1) for _ in range(self.bits_per_variable)]
                genes.extend(bits)
            return Individual(genes)
        else:  # real-valued
            genes = [
                random.uniform(self.domain[i][0], self.domain[i][1]) for i in range(2)
            ]
            return Individual(genes)

    def decode_binary(self, binary_genes: List[int]) -> List[float]:
        """Decode binary genes to real values"""
        decoded = []
        for i in range(2):
            start_idx = i * self.bits_per_variable
            end_idx = start_idx + self.bits_per_variable
            binary_str = "".join(map(str, binary_genes[start_idx:end_idx]))
            decimal = int(binary_str, 2)
            max_decimal = 2**self.bits_per_variable - 1

            # Scale to domain
            min_val, max_val = self.domain[i]
            real_val = min_val + (decimal / max_decimal) * (max_val - min_val)
            decoded.append(real_val)
        return decoded

    def evaluate_fitness(self, individual: Individual) -> float:
        """Evaluate fitness of an individual"""
        if self.evaluation_count >= self.max_evaluations:
            return (
                individual.fitness if individual.fitness is not None else float("inf")
            )

        if self.representation == "binary":
            real_genes = self.decode_binary(individual.genes)
        else:
            real_genes = individual.genes

        fitness = self.objective_function(real_genes[0], real_genes[1])
        self.evaluation_count += 1
        return fitness

    def tournament_selection(
        self, population: List[Individual], tournament_size: int = 3
    ) -> Individual:
        """Tournament selection"""
        tournament = random.sample(population, tournament_size)
        return min(tournament, key=lambda x: x.fitness)

    def crossover_binary_1point(
        self, parent1: Individual, parent2: Individual
    ) -> Tuple[Individual, Individual]:
        """1-point crossover for binary representation"""
        if random.random() > self.crossover_rate:
            return Individual(parent1.genes[:]), Individual(parent2.genes[:])

        point = random.randint(1, len(parent1.genes) - 1)
        child1_genes = parent1.genes[:point] + parent2.genes[point:]
        child2_genes = parent2.genes[:point] + parent1.genes[point:]

        return Individual(child1_genes), Individual(child2_genes)

    def crossover_binary_2point(
        self, parent1: Individual, parent2: Individual
    ) -> Tuple[Individual, Individual]:
        """2-point crossover for binary representation"""
        if random.random() > self.crossover_rate:
            return Individual(parent1.genes[:]), Individual(parent2.genes[:])

        point1 = random.randint(1, len(parent1.genes) - 2)
        point2 = random.randint(point1 + 1, len(parent1.genes) - 1)

        child1_genes = (
            parent1.genes[:point1]
            + parent2.genes[point1:point2]
            + parent1.genes[point2:]
        )
        child2_genes = (
            parent2.genes[:point1]
            + parent1.genes[point1:point2]
            + parent2.genes[point2:]
        )

        return Individual(child1_genes), Individual(child2_genes)

    def crossover_arithmetic(
        self, parent1: Individual, parent2: Individual
    ) -> Tuple[Individual, Individual]:
        """Arithmetic crossover for real-valued representation"""
        if random.random() > self.crossover_rate:
            return Individual(parent1.genes[:]), Individual(parent2.genes[:])

        alpha = random.random()
        child1_genes = [
            alpha * p1 + (1 - alpha) * p2
            for p1, p2 in zip(parent1.genes, parent2.genes)
        ]
        child2_genes = [
            (1 - alpha) * p1 + alpha * p2
            for p1, p2 in zip(parent1.genes, parent2.genes)
        ]

        # Ensure genes stay within domain bounds
        for i in range(len(child1_genes)):
            child1_genes[i] = np.clip(
                child1_genes[i], self.domain[i][0], self.domain[i][1]
            )
            child2_genes[i] = np.clip(
                child2_genes[i], self.domain[i][0], self.domain[i][1]
            )

        return Individual(child1_genes), Individual(child2_genes)

    def crossover_blx_alpha(
        self, parent1: Individual, parent2: Individual, alpha: float = 0.5
    ) -> Tuple[Individual, Individual]:
        """BLX-α crossover for real-valued representation"""
        if random.random() > self.crossover_rate:
            return Individual(parent1.genes[:]), Individual(parent2.genes[:])

        child1_genes = []
        child2_genes = []

        for i in range(len(parent1.genes)):
            min_val = min(parent1.genes[i], parent2.genes[i])
            max_val = max(parent1.genes[i], parent2.genes[i])
            interval = max_val - min_val

            lower_bound = max(self.domain[i][0], min_val - alpha * interval)
            upper_bound = min(self.domain[i][1], max_val + alpha * interval)

            child1_genes.append(random.uniform(lower_bound, upper_bound))
            child2_genes.append(random.uniform(lower_bound, upper_bound))

        return Individual(child1_genes), Individual(child2_genes)

    def mutate(self, individual: Individual) -> Individual:
        """Mutation operator"""
        if self.representation == "binary":
            mutated_genes = individual.genes[:]
            for i in range(len(mutated_genes)):
                if random.random() < self.mutation_rate:
                    mutated_genes[i] = 1 - mutated_genes[i]  # Flip bit
            return Individual(mutated_genes)
        else:  # real-valued
            mutated_genes = individual.genes[:]
            for i in range(len(mutated_genes)):
                if random.random() < self.mutation_rate:
                    # Gaussian mutation with adaptive step size
                    sigma = (self.domain[i][1] - self.domain[i][0]) * 0.1
                    mutation = random.gauss(0, sigma)
                    mutated_genes[i] = np.clip(
                        mutated_genes[i] + mutation,
                        self.domain[i][0],
                        self.domain[i][1],
                    )
            return Individual(mutated_genes)

    def get_crossover_function(self) -> Callable:
        """Get the appropriate crossover function"""
        if self.representation == "binary":
            if self.crossover_type == "1-point":
                return self.crossover_binary_1point
            elif self.crossover_type == "2-point":
                return self.crossover_binary_2point
        else:  # real-valued
            if self.crossover_type == "arithmetic":
                return self.crossover_arithmetic
            elif self.crossover_type == "blx-alpha":
                return self.crossover_blx_alpha

        raise ValueError(f"Invalid crossover type: {self.crossover_type}")

    def run(self) -> Dict[str, Any]:
        """Run the genetic algorithm"""
        self.evaluation_count = 0

        # Initialize population
        population = [self.create_individual() for _ in range(self.population_size)]

        # Evaluate initial population
        for individual in population:
            individual.fitness = self.evaluate_fitness(individual)

        best_fitness_history = []
        avg_fitness_history = []
        generation = 0
        crossover_func = self.get_crossover_function()

        while self.evaluation_count < self.max_evaluations:
            # Selection and reproduction
            new_population = []

            while (
                len(new_population) < self.population_size
                and self.evaluation_count < self.max_evaluations
            ):
                parent1 = self.tournament_selection(population)
                parent2 = self.tournament_selection(population)

                child1, child2 = crossover_func(parent1, parent2)

                child1 = self.mutate(child1)
                child2 = self.mutate(child2)

                if self.evaluation_count < self.max_evaluations:
                    child1.fitness = self.evaluate_fitness(child1)
                if (
                    self.evaluation_count < self.max_evaluations
                    and len(new_population) < self.population_size - 1
                ):
                    child2.fitness = self.evaluate_fitness(child2)

                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)

            population = new_population[: self.population_size]

            # Track fitness statistics
            best_individual = min(population, key=lambda x: x.fitness)
            best_fitness_history.append(best_individual.fitness)
            avg_fitness = np.mean([ind.fitness for ind in population])
            avg_fitness_history.append(avg_fitness)

            generation += 1

        best_individual = min(population, key=lambda x: x.fitness)
        if self.representation == "binary":
            best_solution = self.decode_binary(best_individual.genes)
        else:
            best_solution = best_individual.genes

        return {
            "best_fitness": best_individual.fitness,
            "best_solution": best_solution,
            "fitness_history": best_fitness_history,
            "avg_fitness_history": avg_fitness_history,
            "generations": generation,
            "evaluations": self.evaluation_count,
        }

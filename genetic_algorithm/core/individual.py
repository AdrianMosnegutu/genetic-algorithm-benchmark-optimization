from typing import List, Optional, Any


class Individual:
    """Represents an individual in the genetic algorithm population"""

    def __init__(self, genes: List[Any], fitness: Optional[float] = None):
        self.genes = genes
        self.fitness = fitness

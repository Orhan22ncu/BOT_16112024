import numpy as np
from deap import base, creator, tools, algorithms
import random
from typing import List, Tuple, Callable

class EvolutionaryOptimizer:
    def __init__(self, 
                 param_bounds: dict,
                 population_size: int = 50,
                 n_generations: int = 100):
        self.param_bounds = param_bounds
        self.population_size = population_size
        self.n_generations = n_generations
        self._setup_evolutionary_algorithm()
        
    def _setup_evolutionary_algorithm(self):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        self.toolbox = base.Toolbox()
        
        # Attribute generator
        for i, (param, bounds) in enumerate(self.param_bounds.items()):
            self.toolbox.register(
                f"attr_{i}",
                random.uniform,
                bounds[0],
                bounds[1]
            )
        
        # Structure initializers
        self.toolbox.register(
            "individual",
            tools.initCycle,
            creator.Individual,
            [getattr(self.toolbox, f"attr_{i}") 
             for i in range(len(self.param_bounds))],
            n=1
        )
        
        self.toolbox.register(
            "population",
            tools.initRepeat,
            list,
            self.toolbox.individual
        )
        
        # Genetic operators
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", self._mutate_individual)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        
    def _mutate_individual(self, individual: List[float], 
                          indpb: float = 0.1) -> Tuple[List[float]]:
        """Custom mutation operator"""
        for i, (param, bounds) in enumerate(self.param_bounds.items()):
            if random.random() < indpb:
                individual[i] = random.uniform(bounds[0], bounds[1])
        return individual,
    
    def optimize(self, fitness_function: Callable) -> Tuple[List[float], float]:
        """Run evolutionary optimization"""
        # Register evaluation function
        self.toolbox.register("evaluate", fitness_function)
        
        # Initialize population
        pop = self.toolbox.population(n=self.population_size)
        hof = tools.HallOfFame(1)
        
        # Statistics setup
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        # Run evolution
        pop, log = algorithms.eaSimple(
            pop,
            self.toolbox,
            cxpb=0.7,  # Crossover probability
            mutpb=0.2,  # Mutation probability
            ngen=self.n_generations,
            stats=stats,
            halloffame=hof,
            verbose=True
        )
        
        return hof[0], hof[0].fitness.values[0]
    
    def _validate_bounds(self, individual: List[float]) -> List[float]:
        """Ensure individual stays within bounds"""
        for i, (param, bounds) in enumerate(self.param_bounds.items()):
            individual[i] = max(bounds[0], min(bounds[1], individual[i]))
        return individual
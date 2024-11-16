from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import numpy as np
from scipy.stats import norm
from typing import Callable, Dict, List

class BayesianOptimizer:
    def __init__(self, 
                 param_space: Dict[str, tuple],
                 n_initial_points: int = 5):
        self.param_space = param_space
        self.n_initial_points = n_initial_points
        self.X_observed = []
        self.y_observed = []
        
        # Initialize Gaussian Process
        kernel = C(1.0) * RBF([1.0] * len(param_space))
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=10
        )
    
    def optimize(self, objective_function: Callable, n_iterations: int = 50):
        # Initial random sampling
        X_init = self._sample_initial_points()
        y_init = [objective_function(x) for x in X_init]
        
        self.X_observed.extend(X_init)
        self.y_observed.extend(y_init)
        
        # Bayesian optimization loop
        for i in range(n_iterations):
            # Update GP
            X = np.array(self.X_observed)
            y = np.array(self.y_observed)
            self.gp.fit(X, y)
            
            # Find next point to evaluate
            next_point = self._get_next_point()
            next_value = objective_function(next_point)
            
            self.X_observed.append(next_point)
            self.y_observed.append(next_value)
            
            print(f"Iteration {i + 1}/{n_iterations}")
            print(f"Best value: {max(self.y_observed)}")
        
        # Return best parameters
        best_idx = np.argmax(self.y_observed)
        return self.X_observed[best_idx], self.y_observed[best_idx]
    
    def _sample_initial_points(self):
        points = []
        for _ in range(self.n_initial_points):
            point = []
            for param_range in self.param_space.values():
                if isinstance(param_range[0], int):
                    value = np.random.randint(param_range[0], param_range[1])
                else:
                    value = np.random.uniform(param_range[0], param_range[1])
                point.append(value)
            points.append(point)
        return points
    
    def _get_next_point(self):
        def expected_improvement(X):
            mu, sigma = self.gp.predict(X.reshape(1, -1), return_std=True)
            best_y = max(self.y_observed)
            
            with np.errstate(divide='warn'):
                imp = mu - best_y
                Z = imp / sigma
                ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
                ei[sigma == 0.0] = 0.0
                
            return -ei
        
        # Random search for optimization
        best_ei = float('inf')
        best_point = None
        
        for _ in range(100):
            point = self._sample_initial_points()[0]
            ei = expected_improvement(np.array(point))
            
            if ei < best_ei:
                best_ei = ei
                best_point = point
        
        return best_point
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from tabulate import tabulate
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict

class DifferentialEvolution:
    def __init__(
        self,
        pop_size: int = 50,
        F: float = 0.8,
        CR: float = 0.7,
        generations: int = 100
    ):
        """
        Initialize Differential Evolution optimizer.

        Args:
            pop_size: Population size
            F: Mutation factor
            CR: Crossover rate
            generations: Number of generations
        """
        self.pop_size = pop_size
        self.F = F
        self.CR = CR
        self.generations = generations
        self.best_solution = None
        self.best_fitness = float('inf')
        self.fitness_history = []

    def initialize_population(self, dimensions: int) -> np.ndarray:
        """Initialize random population."""
        return np.random.uniform(-1, 1, (self.pop_size, dimensions))

    def mutate(self, population: np.ndarray) -> np.ndarray:
        """Apply mutation using DE/rand/1 strategy."""
        mutated = np.zeros_like(population)

        for i in range(self.pop_size):
            # Select three random vectors
            candidates = list(range(self.pop_size))
            candidates.remove(i)
            r1, r2, r3 = np.random.choice(candidates, 3, replace=False)

            # Create mutant
            mutated[i] = population[r1] + self.F * (population[r2] - population[r3])

        return mutated

    def crossover(self, population: np.ndarray, mutated: np.ndarray) -> np.ndarray:
        """Apply binomial crossover."""
        mask = np.random.rand(*population.shape) < self.CR
        return np.where(mask, mutated, population)

    def compute_fitness(self, solution: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        """Compute MSE fitness."""
        y_pred = np.dot(X, solution)
        return mean_squared_error(y, y_pred)

    def optimize(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, List[float]]:
        """Main optimization loop."""
        dimensions = X.shape[1]
        population = self.initialize_population(dimensions)

        for gen in range(self.generations):
            # Create trial population
            mutated = self.mutate(population)
            trial_pop = self.crossover(population, mutated)

            # Selection
            for i in range(self.pop_size):
                trial_fitness = self.compute_fitness(trial_pop[i], X, y)
                current_fitness = self.compute_fitness(population[i], X, y)

                if trial_fitness < current_fitness:
                    population[i] = trial_pop[i]
                    if trial_fitness < self.best_fitness:
                        self.best_fitness = trial_fitness
                        self.best_solution = trial_pop[i].copy()

            self.fitness_history.append(self.best_fitness)

        return self.best_solution, self.fitness_history

def create_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create synthetic dataset."""
    X, y = make_regression(n_samples=200, n_features=5, noise=0.1, random_state=42)
    return train_test_split(X, y, test_size=0.2, random_state=42)

def evaluate_model(
    model: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, float]:
    """Evaluate model performance."""
    y_train_pred = np.dot(X_train, model)
    y_test_pred = np.dot(X_test, model)

    return {
        'Train MSE': mean_squared_error(y_train, y_train_pred),
        'Test MSE': mean_squared_error(y_test, y_test_pred),
        'Train R2': r2_score(y_train, y_train_pred),
        'Test R2': r2_score(y_test, y_test_pred)
    }

def plot_convergence(fitness_history: List[float]) -> None:
    """Plot convergence curve."""
    plt.figure(figsize=(10, 6))
    plt.plot(fitness_history, 'b-', label='Best Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Mean Squared Error')
    plt.title('Convergence Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_predictions(
    model: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> None:
    """Plot predicted vs actual values."""
    # Calculate predictions
    y_train_pred = np.dot(X_train, model)
    y_test_pred = np.dot(X_test, model)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Training data plot
    ax1.scatter(y_train, y_train_pred, alpha=0.5, label='Training Data')
    ax1.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', label='Perfect Fit')
    ax1.set_xlabel('Actual Values')
    ax1.set_ylabel('Predicted Values')
    ax1.set_title('Training Set: Predicted vs Actual')
    ax1.legend()
    ax1.grid(True)

    # Test data plot
    ax2.scatter(y_test, y_test_pred, alpha=0.5, label='Test Data')
    ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Perfect Fit')
    ax2.set_xlabel('Actual Values')
    ax2.set_ylabel('Predicted Values')
    ax2.set_title('Test Set: Predicted vs Actual')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

def main():
    # Create dataset
    X_train, X_test, y_train, y_test = create_dataset()

    # Initialize and run optimizer
    de = DifferentialEvolution(pop_size=50, F=0.8, CR=0.7, generations=100)
    best_solution, fitness_history = de.optimize(X_train, y_train)

    # Evaluate model
    metrics = evaluate_model(best_solution, X_train, y_train, X_test, y_test)

    # Display results
    headers = ['Metric', 'Value']
    table_data = [[k, f"{v:.6f}"] for k, v in metrics.items()]
    print("\nModel Performance Metrics:")
    print(tabulate(table_data, headers=headers, tablefmt='grid'))

    # Plot convergence
    plot_convergence(fitness_history)

    # Plot predictions vs actual values
    plot_predictions(best_solution, X_train, y_train, X_test, y_test)

    # Model analysis
    print("\nModel Analysis:")
    if metrics['Train MSE'] > 1.0:
        print("⚠️ No improvement: The model failed to learn from the training data")
    else:
        print("✅ Good optimization: Model successfully minimized the error")

    train_test_diff = abs(metrics['Train MSE'] - metrics['Test MSE'])
    if train_test_diff < 0.1:
        print("✅ Good generalization: Model performs similarly on training and test data")
    else:
        print("⚠️ Poor generalization: Large difference between training and test performance")

    if metrics['Train R2'] > 0.9 and metrics['Test R2'] > 0.9:
        print("✅ Excellent fit: High R² scores indicate strong predictive power")
    elif metrics['Train R2'] > 0.7 and metrics['Test R2'] > 0.7:
        print("✅ Good fit: Acceptable R² scores for practical use")
    else:
        print("⚠️ Poor fit: Low R² scores indicate weak predictive power")

if __name__ == "__main__":
    main()
import numpy as np
import pandas as pd
import pathlib


class DataCollector:
    """Collect data for analysis."""
    def __init__(self, n_features, n_iterations):
        self.stds = np.empty([n_iterations+1, n_features])
        self.means = np.empty([n_iterations+1, n_features])
        self.best_scores = np.empty([n_iterations+1, 1])
        self.best_features = np.empty([n_iterations+1, n_features])
        self.results = None

    def add_metrics(self, iteration, population, population_scores, best_individual_features, best_individual_score):
        """Add metrics from single iteration."""
        self.stds[iteration] = population.std(axis=0)
        self.means[iteration] = population.mean(axis=0)
        self.best_scores[iteration] = best_individual_score
        self.best_features[iteration] = best_individual_features

    def create_dirs(self, file_path):
        """Create directory where data will be stored in csv format."""
        pathlib.Path(file_path.parents[0]).mkdir(parents=True, exist_ok=True)

    def save_data(self, file_path):
        """Save data to csv."""
        self.create_dirs(file_path)
        results = pd.DataFrame(np.concatenate([self.stds, self.means, self.best_features, self.best_scores], axis=1),
                               columns=
                               ["std_x0", "std_x1", "std_x2", "std_x3", "std_x4", "std_x5", "std_x6", "std_x7", "std_x8",
                                "std_x9",
                                'mean_x0', 'mean_x1', 'mean_x2', 'mean_x3', 'mean_x4', 'mean_x5', 'mean_x6', 'mean_x7',
                                'mean_x8', 'mean_x9',
                                'best_x0', 'best_x1', 'best_x2', 'best_x3', 'best_x4', 'best_x5', 'best_x6', 'best_x7', 'best_x8', 'best_x9',
                                'best_value'])
        results.index.name = "idx"
        results.to_csv(file_path)
        self.results = results


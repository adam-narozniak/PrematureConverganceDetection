import matplotlib.pyplot as plt
import pathlib
import pandas as pd
import numpy as np


class Plotter:
    """Visualize data to gain insides."""
    def __init__(self, data, stopped_in_iteration, name, plots_path=pathlib.Path.cwd() / "plots" / "all_fnc"):
        self.plots_path = plots_path
        self.data = data
        self.stopped_in_iteration = stopped_in_iteration
        self.name = name
        self.best_instance_cmp_path = plots_path / "best_instance_cmp"
        self.create_dirs()

    def create_dirs(self):
        """Create directories."""
        self.plots_path.mkdir(parents=True, exist_ok=True)
        self.best_instance_cmp_path.mkdir(parents=True, exist_ok=True)

    def plot_std_on_best_x(self):
        fig, ax = plt.subplots()
        ax.set_title(f"{self.name} " + f"std_x0 vs best individual x0")
        ax.set_xlabel("iterations")
        ax.plot(self.data.index.values, self.data.std_x0, label=self.data.std_x0.name)
        ax.plot(self.data.index.values, self.data.best_x0, label=self.data.best_x0.name)
        if self.stopped_in_iteration != -1:
            ax.axvline(self.stopped_in_iteration, ls="--", color="r", label="stop")
        ax.legend()
        plt.savefig(self.plots_path / (f"{self.name}_" + f"std_x0_vs_best_x0.jpg"))
        plt.close(fig)

    def plot_mean_on_best_x(self):
        fig, ax = plt.subplots()
        ax.set_title(f"{self.name} " + f"mean_x0 vs best individual x0")
        ax.set_xlabel("iterations")
        ax.plot(self.data.index.values, self.data.mean_x0, label=self.data.mean_x0.name)
        ax.plot(self.data.index.values, self.data.best_x0, label=self.data.best_x0.name)
        if self.stopped_in_iteration != -1:
            ax.axvline(self.stopped_in_iteration, ls="--", color="r", label="stop")
        ax.legend()
        plt.savefig(self.plots_path / (f"{self.name}_" + f"mean_x0_vs_best_x0.jpg"))
        plt.close(fig)

    def plot_best_individual_value_vs_std_x(self):
        fig, ax = plt.subplots()
        ax.set_title(f"{self.name} " + f"std_x0 vs best individual score")
        ax.set_xlabel("iterations")
        ax.set_ylabel("best value", color='b')
        line1 = ax.plot(self.data.index.values, self.data.best_value, label=self.data.best_value.name, color='b')
        ax2 = ax.twinx()
        line2 = ax2.plot(self.data.index.values, self.data.std_x0, label=self.data.std_x0.name, color='orange')
        ax2.set_ylabel("std_x0 value", color='orange')
        lines = line1 + line2
        if self.stopped_in_iteration != -1:
            vertical = ax.axvline(self.stopped_in_iteration, ls="--", color="r", label="stop")
            lines += [vertical]

        ax2.legend(lines, [line.get_label() for line in lines])
        plt.savefig(self.plots_path / (f"{self.name}_" + f"best_individual_value_vs_std_x0.jpg"))
        plt.close(fig)

    def plot_best_individual_value_vs_mean_x(self):
        fig, ax = plt.subplots()
        ax.set_title(f"{self.name}" + f" mean_x0 vs best individual score")
        ax.set_xlabel("iterations")
        ax.set_ylabel("best value", color='b')
        line1 = ax.plot(self.data.index.values, self.data.best_value, label=self.data.best_value.name, color='b')
        ax2 = ax.twinx()
        line2 = ax2.plot(self.data.index.values, self.data.mean_x0, label=self.data.mean_x0.name, color='orange')
        ax2.set_ylabel("mean_x0 value", color='orange')
        lines = line1 + line2
        if self.stopped_in_iteration != -1:
            vertical = ax.axvline(self.stopped_in_iteration, ls="--", color="r", label="stop")
            lines += [vertical]

        ax2.legend(lines, [line.get_label() for line in lines])
        plt.savefig(self.plots_path / (f"{self.name}_" + f"best_individual_value_vs_mean_x0.jpg"))
        plt.close(fig)

    def plot_cmp_bests_variants(self):
        """Plot stops on scores of best instance for all different stopping strategies"""
        color_representation = self.make_color_representation_for_best_individual()
        mins, maks = self.make_line_len_representation_for_best_individual()
        fig, ax = plt.subplots()
        variants = self.stopped_in_iteration
        ax.set_xlabel("iterations")
        ax.set_ylabel("scores")
        plt.title("Comparison of stop variants")
        line1 = ax.plot(self.data.index.values, self.data.best_value, label=self.data.best_value.name, color='b')
        stop_lines = []
        for look_back_by in variants.index.values:
            for ratio in variants.columns.values:
                if variants.loc[look_back_by, ratio] != 0:

                    line = ax.axvline(x=variants.loc[look_back_by, ratio], ymin = mins.loc[look_back_by, ratio], ymax = maks.loc[look_back_by, ratio], ls="--", color=color_representation.loc[look_back_by, ratio], label=f"stop, ratio: {ratio}, look_back: {look_back_by}")
                    stop_lines.append(line)
        lines = line1 + stop_lines
        ax.legend(lines, [l.get_label() for l in lines])
        plt.savefig(self.best_instance_cmp_path / (f"{self.name}_" + f"cmp_stop_conditions.jpg"))
        plt.close(fig)

    def make_color_representation_for_best_individual(self):
        """Choose colors for stop places for 3 by 3 search."""
        look_back_by = [10, 25, 50]
        count_as_stuck_if_ratio_is_less_than = [1.025, 1.05, 1.1]
        return pd.DataFrame(np.array(["r", "g", "c", "m", "y", "brown", "pink", "orange", "teal"]).reshape(3, 3),
                            columns=count_as_stuck_if_ratio_is_less_than, index=look_back_by)


    def make_line_len_representation_for_best_individual(self):
        """Prepare line widths for parameters range search to differentiate them. This is crucial cause otherwise
        they were plotted on each other. """
        look_back_by = [10, 25, 50]
        count_as_stuck_if_ratio_is_less_than = [1.025, 1.05, 1.1]
        part = 1/9
        min = -part
        max = 0
        mins = []
        maks = []
        for stop in range(1,10):
            min +=part
            max +=part
            mins.append(min)
            maks.append(max)
        mins = pd.DataFrame(np.array(mins).reshape(3, 3), columns=count_as_stuck_if_ratio_is_less_than, index=look_back_by)
        maks = pd.DataFrame(np.array(maks).reshape(3, 3),  columns=count_as_stuck_if_ratio_is_less_than, index=look_back_by)
        return mins, maks

    def plot_cmp_stds_variants(self):
        """Plot stops on scores of stds for all different stopping strategies"""
        color_representation = self.make_color_representation_for_stds()
        mins, maks = self.make_line_len_representation_for_stds()
        fig, ax = plt.subplots()
        variants_std = self.stopped_in_iteration
        ax.set_xlabel("iterations")
        ax.set_ylabel("scores")
        plt.title("Comparison of stop variants")
        line1 = ax.plot(self.data.index.values, self.data.best_value, label=self.data.best_value.name, color='b')
        stop_lines = []
        for th in variants_std.index.values:
            if variants_std.loc[th] != 0:
                line = ax.axvline(x=variants_std.loc[th], ymin = mins.loc[th], ymax = maks.loc[th], ls="--", color=color_representation.loc[th], label=f"stop, threshold: {th}")
                stop_lines.append(line)
        lines = line1 + stop_lines
        ax.legend(lines, [l.get_label() for l in lines])
        plt.savefig(self.best_instance_cmp_path / (f"{self.name}_" + f"cmp_stop_conditions.jpg"))
        plt.close(fig)

    def make_line_len_representation_for_stds(self):
        """Prepare line widths for parameters range search to differentiate them. This is crucial cause otherwise
        they were plotted on each other. """
        thresholds = [2, 0.1, 0.001]
        part = 1/3
        min = -part
        max = 0
        mins = []
        maks = []
        for stop in range(1, 4):
            min +=part
            max +=part
            mins.append(min)
            maks.append(max)
        mins = pd.Series(np.array(mins), index=thresholds)
        maks = pd.Series(np.array(maks), index=thresholds)
        return mins, maks

    def make_color_representation_for_stds(self):
        """Choose colors for stop places for 3 param search."""
        thresholds = [2, 0.1, 0.001]
        return pd.Series(np.array(["r", "g", "c"] ), index=thresholds)
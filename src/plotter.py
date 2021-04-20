import matplotlib.pyplot as plt
import pathlib


class Plotter:
    def __init__(self, data, stopped_in_iteration, name, plots_path=pathlib.Path.cwd()/"plots"/"all_fnc"):
        self.plots_path = plots_path
        self.data = data
        self.stopped_in_iteration = stopped_in_iteration
        self.name = name
        self.create_dirs()
    def create_dirs(self):
        self.plots_path.mkdir(parents=True, exist_ok=True)
        
    def plot_std_on_best_x(self):
        fig, ax = plt.subplots()
        ax.set_title(f"{self.name} "+f"std_x0 vs best individual x0")
        ax.set_xlabel("iterations")
        ax.plot(self.data.index.values, self.data.std_x0, label=self.data.std_x0.name)
        ax.plot(self.data.index.values, self.data.best_x0, label=self.data.best_x0.name)
        if self.stopped_in_iteration != -1:
            ax.axvline(self.stopped_in_iteration, ls="--", color="r", label="stop")
        ax.legend()
        plt.savefig(self.plots_path/(f"{self.name}_"+f"std_x0_vs_best_x0.jpg"))
        plt.close(fig)
    
    def plot_mean_on_best_x(self):
        fig, ax = plt.subplots()
        ax.set_title(f"{self.name} "+f"mean_x0 vs best individual x0")
        ax.set_xlabel("iterations")
        ax.plot(self.data.index.values, self.data.mean_x0, label=self.data.mean_x0.name)
        ax.plot(self.data.index.values, self.data.best_x0, label=self.data.best_x0.name)
        if self.stopped_in_iteration != -1:
            ax.axvline(self.stopped_in_iteration, ls="--", color="r", label="stop")
        ax.legend()
        plt.savefig(self.plots_path/(f"{self.name}_"+f"mean_x0_vs_best_x0.jpg"))
        plt.close(fig)
    
    def plot_best_individual_value_vs_std_x(self):
        fig, ax = plt.subplots()
        ax.set_title(f"{self.name} "+f"std_x0 vs best individual score")
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
        plt.savefig(self.plots_path/(f"{self.name}_"+f"best_individual_value_vs_std_x0.jpg"))
        plt.close(fig)
    
    
    def plot_best_individual_value_vs_mean_x(self):
        fig, ax = plt.subplots()
        ax.set_title(f"{self.name}"+f" mean_x0 vs best individual score")
        ax.set_xlabel("iterations")
        ax.set_ylabel("best value", color='b')
        line1 = ax.plot(self.data.index.values, self.data.best_value, label=self.data.best_value.name, color='b')
        ax2 = ax.twinx()
        line2 = ax2.plot(self.data.index.values, self.data.mean_x0, label=self.data.mean_x0.name, color='orange')
        ax2.set_ylabel("mean_x0 value", color='orange')
        lines = line1+line2
        if self.stopped_in_iteration != -1:
            vertical = ax.axvline(self.stopped_in_iteration, ls="--", color="r", label="stop")
            lines+=[vertical]
    
        ax2.legend(lines, [line.get_label() for line in lines])
        plt.savefig(self.plots_path/(f"{self.name}_"+f"best_individual_value_vs_mean_x0.jpg"))
        plt.close(fig)

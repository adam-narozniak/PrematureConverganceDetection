import matplotlib.pyplot as plt
import pathlib
plots_path = pathlib.Path.cwd()/"plots"/"all_fnc"

def plot_std_on_best_x(data, name):
    fig, ax = plt.subplots()
    ax.set_title(f"{name} "+f"std_x0 vs best individual x0")
    ax.set_xlabel("iterations")
    ax.plot(data.index.values, data.std_x0, label=data.std_x0.name)
    ax.plot(data.index.values, data.best_x0, label=data.best_x0.name)
    ax.legend()
    plt.savefig(plots_path/(f"{name}_"+f"std_x0_vs_best_x0.jpg"))
    plt.close(fig)

def plot_mean_on_best_x(data, name):
    fig, ax = plt.subplots()
    ax.set_title(f"{name} "+f"mean_x0 vs best individual x0")
    ax.set_xlabel("iterations")
    ax.plot(data.index.values, data.mean_x0, label=data.mean_x0.name)
    ax.plot(data.index.values, data.best_x0, label=data.best_x0.name)
    ax.legend()
    plt.savefig(plots_path/(f"{name}_"+f"mean_x0_vs_best_x0.jpg"))
    plt.close(fig)

def plot_best_individual_value_vs_std_x(data, name):
    fig, ax = plt.subplots()
    ax.set_title(f"{name} "+f"std_x0 vs best individual score")
    ax.set_xlabel("iterations")
    ax.set_ylabel("best value", color='b')
    line1 = ax.plot(data.index.values, data.best_value, label=data.best_value.name, color='b')
    ax2 = ax.twinx()
    line2 = ax2.plot(data.index.values, data.std_x0, label=data.std_x0.name, color='orange')
    ax2.set_ylabel("std_x0 value", color='orange')
    lines = line1 + line2
    ax2.legend(lines, [line.get_label() for line in lines])
    plt.savefig(plots_path/(f"{name}_"+f"best_individual_value_vs_std_x0.jpg"))
    plt.close(fig)


def plot_best_individual_value_vs_mean_x(data, name):
    fig, ax = plt.subplots()
    ax.set_title(f"{name}"+f" mean_x0 vs best individual score")
    ax.set_xlabel("iterations")
    ax.set_ylabel("best value", color='b')
    line1 = ax.plot(data.index.values, data.best_value, label=data.best_value.name, color='b')
    ax2 = ax.twinx()
    line2 = ax2.plot(data.index.values, data.mean_x0, label=data.mean_x0.name, color='orange')
    ax2.set_ylabel("mean_x0 value", color='orange')
    lines = line1+line2
    ax2.legend(lines, [line.get_label() for line in lines])
    plt.savefig(plots_path/(f"{name}_"+f"best_individual_value_vs_mean_x0.jpg"))
    plt.close(fig)

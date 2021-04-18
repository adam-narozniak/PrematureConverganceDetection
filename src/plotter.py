import matplotlib.pyplot as plt
import pathlib
plots_path = pathlib.Path.cwd()/"plots"

def plot_std_on_best_x(data):
    plt.figure()
    plt.title("std_x0 vs best individual x0")
    plt.xlabel("iterations")
    plt.plot(data.index.values, data.std_x0, label=data.std_x0.name)
    plt.plot(data.index.values, data.best_x0, label=data.best_x0.name)
    plt.legend()
    plt.savefig(plots_path/"std_x0_vs_best_x0.jpg")

def plot_mean_on_best_x(data):
    plt.figure()
    plt.title("mean_x0 vs best individual x0")
    plt.xlabel("iterations")
    plt.plot(data.index.values, data.mean_x0, label=data.mean_x0.name)
    plt.plot(data.index.values, data.best_x0, label=data.best_x0.name)
    plt.legend()
    plt.savefig(plots_path/"mean_x0_vs_best_x0.jpg")

def plot_best_individual_value_vs_std_x(data):
    fig, ax = plt.subplots()
    ax.set_title("std_x0 vs best individual score")
    ax.set_xlabel("iterations")
    ax.set_ylabel("best value", color='blue')
    line1 = ax.plot(data.index.values, data.best_value, label=data.best_value.name, color='blue')
    ax2 = ax.twinx()
    line2 = ax2.plot(data.index.values, data.std_x0, label=data.std_x0.name, color='orange')
    ax2.set_ylabel("std x0 value", color='orange')
    lines = line1 + line2
    ax2.legend(lines, [line.get_label() for line in lines])
    plt.savefig(plots_path/"best_individual_value_vs_std_x0.jpg")


def plot_best_individual_value_vs_mean_x(data):
    fig, ax = plt.subplots()
    ax.set_title("mean_x0 vs best individual score")
    ax.set_xlabel("iterations")
    ax.set_ylabel("best value", color='blue')
    line1 = ax.plot(data.index.values, data.best_value, label=data.best_value.name, color='blue')
    ax2 = ax.twinx()
    line2 = ax2.plot(data.index.values, data.mean_x0, label=data.mean_x0.name, color='orange')
    ax2.set_ylabel("std x0 value", color='orange')
    lines = line1+line2
    ax2.legend(lines, [line.get_label() for line in lines])
    plt.savefig(plots_path/"best_individual_value_vs_mean_x0.jpg")

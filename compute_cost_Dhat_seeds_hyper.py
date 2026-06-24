from EncoderDecoder.utils import loadData, compute_avg

from PCAfold import compute_normalized_variance, normalized_variance_derivative, cost_function_normalized_variance_derivative, plot_normalized_variance_derivative
import numpy as np
import matplotlib.pyplot as plt
import sys
from itertools import product
import torch
import logging
logging.disable(logging.CRITICAL)

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

path_data = "data-files/"
dataset_type = "flamelet"
penalty_function = 'log-sigma-over-peak'
start_bw = -6
end_bw = 2
nbr_points_bw = 100
bandwidth_values = np.logspace(start_bw, end_bw, nbr_points_bw)
power = 4
vertical_shift = 1

nbr_seeds = 6

learning_rates = [0.025]
optimizers = ["RMSprop"]
lists_species_output_QoI = [
    ("lin", ['H2O2', 'H2O', 'H2', 'HO2', 'N2O', 'NO2', 'NO', 'O2', 'OH']),
    ("linLog", ['H2O2', 'H2O', 'H2', 'HO2', 'N2O', 'NO2', 'NO', 'O2', 'OH', 'logH2O2', 'logH2O', 'logH2', 'logHO2', 'logN2O', 'logNO2', 'logNO', 'logO2', 'logOH']),
    ("log", ['logH2O2', 'logH2O', 'logH2', 'logHO2', 'logN2O', 'logNO2', 'logNO', 'logO2', 'logOH'])
]
list_input_scaling_name = ["0to1", "-1to1", "std", "pareto", "mean-pareto"] #"0to1", "-1to1", "std", "pareto", "mean-pareto"
list_species_scaling_layer = [False, True]
seeds = list(range(nbr_seeds))

experiment_configs = []

for lr_i, opt_i, (species_tag, species_i), input_scaling_name, species_scaling_layer, seed_i in product(
    learning_rates,
    optimizers,
    lists_species_output_QoI,
    list_input_scaling_name,
    list_species_scaling_layer,
    seeds):

    config = {
        "lr": lr_i,
        "optimizer": opt_i,
        "output_species": species_i,
        "species_tag": species_tag,
        "input_scaling_name": input_scaling_name,
        "species_scaling_layer": species_scaling_layer,
        "seed": seed_i,
    }
    experiment_configs.append(config)

nbr_experiment_configs = len(experiment_configs)
print(f"Total number of runs: {nbr_experiment_configs}")

list_avg_cost = []

for idxConfig, config in enumerate(experiment_configs):

    optimizer_name = config["optimizer"]
    lr = config["lr"]
    list_species_output = config["output_species"]
    species_tag = config["species_tag"]
    input_scaling_name = config["input_scaling_name"]
    species_scaling_layer = config["species_scaling_layer"]
    my_seed = config["seed"]

    filename = f"Tr2_2PV_RMSprop_250_{species_tag}_{input_scaling_name}_scaling{species_scaling_layer}_s{my_seed}-AE-date_23Jun2026-hour_15h47_Xu-flamelet-augm"

    loader = loadData(filename)
    input, output = loader.getInputOutputAnalysis(path_data, dataset_type)

    #scale every column of the input tensor between 0 and 1
    min_vals = np.min(input, axis=0, keepdims=True)
    max_vals = np.max(input, axis=0, keepdims=True)
    input_scaled = (input - min_vals) / (max_vals - min_vals)

    indepVars = input_scaled
    depVars = output

    depvar_names = loader.metadata["list_species_output_evaluation"]
    if(loader.metadata["temperature_output"]):
        depvar_names.append("T")
    for i in range(1,1+loader.metadata["PV_dim"]):
        depvar_names.append(f"PV{i}")

    print(filename)
    print(indepVars)
    print(depVars)

    variance_data = compute_normalized_variance(indepVars,
                                                    depVars,
                                                    depvar_names=depvar_names,
                                                    bandwidth_values=bandwidth_values)
    np.save(f"data-files/costs/variance/variance_{filename}-dataset_{dataset_type}.npy", variance_data)

    costs = cost_function_normalized_variance_derivative(variance_data,
                                                        penalty_function=penalty_function,
                                                        power=power,
                                                        vertical_shift=vertical_shift,
                                                        norm=None)
    np.save(f"data-files/costs/costs/costs_{filename}-dataset_{dataset_type}.npy", costs)

    (derivative, bandwidth_values, max_derivative) = normalized_variance_derivative(variance_data)

    plt = plot_normalized_variance_derivative(variance_data)
    plt.savefig(f"data-files/costs/figure/plot_Dhat_{filename}-dataset_{dataset_type}.png")
    plt.close()

    list_avg_cost.append(compute_avg(np.array(costs)))
    print(f"{idxConfig+1}/{nbr_experiment_configs}: {filename} done. Average cost of {np.round(compute_avg(np.array(costs)),2)}.")

print()
print("Computation complete")
print()

for cost in list_avg_cost:
    print(f"{np.round(cost,2)}")
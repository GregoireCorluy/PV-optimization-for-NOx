"""
PV optimization on DNS dataset of Xu using an encoder-decoder architecture (Kamila Zdybal)
Version: Sped up version without using the dataloader during the training and using manual batches - train multiple times the same NN with different seeds
Author: Grégoire Corlùy (gregoire.stephane.corluy@ulb.be)
Date: September 2024
Python version: 3.10.10
"""


########
#Imports
########

from EncoderDecoder.models import PV_autoencoder
from EncoderDecoder.utils import Species, create_dirs, get_dataset_training, get_optimizer, get_loss_criterion, compute_Kreg, rescale_PVsource, cosine_decay
import time
import numpy as np
import copy
import logging
from datetime import datetime
from itertools import product
import torch
import torch.nn as nn

###########
#Parameters
###########

# General information
path_data = 'data-files/'
general_dataset_type = "Xu"
dataset_type = "autoignition_augm2"
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
logging.info(f"My device: {device}")
current_time = datetime.now()

# Optimizer
loss_name = "mse" #"MSE"
lambda_reg = 1
learning_rate_decay = "Cosine"
cosine_alpha = 0.01
max_epo = 100000
cosine_decay_steps = 100000
optimizer_alpha = 0.9
optimizer_momentum = 0.3
epo_show_loss = 1000000
batch_size = "all"

# Manifold
PV_rescaling_init = True
PV_rescaling_batch = True
scale_PV = 0.001
always_rescale_PV = False
PV_dim = 1
extra_manifold_parameters = ["mf"]
range_extra_manifold_parameters = 1 #from -x/2 to x/2

# Input/output data
perc_val = 0.1 #0.2 #percentage of validation data
list_species_input = ['H2NN', 'H2O2', 'H2O', 'H2', 'HNO', 'HO2', 'HONO2', 'HONO', 'H', 'N2O', 'NH2', 'NH', 'NNH', 'NO2', 'NO', 'N', 'O2', 'OH', 'O']
list_species_output_evaluation = ['H2O2', 'H2O', 'H2', 'HO2', 'N2O', 'NO2', 'NO', 'O2', 'OH']
input_scaling_name = "None"
temperature_output = True
output_scaling = "-1to1"

# Encoder-decoder architecture
species_scaling_layer = True
init_species_scaling_range = (1.0, 2.0)
init_name_enc = "Normal"
init_name_dec = "Normal"
std_init_enc = 0.05 #standard deviation for initialization of the encoder weights
std_init_dec = 0.05
decoder_layers = [0, 10, 10] #decoder architecture
activation_function = "tanh"
activation_function_output = "tanh"

# Extra
header_data = 'infer'
bool_compute_Kreg = False
nbr_seeds = 1

####################################
#Set name of file with species names
####################################

file_species_names = f"Xu-state-space-names-{dataset_type}.csv" #f"Xu-state-space-names-{dataset_type}.csv" #"Xu-state-space-names.csv"

learning_rates = [0.025]
optimizers = ["adam"]
lists_species_output_QoI = [
    #("noLog", ['H2O2', 'H2O', 'H2', 'HO2', 'N2O', 'NO2', 'NO', 'O2', 'OH']),
    #("log10", ['logH2O2-10', 'logH2O-10', 'logH2-10', 'logHO2-10', 'logN2O-10', 'logNO2-10', 'logNO-10', 'logO2-10', 'logOH-10']),
    ("log20", ['logH2O2-20', 'logH2O-20', 'logH2-20', 'logHO2-20', 'logN2O-20', 'logNO2-20', 'logNO-20', 'logO2-20', 'logOH-20'])
]
seeds = list(range(nbr_seeds))

experiment_configs = []

for lr_i, opt_i, (species_tag, species_i), seed_i in product(
    learning_rates,
    optimizers,
    lists_species_output_QoI,
    seeds):

    config = {
        "lr": lr_i,
        "optimizer": opt_i,
        "output_species": species_i,
        "species_tag": species_tag,
        "seed": seed_i,
    }
    experiment_configs.append(config)

print(f"Total number of runs: {len(experiment_configs)}")

MSE_vals = np.zeros(len(experiment_configs))
MSE_kr_vals = []
list_ids = []

for idxConfig, config in enumerate(experiment_configs):

    optimizer_name = config["optimizer"]
    lr = config["lr"]
    list_species_output = config["output_species"]
    species_tag = config["species_tag"]
    my_seed = config["seed"]

    training_nbr = f"35aTestAutoignitionNewLib_{optimizer_name}_{int(lr*10000)}_{species_tag}"
    training_id = f"Tr{training_nbr}_s{my_seed}"
    list_ids.append(training_id)
    print(training_id)

    ###############################
    #Initialization of the training
    ###############################

    epo = 1
    training_loss_list = np.zeros(max_epo)
    validation_loss_list = np.zeros(max_epo)
    best_training_loss = np.inf
    best_validation_loss = np.inf
    epo_best_model = 0


    #############################
    #Set seed for reproducibility
    #############################

    generator = torch.Generator(device = device)
    generator.manual_seed(my_seed)

    generator_cpu = torch.Generator(device = "cpu")
    generator_cpu.manual_seed(my_seed)

    all_output = copy.deepcopy(list_species_output)
    if(temperature_output):
        all_output.append("T")
    for i in range(1,1+PV_dim):
        all_output.append(f"PV{i}")
    output_dim = len(all_output) #species + Temperature and Source PV

    variable_headers = ["training_id", "training_name","model_name", "curve_name", "metadata_name",
                        "general_dataset_type", "dataset_type", 
                        "nbr_total_datapoints", "max_epo", "epo_best_model", "optimizer_name",
                        "output_scaling", "loss_name", "lambda_reg", "PV_rescaling_init", "PV_rescaling_batch", "always_rescale_PV",
                        "lr", "my_seed", "batch_size", "nbr_input_species", "PV_dim", "date", "hour",
                        "perc_val", "list_species_input", "lists_species_output_QoI", "list_species_output_evaluation",
                        "output_dim", "all_output", "temperature_output",
                        "init_name_enc", "init_name_dec", "std_init_enc", "std_init_dec",
                        "decoder_layers", "elapsed_time",
                        "learning_rate_decay", "cosine_alpha", "cosine_decay_steps", "optimizer_alpha", "optimizer_momentum",
                        "extra_manifold_parameters", "range_extra_manifold_parameters", "scale_PV", "model_params",
                        "input_scaling_name", "species_scaling_layer", "init_species_scaling_range", "input_species_scaling", "input_species_bias",
                        "activation_function", "activation_function_output",
                        "best_training_loss", "best_validation_loss", "avg_std_MSE_Kreg"]


    ##############
    #Load the data
    ##############

    logging.info("Load the data")
    train_input, train_output, val_input, val_output, nbr_total_datapoints, input_species_scaling, input_species_bias = get_dataset_training(path_data, general_dataset_type, dataset_type, generator_cpu, perc_val,
                                                                                                                                             list_species_input, list_species_output, input_scaling_name, output_scaling,
                                                                                                                                             temperature_output, header_data, extra_manifold_parameters, range_extra_manifold_parameters)

    nbr_training_datapoints = train_input.size(0)
    if(batch_size == "all"):
        batch_size = nbr_training_datapoints
    else:
        batch_size = batch_size

    nbr_input_species = train_input.size(1)-1 #all except f (last column)

    train_input, train_output = train_input.to(device), train_output.to(device)
    val_input, val_output = val_input.to(device), val_output.to(device)


    #################################
    #Create directories and filenames
    #################################

    dirs = create_dirs( overall_dataset=general_dataset_type,
                        dataset_type=dataset_type,
                        current_time = current_time,
                        training_id = training_id)
    dirs.create(variable_headers)


    ###############
    #Load the model
    ###############

    logging.info("Load the model")
    model_params = {"nbr_species": nbr_input_species,
                    "PV_dim": PV_dim,
                    "output_dim": output_dim,
                    "decoder_layers": decoder_layers,
                    "species_scaling_layer": species_scaling_layer,
                    "activation_function": activation_function,
                    "activation_function_output": activation_function_output,
                    "extra_manifold_parameters": extra_manifold_parameters}
    model = PV_autoencoder(**model_params)
    model.to(device)

    #initialize the weights of the model
    model.initialize_model_weights(generator, std_init_enc, std_init_dec, init_species_scaling_range)

    #scale the weights to have the PV having a range of 1
    if(PV_rescaling_init):    
        model.rescale_encoder_data(train_input, scale_PV)
        logging.info("PV rescaled")

    ##############################
    #Intialize the optimizer tools
    ##############################

    optimizer = get_optimizer(model.parameters(), optimizer_name, lr, optimizer_alpha, optimizer_momentum)  #torch.optim.Adam(model.parameters(), lr=lr)
    loss_criterion = get_loss_criterion(loss_name, lambda_reg=lambda_reg)
    scheduler = torch.optim.lr_scheduler.LambdaLR( #cosine decay learning rate scheduler
    optimizer, lr_lambda=lambda epoch: cosine_decay(cosine_alpha, epoch, cosine_decay_steps)
    )


    ###############
    #Start training
    ###############

    logging.info("Start of the training")
    start_time = time.time()
    while(epo<=max_epo):
        #add criterion stop model
        
        training_loss = 0
        validation_loss = 0

        if(PV_rescaling_batch and epo>1):
            model.rescale_encoder_data(train_input, scale_PV, always_rescale = always_rescale_PV)

        #Perform the minibatching
        torch.manual_seed(epo) #seed value is the epoch number
        indices = torch.randperm(nbr_training_datapoints) #shuffle the indices
        split_indices = torch.split(indices, batch_size) #split in subtensors for the batches
        
        #Start training
        for batch_idx in split_indices:

            output_model = model(train_input[batch_idx,:])
            
            #prepare the output batch, create PV source and scale it between -1 and 1
            batch_output_mod = model.get_source_PV(train_output[batch_idx,:], input_species_scaling)
            
            #rescale source PV between -1 and 1
            batch_output_rescaled = rescale_PVsource(batch_output_mod, PV_dim) #rescale the last feature
        
            #get MSE loss
            if loss_name.lower() == "mse_orth_multipv":
                loss = loss_criterion(output_model, batch_output_rescaled, model, train_input[batch_idx,:])
            elif loss_name.lower() == "mse_orth_W_multipv":
                loss = loss_criterion(output_model, batch_output_rescaled, model)
            else:
                loss = loss_criterion(output_model, batch_output_rescaled)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_loss += len(batch_idx)*loss.detach().cpu().numpy()

        #############
        #End training
        #############

        #################
        #Begin validation
        #################

        output_model = model(val_input)

        #prepare the output batch, create PV source and scale it between -1 and 1
        batch_output_mod = model.get_source_PV(val_output, input_species_scaling)

        #rescale source PV between -1 and 1
        batch_output_rescaled = rescale_PVsource(batch_output_mod, PV_dim)

        #get MSE loss
        #loss = loss_criterion(output_model, batch_output_rescaled, model, val_input)
        #loss = loss_criterion(output_model, batch_output_rescaled)
        loss = nn.MSELoss()(output_model, batch_output_rescaled)

        validation_loss += loss.detach().cpu().numpy()
        ###############
        #End validation
        ###############
        
        #########################
        #Post-processing of epoch
        #########################

        #save training and validation loss
        training_loss_list[epo-1] = training_loss/nbr_training_datapoints #weighted average of all the training losses
        validation_loss_list[epo-1] = validation_loss
        
        #checkpoint save model, in case it is a better model
        if(validation_loss < best_validation_loss):
            dirs.save_model(model.state_dict())
            best_training_loss = training_loss/nbr_training_datapoints
            best_validation_loss = validation_loss
            epo_best_model = epo
        
        epo += 1
        scheduler.step() #next learning rate in the scheduler

        if(epo%epo_show_loss==0):
            logging.info(f"Current epoch: {epo} - Validation loss (1e-04): {np.round(validation_loss_list[epo-2]*10000,1)}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info(f"Training finished - {np.round(elapsed_time,2)} seconds")
    #############
    #End training
    #############

    MSE_vals[idxConfig] = best_validation_loss

    ###########################
    #Assess model's performance
    ###########################

    best_model = dirs.load_model(model_params)
    avg_std_MSE_Kreg = [-1, -1]

    if(bool_compute_Kreg):

        logging.info("Start computing the Kreg performance")

        #determine which idx to remove for the source terms
        #needed when there the dataset for training is different than the dataset for Kreg
        
        avg_std_MSE_Kreg = compute_Kreg(path_data,
                                        general_dataset_type,
                                        dataset_type,
                                        list_species_input,
                                        list_species_output_evaluation,
                                        input_scaling_name,
                                        input_species_scaling,
                                        input_species_bias,
                                        extra_manifold_parameters,
                                        range_extra_manifold_parameters,
                                        model, device)
        
    MSE_kr_vals.append(avg_std_MSE_Kreg)

    ################
    #Post-processing
    ################

    variable_data = {"training_id": training_id, "training_name":dirs.training_name,"model_name": dirs.dirout, "curve_name": dirs.dircurves, "metadata_name": dirs.dirMetadata,
                     "general_dataset_type": general_dataset_type, "dataset_type": dataset_type,
                     "nbr_total_datapoints": nbr_total_datapoints, "max_epo": max_epo, "epo_best_model": epo_best_model, "optimizer_name": optimizer_name,
                     "output_scaling": output_scaling, "loss_name": loss_name, "lambda_reg": lambda_reg, "PV_rescaling_init": PV_rescaling_init, "PV_rescaling_batch": PV_rescaling_batch, "always_rescale_PV": always_rescale_PV,
                     "lr": lr, "my_seed": my_seed, "batch_size": batch_size, "nbr_input_species": nbr_input_species, "PV_dim": PV_dim, "date": dirs.formatted_date, "hour": dirs.formatted_time,
                     "perc_val": perc_val, "list_species_input": list_species_input, "lists_species_output_QoI": list_species_output, "list_species_output_evaluation": list_species_output_evaluation,
                     "output_dim": output_dim, "all_output": all_output, "temperature_output": temperature_output, 
                     "init_name_enc": init_name_enc, "init_name_dec": init_name_dec, "std_init_enc": std_init_enc, "std_init_dec": std_init_dec,
                     "decoder_layers": decoder_layers, "elapsed_time": elapsed_time,
                     "learning_rate_decay": learning_rate_decay, "cosine_alpha": cosine_alpha, "cosine_decay_steps": cosine_decay_steps, "optimizer_alpha": optimizer_alpha, "optimizer_momentum": optimizer_momentum,
                     "extra_manifold_parameters": extra_manifold_parameters, "range_extra_manifold_parameters": range_extra_manifold_parameters, "scale_PV": scale_PV, "model_params": model_params,
                     "input_scaling_name": input_scaling_name, "species_scaling_layer": species_scaling_layer, "init_species_scaling_range": init_species_scaling_range, "input_species_scaling": input_species_scaling, "input_species_bias": input_species_bias,
                     "activation_function": activation_function, "activation_function_output": activation_function_output,
                     "best_training_loss": best_training_loss, "best_validation_loss": best_validation_loss, "avg_std_MSE_Kreg":avg_std_MSE_Kreg}

    dirs.save_train_info_model(variable_data)
    dirs.save_train_val_curves(training_loss_list, validation_loss_list)
    dirs.save_metadata(variable_data)

    logging.info(f"Training {training_id} finished with MSE {np.round(best_validation_loss*10000,2)}")

for idxConfig in range(len(experiment_configs)):
    print(f"{list_ids[idxConfig]}: {np.round(MSE_vals[idxConfig]*10000,2)} - {round(MSE_kr_vals[idxConfig][0]*10000,3)} (\u00B1 {round(MSE_kr_vals[idxConfig][1]*10000, 3)}) - ")

logging.info("Script finished")
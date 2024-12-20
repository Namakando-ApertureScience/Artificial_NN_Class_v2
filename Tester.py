from Network_Constructor import Construct
from Useful_methods import *


# Do you want a file
file = open("Training_data.txt", 'a') if (input("Would you like a file? (yes/no): ").lower()) == "yes" else None

########################################################################################################################
# Training options #
####################

Patterned_data_set = []

if "patterned" == input("Would you like to train on random data or patterned data? (random/patterned): "):
    pattern = True
else:
    pattern = False

Data_set=None
if pattern:
    # Pattern function
    function = [lambda arr: arr[0] + arr[1] + arr[2],
                lambda arr: 2 * arr[0] - arr[1]]

    # Pattern data generation
    Patterned_data_set = patterned_data_generator(function, 3, Number_of_datapoints=20)

else:
    # Random data
    Data_set = random_data_generator(3, [-1, 1], 2, [0, 3], Number_of_datapoints=20)

########################################################################################################################
# Neural network sections #
###########################

# Network parameters
layers = [3, 10, 15, 2]                                                                                            # Options: layers (e.g. layers = [l1, l2, ...]), shape (e.g. shape = [depth, width])
shape = [5, 6]
approximation_interval = [-1, 4]

if Patterned_data_set:
    approximation_interval = [Patterned_data_set[2][0] - 1, Patterned_data_set[2][1] + 1]

activation_functions = ["Sigmoid", "ReLU", ("Approximation_interval", approximation_interval)]                     # Can take sting of  1 AF or list of multiple. Options: ("Approximation_interval", approximation_interval), "Output_function", "Sigmoid", "ReLU"
optimizers = "Adam"                                                                                                # Options: "Backpropagation", "Resilient_Backpropagation", "Adam"

network = Construct(layers, activation_functions, optimizers, weight_decay_rate=0.001, learning_rate_adam=0.005)

########################################################################################################################
# Training #
############

# training iterations
iterations = 10_000

train(pattern, Patterned_data_set, network, file, activation_functions, optimizers, iterations, Data_set)

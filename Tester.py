from Network_Constructor import Construct
from Useful_methods import *

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Do you want a file
file = open("Training_data.txt", 'a') if (input("Would you like a file? (yes/no): ").lower()) == "yes" else None

########################################################################################################################
# Training options #
####################

# Function and variance to be approximated
function_variance = [(lambda arr: arr[0] + arr[1] + arr[2], 2),
                     (lambda arr: 2 * arr[0] - arr[1], 3)]

# Data generation
Data_set = data_generator(function_variance, 3, input_interval=[-25, 25],
                          Number_of_datapoints=500, training_to_validation_ratio=0.05)

########################################################################################################################
# Neural network sections #
###########################

# Network parameters
layers = [3, 7, 7, 2]  # Options: layers (e.g. layers = [l1, l2, ...]), shape (e.g. shape = [depth, width])
shape = [5, 6]

approximation_interval = [Data_set[2][0] - 1, Data_set[2][1] + 1]
activation_functions = ["ReLU", "ReLU", ("Approximation_interval", approximation_interval)]  # Can take sting of  1 AF or list of multiple. Options: ("Approximation_interval", approximation_interval), "Output_function", "Sigmoid", "ReLU"
optimizers = "Adam"  # Options: "Backpropagation", "Resilient_Backpropagation", "Adam"

network = Construct(layers, activation_functions, optimizers, weight_decay_rate=0.00001, learning_rate_adam=0.001)

########################################################################################################################
# Training #
############

# training iterations / batch size
iterations, batch_size = 2000, 20
train(Data_set, network, file, activation_functions, optimizers, iterations, batch_size)

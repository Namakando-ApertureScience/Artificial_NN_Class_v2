from Network_Constructor import Construct
from Useful_methods import *

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

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

Data_set = None
function = []
if pattern:
    # Pattern function
    function = [lambda arr: arr[0]**2 + 2,
                lambda arr: 5**arr[0]]

    # Pattern data generation
    Patterned_data_set = patterned_data_generator(function, 1, Number_of_datapoints=200, noise=0.05)

else:
    # Random data
    Data_set = random_data_generator(2, [-1, 1], 2, [0, 3], Number_of_datapoints=20)

########################################################################################################################
# Neural network sections #
###########################

# Network parameters
in_num, out_num = 1, 2
layers = [in_num, 15, 15, out_num]                                                                                            # Options: layers (e.g. layers = [l1, l2, ...]), shape (e.g. shape = [depth, width])
shape = [5, 6]
approximation_interval = [-1, 4]

if Patterned_data_set:
    approximation_interval = [Patterned_data_set[2][0] - 1, Patterned_data_set[2][1] + 1]

activation_functions = ["Sigmoid", "ReLU", ("Approximation_interval", approximation_interval)]                     # Can take sting of  1 AF or list of multiple. Options: ("Approximation_interval", approximation_interval), "Output_function", "Sigmoid", "ReLU"
optimizers = "Adam"                                                                                                # Options: "Backpropagation", "Resilient_Backpropagation", "Adam"

network = Construct(layers, activation_functions, optimizers, weight_decay_rate=0.0005, learning_rate_adam=0.01)

########################################################################################################################
# Training #
############

# training iterations
iterations = 5_000

network.copy_()
train(pattern, Patterned_data_set, network, file, activation_functions, optimizers, iterations, Data_set)

# visualisation
training_data, testing_data = Patterned_data_set[:2]
training_dataT, testing_dataT = list(zip(*training_data)), list(zip(*testing_data))

if pattern and in_num == 1:

    for index in range(len(function)):
        buffer_1 = list(zip(*training_dataT[1]))
        buffer_2 = list(zip(*testing_dataT[1]))

        plot(training_dataT[0], testing_dataT[0], buffer_1[index], buffer_2[index])

    net_out_old = list(zip(*[network.compute(x, old_eval=True) for x in testing_dataT[0]]))
    for index in range(len(function)):
        buffer_1 = list(zip(*training_dataT[1]))
        plot(training_dataT[0], testing_dataT[0], buffer_1[index], net_out_old[index])

    net_out = list(zip(*[network.compute(x) for x in testing_dataT[0]]))
    for index in range(len(function)):
        buffer_1 = list(zip(*training_dataT[1]))
        plot(training_dataT[0], testing_dataT[0], buffer_1[index], net_out[index])

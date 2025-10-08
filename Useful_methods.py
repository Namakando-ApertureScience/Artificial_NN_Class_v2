import matplotlib.pyplot as plt
import pandas as pd
import random as rd
import numpy as np


# Standard error measure
def error_measure(Data_set_, network_, file=None, activation_functions_optimizers=None, old_error=None,
                  training=False, validation=False, close=False, weight_sum=0):
    data_frame_A = pd.DataFrame()

    data_frame_A["Approximation"] = [network_.compute(exa[0]).round(2) for exa in Data_set_]
    data_frame_A["Expected output"] = [np.array(exa[1]).round(2) for exa in Data_set_]
    data_frame_A.index += 1

    if activation_functions_optimizers is not None:
        word1 = "activation functions" if type(activation_functions_optimizers[0]) is list else "activation function"
        word2 = "optimizes" if type(activation_functions_optimizers[1]) is list else "optimizer"

    if file is not None:
        if activation_functions_optimizers is not None and not validation:
            file.write("You're using the {} {}".format(activation_functions_optimizers[0], word1))
            file.write("\n")

            file.write("and the {} {}. ".format(activation_functions_optimizers[1], word2))
            file.write("Your networks shape is: {}.".format(network_.layers))
            file.write("\n\n")

        if training:
            file.write("Training_Data")
            file.write("\n")

        if validation:
            file.write("Validation_Data")
            file.write("\n")

        data_frame_A_str = data_frame_A.to_string(header=True, index=True)
        file.write(data_frame_A_str)
        file.write("\n\n")

    else:
        if activation_functions_optimizers is not None and not validation:
            print("You're using the {} {}".format(activation_functions_optimizers[0], word1))
            print("and the {} {}. ".format(activation_functions_optimizers[1], word2), end='')
            print("Your networks shape is: {}.".format(network_.layers), end="\n\n")

        if training:
            print("Training_Data")

        if validation:
            print("Validation_Data")

        print(data_frame_A)

    print()

    error_ = [0, weight_sum]
    for exa in Data_set_:
        error_[0] += abs(network_.compute(exa[0]) - exa[1]).sum()

    error_[1] += abs(network_.sum())

    if file is not None:
        file.write("Approximation error: " + str(round(error_[0], 2)) + "\n")
        if validation:
            file.write("Sum of the absolute values of all weights: " + str(round(error_[1], 2)) + "\n")

        if old_error is None:
            file.write("\n")

    else:
        print("Approximation error: " + str(round(error_[0], 2)))
        if validation:
            print("Sum of the absolute values of all weights: " + str(round(error_[1], 2)))

        if old_error is None:
            input()

    if old_error is not None:
        error_improvement_percentage = 100 * (1 - error_[0] / old_error[0])
        weight_improvement = 100 * (1 - error_[1] / old_error[1])

        if file is not None:
            file.write("Approximation error relative improvement: " +
                       str(round(error_improvement_percentage, 2)) + "%" + "\n")
            if not close:
                file.write("\n")

            if validation:
                file.write("Sum of the absolute values of all weights relative improvement: " + str(
                    round(weight_improvement, 2)) + "%")

            if close:
                file.close()

        else:
            print("Approximation error relative improvement: " + str(round(error_improvement_percentage, 2)) + "%")
            if validation:
                print("Sum of the absolute values of all weights relative improvement: " + str(
                    round(weight_improvement, 2)) + "%")
            input()

    return error_


# Random data generator
def random_data_generator(number_of_inputs=3, input_interval=None, number_of_outputs=2, output_interval=None,
                          Number_of_datapoints=10):
    if input_interval is None:
        input_interval = [0, 1]

    if output_interval is None:
        output_interval = [0, 1]

    DS = []
    for i in range(Number_of_datapoints):

        input_vector = []
        for j in range(number_of_inputs):
            value = rd.random() * (input_interval[1] - input_interval[0]) + input_interval[0]
            input_vector.append(value)

        output_vector = []
        for k in range(number_of_outputs):
            value = rd.random() * (output_interval[1] - output_interval[0]) + output_interval[0]
            output_vector.append(value)

        DS.append([input_vector, output_vector])

    return DS


# Pattern generator
def patterned_data_generator(pattern_functions, number_of_inputs=3, input_interval=None, Number_of_datapoints=10,
                             training_to_validation_ratio=0.2, noise=0.0):

    if input_interval is None:
        input_interval = [0, 1]

    data_set, training_set, validation_set, min_val, max_val = [], [], [], float("inf"), float("-inf")
    for i in range(Number_of_datapoints):

        input_vector = []
        for j in range(number_of_inputs):
            value = rd.random() * (input_interval[1] - input_interval[0]) + input_interval[0]
            input_vector.append(value)

        output_vector = []
        for pattern_function in pattern_functions:
            value = pattern_function(input_vector) + rd.gauss(0, noise)
            min_val = min(min_val, value)
            max_val = max(max_val, value)
            output_vector.append(value)

        data_set.append([input_vector, output_vector])

    training_set = data_set[:int(training_to_validation_ratio * len(data_set))]
    validation_set = data_set[int(training_to_validation_ratio * len(data_set)):]

    return training_set, validation_set, [min_val, max_val]


def train(pattern, Patterned_data_set, network, file, activation_functions, optimizers, iterations, Data_set):
    if pattern:

        error1 = error_measure(Patterned_data_set[0], network, file,
                               activation_functions_optimizers=[activation_functions, optimizers], training=True)
        error2 = error_measure(Patterned_data_set[1], network, file,
                               activation_functions_optimizers=[activation_functions, optimizers], validation=True,
                               weight_sum=error1[1])

        for i in range(iterations):
            print("Iterations: " + str(i + 1), end='\r')
            Data = rd.choice(Patterned_data_set[0])
            network.Optimize(Data[0], Data[1])
        input()
        print(end="\n")

        error_measure(Patterned_data_set[0], network, file, old_error=error1, training=True)
        error_measure(Patterned_data_set[1], network, file, old_error=error2, validation=True, close=True)

    else:

        error = error_measure(Data_set, network, file,
                              activation_functions_optimizers=[activation_functions, optimizers])

        for i in range(iterations):
            print("Iterations: " + str(i + 1), end='\r')
            Data = rd.choice(Data_set)
            network.Optimize(Data[0], Data[1])
        input()
        print(end="\n")

        error_measure(Data_set, network, file, old_error=error, close=True)


def plot(X_train, X_test, y_train, y_test, title):
    plt.figure(figsize=(10, 6), facecolor="lightcyan")
    plt.title(title)
    plt.scatter(X_train, y_train, s=4, c="blue", label="Training Data")
    plt.scatter(X_test, y_test, s=4, c="red", label="Testing Data")
    plt.legend(prop={"size": 14})
    plt.show()

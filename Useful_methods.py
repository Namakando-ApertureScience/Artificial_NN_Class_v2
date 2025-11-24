import pandas as pd
import random as rd
import numpy as np


# Standard error measure
def error_measure(Data_set_, network_, file=None, activation_functions_optimizers=None, old_error=None,
                  training=False, validation=False, close=False, weight_sum=0):
    data_frame_A = pd.DataFrame()

    data_frame_A["Approximation"] = [network_.forward(exa[0]).round(2) for exa in Data_set_]
    data_frame_A["Expected output"] = [np.array(exa[1]).round(2) for exa in Data_set_]
    data_frame_A.index += 1

    if activation_functions_optimizers is not None:
        word1 = "activation functions" if type(activation_functions_optimizers[0]) is list else "activation function"
        word2 = "optimizes" if type(activation_functions_optimizers[1]) is list else "optimizer"

    if file is not None:
        if activation_functions_optimizers is not None and not validation:
            file.write("You're using the {} {}".format(activation_functions_optimizers[0], word1))
            file.write("\n")

            file.write("and the {} {}.".format(activation_functions_optimizers[1], word2))
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
            print("and the {} {}.".format(activation_functions_optimizers[1], word2), end="\n\n")

        if training:
            print("Training_Data")

        if validation:
            print("Validation_Data")

        print(data_frame_A)

    print()

    error_ = [0, weight_sum]
    for exa in Data_set_:
        error_[0] += abs(network_.forward(exa[0]) - exa[1]).sum()

    error_[1] += abs(network_.sum())

    if file is not None:
        file.write("Approximation error: " + str(round(error_[0], 2)) + "\n")
        if validation:
            file.write("Absolute value of the sum of all weights: " + str(round(error_[1], 2)) + "\n")

        if old_error is None:
            file.write("\n")

    else:
        print("Approximation error: " + str(round(error_[0], 2)))
        if validation:
            print("Absolute value of the sum of all weights: " + str(round(error_[1], 2)))

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
                file.write("Absolute value of the sum of all weights relative improvement: " + str(
                    round(weight_improvement, 2)) + "%")

            if close:
                file.close()

        else:
            print("Approximation error relative improvement: " + str(round(error_improvement_percentage, 2)) + "%")
            if validation:
                print("Absolute value of the sum of all weights relative improvement: " + str(
                    round(weight_improvement, 2)) + "%")
            input()

    return error_


# Data generator
def data_generator(function_variance, number_of_inputs=3, input_interval=None,
                   Number_of_datapoints=10, training_to_validation_ratio=0.2):
    if input_interval is None:
        input_interval = [0, 1]

    data_set, training_set, validation_set, min_val, max_val = [], [], [], float("inf"), float("-inf")
    for i in range(Number_of_datapoints):

        input_vector = []
        for j in range(number_of_inputs):
            value = rd.random() * (input_interval[1] - input_interval[0]) + input_interval[0]
            input_vector.append(value)

        output_vector = []
        for func_var in function_variance:
            value = (func_var[0](input_vector) +
                     np.random.normal(loc=0, scale=np.sqrt(func_var[1])))
            min_val, max_val = (min(min_val, value),
                                max(max_val, value))
            output_vector.append(value)

        data_set.append([input_vector, output_vector])

    training_set = data_set[:int(training_to_validation_ratio * len(data_set))]
    validation_set = data_set[int(training_to_validation_ratio * len(data_set)):]

    return training_set, validation_set, [min_val, max_val]


# Training and documentation
def train(Data_set, network, file, activation_functions, optimizers, iterations, batch_size):

    error1 = error_measure(Data_set[0],
                           network,
                           file,
                           activation_functions_optimizers=[activation_functions, optimizers],
                           training=True)

    error2 = error_measure(Data_set[1],
                           network,
                           file,
                           activation_functions_optimizers=[activation_functions, optimizers],
                           validation=True,
                           weight_sum=error1[1])

    for i in range(iterations):
        print("Iterations: " + str(i + 1), end='\r')
        Data = rd.choices(Data_set[0], k=batch_size)
        Data0, Data1 = [d[0] for d in Data], [d[1] for d in Data]
        network.Optimize(Data0, Data1)
    input()
    print(end="\n")

    error_measure(Data_set[0], network, file, old_error=error1, training=True)
    error_measure(Data_set[1], network, file, old_error=error2, validation=True, close=True)

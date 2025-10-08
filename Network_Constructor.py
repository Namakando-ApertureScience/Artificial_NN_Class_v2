import numpy as np
import math as m


def derivative(func, x, quant=1e-13):
    x = np.array(x, dtype=np.float128)
    return (func(x + quant) - func(x - quant)) / (2 * quant)


class Activation_functions:

    def __init__(self, steepness_value, approximation_interval=None, leaky_slope=0.01):
        if approximation_interval is None:
            self.approximation_interval = [0, 1]
        else:
            self.approximation_interval = approximation_interval

        self.steepness_value = steepness_value
        self.leaky_slope = leaky_slope

    # Input normalized data and output within the approximation interval
    def Approximation_interval(self, x):
        return ((self.approximation_interval[1] - self.approximation_interval[0]) *
                self.Sigmoid(x) + self.approximation_interval[0])

    # Identity
    @staticmethod
    def Output_function(x):
        return x

    # Input real data and output normalized data
    def Sigmoid(self, x):
        x = np.array(x, dtype=np.float128)
        return 1 / (1 + np.e ** (-x / self.steepness_value))

    def Hyperbolic_tangent(self):
        return

    # Input real data output non-linear data
    def ReLU(self, x):
        x = np.array(x, dtype=np.float128)
        return np.array(list(map(lambda entry: max(0, entry), x * self.steepness_value)))

    # Input real data output non-linear data
    def Leaky_ReLU(self, x):
        x = np.array(x, dtype=np.float128)
        return np.array(list(map(lambda entry: max(0, entry), x * self.steepness_value)))

    def Soft_max(self):
        return

class Construct:

    def __init__(self, layers=None, str_activation_functions=None, optimizers=None, shape=None,
                 weight_interval=None, input_output_steepness=None, weight_decay_rate=0.000_100,
                 learning_rate_BP=0.1, learning_rate_up=1.05, learning_rate_down=0.5,
                 learning_rate_adam=0.001, rho1=0.9, rho2=0.999):

        ################################################################################################################
        # Neural network constructor #
        ##############################
        # Hyper parameter setting

        if layers is None and shape is None:
            raise Exception("Necessary parameters missing.")

        if layers is None:
            layers = [shape[1] for i in range(shape[0])]

        self.layers = layers

        if input_output_steepness is None:
            self.steepness_values = np.full(len(layers) - 1, 1)
        else:
            self.steepness_values = np.linspace(input_output_steepness[0], input_output_steepness[1], len(layers) - 1)

        if str_activation_functions is None:
            str_activation_functions = ["Sigmoid" for i in layers][:-1]
        elif type(str_activation_functions) is not list:
            str_activation_functions = [str_activation_functions for i in layers][:-1]

        if weight_interval is None:
            weight_interval = [-0.5, 0.5]

        if optimizers is None:
            optimizers = ["Backpropagation" for i in layers][:-1]
        elif type(optimizers) is not list:
            optimizers = [optimizers for i in layers][:-1]

        self.optimizers = optimizers

        # Activation functions
        self.activation_functions = []

        for index in range(len(layers) - 1):
            if type(str_activation_functions[index]) is tuple:
                if str_activation_functions[index][0] == "Approximation_interval":
                    self.activation_functions.append(Activation_functions(
                        steepness_value=self.steepness_values[index],
                        approximation_interval=str_activation_functions[index][1]).Approximation_interval)
            else:
                if str_activation_functions[index] == "Output_function":
                    self.activation_functions.append(Activation_functions(
                        steepness_value=self.steepness_values[index]).Output_function)

                if str_activation_functions[index] == "Sigmoid":
                    self.activation_functions.append(Activation_functions(
                        steepness_value=self.steepness_values[index]).Sigmoid)

                if str_activation_functions[index] == "ReLU":
                    self.activation_functions.append(Activation_functions(
                        steepness_value=self.steepness_values[index]).ReLU)

        # Important matrices
        self.neural_network_copy = []
        self.weighted_sums_matrix = []

        # Layer construction
        self.neural_network = []
        for index in range(1, len(layers)):
            self.neural_network.append((weight_interval[1] - weight_interval[0]) *
                                       np.random.rand(layers[index], layers[index - 1] + 1) + weight_interval[0])

        self.neural_network_copy = self.neural_network.copy()

        ################################################################################################################
        # optimizers constructor #
        ##########################

        # Error vector
        self.error_vector = np.array([])

        # Weight decay rate
        self.weight_decay_rate = weight_decay_rate

        ################################################################################################################
        # For backpropagation

        # Learning rate
        self.learning_rate_BP = learning_rate_BP

        ################################################################################################################
        # For resilient backpropagation

        self.learning_rate_up = learning_rate_up
        self.learning_rate_down = learning_rate_down

        self.rho1 = rho1
        self.rho2 = rho2

        # Important matrices
        self.ILG = {}

        if "Resilient_Backpropagation" in self.optimizers:
            self.Resilient_Init()

        ################################################################################################################
        # For Adam

        # Bias correction
        self.epsilon = 1e-10
        self.bias_correction = {}

        # Learning rate
        self.learning_rate_adam = learning_rate_adam

        # Time step
        self.time_step = {}
        self.time_step_bool = {}

        # Momenta
        self.AF = {}

        if "Adam" in self.optimizers:
            self.Adam_Init()

    ####################################################################################################################
    # Optimizer Initialization #
    ############################

    # Resilient BP
    def Resilient_Init(self):
        # Important matrices for resilient propagation

        Indices = [i for i in range(len(self.optimizers)) if self.optimizers[i] == "Resilient_Backpropagation"]
        self.ILG = {}

        for index in Indices:
            number_of_neurons, number_of_weights = self.layers[index + 1], self.layers[index]

            identity = np.full((number_of_neurons, number_of_weights + 1), 1.0)
            learning_rate_matrix = np.full((number_of_neurons, number_of_weights + 1), 0.1)
            gradients = np.full((number_of_neurons, number_of_weights + 1), 0.0)

            self.ILG[index] = [identity, learning_rate_matrix, gradients]

    # Adam
    def Adam_Init(self):
        # Momenta

        Indices = [i for i in range(len(self.optimizers)) if self.optimizers[i] == "Adam"]
        self.AF = {}

        for index in Indices:
            number_of_neurons, number_of_weights = self.layers[index + 1], self.layers[index]

            A = np.zeros((number_of_neurons, number_of_weights + 1))
            F = np.zeros((number_of_neurons, number_of_weights + 1))

            self.AF[index] = [A, F]

            self.time_step[index] = 0
            self.time_step_bool[index] = True

    ####################################################################################################################
    # Neural network #
    ##################

    def compute(self, input_vector, old_eval=False):

        # Bias neuron
        input_vector = input_vector.copy()
        input_vector.append(1)

        if old_eval:
            vector = np.matmul(self.neural_network_copy[0], input_vector)

            for index in range(1, len(self.neural_network)):
                vector = np.matmul(self.neural_network_copy[index],
                                   np.append(self.activation_functions[index - 1](vector), 1))
            return self.activation_functions[-1](vector)

        else:
            self.weighted_sums_matrix = []
            self.weighted_sums_matrix.append(np.matmul(self.neural_network[0], input_vector))

            for index in range(1, len(self.neural_network)):
                self.weighted_sums_matrix.append(np.matmul(self.neural_network[index],
                np.append(self.activation_functions[index - 1](self.weighted_sums_matrix[-1]), 1)))
            return self.activation_functions[-1](self.weighted_sums_matrix[-1])

    def sum(self):
        return sum([sum([sum([abs(weight) for weight in neuron])
                         for neuron in layer]) for layer in self.neural_network])

    def copy_(self):
        self.neural_network_copy = self.neural_network.copy()

    ####################################################################################################################
    # optimizer #
    #############

    # Get/Set optimizers

    def Get_optimizers(self):
        return self.optimizers

    def Set_optimizers(self, optimizers):
        self.optimizers = optimizers
        self.time_step = {}
        self.time_step_bool = {}

        if "Resilient_Backpropagation" in self.optimizers:
            self.Resilient_Init()

        if "Adam" in self.optimizers:
            self.Adam_Init()

    ####################################################################################################################
    # Optimizer

    def Optimize(self, input_vector, target_output_vector):
        self.error_vector = target_output_vector - self.compute(input_vector)

        gradient = derivative(self.activation_functions[len(self.layers) - 2],
                              self.weighted_sums_matrix[len(self.layers) - 2]) * self.error_vector
        self.distributer(gradient)

        for index in range(len(self.layers) - 3, 0, -1):
            gradient = (derivative(self.activation_functions[index], self.weighted_sums_matrix[index]) *
                        np.matmul(np.delete(self.neural_network[index + 1].transpose(),
                                            self.layers[index + 1], axis=0), gradient))
            self.distributer(gradient, index)

    ####################################################################################################################
    # Distributer

    def distributer(self, gradient, index=None):
        if index is None:
            index = len(self.layers) - 2

        if self.optimizers[index] == "Adam" and self.time_step_bool[index]:
            self.time_step[index] += 1
            self.bias_correction[index] = (m.sqrt(1 - self.rho2 ** self.time_step[index]) /
                                           (1 - self.rho1 ** self.time_step[index]))

            if abs(self.bias_correction[index] - 1) < self.epsilon:
                self.time_step_bool[index] = False
                self.bias_correction[index] = 1

        (self.Backpropagation_prop_update(gradient, index) if self.optimizers[index] == "Backpropagation"
         else (self.Resilient_prop_update(gradient, index) if self.optimizers[index] == "Resilient_Backpropagation"
               else (self.Adam_update(gradient, index) if self.optimizers[index] == "Adam"
                     else print("Error no optimizers specified!!!"))))

    ####################################################################################################################
    # Backpropagation

    def Backpropagation_prop_update(self, gradient, index):
        self.neural_network[index] = (self.neural_network[index] + self.learning_rate_BP * np.matmul(
            np.array([gradient]).transpose(), np.array([np.append(self.activation_functions[index](
                self.weighted_sums_matrix[index - 1]), 1)])) - self.weight_decay_rate * self.neural_network[index])

    ####################################################################################################################
    # Resilient Backpropagation

    def Resilient_prop_update(self, gradient, index):
        gradients = - np.matmul(np.array([gradient]).transpose(), np.array([np.append(self.activation_functions[index](
            self.weighted_sums_matrix[index - 1]), 1)]))

        self.ILG[index][1] = self.ILG[index][1] * ((np.sign(gradient * self.ILG[index][2]) *
                                                    (self.learning_rate_up - self.learning_rate_down) + abs(
                    np.sign(gradient *
                            self.ILG[index][2])) * (self.learning_rate_up + self.learning_rate_down)) / 2 +
                                                   abs(self.ILG[index][0] - abs(np.sign(gradient * self.ILG[index][2]))))

        self.ILG[index][2] = gradients

        self.neural_network[index] = ((self.neural_network[index] - self.ILG[index][1] *
                                       np.sign(gradient)) - self.weight_decay_rate * self.neural_network[index])

    ####################################################################################################################
    # Adam and Adam-hybrid update

    def Adam_update(self, gradient, index):
        grad = (np.matmul(np.array([gradient]).transpose(), np.array([np.append(self.activation_functions[index]
                                                                                (self.weighted_sums_matrix[index - 1]),
                                                                                1)])))

        self.AF[index][0] = self.rho2 * self.AF[index][0] + (1 - self.rho2) * grad ** 2
        self.AF[index][1] = self.rho1 * self.AF[index][1] + (1 - self.rho1) * grad

        self.neural_network[index] = ((self.neural_network[index] +
                                       (self.learning_rate_adam * self.bias_correction[index]) *
                                       (self.AF[index][1] / (self.AF[index][0] + self.epsilon) ** 0.5)) -
                                      self.weight_decay_rate * self.neural_network[index])

from brian2 import *
import numpy as np
import matplotlib.pyplot as plt

alpha = 0.05  # learning rate
threshold = 0.001  # error required for convergence
theta = 1  # threshold for activation
base_error = 1  # starting error

# XOR training data
training_data_xor = [
    (np.array([1, 1, -1]), 0),
    (np.array([1, 0, -1]), 1),
    (np.array([0, 1, -1]), 1),
    (np.array([0, 0, -1]), 0),
]

# initialize weights randomly with mean 0
syn0_start = np.array([[0.5, 0.9], [0.4, 1.0], [0.8, -0.1]])
syn0_start = syn0_start.T
syn1_start = np.array([-1.2, 1.1, 0.3])


# sigmoid function
def nonlin(x, deriv=False):
    if deriv:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))


# activation function
def activation(neuron):
    if neuron < nonlin(theta):
        return 0
    else:
        return 1


# calculate output based on current weights
def forward_prop(syn0, syn1, training_data, result=False):
    l0, y = training_data
    l0 = l0.T
    l1_out = nonlin(np.dot(syn0, l0))  # first layer output
    l2_in = np.append(l1_out, -1)
    l2_out = nonlin(np.dot(syn1, l2_in))  # second layer output
    if result:
        print("{}: {} -> {}".format(y, l2_out, activation(l2_out)))
    else:
        return y, l0, l1_out, l2_in, l2_out


# update weights based on error from current output
def backward_prop(y, syn0, syn1, l0, l1_out, l2_in, l2_out):
    e = y - l2_out  # error
    gradient5 = nonlin(l2_out, deriv=True) * e  # gradient for second layer
    deltasyn1 = alpha * gradient5 * l2_in.T  # weight change for second layer
    gradient3 = nonlin(l1_out[0], deriv=True) * gradient5 * syn1[0]  # gradients for first layer
    gradient4 = nonlin(l1_out[1], deriv=True) * gradient5 * syn1[1]
    deltasyn0_top = np.array([alpha * gradient3 * l0])  # weight changes for first layer
    deltasyn0_bot = np.array([alpha * gradient4 * l0])
    deltasyn0 = np.concatenate((deltasyn0_top, deltasyn0_bot))
    syn0 = syn0 + deltasyn0  # update the weights locally
    syn1 = syn1 + deltasyn1
    return syn0, syn1, e


# combines forward_prop and backward_prop to train the ANN
def ann(ss_error, convergence, syn0, syn1, training_data):
    num_iter = 0
    ss_array = []
    while ss_error >= convergence:
        error_array = []
        for i in range(4):  # train each of the 4 inputs
            (y, l0, l1_out, l2_in, l2_out) = forward_prop(syn0, syn1, training_data[i])
            (new_syn0, new_syn1, error) = backward_prop(y, syn0, syn1, l0, l1_out, l2_in, l2_out)
            syn0 = new_syn0  # update the weights globally
            syn1 = new_syn1
            error_array.append(error)  # used to calculate sums square of errors
        ss_error = .5 * (np.sum(np.power(error_array, 2)))
        num_iter += 1  # new epoch when all 4 inputs are used
        ss_array.append(ss_error)
    converge_line = convergence * np.ones(num_iter)
    # plt.plot(ss_array)
    # plt.plot(converge_line)
    # plt.show()
    return num_iter, syn0, syn1


if __name__ == '__main__':
    # (total, syn0_final, syn1_final) = ann(base_error, threshold, syn0_start, syn1_start, training_data_xor)
    # print(f"Epochs: {total}")
    # for j in range(4):
    #     forward_prop(syn0_final, syn1_final, training_data_xor[j], True)
    alpha_arr = []
    epoch_arr = []
    for j in range(100):
        (total, syn0_final, syn1_final) = ann(base_error, threshold, syn0_start, syn1_start, training_data_xor)
        alpha_arr.append(alpha)
        epoch_arr.append(total)
    plt.plot(alpha_arr, epoch_arr)


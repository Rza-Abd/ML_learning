# Create a Neural Network process with vector 
# This was done in Mr. Arabi's class episode: 3, 4
#    input     | target
# ------------------------
# (1.66, 1.56) |   1
#   (2, 1.5)   |   0

import numpy as np

# Define Condition
input_vector_1 = np.array([1.66, 1.56])
input_vector_2 = np.array([2, 1.5])
true_values = np.array([1, 0])

# Create a random initial weight and bias
weight_1 = np.array([1.45, -0.66])
bias = np.array([0.0])

# Activation function
def sigmoid(x):
    return 1/ (1 + np.exp(-x))

# Prediction Process
def make_prediction(input_vector, weight, bias):
    layer_1 = np.dot(input_vector, weight) + bias
    layer_2 = sigmoid(layer_1)
    if layer_2 >= 0.5 :
        return 1, layer_2
    else:
        return 0, layer_2

# Prediction values for input_1
acitve, prediction_1 = make_prediction(input_vector_1, weight_1, bias)
print(f"Prediction result for input_1 : {prediction_1.item()} ,  activation : {acitve}")

# Error
target_1 = 1
mse = np.square(prediction_1 - target_1)
print(f"Prediction = {prediction_1} ,  Error = {mse}")

# Prediction values for input_2
acitve, prediction_2 = make_prediction(input_vector_2, weight_1, bias)
print(f"Prediction result for input_2 : {prediction_2.item()} ,  activation : {acitve}")

# Error
target_2 = 0
mse = np.square(prediction_2 - target_2)
print(f"Prediction = {prediction_2} ,  Error = {mse}")

# Derivative
derivative_2 = 2 * (prediction_2 - target_2)
print(f"The Derivative is : {derivative_2}")

# Updating the weights
weight_1 = weight_1 - derivative_2

acitve , prediction_2 = make_prediction(input_vector_2, weight_1, bias)
mse = np.square(prediction_2 - target_2)

print(f"Prediction: {prediction_2}, Erroe: {mse}, Activation: {acitve}")

# The new weight is good for input_2 now try it for input_1

acitve , prediction_1 = make_prediction(input_vector_1, weight_1, bias)
mse = np.square(prediction_1 - target_1)
print(f"Prediction: {prediction_1}, Erroe: {mse}, Activation: {acitve}")

# We see Its not good weight for input_1. we shoud try again and do it better.

def sigmoid_deriv(x):
    return sigmoid(x) * (1-sigmoid(x))

derror_dprediction = 2 * (prediction_1 - target_1)
layer_1 = np.dot(input_vector_1, weight_1) + bias
dprediction_dlayer1 = sigmoid_deriv(layer_1)
dlayer1_dbias = 1

derror_dbias = (derror_dprediction * dprediction_dlayer1 * dlayer1_dbias)

print(derror_dbias)



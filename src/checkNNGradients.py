import numpy as np
from debugInitializeWeights import debugInitializeWeights
from nnCostFunction import nnCostFunction
from computeNumericalGradient import computeNumericalGradient

def checkNNGradients(lambda_value=0):
    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5

# We generate some 'random' test data
    Theta1 = debugInitializeWeights(hidden_layer_size, input_layer_size)
    Theta2 = debugInitializeWeights(num_labels, hidden_layer_size)
# Reusing debugInitializeWeights to generate X
    X  = debugInitializeWeights(m, input_layer_size - 1)
    y  = 1 + np.transpose(np.mod(range(1, m+1), num_labels))
#    y=np.expand_dims(y,axis=1)
    
    
# Unroll parameters
    Theta1_1d = np.reshape(Theta1, Theta1.size, order='F')
    Theta2_1d = np.reshape(Theta2, Theta2.size, order='F')

    nn_params = np.hstack((Theta1_1d, Theta2_1d))

# Short hand for cost function
    costFunc = lambda p : nnCostFunction(p, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_value)
    
    cost, grad = costFunc(nn_params)
    numgrad = computeNumericalGradient(costFunc, np.expand_dims(nn_params, axis=1))


# Visually examine the two gradient computations.  The two columns
# you get should be very similar. 
    print(numgrad, grad)
    print('The above two columns you get should be very similar.\n (Left-Numerical Gradient, Right-(Your) Analytical Gradient)\n\n')

# Evaluate the norm of the difference between two solutions.  
# If you have a correct implementation, and assuming you used EPSILON = 0.0001 
# in computeNumericalGradient.m, then diff below should be less than 1e-9
    diff = np.linalg.norm(numgrad-grad)/np.linalg.norm(numgrad+grad)
    print('If your backpropagation implementation is correct, then \n the relative difference will be small (less than 1e-9). \n \nRelative Difference: ', diff)

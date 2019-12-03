import numpy as np
from sigmoid import sigmoid

def predict(Theta1, Theta2, X):
#PREDICT Predict the label of an input given a trained neural network
#   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
#   trained weights of a neural network (Theta1, Theta2)

# Useful values
    m = np.shape(X)[0]              #number of examples
    
# You need to return the following variables correctly 
    p = np.zeros(m);


    #theta1 = sigmoid(np.dot(m, Theta1))
    #theta2 = sigmoid(np.dot(theta1, Theta2))

    #p = max(theta2, [], 2)
# ====================== YOUR CODE HERE ======================
# Instructions: Complete the following code to make predictions using
#               your learned neural network. You should set p to a 
#               vector containing labels between 1 to num_labels.
#
# Hint: The max function might come in useful. In particular, the max
#       function can also return the index of the max element, for more
#       information see 'help max'. If your examples are in rows, then, you
#       can use max(A, [], 2) to obtain the max for each row.
    firstTheta = sigmoid(np.dot(np.hstack((np.ones((m, 1)), X)), np.transpose(Theta1)))
    secondTheta = sigmoid(np.dot(np.hstack((np.ones((m,1)), firstTheta)), np.transpose(Theta2)))

    #print(np.hstack(np.amax(secondTheta, 1)))

    return (np.argmax(secondTheta, axis = 1)+1)

# =========================================================================

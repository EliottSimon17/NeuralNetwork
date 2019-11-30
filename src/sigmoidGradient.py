from src.sigmoid import sigmoid

def sigmoidGradient(z):
#SIGMOIDGRADIENT returns the gradient of the sigmoid function
#evaluated at z
#   g = SIGMOIDGRADIENT(z) computes the gradient of the sigmoid function
#   evaluated at z. This should work regardless if z is a matrix or a
#   vector. In particular, if z is a vector or matrix, you should return
#   the gradient for each element.

# The value g should be correctly computed by your code below.
    g = 0

# ====================== YOUR CODE HERE ======================
# Instructions: Compute the gradient of the sigmoid function evaluated at
#               each value of z (z can be a matrix, vector or scalar).

#Formula for the sigmoid gradient
#     dg(𝑥) = g(𝑥)(1 − g(𝑥))
    g = sigmoid(z) * (1 - sigmoid(z))



# =============================================================
    
    return g
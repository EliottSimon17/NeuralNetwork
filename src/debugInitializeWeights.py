import numpy as np

def debugInitializeWeights(fan_out, fan_in):
#DEBUGINITIALIZEWEIGHTS Initialize the weights of a layer with fan_in
#incoming connections and fan_out outgoing connections using a fixed
#strategy, this will help you later in debugging
#   W = DEBUGINITIALIZEWEIGHTS(fan_in, fan_out) initializes the weights 
#   of a layer with fan_in incoming connections and fan_out outgoing 
#   connections using a fix set of values
#
#   Note that W should be set to a matrix of size(1 + fan_in, fan_out) as
#   the first row of W handles the "bias" terms
#

# Set W to zeros
    W = np.zeros((fan_out, 1 + fan_in))

# Initialize W using "sin", this ensures that W is always of the same
# values and will be useful for debugging
#    W = [[0.0841, -0.0279, -0.1000, -0.0288], [0.0909, 0.0657, -0.0537, -0.0961], [0.0141, 0.0989, 0.0420, -0.0751], [-0.0757, 0.0412, 0.0991, 0.0150], [-0.0959, -0.0544, 0.0650, 0.0913]]
    W = np.reshape(np.sin(range(1,np.size(W)+1))/10, np.shape(W), order='F')

# =========================================================================

    return W
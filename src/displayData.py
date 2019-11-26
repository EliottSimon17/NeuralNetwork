import numpy as np
import matplotlib.pyplot as plt

def displayData(data):
    fig = plt.figure()
    plt.gray()
    for i in range(data.shape[0]):
        ax = fig.add_subplot(np.sqrt(data.shape[0]), np.sqrt(data.shape[0]), i+1)
        ax.imshow(np.transpose(np.reshape(data[i,:], (-1, 20))))
        ax.set_axis_off()
    plt.show()
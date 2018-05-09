from scipy import misc
import numpy as np
import random
from pylab import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

def getImage(image_path):
    image = misc.imread(image_path)
    image = np.asarray(image)
    return where(image<120,-1,1)

def getNoiseAddedImage(image):
    noise = np.zeros((image.shape))
    noise = np.random.rand(512,512)
    noisy = where(noise<0.1,-1,1)
    return image*noisy

image = getImage('lena512.bmp')
noisy_image = getNoiseAddedImage(image)

noise_added = noisy_image

def getTotalEnergy(x_i,y_i,noisy_image):
    h = 1
    beta = 1.0
    eta = 2.1
    neighbour_correlation = 0
    variable_correlation = 0
    bias = 0 
    if i-1 >= 0:
        neighbour_correlation += noisy_image[i-1][j]*x_i
    if i+1 < 512:
        neighbour_correlation += noisy_image[i+1][j]*x_i
    if j-1 >= 0:
        neighbour_correlation += noisy_image[i][j-1]*x_i
    if j+1 < 512:
        neighbour_correlation += noisy_image[i][j+1]*x_i
    variable_correlation = x_i*y_i
    bias = x_i
    return h*bias - beta*neighbour_correlation - eta*variable_correlation

for i in range(0,512):
    for j in range(0,512):
        one = getTotalEnergy(1,noisy_image[i][j],noisy_image)
        minus_one = getTotalEnergy(-1,noisy_image[i][j],noisy_image)
        if one < minus_one:
            noisy_image[i][j] = 1
        else:
            noisy_image[i][j] = -1

imgplot = plt.imshow(noisy_image, cmap = cm.Greys_r)
plt.show()

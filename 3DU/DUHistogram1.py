import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os


def range_by_quantiles(img, p_low, p_high):
    """ Function finds quantiles based on p_low and p_high

    Args:
        img (np.array): 2D matrix (numbers from 0.0 - 1.0) representing
            black&white image
        p_low (float): low quantile to map it to 0.0
        p_high (float): high quantile to map it to 1.0

    Returns:
        x_low (float): point where quantile defined by p_low ends
        x_high (float): point where quantile defined by p_high starts
    """
    # YOUR IMPLEMENTATION HERE (also change return)

    x_low = np.quantile(img, p_low)
    x_high = np.quantile(img, p_high)

    return x_low, x_high


def transform_by_lut(img, x_low, x_high):
    """ Function transforms image based x_low and x_high

    Args:
        img (np.array): 2D matrix (numbers from 0.0 - 1.0) representing
            black&white image
        x_low (float): point where low quantile ends
        x_high (float): point where high quantile starts

    Returns:
        transformed image
    """
    imgcopy = img.copy()
    imgcopy[imgcopy < x_low] = 0
    imgcopy[imgcopy > x_high] = 1

    for i in range(np.shape(imgcopy)[0]):
        for j in range(np.shape(imgcopy)[1]):
            if (imgcopy[i, j] != 0) and (imgcopy[i, j] != 1):
                imgcopy[i, j] = 1 / (x_high - x_low) * (imgcopy[i, j] - x_low)

    return imgcopy


# Don't change this code, you can change P_LOW and P_HIGH of course
img = cv.imread(os.path.join("data", "P.jpg"))
grayscaled = cv.cvtColor(img, cv.COLOR_BGR2GRAY) / 255
P_LOW = 0.01
P_HIGH = 0.99
x_low, x_high = range_by_quantiles(grayscaled, P_LOW, P_HIGH)
transformed = transform_by_lut(grayscaled, x_low, x_high)

# PLOT HISTOGRAMS HERE (CHECK IF IT IS CORRECT)

plt.subplot(221)
plt.imshow(img, cmap="gray")
plt.title(("Source img"))

plt.subplot(222)
plt.imshow(transformed, cmap="gray")
plt.title(("LUT img"))

plt.subplot(223)
plt.hist(grayscaled.flatten(), bins=256, color="r")
plt.title(("Source img hist"))
plt.xlim(0, 1)

plt.subplot(224)
plt.hist(transformed.flatten(), bins=256, color="b")
plt.title(("LUT img hist"))
plt.xlim(0, 1)

plt.show()

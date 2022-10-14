import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os


def advanced_lut(img, p_high, p_low, **kwargs):
    """ Function transformes image based p_low and p_high

    This function also try to do it without artifacts in transformed
    image.

    Args:
        img (np.array): 2D matrix (numbers from 0.0 - 1.0) representing
            black&white image
        p_low (float): low quantile to map it to 0.0
        p_high (float): high quantile to map it to 1.0
        **kwargs:
            your keyword arguments

    Returns:
        transformed image
    """
    # YOUR IMPLEMENTATION HERE (also change return)
    return "do some magic"

# dont change this code
img = cv.imread(os.path.join("data", "L.jpg"))
grayscaled = cv.cvtColor(img, cv.COLOR_BGR2GRAY).astype(float) / 255.
P_LOW = 0.01
P_HIGH = 0.99
transformed = advanced_lut(grayscaled, P_LOW, P_HIGH)


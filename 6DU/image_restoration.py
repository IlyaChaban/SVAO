import cv2
import numpy as np
import matplotlib.pyplot as plt


def uniform_noise(img, low, high):
    """
    Application of the uniform noise to the image in range low, high
    
    Parameters
    ----------
    img : ndarray
        Image
    low : float
        lower boundary of the range for the noise generation
    high : float
        upper boundary of the range for the noise generation
    
    Returns
    ----------
    tuple
        img+noise, noise as img, generated noise
    """
    
    noise_rnd = np.random.uniform(0, 1, img.shape) # Generation of the 3d matrix of random numbers in range 0, 1 with the same shape as the image 
    mask = np.where(np.logical_and(noise_rnd>low, noise_rnd<high)) #mask - indices of ganareted values in the range 
    noise = np.zeros(img.shape)
    noise[mask] = (noise_rnd[mask]) * 255 # Generation of noise matrix for direct application to the image
    img_noise = img + noise # noise application
    
    # normalization of the new noisy imge in to the range 0 - 255 
    img_noise[img_noise<0] = 0
    img_noise[img_noise>255] = 255
    
    return img_noise.astype(np.uint8), noise.astype(np.uint8), noise.flatten()
    
def gaussian_noise(img, mu, sigma):
    """
    Application of the gaussian noise to the image
    
    Parameters
    ----------
    img : ndarray
        Image
    mu : float
        gaussian distribution parameter
    sigma : float
        gaussian distribution parameter
    
    Returns
    ----------
    tuple
        img+noise, noise as img, generated noise
    """
    
    # replace following dummy code by your code
    
    img_copy = img.copy()
    
    noise_gauss = np.random.normal(mu, sigma, img.shape)

    img_noise = img + noise_gauss# noise application
    
    img_noise[img_noise<0] = 0
    img_noise[img_noise>255] = 255
       
    return img_noise.astype(np.uint8), np.abs(noise_gauss).astype(np.uint8), noise_gauss.flatten()
    
def saltnpepper_noise(img, p_pepper, p_salt):
    """
    Application of the salt&pepper noise to the image, use the uniform probability distribution.
    
    Parameters
    ----------
    img : ndarray
        Image
    p_pepper : float
        Pepper probability
    p_salt : float
        Salt probability
        
    Returns
    ----------
    tuple
        img+noise, noise as img, generated noise
    """
    
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    noise_rnd = np.random.uniform(0, 1, img_gray.shape) 

    pepper_mask = np.where(noise_rnd > (1-p_pepper))
    salt_mask = np.where(noise_rnd < (p_salt))

    noise = np.zeros(img_gray.shape)
    noise_pepper = np.zeros(img_gray.shape)
    noise_salt = np.zeros(img_gray.shape)
    
    noise_pepper[pepper_mask] = -256
    noise_salt[salt_mask] = 255
    
    
    noise_pepper = np.expand_dims(noise_pepper,axis = 2)
    noise_pepper = np.concatenate((noise_pepper, noise_pepper, noise_pepper), axis = 2)
    
    noise_salt = np.expand_dims(noise_salt,axis = 2)
    noise_salt = np.concatenate((noise_salt, noise_salt, noise_salt), axis = 2)
    print(noise_pepper)
    
    noise = noise_salt + noise_pepper
    

    img_noise = img + noise # noise application
    
    img_noise[img_noise<0] = 0
    img_noise[img_noise>255] = 255
    
    return img_noise.astype(np.uint8), np.abs(noise).astype(np.uint8), noise.flatten()
    
def exponential_noise(img, _lambda):
    """
    Application of the exponential noise to the image.
    
    Parameters
    ----------
    img : ndarray
        Image
    _lambda : float
        Exponential probability parameter
        
    Returns
    ----------
    tuple
        img+noise, noise as img, generated noise
    """
    
    # replace following dummy code by your code
    
    noise_exp = np.random.exponential(_lambda, img.shape)
    
    img_noise = img + noise_exp # noise application
    img_noise[img_noise<0] = 0
    img_noise[img_noise>255] = 255
    return img_noise.astype(np.uint8), np.abs(noise_exp).astype(np.uint8), noise_exp.flatten()
    
def averaging(img, kernel_size, sigma_x):
    """
    Application of the mean filter to the noisy image.

    Parameters
    ----------
    img : ndarray
        Image
    k_size : int
        blurring kernel size.
        
    Returns
    ----------
    tuple
        processed image
    """
    img_copy = img.copy()
    img_copy = cv2.filter2D(img,sigma_x,kernel_size)

    return img_copy

def median_blurring(img, ksize):
    """
    Application of the median blurring to the noisy image.
    
    Parameters
    ----------
    img : ndarray
        Image
    ksize: int
        aperture linear size
            
    Returns
    ----------
    tuple
        processed image
    """
    
    img_copy = img.copy()
    img_copy = cv2.medianBlur(img_copy, ksize)
    return img_copy
    
def min_filter(img, kernel):
    """
    Application MIN filter to the noisy image.
    
    Parameters
    ----------
    img : ndarray
        Image
    kernel: np.ndarray
        structuring element used for erosion
            
    Returns
    ----------
    tuple
        processed image
    """
    
    img_copy = img.copy()
    img_copy = cv2.erode(img,kernel,iterations = 1)
    return img_copy
    
def max_filter(img, kernel):
    """
    Application MAX filter to the noisy image.
    
    Parameters
    ----------
    img : ndarray
        Image
    kernel: np.ndarray
        structuring element used for dilation
            
    Returns
    ----------
    tuple
        processed image
    """
    
    img_copy = img.copy()
    img_copy = cv2.dilate(img,kernel,iterations = 1)
    return img_copy
    
img =  cv2.imread("data/tower.jpg")

# PARAMETERS (have to be set)

# Uniform noise
low = 0.2
high = 0.5
# Gaussian noise
mu = 20
sigma = 10
# Salt&Pepper noise
p_pepper = .05
p_salt = .05
# Exponential noise
_lambda = 50

# Mean filter
avg_kernel_size = np.ones((5,5),np.float32)/25
avg_sigma_x = -1
# Median blurring
median_blur_ksize = 3
# MIN filter
min_kernel = np.ones((5,5),np.float32)/25
# MAX filter
max_kernel = np.ones((5,5),np.float32)/25

# Noise application
noise_source = {'Uniform Noise': uniform_noise(img, low, high),
                'Gaussian Noise': gaussian_noise(img, mu, sigma),
                'Salt&Pepper Noise': saltnpepper_noise(img, p_pepper, p_salt),
                'Exponential Noise': exponential_noise(img, _lambda)
                }

# Application of filters and plotting

for key, val in noise_source.items():
    fig = plt.figure(figsize= (24,8))
    plt.subplot(241)
    plt.suptitle(key)
    
    # Noise
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB));
    plt.title('original')
    plt.subplot(242)
    plt.imshow(cv2.cvtColor(val[1], cv2.COLOR_BGR2RGB));
    plt.title('noise')
    plt.subplot(243)
    plt.imshow(cv2.cvtColor(val[0], cv2.COLOR_BGR2RGB));
    plt.title('original + noise')
    plt.subplot(244)
    plt.hist(val[2], density=True, bins=np.linspace(1,255,255));
    plt.title('noise histogram');
    
    # Filters
    plt.subplot(245)
    plt.imshow(cv2.cvtColor(averaging(val[0], avg_kernel_size, avg_sigma_x), cv2.COLOR_BGR2RGB));
    plt.title('Mean Filter')
    plt.subplot(246)
    plt.imshow(cv2.cvtColor(median_blurring(val[0], median_blur_ksize), cv2.COLOR_BGR2RGB));
    plt.title('Median Filter')
    plt.subplot(247)
    plt.imshow(cv2.cvtColor(min_filter(val[0], min_kernel), cv2.COLOR_BGR2RGB));
    plt.title('MIN Filter')
    plt.subplot(248)
    plt.imshow(cv2.cvtColor(max_filter(val[0], max_kernel), cv2.COLOR_BGR2RGB));
    plt.title('MAX Filter');
    plt.show()

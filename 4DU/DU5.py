import exif
import cv2
import matplotlib.pyplot as plt
import glob
import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack
from scipy.sparse.linalg import lsqr


def read_images(file_pattern, scale_percent = 5):
    files = glob.glob(file_pattern) # Load path to all images to process
    # lists preparation
    imgs = [] #images
    t = [] #exposure times

    # cylcle over all images
    for file in files:
        tmp_img = cv2.imread(file) # load image
        width = int(tmp_img.shape[1] * scale_percent / 100) #new image width after shrink
        height = int(tmp_img.shape[0] * scale_percent / 100) #new image heght after shrink
        dim = (width, height) #new image dimension after shrink
        imgs.append(cv2.resize(tmp_img, dim, interpolation = cv2.INTER_AREA).flatten()) # shrink the image
        info = exif.Image(file) # Load the EXIF information
        t.append(info.exposure_time) # Save exposure time to the list
    t = np.array(t)
    return dim, np.array(imgs), t # return the data

def bgr2rgb(bgr_image):
    b,g,r = cv2.split(bgr_image) # Split color channels
    return cv2.merge([r,g,b]) # merge color channels in new order RGB

def get_weights(Z, L):
    #function for calculation of weights as a penalty of a pixel value
    return np.interp(Z, [0, (L-1)/2, L-1], [0, 1, 0]).flatten() 
    
def estimate_exposure(Z, w):

    t_ind, Z_ind = np.indices(Z.shape)
    E_ind=Z_ind.flatten()
    t_ind=t_ind.flatten()
    Z = Z.astype('float32')
    result__ = np.where(Z == 0)
    
    for i in range(len(result__[0])):
        Z[result__[0][i]][result__[1][i]] = 0.000001

    b = np.log(Z.flatten())*np.sqrt(w)

    for i in range(Z.shape[0]):
        
        logE = csr_matrix((np.sqrt(get_weights(Z[i], 2**8)), (Z_ind[0], Z_ind[0])), shape=(Z.shape[1], Z.shape[1]))
        logT = csr_matrix((np.sqrt(get_weights(Z[i], 2**8)), (Z_ind[0], np.full((Z_ind.shape[1], 1),i, dtype=int).flatten())), shape=(Z.shape[1], Z.shape[0]))
        A_t = hstack((logE, logT))
        if i == 0:
            A = A_t
        else:
            A = vstack((A, A_t))
    
    sol = lsqr(A, b)
    
    return np.exp(sol[0][: np.max(E_ind)+1]), np.exp(sol[0][np.max(E_ind)+1:])

    return Ei_shape, tj

if __name__ == "__main__":
    
    dim, Z, t = read_images("./images/*.jpg",20)
    

    L = np.max(Z) + 1
    w = get_weights(Z,L)

    Ei_shape, tj = estimate_exposure(Z, w)
    
    hdr_img, times = estimate_exposure(Z,w)
    hdr_img = np.reshape(hdr_img, (60*4, 80*4, 3))
    hdr_img_rgb = bgr2rgb(hdr_img)
    hdr_img.shape
    
    plt.subplot(1, 2, 1)
    plt.plot(tj)
    
    plt.subplot(1, 2, 2)
    plt.imshow(hdr_img_rgb)
    plt.title("HDR Image")
    
    plt.show()
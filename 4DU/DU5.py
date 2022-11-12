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

def estimate_exposure(Z, w, t, dim):
    """
    Estimate the chip cells exposure and exposure time from the pixels intensities.

    Assume taht the response function f is the identity.

    Z(j,i) is the intensity of i-th pixel in j-th image
    w - weights
    """
    w_sqrt=np.sqrt(w)
    
    Z_flat=Z.flatten()
    Z_log=np.log(Z_flat,where=(Z_flat!=0))
    Z_w=Z_log*w_sqrt
    pixels=dim[0]*dim[1]
    N=pixels*3
    P=len(t)
    
    first=1
    for j in range(P):
        row = np.array(list(range(N)))
        col = np.array(list(range(N)))
        data = np.array(w[j*N:N*(j+1)])
        irra_matrix=csr_matrix((data, (row, col)), shape=(len(row),len(col)))
        
        row = np.array(list(range(N)))
        col_cur = np.array(np.ones(N)*j)
        data = np.array(w[j*N:N*(j+1)])
        
        if first == 1:
            data = np.array(np.zeros(N))
        column_matrix=csr_matrix((data, (row, col_cur)), shape=(len(row),len(col_cur)))
        
        select_ind = np.array(list(range(P)))
        time_matrix=column_matrix.tocsr()[:,select_ind]
        
        combined=hstack([irra_matrix,time_matrix])
        
        if first == 1:
            A_matrix = combined
            first = 0
            continue
        
        A_matrix=vstack([A_matrix,combined])
        
    A_csr_matrix=A_matrix.asformat("csr")
    
    sol = lsqr(A_csr_matrix, Z_w)
    solution_array=sol[0]
    
    tj=np.exp(solution_array[3*pixels:3*pixels+len(t)])
    Ei=np.exp(solution_array[:3*pixels])
    Ei_shape=Ei.reshape(dim[1],dim[0],3,order='F')

    return Ei_shape, tj
if __name__ == "__main__":

    dim, Z, t = read_images("./images/*.jpg")

    L = np.max(Z) + 1
    w = get_weights(Z,L)

    Ei_shape, tj = estimate_exposure(Z, w,t,dim)

    b,g,r = cv2.split(Ei_shape)
    pic = b*0.3+g*0.6+r*0.1
    
    plt.subplot(1, 2, 1)
    plt.imshow(pic.astype(np.uint16),cmap='rainbow')
    
    plt.subplot(1, 2, 2)
    plt.plot(tj)

    plt.show()
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np


class productionLine:

    def preparePictures(self, nails_img):
        # basic pictures processing: turning it to grayscale and using threshold on it
        nails_img_gray = [None] * len(nails_img)
        nails_img_thresh = [None] * len(nails_img)
        for i, nail_img in enumerate(nails_img):
            nail_img_copy = nail_img.copy()
            nails_img_gray[i] = cv2.cvtColor(nail_img_copy, cv2.COLOR_BGR2GRAY)
            _, nails_img_thresh[i] = cv2.threshold(nails_img_gray[i], 120, 255, cv2.THRESH_BINARY_INV)
        return nails_img_thresh

    def thereIsNail(self, nails_img):
        """
        function that return if there is a nail on the picture
        """
        nails_img_thresh = self.preparePictures(nails_img)
        # counting all white pixels that will allow us to define if there is a nail on the picture
        result = [None] * len(nails_img)
        for i, nail_img_thresh in enumerate(nails_img_thresh):
            if np.sum(nail_img_thresh == 0) > 50:
                result[i] = True
            else:
                result[i] = False

        # ax = plt.subplot(4,2,1)
        # ax.set_title(f"{result[0]}")
        # plt.imshow(nails_img_thresh[0],cmap="gray")

        # ax = plt.subplot(4,2,2)
        # ax.set_title(f"{result[1]}")
        # plt.imshow(nails_img_thresh[1],cmap="gray")

        # ax = plt.subplot(4,2,3)
        # ax.set_title(f"{result[2]}")
        # plt.imshow(nails_img_thresh[2],cmap="gray")

        # ax = plt.subplot(4,2,4)
        # ax.set_title(f"{result[3]}")
        # plt.imshow(nails_img_thresh[3],cmap="gray")

        # ax = plt.subplot(4,2,5)
        # ax.set_title(f"{result[4]}")
        # plt.imshow(nails_img_thresh[4],cmap="gray")

        # ax = plt.subplot(4,2,6)
        # ax.set_title(f"{result[5]}")
        # plt.imshow(nails_img_thresh[5],cmap="gray")

        # ax = plt.subplot(4,2,7)
        # ax.set_title(f"{result[6]}")
        # plt.imshow(nails_img_thresh[6],cmap="gray")

        # plt.show()
        return result

    def deleteErrors(self,nails_img):
        contours = [None]*len(nails_img)
        hierarchy = [None]*len(nails_img)
        nails_img_thresh = self.preparePictures(nails_img)
        for i, nail_img_thresh in enumerate(nails_img_thresh):
            contours[i], _ = cv2.findContours(nail_img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            print(len(contours[i]))
        
        newContours = [None]*len(contours)
        for i, contoursSet in enumerate(contours):
            if len(contoursSet)>1:
                newContours[i]= max(contoursSet, key = cv2.contourArea)
            else: 
                newContours[i] = contours[i]
        
        
        # for i in range(len(contours)):
            # cv2.drawContours(nails_img[i],newContours[i], -1, (255, 255, 0),thickness = 2)
        
        # for i in range(len(contours)):
            # ax = plt.subplot(4,2,i+1)
            # plt.imshow(nails_img[i])
        # plt.show()
        
        return newContours

    
    def findActualNailVector(self):
        pass

    def nailStartPoint(self):
        """
        function that return the beginning of the nail
        """
        pass

    def nailEndPoint(self):
        """
        function that return the end of the nail
        """
        pass

    def nailLength(self):
        """
        returns nail length
        """
        pass


if __name__ == "__main__":

    PATH = os.path.join("data", "homework")
    """
    str,
    folder, where are all images for homework
    """
    LENGTH = 0.8
    """
    float,
    length (letter 'L' in diagram above), this number represents
    length between plane and camera lens.
    """
    RESOLUTION = (600, 397)
    """
    tuple of int,
    resolution of camera('w x h' in diagram above)
    """
    ALPHA = 15
    """
    int,
    angle of camera (letter alpha in diagram above)
    """

    empty = cv2.imread(os.path.join(PATH, "nail_empty.jpg"))
    empty_color = cv2.cvtColor(empty, cv2.COLOR_BGR2RGB)

    error = cv2.imread(os.path.join(PATH, "nail_error.jpg"))
    error_color = cv2.cvtColor(error, cv2.COLOR_BGR2RGB)

    nails = []
    for i in range(1, 8):
        im = cv2.imread(os.path.join(PATH, "nail_0{}.jpg".format(i)))
        im_color = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        nails.append(im_color)
        
        
        
        
    pLine = productionLine()
    
    boolNails = pLine.thereIsNail(nails)
    contours = pLine.deleteErrors(nails)
    
    
    
    # ax = plt.subplot(3,2,1)
    # ax.set_title("Empty - no nail")
    # plt.imshow(empty_color)

    # ax = plt.subplot(3,2,2)
    # ax.set_title("Example of error")
    # plt.imshow(error_color)

    # ax = plt.subplot(3,2,3)
    # ax.set_title("Example 1")
    # plt.imshow(nails[0])

    # ax = plt.subplot(3,2,4)
    # ax.set_title("Example 2")
    # plt.imshow(nails[1])

    # ax = plt.subplot(3,2,5)
    # ax.set_title("Example 3")
    # plt.imshow(nails[2])

    # ax = plt.subplot(3,2,6)
    # ax.set_title("Example 4")
    # plt.imshow(nails[3])
    
    #plt.show()
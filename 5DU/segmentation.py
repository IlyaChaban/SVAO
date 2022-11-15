import cv2
import matplotlib.pyplot as plt
import os
import numpy as np


class productionLine:
    def __init__(self,nails_img):
        self.nails_img = nails_img
        self.filtered_picture = self.preparePictures()
        self.contours = self.findNailContours()
        self.bBoxes = self.createBBoxAroundContours()
        self.nailsLengths = self.nailLength()
        self.midpoints = self.nailStartEndPoints()

    def preparePictures(self):
        # basic pictures processing: turning it to grayscale and using threshold on it
        nails_img_gray = [None] * len(self.nails_img)
        nails_img_thresh = [None] * len(self.nails_img)
        
        kernel = np.ones((4, 4), np.uint8)
        
        for i, nail_img in enumerate(self.nails_img):
            nail_img_copy = nail_img.copy()
            nails_img_gray[i] = cv2.cvtColor(nail_img_copy, cv2.COLOR_BGR2GRAY)
            _, nails_img_thresh[i] = cv2.threshold(nails_img_gray[i], 120, 255, cv2.THRESH_BINARY_INV)
            
            nails_img_thresh[i] = cv2.dilate(nails_img_thresh[i], kernel, iterations=2)
            
            _, nails_img_thresh[i] = cv2.threshold(nails_img_thresh[i], 120, 255, cv2.THRESH_BINARY)
            
            nails_img_thresh[i] = cv2.erode(nails_img_thresh[i], kernel, iterations=2)
        
        # for i in range(len(nails_img_thresh)):
            # ax = plt.subplot(4,2,i+1)
            # plt.imshow(nails_img_thresh[i],cmap = "gray")
        # plt.show()
        
        return nails_img_thresh

    def thereIsNail(self):
        """
        function that return if there is a nail on the picture
        """
        nails_img_thresh = self.filtered_picture.copy()
        # counting all white pixels that will allow us to define if there is a nail on the picture
        result = [None] * len(self.nails_img)
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

    def findNailContours(self):
        """
        Function that creates contours around nails
        """
        contours = [None]*len(self.nails_img)
        hierarchy = [None]*len(self.nails_img)
        nails_img_thresh = self.filtered_picture.copy()
        for i, nail_img_thresh in enumerate(nails_img_thresh):
            contours[i], _ = cv2.findContours(nail_img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        
        #finding that one correct contour
        newContours = [None]*len(contours)
        # TODO: when i'm using contours[i] to describe contour it works. But when i use contoursSet[x] it creates multiple contours that are 1 point long. Final result has to be 1 continuous contour.
        # HINT: contours[i] is tuple, contoursSet[x] numpy.ndarray'
        for i, contoursSet in enumerate(contours):
            if len(contoursSet)>1:
                
                if cv2.contourArea(contoursSet[0]) < cv2.contourArea(contoursSet[1]):
                    newContours[i]= tuple([contoursSet[1]])
                else:
                    newContours[i] = tuple([contoursSet[0]])
                
            else:
                newContours[i] = contours[i]

        # #showing all the images
        # for i in range(len(contours)):
            # cv2.drawContours(self.nails_img[i],newContours[i], -1, (255, 255, 0),thickness = 2)
        # for i in range(len(contours)):
            # ax = plt.subplot(4,2,i+1)
            # plt.imshow(self.nails_img[i])
        # plt.show()
        
        return newContours
    
    def createBBoxAroundContours(self):
        """
        function that creates smalles area bondary box around existing contours 
        """
        contours = self.contours
        
        #creating boundary box around that one contour
        bBoxes = [None]*len(contours)
        
        for i, contour in enumerate(contours):
            if len(contour) > 0:
                bBoxes[i] = cv2.minAreaRect(contour[0])
            else: 
                bBoxes[i] = None
        
        rects = []
        for bBox in bBoxes :
            if bBox != None:
                rects.append(np.int0(cv2.boxPoints(bBox)))
            else:
                rects.append(np.array([[None,None,None,None]]))
        
        # #showing all the images
        # for i in range(len(contours)):
            # cv2.drawContours(self.nails_img[i],contours[i], -1, (255, 255, 0),thickness = 2)
            # if rects[i][0][0] != None:
                # cv2.drawContours(self.nails_img[i],[rects[i]],0,(0,255,255),2)
        
        # for i in range(len(contours)):
            # ax = plt.subplot(4,2,i+1)
            # plt.imshow(self.nails_img[i])
        # plt.show()
        return rects
    
    def findActualNailVector(self):
        contours = self.deleteErrors(self.nails_img)
        
        pass

    def nailStartEndPoints(self):
        """
        function that return the beginning and the end of the nail
        """
        contours = self.contours
        shortest_edge = self.indexOfShortEdge()
        midpoint_of_shortest_edge = self.midPointOfEdge(shortest_edge)
        return midpoint_of_shortest_edge
    
    def midPointOfEdge(self, shortest_edge):
        midpoints = [[None],[None]]*len(self.nails_img)
        for i, bBoxCoords in enumerate(self.bBoxes):
            if bBoxCoords.all() != None:
                if shortest_edge[i] == 0:
                    x1 = round((bBoxCoords[0][0] + bBoxCoords[1][0])/2)
                    y1 = round((bBoxCoords[0][1] + bBoxCoords[1][1])/2)
                    x2 = round((bBoxCoords[2][0] + bBoxCoords[3][0])/2)
                    y2 = round((bBoxCoords[2][1] + bBoxCoords[3][1])/2)
                else:
                    x1 = round((bBoxCoords[1][0] + bBoxCoords[2][0])/2)
                    y1 = round((bBoxCoords[1][1] + bBoxCoords[2][1])/2)
                    x2 = round((bBoxCoords[3][0] + bBoxCoords[0][0])/2)
                    y2 = round((bBoxCoords[3][1] + bBoxCoords[0][1])/2)
                midpoints[i]=[[x1,y1],[x2,y2]]
        return midpoints
    
    def indexOfShortEdge(self):
        shortest_edge_index = [None]*len(self.nails_img)
        for i, points in enumerate(self.bBoxes):
            if points[0][0] != None:
                len1 = np.sqrt((points[0][0]-points[1][0])**2+(points[0][1]-points[1][1])**2)
                len2 = np.sqrt((points[1][0]-points[2][0])**2+(points[1][1]-points[2][1])**2)
                if len1 < len2:
                    shortest_edge_index[i] = 0
                else:
                    shortest_edge_index[i] = 1
            else:
                shortest_edge_index[i] = None
            
        return shortest_edge_index
    
    def lenOfLongestEdge(self, points):

        if points[0].any() != None:
            len1 = np.sqrt((points[0][0]-points[1][0])**2+(points[0][1]-points[1][1])**2)*.00026795*1000
            len2 = np.sqrt((points[1][0]-points[2][0])**2+(points[1][1]-points[2][1])**2)*.00026795*1000
        else:
            return None
        return max(len1,len2)
    
    def nailLength(self):
        """
        returns nail length
        """
        nailLengths = [None]*len(self.nails_img)
        edge_length_mm = [None]*len(self.nails_img)
        for i, bBox in enumerate(self.bBoxes):
                edge_length_mm[i] = self.lenOfLongestEdge(bBox)
            
        return edge_length_mm
        
    def VisualizeResults(self):
        #showing all the images
        nails_img = self.nails_img.copy()
        
        for i in range(len(self.contours)):
            # drawing contours
            cv2.drawContours(nails_img[i],self.contours[i], -1, (255, 0, 0),thickness = 2)
            
            # drawing bBoxes in not empty
            if self.bBoxes[i][0][0] != None:
                cv2.drawContours(nails_img[i],[self.bBoxes[i]], -1, (0, 255, 0),thickness = 2)
                
            # drawing midpoints
            if self.midpoints[i] != [None]:
                cv2.circle(nails_img[i], (int(round(self.midpoints[i][0][0])),int(round(self.midpoints[i][0][1]))), radius=4, color=(0, 0, 255), thickness=4)
                cv2.circle(nails_img[i], (int(round(self.midpoints[i][1][0])),int(round(self.midpoints[i][1][1]))), radius=4, color=(0, 0, 255), thickness=4)
                
        for i in range(len(self.contours)):
            plt.subplot(3,3,i+1)
            if self.nailsLengths[i] != None:
                plt.title(f"Length of the nail is:{round(self.nailsLengths[i])} [mm]")
            else:
                plt.title(f"Length of the nail is: None")
            plt.imshow(nails_img[i])
        plt.tight_layout()
        plt.show()


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
        
    pLine = productionLine(nails)
    pLine.VisualizeResults()


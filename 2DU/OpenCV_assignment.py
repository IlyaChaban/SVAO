import cv2 as cv
import numpy as np
import matplotlib.pyplot as plts

class ImageBGR:

    def __init__(self, file: str = None, image: np.ndarray = None):
        if file:
            self.__image = cv.imread(file)  # load the image from file -- your code here
        elif image:
            self.__image = self.image  # load the image from ndarray -- your code here
        else:
            raise AttributeError(
                "There is no such file or it is not an array ")  # add your code for the case of the error

    def gray(self) -> np.ndarray:
        """
        Returns the image in grayscale
        """

        return cv.cvtColor(self.__image, cv.COLOR_BGR2GRAY)

    def lab(self) -> np.ndarray:
        """
        Returns the image in Lab color format
        """

        return cv.cvtColor(self.__image, cv.COLOR_BGR2Lab)

    def rgb(self) -> np.ndarray:
        """
        Returns the image in RGB color format
        """

        return cv.cvtColor(self.__image, cv.COLOR_BGR2RGB)

    def bgr(self) -> np.ndarray:
        """
        Returns the image in BGR color format
        """

        return self.__image

    def resize(self, width: int, height: int) -> 'ImageBGR':
        """
        Returns a new instance of ImageBGR class but with a new image shape that is gotten by OpenCV resize function
        """

        return cv.resize(self.__image, [height, width])

    def rotate(self, angle: int, keep_ratio: bool) -> 'ImageBGR':
        """
        Returns a new instance of ImageBGR class containing the image from the original instance of the ImageBGR class but rotated by the given angle.
        If keep_ratio is set to True, the new image must be the same size as the original. If set to False, the new image must contain the entire image information
        from the original image.
        """

        image = self.__image;
        (centerX, centerY) = (self.__image.shape[1] // 2, self.__image.shape[0] // 2)
        RotMat = cv.getRotationMatrix2D((centerX, centerY), angle, 1)
        if keep_ratio == 1:
            Fin_res = cv.warpAffine(image, RotMat, [self.__image.shape[1], self.__image.shape[0]])

        else:
            cos = np.abs(RotMat[0, 0])
            sin = np.abs(RotMat[0, 1])
            nW = int((self.__image.shape[0] * sin) + (self.__image.shape[1] * cos))
            nH = int((self.__image.shape[0] * cos) + (self.__image.shape[1] * sin))
            RotMat[0, 2] += (nW / 2) - centerX
            RotMat[1, 2] += (nH / 2) - centerY
            Fin_res = cv.warpAffine(image, RotMat, (nW, nH))

        return Fin_res

    def histogram(self) -> np.ndarray:
        """
        Returns the histogram of the image from its grayscale version.
        """

        return np.histogram(self.gray(), 256, [0, 256])

    @property
    def shape(self) -> tuple:
        """
        A function decorated as an attribute that returns the dimensions of the stored image.
        """

        return (self.__image.shape[0], self.__image.shape[1], self.__image.shape[2])

    @property
    def size(self) -> int:
        """
        A function decorated as an attribute that returns the memory occupied by the image (purely the field in which the image is stored).
        """

        return self.__image.itemsize * self.__image.size


if __name__ == "__main__":
    image = ImageBGR(file='./image.jpg')

    fig, ax = plt.subplots(4, 3)

    ax[0, 0].axis('off')
    ax[0, 0].title.set_text('Init')

    ax[0, 1].imshow(image.gray(), cmap='gray')
    ax[0, 1].title.set_text('Black and White')
    ax[0, 1].axis('off')

    ax[0, 2].imshow(image.lab())
    ax[0, 2].title.set_text('Lab')
    ax[0, 2].axis('off')

    ax[1, 0].imshow(image.rgb())
    ax[1, 0].title.set_text('RGB')
    ax[1, 0].axis('off')

    ax[1, 1].imshow(image.bgr())
    ax[1, 1].title.set_text('BGR')
    ax[1, 1].axis('off')

    ax[1, 2].imshow(cv.cvtColor(image.resize(60, 60), cv.COLOR_BGR2RGB))
    ax[1, 2].title.set_text('Resize')
    ax[1, 2].axis('off')

    ax[2, 0].imshow(image.rotate(60, 1))
    ax[2, 0].title.set_text('Rotate')
    ax[2, 0].axis('off')

    #print(image.histogram()[0])
    ax[2, 1].stairs(image.histogram()[0], fill=True)
    ax[2, 1].title.set_text('Histogram')

    image_shape = np.asarray(image.shape)
    ax[2, 2].text(0, 0, ''.join(str(image.shape)), fontsize=12)
    ax[2, 2].title.set_text('Shape')
    ax[2, 2].axis('off')
    ax[2, 2].axis('off')

    ax[3, 0].text(0, 0, str(image.size), fontsize=12)
    ax[3, 0].title.set_text('Size')
    ax[3, 0].axis('off')

    ax[3, 0].axis('off')
    ax[3, 1].axis('off')
    ax[3, 2].axis('off')

    plt.tight_layout()
    plt.show()


image.bgr()

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from copy import copy


def fft_plot(img, fft):
    magnitude_spectrum = np.abs(fft)
    magnitude_spectrum[magnitude_spectrum==0] = 1e-6
    phase_spectrum = np.arctan2(fft.real, fft.imag)
    # fig = plt.figure(figsize= (20,15))
    # plt.subplot(221),plt.imshow(img, cmap = 'gray')
    # plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    # plt.subplot(222),plt.imshow(np.log10(magnitude_spectrum), cmap = 'gray')
    # plt.title('Magnitude Spectrum Log'), plt.xticks([]), plt.yticks([])
    # plt.subplot(223),plt.imshow(phase_spectrum, cmap = 'gray')
    # plt.title('Phase Spectrum'), plt.xticks([]), plt.yticks([])
    # plt.subplot(224),plt.imshow(magnitude_spectrum, cmap = 'gray')
    # plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    
    return magnitude_spectrum


original_image = cv2.cvtColor(cv2.imread("data/fft_org.png"), cv2.COLOR_BGR2GRAY)
noisy_image = cv2.cvtColor(cv2.imread("data/fft_1.png"), cv2.COLOR_BGR2GRAY)

fft_original = np.fft.fft2(original_image)
fft_noisy = np.fft.fft2(noisy_image)

fft_shift_original = np.fft.fftshift(fft_original)
fft_shift_noisy = np.fft.fftshift(fft_noisy)

original_magnitude = fft_plot(original_image,fft_shift_original)
noisy_magnitude = fft_plot(noisy_image, fft_shift_noisy)


difference = original_magnitude - noisy_magnitude
# converting to 0 to 255 values

#difference = cv2.normalize(difference, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
#difference = difference.astype(np.uint8)

# plt.imshow(np.log10(difference), cmap = 'gray')
# plt.show()


def f(difference, fft_shift_noisy,constant = 1):
    # creating mask from difference 
    mask_from_difference = np.zeros(difference.shape)
    #print(difference)

    mask_from_difference[difference < constant] = 0
    mask_from_difference[difference >= constant] = 1

    # plt.imshow(mask_from_difference, cmap ="gray")
    # plt.show()
    fft_noisy_hpf = copy(fft_shift_noisy)
    fft_noisy_hpf[mask_from_difference == 0] = 0
    f_ishift_hpf = np.fft.ifftshift(fft_noisy_hpf)
    img_noisy_hpf = np.fft.ifft2(f_ishift_hpf)
    img_noisy_hpf = np.real(img_noisy_hpf)
    return img_noisy_hpf

fig, ax = plt.subplots()
picture = ax.imshow(f(difference, fft_shift_noisy, 1), cmap = "gray")
fig.subplots_adjust(left=0.25)

axamp = fig.add_axes([0.1, 0.25, 0.0225, 0.63])
amp_slider = Slider(
    ax=axamp,
    label="Amplitude",
    valmin=-1,
    valmax=100000,
    valinit=10,
    orientation="vertical"
)

def update(val):
    ax.imshow(f(difference, fft_shift_noisy, amp_slider.val), cmap = "gray")
    fig.canvas.draw_idle()
amp_slider.on_changed(update)
plt.show()
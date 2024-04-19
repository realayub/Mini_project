import cv2
import numpy as np
from skimage import io, img_as_float
from skimage.restoration import denoise_nl_means, estimate_sigma

img = img_as_float(io.imread('images/BSE_25sigma_noisy.jpg'))
sigma_est=np.mean(estimate_sigma(img,multichannel=false))

denoise_img=denoise_nl_means(img,h=1.*sigma_est,fast_mode=True,
                             patch_size=5,patch_distance=3,multichannel=False)
cv2.imshow("original",img)
cv2.imshow("denoised",denoise_img)
cv2.waitkey(0)
cv2.destroyAllwindow()
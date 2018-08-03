#import packages
from skimage.filters import hessian, sobel_h
import numpy as np
import cv2

def HHF(image):
    horizontal_gradient = sobel_h(image)
    hessian_filter = hessian(horizontal_gradient,scale_range=(1,10),scale_step=2,beta1=0.5,beta2=10)
    norm_image = cv2.normalize(hessian_filter, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return norm_image
    
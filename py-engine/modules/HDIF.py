import cv2
import numpy as np

# blur function
def blur_n_times(img: list, n: int, kernel: tuple):
    blur_img = img
    while(n > 0):
        blur_img = cv2.boxFilter(blur_img, 0, kernel)
        n = n - 1
    return blur_img

# get difference between image
def get_diff(img_1: list, img_2: list):
    return img_1 - img_2

# get the maximum pixel value of difference image
def get_max(img_diff):
    X, Y = img_diff.shape
    img_max = np.zeros((X,Y))
    for x in range(0,X):
        for y in range(0,Y):
            img_max[x][y] = max(img_diff[x][y],img_diff[x][y],img_diff[x][y],0)
    return img_max

# HDIF function (all in one)
def HDIF(img: list, v: int, u: int, kernel: tuple):
    """
    This function returns the result of a basic HDIF test.
    v has to be larger than u.
    """
    if u >= v:
        print("v has to be larger than u")
    else:
        diff = v - u
        blur_u = blur_n_times(img, v, kernel)
        result = get_max(get_diff(blur_n_times(blur_u, diff, kernel), blur_u))
        return result

# implement the improved function
def HDIF_plus(img: list, v: int, u: int, kernel: tuple):
    img_hdif = HDIF(img, v, u, kernel)
    img_hdif_plus = HDIF(img, v+1, u+1, kernel)
    diff = img_hdif - img_hdif_plus
    hdif_plus = get_max(diff)
    return hdif_plus


def HDIF_bgr(img: list, v: int, u:int, kernel: tuple):
    """
    returns a list of hdif results for all channels
    """
    bgr = cv2.split(img)
    hdif_bgr = []
    for channel in bgr:
        hdif_bgr.append(HDIF(channel, v, u, kernel))

    return hdif_bgr

def HDIF_plus_bgr(img: list, v: int, u:int, kernel: tuple):
    """
    returns a list of hdif_plus results for all channels
    """
    bgr = cv2.split(img)
    hdif_bgr = []
    for channel in bgr:
        hdif_bgr.append(HDIF_plus(channel, v, u, kernel))

    return hdif_bgr


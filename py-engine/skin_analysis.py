import cv2
import sys
import face_recognition
import numpy as np
from modules import HDIF, HHF, mask, LandVis

PATH_TO_IMG="/proj/aimirror/public/pictures/target.jpg"

# Load Original Image
image = cv2.imread(PATH_TO_IMG, 1)
image_gray = cv2.imread(PATH_TO_IMG, 0)

# Obtain FLP 
face_landmarks = face_recognition.face_landmarks(image)

# Specify Areas for Analysis
## For HDIF
cheeks = [[14, 15, 16, 47, 48, 36],[4, 3, 2, 42, 41, 32]]
## For Hessian 
others = [[62, 61, 54, 14, 13, 12],[66, 67, 50, 4, 5, 6],[46, 27, 17, 16],[37, 2, 1, 18]]

# Create Alpha Masks
alpha_cheek = mask.polygon_mask(image, face_landmarks, cheeks)
alpha_other = mask.polygon_mask(image, face_landmarks, others)

# Perform HDIF analysis and get alpha mask
hdif_results = HDIF.HDIF_plus_bgr(image, 8, 7, (3,3))
alpha_hdif = mask.mask_hdif(hdif_results)

# Perform HHF analysis and get alpha mask
hhf_result = HHF.HHF(image_gray)
alpha_hhf = mask.mask_hessian(hhf_result)

# Prepare Foreground for HDIF
blue = np.zeros(image.shape, float)
blue[:,:,0] = 255
green = np.zeros(image.shape, float)
green[:,:,1] = 255
red = np.zeros(image.shape, float)
red[:,:,2] = 255

#HDIF
first_layer = mask.mask(blue, image, alpha_hdif[0])
second_layer = mask.mask(green, first_layer, alpha_hdif[1])
third_layer = mask.mask(red, second_layer, alpha_hdif[2])
hdif_layer = mask.mask(third_layer,image,alpha_cheek)

#ここまでオッケー
orange = np.zeros(image.shape, float)
orange[:,:,0] = 100
orange[:,:,1] = 100
orange[:,:,1] = 255

#HHF
hhf_first_layer = mask.mask(orange, hdif_layer, alpha_hhf)
hhf_layer = mask.mask(hhf_first_layer, hdif_layer, alpha_other)

#Visualize Points and Polygons
out_image = LandVis.visualize_polygons(hhf_layer, face_landmarks, cheeks)
out_image = LandVis.visualize_polygons(out_image, face_landmarks, others, line_color=(0,200,255))
out_image = LandVis.visualize_flp(out_image, face_landmarks)

cv2.imwrite("/proj/aimirror/public/pictures/masked_face.jpg", out_image)





import cv2
import mask
import HHF

image = cv2.imread("../../photo/satomi.png", 0)

hessian = HHF.HHF(image)

cv2.imwrite("./test.jpg", hessian)

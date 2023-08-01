from sewar.full_ref import mse, rmse, psnr, uqi, ssim, ergas, scc, rase, sam, msssim, vifp
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt 
import cv2

blur = np.array(Image.open('/home/holmes/code/cat_face_detection/cat.jpg')) #you can pass multiple arguments in single line
org  =  np.array(Image.open('/home/holmes/code/cat_face_detection/result1.jpg')) #you can pass multiple arguments in single line

print("MSE: ", mse(blur,org))
print("RMSE: ", rmse(blur, org))
print("PSNR: ", psnr(blur, org))
print("SSIM: ", ssim(blur, org))
print("UQI: ", uqi(blur, org))
print("MSSSIM: ", msssim(blur, org))
print("ERGAS: ", ergas(blur, org))
print("SCC: ", scc(blur, org))
print("RASE: ", rase(blur, org))
print("SAM: ", sam(blur, org))
print("VIF: ", vifp(blur, org))
# plt.imshow(im)
# plt.show()
import cv2
from sewar.full_ref import mse, rmse, psnr, uqi, ssim, ergas, scc, rase, sam, msssim, vifp
# import numpy as np
# from PIL import Image
# import matplotlib.pyplot as plt 


shape = (224,224)

cat_1 = cv2.imread('/home/holmes/code/cat_face_detection/cf_0_0.jpg')#you can pass multiple arguments in single line
cat_2 = cv2.imread('/home/holmes/code/cat_face_detection/data/cat_face/cf_0_0.jpg')#you can pass multiple arguments in single line
cat_3 = cv2.imread('/home/holmes/code/cat_face_detection/data/cat_face/cf_120_0.jpg')#you can pass multiple arguments in single line
cat_1 = cv2.resize(cat_1,shape)
cat_2 = cv2.resize(cat_2,shape)
cat_3 = cv2.resize(cat_3,shape)

# cat_1 = np.array(cat_1)
# cat_2 = np.array(cat_2)
# cat_3 = np.array(cat_3)

print("MSE: ", mse(cat_1,cat_2))
print("RMSE: ", rmse(cat_1, cat_2))
print("PSNR: ", psnr(cat_1, cat_2))
print("SSIM: ", ssim(cat_1, cat_2))
print("UQI: ", uqi(cat_1, cat_2))
print("MSSSIM: ", msssim(cat_1, cat_2))
print("ERGAS: ", ergas(cat_1, cat_2))
print("SCC: ", scc(cat_1, cat_2))
print("RASE: ", rase(cat_1, cat_2))
print("SAM: ", sam(cat_1, cat_2))
print("VIF: ", vifp(cat_1, cat_2))
cat_1 = cat_3
print("2,3")
print("MSE: ", mse(cat_1,cat_2))
print("RMSE: ", rmse(cat_1, cat_2))
print("PSNR: ", psnr(cat_1, cat_2))
print("SSIM: ", ssim(cat_1, cat_2))
print("UQI: ", uqi(cat_1, cat_2))
print("MSSSIM: ", msssim(cat_1, cat_2))
print("ERGAS: ", ergas(cat_1, cat_2))
print("SCC: ", scc(cat_1, cat_2))
print("RASE: ", rase(cat_1, cat_2))
print("SAM: ", sam(cat_1, cat_2))
print("VIF: ", vifp(cat_1, cat_2))

cat_1 = cat_2
print("2,2")
print("MSE: ", mse(cat_1,cat_2))
print("RMSE: ", rmse(cat_1, cat_2))
print("PSNR: ", psnr(cat_1, cat_2))
print("SSIM: ", ssim(cat_1, cat_2))
print("UQI: ", uqi(cat_1, cat_2))
print("MSSSIM: ", msssim(cat_1, cat_2))
print("ERGAS: ", ergas(cat_1, cat_2))
print("SCC: ", scc(cat_1, cat_2))
print("RASE: ", rase(cat_1, cat_2))
print("SAM: ", sam(cat_1, cat_2))
print("VIF: ", vifp(cat_1, cat_2))

#picture preprocessing
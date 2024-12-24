import cv2
import numpy as np
import os
from PIL import Image

def change_color_to_black_white_mask(image_path):
  img = cv2.imread(image_path)
                          #B   G  R
  lower_bound = np.array((142, 0, 0)) - np.array([20, 40, 40])
  upper_bound = np.array((142, 0, 0)) + np.array([20, 40, 40])# typ aut
  combined_mask = cv2.inRange(img, lower_bound, upper_bound)
  lower_bound = np.array((230, 0, 0)) - np.array([20, 40, 40])
  upper_bound = np.array((230, 0, 0)) + np.array([20, 40, 40])
  mask = cv2.inRange(img, lower_bound, upper_bound)
  combined_mask = cv2.bitwise_or(combined_mask, mask)
  
  lower_bound = np.array((100, 60, 0)) - np.array([20, 40, 40])
  upper_bound = np.array((100, 60, 0)) + np.array([20, 40, 40])
  mask = cv2.inRange(img, lower_bound, upper_bound)
  combined_mask = cv2.bitwise_or(combined_mask, mask)
  lower_bound = np.array((70, 0, 0)) - np.array([20, 40, 40])
  upper_bound = np.array((70, 0, 0)) + np.array([20, 40, 40])
  mask = cv2.inRange(img, lower_bound, upper_bound)
  combined_mask = cv2.bitwise_or(combined_mask, mask)
  lower_bound = np.array((0, 0, 255)) - np.array([20, 40, 40])
  upper_bound = np.array((0, 0, 255)) + np.array([20, 40, 40])#chodci 
  mask = cv2.inRange(img, lower_bound, upper_bound)
  combined_mask = cv2.bitwise_or(combined_mask, mask)
  # Set target color pixels to black (0 intensity) in all channels
  img[combined_mask == 0] =np.array([255, 255, 255]) #np.array([0, 0, 0])
  img[combined_mask > 0] =np.array([0, 0, 0]) #np.array([255, 255, 255])

  return img

seg_path = r"D:\mBuzogan\Runs\runs\16-06-2024_20-24\segmentation\\"
mask_path = r"D:\mBuzogan\Runs\runs\16-06-2024_20-24\mask\\"
segmentation = os.listdir(seg_path)

for i,file in enumerate(segmentation):
  print(i,file)
  mask = change_color_to_black_white_mask(os.path.join(seg_path,file))
  # Save the modified image and mask
  cv2.imwrite(os.path.join(mask_path,file), mask)

  #Transform from RGB to 1 channel img
  img = Image.open(os.path.join(mask_path,file))
  rgb_img = img.convert('L')
  rgb_img.save(os.path.join(mask_path,file))  


print("Color change and mask creation completed!")
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle

def plot_image(title, image, cmap=None):
    plt.figure(figsize=(20,20))
    plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.show()

# Load Different Image Pairs
img1 = cv2.imread("data/images/image0.jpg")
img2 = cv2.imread("data/images/image1.jpg")

# Initialize the FAST detector and BRIEF descriptor
fast = cv2.xfeatures2d.StarDetector_create()
brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()

# Detector
kp = fast.detect(img1,None)

# Descriptor
kp1, des1 = brief.compute(img1, kp)

# Draw the keypoints on the image
img1_kp = cv2.drawKeypoints(img1, kp1, None, color=(0,255,0), flags=0)
plot_image("Image 1 Keypoints", img1_kp)
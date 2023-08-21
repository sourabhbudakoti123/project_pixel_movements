import cv2
import numpy as np

# Load two images
prev_image = cv2.imread('/Users/sourabhbudakoti/Desktop/project_pixel/venv/bin/img1.jpg', cv2.IMREAD_GRAYSCALE)
curr_image = cv2.imread('/Users/sourabhbudakoti/Desktop/project_pixel/venv/bin/img2.jpg', cv2.IMREAD_GRAYSCALE)

# Calculate optical flow using Farneback method
flow = cv2.calcOpticalFlowFarneback(prev_image, curr_image, None, 0.5, 3, 15, 3, 5, 1.2, 0)

# Convert flow to polar coordinates (magnitude and angle)
magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

# Calculate pixel movement as the sum of magnitudes
total_pixel_movement = np.sum(magnitude)

print("Total pixel movement:", total_pixel_movement)

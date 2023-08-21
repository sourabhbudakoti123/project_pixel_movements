import cv2
import numpy as np

# Load two images
prev_image = cv2.imread('/Users/sourabhbudakoti/Desktop/project_pixel/venv/bin/img1.jpg', cv2.IMREAD_GRAYSCALE)
curr_image = cv2.imread('/Users/sourabhbudakoti/Desktop/project_pixel/venv/bin/img2.jpg', cv2.IMREAD_GRAYSCALE)

# Calculate optical flow using Farneback method
flow = cv2.calcOpticalFlowFarneback(prev_image, curr_image, None, 0.5, 3, 15, 3, 5, 1.2, 0)

# Print pixel movement in both x and y directions
for y in range(flow.shape[0]):
    for x in range(flow.shape[1]):
        flow_x, flow_y = flow[y, x]
        print(f"Pixel ({x}, {y}) - Flow X: {flow_x:.2f}, Flow Y: {flow_y:.2f}")

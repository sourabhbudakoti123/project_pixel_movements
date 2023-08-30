import cv2
import numpy as np

# Load the two images
image1 = cv2.imread('C:\python\IMG_4063 (1) (1).jpg')
image2 = cv2.imread('C:\python\IMG_4098 (1) (1).jpg')

# Convert images to grayscale
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Create a mask to specify the region of interest
roi = cv2.selectROI("Select ROI", image1)
cv2.destroyAllWindows()

# Extract the ROI from both images
roi1 = gray1[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
roi2 = gray2[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]

# Calculate the absolute difference between the ROI in the two images
diff = cv2.absdiff(roi2, roi1)

# Threshold the difference image to create a binary mask
threshold = 30  # Adjust this value based on sensitivity
ret, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

# Resize the mask to match the dimensions of image2
mask_resized = cv2.resize(mask, (image2.shape[1], image2.shape[0]))

# Convert the binary mask to BGR format for overlay
mask_bgr = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR)
mask_bgr[:, :, :2] = 0  # Set blue and green channels to 0 (purple color)

# Apply the mask on the original second image
output_image = cv2.addWeighted(image2, 1, mask_bgr, 0.5, 0)

# Display the output image
cv2.imshow("Moved Part Highlighted", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

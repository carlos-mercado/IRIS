#using canny edge detection to find pupil iris center
import cv2
import numpy as np

# Load the iris image
iris_image = cv2.imread('dataset/001/L/S1001L01.jpg', cv2.IMREAD_GRAYSCALE)

# Apply Gaussian blur to reduce noise
blurred_image = cv2.GaussianBlur(iris_image, (5, 5), 0)

# Apply Canny edge detection
edges = cv2.Canny(blurred_image, 50, 150)


# Find contours in the edge image
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Initialize variables to store pupil center coordinates
pupil_center_x = 0
pupil_center_y = 0

if len(contours) > 0:
    # Find the largest contour (assumed to be the pupil)
    largest_contour = max(contours, key=cv2.contourArea)

    # Calculate the centroid of the largest contour
    M = cv2.moments(largest_contour)
    if M["m00"] != 0:
        pupil_center_x = int(M["m10"] / M["m00"])
        pupil_center_y = int(M["m01"] / M["m00"])

# Display the pupil center on the original image
result_image = cv2.cvtColor(iris_image, cv2.COLOR_GRAY2BGR)
cv2.circle(result_image, (pupil_center_x, pupil_center_y), 3, (0, 0, 255), -1)

# Save or display the result
#cv2.imwrite('pupil_center_detected.jpg', result_image)
#cv2.imshow('Result Image', result_image)
#cv2.waitKey(0)


#print(str(pupil_center_x) + " " + str(pupil_center_y))
# Define the size of the cropped image
crop_size = 256

# Calculate the cropping box
x1 = max(0, pupil_center_x - crop_size // 2)
x2 = min(iris_image.shape[1], pupil_center_x + crop_size // 2)
y1 = max(0, pupil_center_y - crop_size // 2)
y2 = min(iris_image.shape[0], pupil_center_y + crop_size // 2)

# Crop the image
cropped_image = iris_image[y1:y2, x1:x2]

# show the cropped image
#cv2.imshow('cropped image', cropped_image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()



cv2.imwrite('cleaned/001/L/S1001L01.jpg', cropped_image)

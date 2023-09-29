# import the necessary packages
from imutils import contours
from skimage import measure
import numpy as np
import imutils
import cv2 as cv
import os
# load the image

# Define the absolute path to the image
image_path = '/home/pratyush/catkin_ws/src/luminosity_drone/luminosity_drone/scripts/sample.jpg'

# Check if the image file exists
if not os.path.exists(image_path):
    print(f"Error: Image file '{image_path}' not found.")
    exit(1)

# Load the image
image = cv.imread(image_path, 1)
#
#image = cv.imread('/scripts/led.jpg', 1)
#image_path = os.path.abspath('led.jpg')

# Check if the image file exists
if not os.path.exists(image_path):
    print(f"Error: Image file '{image_path}' not found.")
    exit(1)

# Load the image
image = cv.imread(image_path, 1)

if image is None:
    print("Error: Unable to load the image.")
    exit(1)

# convert it to grayscale, and blur it
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
blurred = cv.GaussianBlur(gray, (11, 11), 0)

# threshold the image to reveal light regions in the blurred image
thresh = cv.threshold(blurred, 200, 255, cv.THRESH_BINARY)[1]

# perform a series of erosions and dilations to remove any small blobs of noise from the thresholded image
thresh = cv.erode(thresh, None, iterations=2)
thresh = cv.dilate(thresh, None, iterations=4)

# perform a connected component analysis on the thresholded image, then initialize a mask to store only the "large" components
labels = measure.label(thresh, background=0)
mask = np.zeros(thresh.shape, dtype="uint8")

# loop over the unique components
for label in np.unique(labels):
    # if this is the background label, ignore it
    if label == 0:
        continue

    # otherwise, construct the label mask and count the number of pixels
    labelMask = np.zeros(thresh.shape, dtype="uint8")
    labelMask[labels == label] = 255
    numPixels = cv.countNonZero(labelMask)

    # if the number of pixels in the component is sufficiently large, then add it to our mask of "large blobs"
    if numPixels > 300:  # Adjust this threshold as needed
        mask = cv.add(mask, labelMask)

# Find the contours in the mask
contours, _ = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# Sort the contours from left to right
contours = sorted(contours, key=lambda c: cv.boundingRect(c)[0])

# Initialize lists to store centroid coordinates and area
centroids = []
areas = []

# Loop over the contours
for i, contour in enumerate(contours):
    # Calculate the area of the contour
    area = cv.contourArea(contour)

    # Find the centroid of the contour
    M = cv.moments(contour)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    # Draw the bright spot on the image
    cv.drawContours(image, [contour], -1, (0, 255, 0), 2)
    cv.circle(image, (cX, cY), 7, (255, 0, 0), -1)

    # Append centroid coordinates and area to the respective lists
    centroids.append((cX, cY))
    areas.append(area)


# Save the output image as a PNG file
cv.imwrite("led_detection_results.png", image)

# Open a text file for writing
with open("led_detection_results.txt", "w") as file:
    # Write the number of LEDs detected to the file
    file.write(f"No. of LEDs detected: {len(centroids)}\n")
    # Loop over the contours
    for i, (centroid, area) in enumerate(zip(centroids, areas)):
        # Write centroid coordinates and area for each LED to the file
        file.write(f"Centroid #{i + 1}: {centroid}\nArea #{i + 1}: {area}\n")
# No need to close the text file when using "with open(...) as file"

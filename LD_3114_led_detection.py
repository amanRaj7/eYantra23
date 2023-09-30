# import the necessary packages
from imutils import contours
from skimage import measure
import numpy as np
import imutils
import cv2 as cv
import os

script_dir = os.path.dirname(__file__)
image_name = 'led.jpg'
image_path = os.path.join(script_dir, image_name)

# load the image, 
image = cv.imread(image_path, 1)

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
    #print(numPixels)

	# if the number of pixels in the component is sufficiently large, then add it to our mask of "large blobs"
    if numPixels > 300: 
        mask = cv.add(mask, labelMask)

# find the contours in the mask, then sort them from left to right
contours, _ = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# loop over the contours
contours = sorted(contours, key=lambda c: cv.minEnclosingCircle(c)[0])

# Initialize lists to store centroid coordinates and area
centroid = []
area = []

# Loop over the contours
for i, contour in enumerate(contours):
    # Calculate the area of the contour
    areas = cv.contourArea(contour)
    M = cv.moments(contour)
    c_X = round(M["m10"] / M["m00"],10)
    c_Y = round(M["m01"] / M["m00"],10)
    cX = int(c_X)
    cY = int(c_Y)

    # Draw the bright spot on the image
    cv.drawContours(image, [contour], -1, (0, 0, 255), 3)
    cv.putText(image, f'#{i + 1}', (cX -15, cY - 23), cv.FONT_HERSHEY_SIMPLEX,0.6, (0, 0, 255), 2)

    # Append centroid coordinates and area to the respective lists
    centroid.append((c_X, c_Y))
    area.append(areas)

a = len(centroid)

# Save the output image as a PNG file
cv.imwrite("led_detection_results.png", image)

# Open a text file for writing
with open("led_detection_results.txt", "w") as file:
    # Write the number of LEDs detected to the file
    file.write(f"No. of LEDs detected: {a}\n")
    # Loop over the contours
    for i, (centroid, area) in enumerate(zip(centroid, area)):
        # Write centroid coordinates and area for each LED to the file
        file.write(f"Centroid #{i + 1}: {centroid}\nArea #{i + 1}: {area}\n")
# Close the text file
file.close()

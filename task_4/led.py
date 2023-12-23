'''
# * Team Id : #3114
# * Author List : Aman Raj, Pratyush Roshan Mallik, Gaurav Kumar Sharma, Chandan Priyadarshi
# * Filename: = led_detection.py
# * Theme: Luminosity Drone
# * Functions: distance
# * Global Variables: NONE
'''
# import the necessary packages
from imutils import contours
from skimage import measure
import numpy as np
import imutils
import cv2 as cv
import os
import argparse


'''
# * Function Name: distance
# * Input: point1, point2
# * Output: distance between point1 and point2
# * Logic: Calculates the distance between two points
# * Example Call: distance((1,2), (3,4))
'''
def distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) 

# Construct the argument parser and parse the arguments
parser = argparse.ArgumentParser(prog='led_detection', description="process local images") # Create Argument Parser object
parser.add_argument('--image', type=str) # Add the image argument to the parser
args = parser.parse_args() # Parse the command-line arguments
image_name = args.image    # Access the value of the image argument
script_dir = os.path.dirname(__file__) # Get the path of the directory where this script is located
image_path = os.path.join(script_dir, image_name) # Create the full path of the image file

print("Image Path:", image_path) # Print the image path(debugging)

# Dictionary to store the number of LEDs in each organism
alien_names = {
    2: 'alien_a',
    3: 'alien_b',
    4: 'alien_c',
    5: 'alien_d',
}

# load the image, 
image = cv.imread(image_path, 1)

# convert it to grayscale, and blur it
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# apply a Gaussian blur with a 9X9 kernel to the image to smooth it, useful when reducing high frequency noise
blurred = cv.GaussianBlur(gray, (9, 9), 0)

# threshold the image to reveal light regions in the blurred images
thresh = cv.threshold(blurred, 200, 255, cv.THRESH_BINARY)[1]

# perform a series of erosions and dilations to remove any small blobs of noise from the thresholded image
thresh = cv.erode(thresh, None, iterations=2)
thresh = cv.dilate(thresh, None, iterations=2)

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
    #print(numPixels) # Print the number of pixels in each component (debugging)

	# if the number of pixels in the component is sufficiently large, then add it to our mask of "large blobs"
    if numPixels > 20: 
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
    cv.putText(image, f'#{i + 1}', (cX, cY - 18), cv.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 255), 2)

    # Append centroid coordinates and area to the respective lists
    centroid.append((c_X, c_Y))
    area.append(areas)

# Calculate the radius and length of the bounding box
radius = int(np.sqrt(max(area) / np.pi))
length = radius*8

# Draw the bounding box and centroid on the image
led_coordinates = centroid
a = len(centroid)

threshold_distance = radius*8.8   # Set a threshold distance to consider LEDs in the same cluster

led_clusters = []  # Initialize a list to store LED clusters

# Iterate through LED coordinates to identify clusters
for led in led_coordinates:

    added_to_cluster = False # Check if the LED can be added to an existing cluster
    for cluster in led_clusters:
        if any(distance(led, cluster_led) < threshold_distance for cluster_led in cluster):
            cluster.append(led)
            added_to_cluster = True
            break

    # If the LED couldn't be added to any existing cluster, create a new cluster
    if not added_to_cluster:
        led_clusters.append([led])

# Draw the bounding box and centroid on the image
k = len(led_clusters)
clusters = []
for i in range(k):
    g = len(led_clusters[i])  # g is the number of leds in the cluster
    # x,y of centroid of cluster by averaging up all the x,y coordinates of the leds
    x=0
    y=0
    for j in range(g):
        x += led_clusters[i][j][0]/g
        y += led_clusters[i][j][1]/g
   
    cluster_center = (round(x,4), round(y,4))
    clusters.append(cluster_center)
    cv.circle(image, (int(x),int(y)), length, (0, 0, 255), 3)

cv.imshow("Image", image) # Display the image(For debugging)
cv.waitKey(0)
# Save the output image as a PNG file
cv.imwrite("led_detection_results.png", image)

filename = image_name.replace(".png", "") # Get the filename without the extension

# Open a text file for writing
with open(f"{filename}.txt", "w") as file:
    # Loop over the contours
    for i, clusters in enumerate(clusters):
        # Write centroid coordinates and area for each LED to the file
        g= len(led_clusters[i])
        file.write(f"Organism Type: {alien_names[g]}\n")
        print(f"Organism Type: {alien_names[g]}\n") # Print the type of organisms (debugging)
        file.write(f"Centroid: {clusters}\n\n")
        print(f"Centroid: {clusters}\n\n") # Print the centroid (debugging)
# Close the text file
file.close()


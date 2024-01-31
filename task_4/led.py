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


'''
# * Function Name: load_image
# * Input: NONE
# * Output: image_name, image_path
# * Logic: Loads the image from the local directory
# * Example Call: load_image()
'''
def load_image():
    parser = argparse.ArgumentParser(prog='led_detection', description="process local images") # Create Argument Parser object
    parser.add_argument('--image', type=str) # Add the image argument to the parser
    args = parser.parse_args() 
    image_name = args.image    # Access the value of the image argument
    # script_dir = os.path.dirname(__file__) # Get the path of the directory where this script is located
    # image_path = os.path.join(script_dir, image_name) # Create the full path of the image file
    image_path = image_name
    return image_name, image_path
image_name, image_path = load_image() # Load the image


image = cv.imread(image_path, 1) # Read the image
# print("Image Path:", image_path) # Print the image path(debugging)


''' Dictionary to store the number of LEDs in each organism '''
alien_names = {
    2: 'alien_a',
    3: 'alien_b',
    4: 'alien_c',
    5: 'alien_d',
}

'''
# * Function Name: image_processing
# * Input: image
# * Output: thresh
# * Logic: Preprocesses the image
# * Example Call: image_processing(image)
'''
def image_processing(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY) # convert the image to grayscale
    blurred = cv.GaussianBlur(gray, (7, 7), 0) # apply a Gaussian blur with a 7x7 kernel to the grayscale image
    thresh = cv.threshold(blurred, 200, 255, cv.THRESH_BINARY)[1] # threshold the image to reveal light regions in the blurred image

    # perform a series of erosions and dilations to remove any small blobs of noise from the thresholded image
    thresh = cv.erode(thresh, None, iterations=2)
    thresh = cv.dilate(thresh, None, iterations=2)
    return thresh
thresh = image_processing(image) # Preprocess the image
# cv.imshow("Image", thresh) # Display the image(For debugging)
# cv.waitKey(1000) # Wait for 1 second(For debugging)

'''
# * Function Name: image_masking
# * Input: thresh
# * Output: mask
# * Logic: Masks the image
# * Example Call: image_masking(thresh)
'''
def image_masking(thresh):
    labels = measure.label(thresh, background=0) # label the connected components in the image
    mask = np.zeros(thresh.shape, dtype="uint8") # initialize the mask to store only the "large" components

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
    return mask
mask = image_masking(thresh) # Mask the image
contours, _ = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE) # find the contours in the mask
contours = sorted(contours, key=lambda c: cv.minEnclosingCircle(c)[0]) # Sort the contours from left to right

''' Initialize lists to store centroid coordinates and area '''
centroid = []
area = []
# debug = []

'''
# * Function Name: centroid_calculation
# * Input: contours
# * Output: led_clusters, clusters
# * Logic: Calculates the centroid coordinates and area of each LED
# * Example Call: centroid_calculation(contours)
'''
def centroid_calculation(contours):
    ''' Loop over the contours '''
    for i, contour in enumerate(contours):
        #Calculate the area of the contour
        areas = cv.contourArea(contour)
        M = cv.moments(contour)
        c_X, c_Y = round(M["m10"] / M["m00"],10), round(M["m01"] / M["m00"],10)
        cX, cY = int(c_X), int(c_Y)

        # put the centroid coordinates in the list
        # debug.append((cX, cY))
        centroid.append((c_X, c_Y))
        area.append(areas)
        
        # Draw the bright spot on the image (debugging)
        cv.drawContours(image, [contour], -1, (0, 0, 255), 3)
        cv.putText(image, f'#{i + 1}', (cX, cY - 18), cv.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 255), 2)

    ''' Assign values to the variables '''
    length_factor = 12 if(sum(area)/len(area) > 900) else 9 # Calculate the length factor based on the maximum area ----> distance from ground
    radius = int(np.sqrt(max(area) / np.pi)) # Calculate the radius of the bounding circle
    led_coordinates = centroid # Store the centroid coordinates in a new list
    a = len(centroid) # a is the number of LEDs
    threshold_distance = radius*length_factor # Set a threshold distance to consider LEDs in the same cluster
    led_clusters = [] # Initialize a list to store the clusters

    # Draw the bounding circle and centroid on the image (debugging)
    # for i in range(len(centroid)):
        # cv.circle(image, debug[i], threshold_distance, (0, 255, 0), 3)

    ''' Iterate through LED coordinates to identify clusters '''
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

    ''' Calculate the centroid coordinates and area of each cluster'''
    k = len(led_clusters) # k is the number of clusters
    clusters = [] # Initialize a list to store the cluster centers
    for i in range(k):
        g = len(led_clusters[i])  # g is the number of leds in the cluster=
        x, y = 0, 0 # Initialize x and y coordinates
        for j in range(g):
            x += led_clusters[i][j][0]/g
            y += led_clusters[i][j][1]/g
    

        cluster_center = (round(x,4), round(y,4)) # Calculate the centroid of the cluster
        clusters.append(cluster_center) # Store the centroid coordinates in a new list


        cv.circle(image, (int(x),int(y)), threshold_distance, (0, 0, 255), 3) # Draw the bounding circle on the image (debugging)
        cv.putText(image, f'x', (int(x), int(y)), cv.FONT_HERSHEY_SIMPLEX,0.5, (0, 255, 0), 2)  # Draw the centroid on the image (debugging)

    return led_clusters, clusters
led_clusters, clusters = centroid_calculation(contours) # Calculate the centroid coordinates and area of each LED

'''
# * Function Name: output_file
# * Input: NONE
# * Output: NONE
# * Logic: Saves the output file
# * Example Call: output_file()
'''
def output_file():
    # cv.imshow("Image", image) # Display the image(For debugging)
    # cv.waitKey(1000) # Wait for 1 second(For debugging)

    filename = image_path.replace(".png", "") # Get the filename without the extension
    # filename = filename.replace(".jpg", "") # Get the filename without the extension (For our convenience)
    # filename = filename.replace(".jpeg", "") # Get the filename without the extension (For our convenience)

    cv.imwrite(f"{filename}_output.png", image) # Save the image

    ''' Open a text file for writing '''
    with open(f"{filename}.txt", "w") as file:

        # Write the number of LEDs in each cluster to the file
        for i, cluster in enumerate(clusters):
            g= len(led_clusters[i]) # g is the number of LEDs in the cluster
            file.write(f"Organism Type: {alien_names[g]}\n") # Write the type of organisms to the file
            file.write(f"Centroid: {cluster}\n\n") # Write the centroid coordinates to the file

            # print(f"Organism Type: {alien_names[g]}\n") # Print the type of organisms (debugging)
            # print(f"Centroid: {cluster}\n\n") # Print the centroid (debugging)
output_file() # Save the output file

import cv2
import numpy as np

def process_figure(image, output_path):
    """Process and save figure image to the specified output path."""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply a binary threshold to the image
    # You might need to adjust the threshold value depending on your image
    ret, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    # Find contours in the thresholded image
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a copy of the original image to draw on
    image_with_boxes = image.copy()

    # Calculate the total image area
    image_area = image.shape[0] * image.shape[1]

    # Define minimum area as a percentage of the image area
    min_area_percentage = 0.1  # Adjust this percentage as needed
    min_area = image_area * min_area_percentage
    
    # Crop and display each detected figure
    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area > min_area: # Use the same minimum area filter as before
        # Crop the figure from the original image
            cropped_figure = image[y:y+h, x:x+w]

            # Save the cropped figure
            figure_path = f"{output_path}_figure_{i+1}.png"
            cv2.imwrite(figure_path, cropped_figure)
            print(f"Saved figure: {figure_path}")
            
image_path = 'samples/layout.png'
image = cv2.imread(image_path)
process_figure(image, 'D:/deep-learning/output/figures/layout')
    
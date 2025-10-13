import cv2
import numpy as np

def process_figure(image, output_path):
    """Process and save figure image to the specified output path."""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply a binary threshold to the image
    ret, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    # Find contours in the thresholded image
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate the total image area
    image_area = image.shape[0] * image.shape[1]

    # Define minimum area as a percentage of the image area
    min_area_percentage = 0.1  
    min_area = image_area * min_area_percentage
    
    figures = []
    # Crop and display each detected figure
    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area > min_area:
            # Crop the figure from the original image
            cropped_figure = image[y:y+h, x:x+w]

            # Save the cropped figure
            figure_path = f"{output_path}_figure_{i+1}.png"
            figures.append(figure_path)
            cv2.imwrite(figure_path, cropped_figure)
            print(f"Saved figure: {figure_path}")
    
    return figures
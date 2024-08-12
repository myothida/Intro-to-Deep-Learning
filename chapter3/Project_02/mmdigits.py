import os
import cv2
import numpy as np
def load_dataset(dataset_dir):
    images = []
    labels = []
    desired_width = 64
    desired_height = 64
    top_crop = 5
    bottom_crop = 20
    left_crop = 50
    right_crop = 50
    

    # Loop through each subfolder in the dataset directory
    for class_label in os.listdir(dataset_dir):
        class_dir = os.path.join(dataset_dir, class_label)
        
        # Check if it's a directory
        if os.path.isdir(class_dir):
            # Get the class label from the subfolder name
            label = int(class_label)

            # Loop through each image file in the subfolder
            for filename in os.listdir(class_dir):
                image_path = os.path.join(class_dir, filename)                
                
                # Read the image and convert it to grayscale
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

                # Resize the image to a fixed size if needed
                # Crop the image if needed
                image = image[top_crop:image.shape[0]-bottom_crop, left_crop:image.shape[1]-right_crop]
                image = cv2.resize(image, (desired_width, desired_height))

                # Add the image and label to the lists
                images.append(image)
                labels.append(label)

    # Convert lists to NumPy arrays
    images = np.array(images)
    labels = np.array(labels)

    return images, labels



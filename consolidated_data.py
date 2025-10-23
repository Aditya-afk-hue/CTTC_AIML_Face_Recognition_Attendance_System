# consolidated_data.py

import cv2
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Define directories for data and images
data_dir = os.path.join(os.getcwd(), 'data')
img_dir = os.path.join(os.getcwd(), 'images')

# Create data directory if it doesn't exist
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

image_data = []
labels = []

# Loop through all files in the images directory
for i in os.listdir(img_dir):
    # Read the image
    image = cv2.imread(os.path.join(img_dir, i))
    # Resize to a consistent 100x100 pixels
    image = cv2.resize(image, (100, 100))
    # Convert to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Add the processed image to our list
    image_data.append(image)
    # Extract the label (name) from the filename (e.g., "shiva_1.jpg" -> "shiva")
    labels.append(str(i).split("_")[0])
    
# Convert lists to NumPy arrays
image_data = np.array(image_data)    
labels = np.array(labels) 

# (Optional) Display a sample image to verify
# plt.imshow(image_data[0], cmap="gray")
# plt.show()

# Save the processed data and labels using pickle
print("Saving processed data...")
with open(os.path.join(data_dir, "images.p"), 'wb') as f:
    pickle.dump(image_data, f)
    
with open(os.path.join(data_dir, "labels.p"), 'wb') as f:
    pickle.dump(labels, f)

print("Data consolidation complete!")
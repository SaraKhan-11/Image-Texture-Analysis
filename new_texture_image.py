import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from skimage.feature import local_binary_pattern
from scipy.spatial import distance

# Load an image from a folder randomly
def load_random_image(folder_path):
    images = [f for f in os.listdir(folder_path) if f.endswith('.jpeg')]
    random_image = random.choice(images)
    image_path = os.path.join(folder_path, random_image)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return image

# Folder where your images are stored
folder_path = "texture image"

# Load a reference image and a random test image
reference_image = cv2.imread("texture image/object_class_3/object 1.jpeg", cv2.IMREAD_GRAYSCALE)
random_image = load_random_image("texture image/object_class_3")

# Parameters for LBP
object_radius = 8
total_points = 16

# Function to calculate LBP and return a normalized histogram
def calculate_texture(image, total_points, object_radius):
    image_texture = local_binary_pattern(image, total_points, object_radius, method='uniform')
    total_bins = int(image_texture.max() + 1)
    texture_hist, _ = np.histogram(image_texture, density=True, bins=total_bins, range=(0, total_bins))
    return texture_hist

# Calculate texture histograms for both images
reference_texture = calculate_texture(reference_image, total_points, object_radius)
random_texture = calculate_texture(random_image, total_points, object_radius)

# Calculate Euclidean distance between the two texture histograms
distance_value = distance.euclidean(reference_texture, random_texture)
print(f'Euclidean distance between textures: {distance_value}')

# Set a threshold to determine if the images are from the same class
threshold = 0.2  # Adjust this value based on experimentation

# Check if the images belong to the same class
if distance_value < threshold:
    print("The two images are from the same class.")
else:
    print("The two images are from different classes.")

# Visualization of images and textures
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))

# Display reference image and its texture
axes[0, 0].imshow(reference_image, cmap='gray')
axes[0, 0].set_title("Reference Image")
axes[0, 1].imshow(reference_image, cmap='gray')
axes[0, 1].set_title("Grayscale Reference")
axes[0, 2].imshow(local_binary_pattern(reference_image, total_points, object_radius, method='uniform'), cmap='gray')
axes[0, 2].set_title("Reference Texture")

# Display random image and its texture
axes[1, 0].imshow(random_image, cmap='gray')
axes[1, 0].set_title("Random Image")
axes[1, 1].imshow(random_image, cmap='gray')
axes[1, 1].set_title("Grayscale Random")
axes[1, 2].imshow(local_binary_pattern(random_image, total_points, object_radius, method='uniform'), cmap='gray')
axes[1, 2].set_title("Random Texture")

plt.tight_layout()
plt.show()

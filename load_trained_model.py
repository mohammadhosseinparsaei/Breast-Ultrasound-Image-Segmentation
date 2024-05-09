import os
import numpy as np
import matplotlib.pyplot as plt
from keras.models import model_from_json
from keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import glob

# Define the path to your model directory
model_directory = "D:/My codes/My DL train/Autoencoder/Breast Ultrasound Images Dataset"

# Load the trained U-Net model
def load_model(model_directory):
    model_json_path = os.path.join(model_directory, 'model.json')
    model_weights_path = os.path.join(model_directory, 'model_weights.h5')
    with open(model_json_path, 'r') as json_file:
        loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(model_weights_path)
    return loaded_model

model = load_model(model_directory)

# Load and preprocess image for model prediction
def load_preprocess_image(path, target_size=(128, 128)):
    img = load_img(path, color_mode='grayscale')
    img_resized = img.resize(target_size)
    img_array = img_to_array(img_resized)
    img_array /= 255.
    return img_array, img

# Predict segmentation
def predict_segmentation(image_path, model, target_size=(128, 128)):
    img_array, original_img = load_preprocess_image(image_path, target_size)
    prediction = model.predict(np.expand_dims(img_array, axis=0))
    predicted_mask = prediction[0, :, :, 0]
    predicted_mask_resized = Image.fromarray((predicted_mask * 255).astype('uint8'))
    predicted_mask_resized = predicted_mask_resized.resize(original_img.size, Image.NEAREST)
    return original_img, predicted_mask_resized

# Automatically find mask for a given image
def find_mask_path(image_path):
    base_path = os.path.splitext(image_path)[0]
    mask_paths = glob.glob(f'{base_path}_mask*.png')
    if mask_paths:
        return mask_paths[0]
    return None

# Function to plot the comparison for a list of images
def all_images(image_paths, model, base_path):
    plt.figure(figsize=(15, 15))
    
    for i, (image_name, label) in enumerate(image_paths):
        full_image_path = os.path.join(base_path, image_name)
        mask_path = find_mask_path(full_image_path)
        if not mask_path:
            print(f"No mask found for {full_image_path}")
            continue
        original_img, predicted_mask = predict_segmentation(full_image_path, model)
        real_mask = load_img(mask_path, color_mode='grayscale')

        plt.subplot(len(image_paths), 4, 4*i + 1)
        plt.imshow(original_img, cmap='gray')
        plt.title(f'Original ({label})', fontsize=16, fontweight='bold')
        plt.axis('off')

        plt.subplot(len(image_paths), 4, 4*i + 2)
        plt.imshow(real_mask, cmap='gray')
        plt.title('True Mask', fontsize=16, fontweight='bold')
        plt.axis('off')

        plt.subplot(len(image_paths), 4, 4*i + 3)
        plt.imshow(predicted_mask, cmap='gray')
        plt.title('Predicted Mask', fontsize=16, fontweight='bold')
        plt.axis('off')

        plt.subplot(len(image_paths), 4, 4*i + 4)
        plt.imshow(original_img, cmap='gray')
        plt.imshow(predicted_mask, cmap='jet', alpha=0.5)
        plt.title('Overlay', fontsize=16, fontweight='bold')
        plt.axis('off')

    plt.tight_layout()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    all_images_path = os.path.join(script_dir, 'selected_images.png')
    plt.savefig(all_images_path, dpi=300, facecolor='white')
    plt.show()

# Base path for images
base_images_path = "D:/Data for machine learning/Breast Ultrasound Images Dataset/Dataset_BUSI_with_GT"

# Example usage
image_paths = [
    ("benign/benign (10).png", "Benign"),
    ("benign/benign (32).png", "Benign"),
    ("benign/benign (36).png", "Benign"),
    ("malignant/malignant (21).png", "Malignant"),
    ("malignant/malignant (28).png", "Malignant"),
    ("malignant/malignant (102).png", "Malignant")
]

all_images(image_paths, model, base_images_path)
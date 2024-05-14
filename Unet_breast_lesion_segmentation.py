# Import library
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array, load_img
from keras import layers, models, callbacks
from keras.utils.vis_utils import plot_model
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

# Set dataset paths
root_dir = "D:/Data for machine learning/Breast Ultrasound Images Dataset/Dataset_BUSI_with_GT"
folders = ["benign", "malignant"]
script_dir = os.path.dirname(os.path.abspath(__file__))

# Initialize lists for storing paths and labels
image_paths, labels, mask_paths = [], [], []

# Collect image and mask paths, along with labels
for folder in folders:
    for file in os.listdir(os.path.join(root_dir, folder)):
        file_path = os.path.join(root_dir, folder, file)
        (image_paths if "mask" not in file else mask_paths).append(file_path)
        labels.append(folder) if "mask" not in file else None

# Create a DataFrame to organize paths and labels
df = pd.DataFrame({"image_path": image_paths, "label": labels})
df["mask_paths"] = df["image_path"].apply(lambda x: [path for path in mask_paths if os.path.splitext(x)[0] in path])

# Function to load and preprocess images
def load_image(path):
    img = load_img(path, target_size=(128, 128), color_mode='grayscale')
    image_array = img_to_array(img)
    image_array /= 255.
    return image_array

# Load all images and masks
images, masks = [], []
for _, row in tqdm(df.iterrows(), total=len(df)):
    image = load_image(row["image_path"])
    mask = np.clip(np.sum([load_image(mask_path) for mask_path in row["mask_paths"]], axis=0), 0, 1)
    images.append(image)
    masks.append(mask)
images, masks = np.array(images), np.array(masks)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(images, masks, test_size=0.05, random_state=2)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.05, random_state=2)

x_train, x_val, x_test = x_train.astype('float32'), x_val.astype('float32'), x_test.astype('float32')
y_train, y_val, y_test = y_train.astype('float32'), y_val.astype('float32'), y_test.astype('float32')

# Define the U-Net model
inputs = layers.Input(shape=(128, 128, 1))
conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

conv2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
conv2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

conv3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
conv3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

conv4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
conv4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
pool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)

conv5 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
conv5 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

up6 = layers.concatenate([layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
conv6 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
conv6 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

up7 = layers.concatenate([layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
conv7 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
conv7 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

up8 = layers.concatenate([layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
conv8 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
conv8 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

up9 = layers.concatenate([layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
conv9 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
conv9 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

conv10 = layers.Conv2D(1, (1, 1), activation='sigmoid')(conv9)

unet = models.Model(inputs=[inputs], outputs=[conv10])
unet.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Plot the model architecture
plot_path = os.path.join(script_dir, 'model_architecture_plot.png')
plot_model(unet, to_file=plot_path, show_shapes=True, show_layer_names=True, dpi=300)

# Define early stopping criteria
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)

# Train the U-Net model with early stopping
history = unet.fit(x_train, y_train,
                   epochs=100,
                   batch_size=16,
                   shuffle=True,
                   validation_data=(x_val, y_val),
                   callbacks=[early_stopping])

# Calculate final accuracy
_, final_accuracy = unet.evaluate(x_test, y_test)
print("Final Accuracy:", final_accuracy)

# Plot the training and validation accuracy and loss
acc_loss_path = os.path.join(script_dir, 'accuracy_loss_plot.png')
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.tight_layout()
plt.savefig(acc_loss_path, dpi=300, facecolor='white')
plt.show()

# Plot the actual image, its mask, and predicted mask for each selected index
evaluation_path = os.path.join(script_dir, 'evaluation.png')

random_indices = np.random.choice(range(len(x_test)), size=5, replace=False)

plt.figure(figsize=(8, 8))
for i, idx in enumerate(random_indices):
    plt.subplot(5, 3, i*3 + 1)
    plt.imshow(x_test[idx].reshape(128, 128), cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(5, 3, i*3 + 2)
    plt.imshow(y_test[idx].reshape(128, 128), cmap='gray')
    plt.title('Ground Truth Mask')
    plt.axis('off')

    plt.subplot(5, 3, i*3 + 3)
    prediction = unet.predict(np.expand_dims(x_test[idx], axis=0))
    plt.imshow(prediction.reshape(128, 128), cmap='gray')
    plt.title('Predicted Mask')
    plt.axis('off')

plt.tight_layout()
plt.savefig(evaluation_path, dpi=300, facecolor='white')
plt.show()

# Saving the Trained Model
model_json = unet.to_json()
model_json_path = os.path.join(script_dir, "model.json")
with open(model_json_path, "w") as json_file:
    json_file.write(model_json)

model_weights_path = os.path.join(script_dir, "model_weights.h5")
unet.save_weights(model_weights_path)

print("Model architecture and weights saved successfully to the script's directory!")

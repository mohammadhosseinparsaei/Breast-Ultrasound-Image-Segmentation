# Import library
import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.preprocessing.image import img_to_array, load_img
from keras.callbacks import Callback, ModelCheckpoint
from keras.utils.vis_utils import plot_model
from keras.preprocessing.image import img_to_array, load_img
from keras import layers, models
from keras.optimizers import Adam

# Define paths and image size
ROOT_DIR = "D:/Data for machine learning/Breast Ultrasound Images Dataset/Dataset_BUSI_with_GT"
CLASSES = ["benign", "malignant"]
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_SIZE = (128, 128)

# Function to load and preprocess images
def load_image(path):
    img = load_img(path, target_size=IMAGE_SIZE, color_mode='grayscale')
    img_array = img_to_array(img) / 255.
    return img_array

# Function to rotate images and masks (data augmentation)
def rotate_images_and_masks(images, masks):
    rotated_images = []
    rotated_masks = []
    angles = np.arange(22.5, 338, 22.5).tolist()
    
    for img, mask in zip(images, masks):
        for angle in angles:
            rows, cols = img.shape[:2]
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
            rotated_img = cv2.warpAffine(img, M, (cols, rows))
            rotated_mask = cv2.warpAffine(mask, M, (cols, rows))
            rotated_images.append(rotated_img)
            rotated_masks.append(rotated_mask)
    
    return np.array(rotated_images).reshape(-1, IMAGE_SIZE[0], IMAGE_SIZE[1], 1), np.array(rotated_masks).reshape(-1, IMAGE_SIZE[0], IMAGE_SIZE[1], 1)

# Function to plot images and masks
def plot_augmented_images_masks(images, masks, title, start_index, num_images):
    for i in range(num_images):
        idx = np.random.randint(len(images))
        image = images[idx]
        mask = masks[idx]

        plt.subplot(2, num_images, start_index + i)
        plt.title(f"{title} Image", fontsize=16)
        plt.imshow(image, cmap='gray')
        plt.axis('off')

        plt.subplot(2, num_images, start_index + num_images + i)
        plt.title(f"{title} Mask", fontsize=16)
        plt.imshow(mask, cmap='gray')
        plt.axis('off')

# U-Net model definition
def unet_model(input_shape):
    inputs = layers.Input(input_shape)
    
    def conv_block(x, filters, kernel_size=3, padding='same', activation='relu'):
        x = layers.Conv2D(filters, kernel_size, padding=padding, activation=activation)(x)
        x = layers.Conv2D(filters, kernel_size, padding=padding, activation=activation)(x)
        return x
    
    def encoder_block(x, filters):
        f = conv_block(x, filters)
        p = layers.MaxPooling2D(pool_size=(2, 2))(f)
        return f, p
    
    def decoder_block(x, conv_output, filters):
        x = layers.Conv2DTranspose(filters, kernel_size=(2, 2), strides=(2, 2), padding='same')(x)
        x = layers.concatenate([x, conv_output])
        x = conv_block(x, filters)
        return x

    f1, p1 = encoder_block(inputs, 32)
    f2, p2 = encoder_block(p1, 64)
    f3, p3 = encoder_block(p2, 128)
    f4, p4 = encoder_block(p3, 256)
    
    bottleneck = conv_block(p4, 512)
    
    d1 = decoder_block(bottleneck, f4, 256)
    d2 = decoder_block(d1, f3, 128)
    d3 = decoder_block(d2, f2, 64)
    d4 = decoder_block(d3, f1, 32)
    
    outputs = layers.Conv2D(1, kernel_size=1, activation='sigmoid')(d4)
    
    model = models.Model(inputs, outputs)
    return model

# Define mean IoU metric
def dice_metric(y_true, y_pred):
    intersection = tf.reduce_sum(tf.cast(y_true, tf.float32) * tf.cast(y_pred, tf.float32))
    return (2. * intersection + 1.) / (tf.reduce_sum(tf.cast(y_true, tf.float32)) + tf.reduce_sum(tf.cast(y_pred, tf.float32)) + 1.)

# Define mean Dice metric
def iou_metric(y_true, y_pred):
    intersection = tf.reduce_sum(tf.cast(y_true, tf.float32) * tf.cast(y_pred, tf.float32))
    union = tf.reduce_sum(tf.cast(y_true, tf.float32)) + tf.reduce_sum(tf.cast(y_pred, tf.float32)) - intersection
    return (intersection + 1.) / (union + 1.)

# Define early stopping callback
class MyEarlyStopping(Callback):
    def __init__(self, monitor='val_mean_iou', patience=0, verbose=0):
        super(MyEarlyStopping, self).__init__()
        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.best_iou = float('-inf')
        self.wait = 0
        self.best_weights = None
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        current_iou = logs.get(self.monitor)
        if current_iou is not None:
            if current_iou > self.best_iou:
                self.best_iou = current_iou
                self.wait = 0
                self.best_weights = self.model.get_weights()
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    self.stopped_epoch = epoch
                    self.model.stop_training = True
                    if self.verbose > 0:
                        print(f"Epoch {epoch + 1}: early stopping")

    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)
        if self.stopped_epoch > 0 and self.verbose > 0:
            best_epoch = self.stopped_epoch - self.wait
            print(f"Restored model weights from epoch {best_epoch + 1}")

# Function to find the best threshold
def find_best_threshold(y_true, y_pred):
    best_threshold = 0
    best_score = 0
    for threshold in np.arange(0.01, 1.0, 0.01):
        thresholded_pred = (y_pred > threshold).astype('float32')
        score = iou_metric(y_true, thresholded_pred).numpy()
        if score > best_score:
            best_score = score
            best_threshold = threshold
    return best_threshold

# Function to apply threshold to predicted mask
def apply_threshold(pred_mask, threshold):
    processed_mask = (pred_mask > threshold).astype('float')
    return processed_mask

# Load dataset paths and labels
image_paths, labels, mask_paths = [], [], []
for class_name in CLASSES:
    for file_name in os.listdir(os.path.join(ROOT_DIR, class_name)):
        file_path = os.path.join(ROOT_DIR, class_name, file_name)
        (image_paths if "mask" not in file_name else mask_paths).append(file_path)
        labels.append(class_name) if "mask" not in file_name else None

# Create DataFrame to organize paths and labels
df = pd.DataFrame({"image_path": image_paths, "label": labels})
df["mask_paths"] = df["image_path"].apply(lambda x: [path for path in mask_paths if os.path.splitext(x)[0] in path])

# Load images and masks
images, masks = [], []
for _, row in tqdm(df.iterrows(), total=len(df)):
    image = load_image(row["image_path"])
    mask = np.clip(np.sum([load_image(mask_path) for mask_path in row["mask_paths"]], axis=0), 0, 1)
    images.append(image)
    masks.append(mask)
images, masks = np.array(images), np.array(masks)

# Split the data into training and testing sets
x_train, temp_images, y_train, temp_masks = train_test_split(images, masks, train_size=0.7, random_state=4)
x_val, x_test, y_val, y_test = train_test_split(temp_images, temp_masks, test_size=0.5, random_state=4)

# Apply data augmentation only to the training data
rotated_images, rotated_masks = rotate_images_and_masks(x_train, y_train)

# Concatenate augmented images and masks to the original dataset
x_train = np.concatenate((x_train, rotated_images), axis=0)
y_train = np.concatenate((y_train, rotated_masks), axis=0)

# Shuffle the indices to mix the original and augmented data
shuffled_indices = np.random.permutation(len(x_train))

# Apply the shuffled indices to both x_train and y_train
x_train, y_train = x_train[shuffled_indices], y_train[shuffled_indices]

# Convert data types
x_train, x_val, x_test = x_train.astype('float32'), x_val.astype('float32'), x_test.astype('float32')
y_train, y_val, y_test = y_train.astype('float32'), y_val.astype('float32'), y_test.astype('float32')

# Plot the rotate images and their masks
plt.figure(figsize=(16, 8))
plot_augmented_images_masks(rotated_images, rotated_masks, "Rotated", 1, num_images=4)
plt.tight_layout()
augmentation_plot_path = os.path.join(SCRIPT_DIR, 'augmented_images_and_masks.png')
plt.savefig(augmentation_plot_path, dpi=300, facecolor='white')

# Compile the model
model = unet_model((IMAGE_SIZE[0], IMAGE_SIZE[1], 1))
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=[iou_metric, dice_metric])

# Plot the model architecture
plot_path = os.path.join(SCRIPT_DIR, 'model_architecture_plot.png')
plot_model(model, to_file=plot_path, show_shapes=True, show_layer_names=True, dpi=300)

# Define checkpoint and early stopping
checkpoint = ModelCheckpoint('model_best.h5', monitor='val_iou_metric', verbose=1, save_best_only=True, mode='max')
early_stopping = MyEarlyStopping(monitor='val_iou_metric', patience=10, verbose=1)

# Train the model
history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[checkpoint, early_stopping]
)

# Predict masks for the testing set
predicted_masks = model.predict(x_test)

# Find the best threshold
best_threshold = find_best_threshold(y_test, predicted_masks)

# Apply the best threshold to the predicted masks
thresholded_masks = (predicted_masks > best_threshold).astype('float32')

# Calculate evaluation metrics
total_iou, total_dice = 0, 0
max_iou, max_dice = 0, 0
for i in range(len(x_test)):
    iou = iou_metric(y_test[i], thresholded_masks[i]).numpy()
    dice = dice_metric(y_test[i], thresholded_masks[i]).numpy()
    total_iou += iou
    total_dice += dice
    if iou > max_iou:
        max_iou = iou
    if dice > max_dice:
        max_dice = dice

average_iou = total_iou / len(x_test)
average_dice = total_dice / len(x_test)

print(f"Average IoU between test data: {average_iou:.4f}")
print(f"Average Dice Coefficient between test data: {average_dice:.4f}")
print(f"Maximum IoU between test data: {max_iou:.4f}")
print(f"Maximum Dice Coefficient between test data: {max_dice:.4f}")

# Plot IoU and Dice coefficients after training
iou_dice_path = os.path.join(SCRIPT_DIR, 'iou_dice_plot.png')
plt.figure(figsize=(10, 10))
epochs = range(1, len(history.history['iou_metric']) + 1)
plt.plot(epochs, history.history['iou_metric'], label='Training IoU')
plt.plot(epochs, history.history['val_iou_metric'], label='Validation IoU')
plt.plot(epochs, history.history['dice_metric'], label='Training Dice')
plt.plot(epochs, history.history['val_dice_metric'], label='Validation Dice')
plt.text(0.5, 1.05, f'Average IoU between test data: {average_iou:.4f}, Average Dice between test data: {average_dice:.4f}\nMaximum IoU between test data: {max_iou:.4f}, Maximum Dice between test data: {max_dice:.4f}', 
         horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
plt.xlabel('Epoch')
plt.ylabel('IoU / Dice Coefficient')
plt.legend()
plt.tight_layout()
plt.savefig(iou_dice_path, dpi=300, facecolor='white')

# Plot the actual image, its mask, and predicted mask for each selected index
evaluation_path = os.path.join(SCRIPT_DIR, 'evaluation.png')
plt.figure(figsize=(25, 25))
for i in range(5):
    idx = np.random.randint(len(x_test))
    image = x_test[idx]
    mask = y_test[idx]
    pred_mask = model.predict(image[np.newaxis, ...])[0]

    plt.subplot(5, 4, i*4 + 1)
    plt.title("Original Image", fontsize=16)
    plt.imshow(image, cmap='gray')
    plt.axis('off')

    plt.subplot(5, 4, i*4 + 2)
    plt.title("Ground Truth Mask", fontsize=16)
    plt.imshow(mask, cmap='gray')
    plt.axis('off')

    plt.subplot(5, 4, i*4 + 3)
    processed_mask = (pred_mask > best_threshold).astype('float32')
    plt.title(f"Predicted Mask (IoU: {iou_metric(mask, processed_mask).numpy():.4f})", fontsize=16)
    plt.imshow(processed_mask, cmap='gray')
    plt.axis('off')

    plt.subplot(5, 4, i*4 + 4)
    plt.title("Predicted Breast Mass", fontsize=16)
    plt.imshow(image, cmap='gray')
    processed_mask = (pred_mask > best_threshold).astype('float32')
    plt.imshow(processed_mask, cmap='coolwarm', alpha=0.4)
    plt.axis('off')

plt.tight_layout()
plt.savefig(evaluation_path, dpi=300, facecolor='white')

# Save the trained model
model_json_path = os.path.join(SCRIPT_DIR, "model.json")
model_weights_path = os.path.join(SCRIPT_DIR, "model_weights.h5")

with open(model_json_path, "w") as json_file:
    json_file.write(model.to_json(indent=4))

model.save_weights(model_weights_path)

print("Model architecture and weights saved successfully to the script's directory!")
"""
Building: #3C1098 voilet
Land (unpaved area): #8429F6 purple
Road: #6EC1E4 blue
Vegetation: #FEDD3A yellow
Water: #E2A929 sandal
Unlabeled: #9B9B9B grey

Use patchify....
Tile 1: 797 x 644 --> 768 x 512 --> 6
Tile 2: 509 x 544 --> 512 x 256 --> 2
Tile 3: 682 x 658 --> 512 x 512  --> 4
Tile 4: 1099 x 846 --> 1024 x 768 --> 12
Tile 5: 1126 x 1058 --> 1024 x 1024 --> 16
Tile 6: 859 x 838 --> 768 x 768 --> 9
Tile 7: 1817 x 2061 --> 1792 x 2048 --> 56
Tile 8: 2149 x 1479 --> 1280 x 2048 --> 40
Total 9 images in each folder * (145 patches) = 1305
Total 1305 patches of size 256x256
"""

import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from patchify import patchify
from PIL import Image
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, concatenate, Input, Dropout, Conv2DTranspose
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
root_directory = r"C:\Users\tandu\Downloads\archive (2)"
patch_size = 256

# Read images from respective 'images' subdirectory
image_dataset = []
for path, subdirs, files in os.walk(root_directory):
    dirname = path.split(os.path.sep)[-1]
    if dirname == 'images':   # Find all 'images' directories
        images = os.listdir(path)
        for image_name in images:
            if image_name.endswith(".jpg"):
                image = cv2.imread(os.path.join(path, image_name), 1)
                SIZE_X = (image.shape[1] // patch_size) * patch_size
                SIZE_Y = (image.shape[0] // patch_size) * patch_size
                image = Image.fromarray(image)
                image = image.crop((0, 0, SIZE_X, SIZE_Y))
                image = np.array(image)
                print("Now patchifying image:", os.path.join(path, image_name))
                patches_img = patchify(image, (patch_size, patch_size, 3), step=patch_size)
                for i in range(patches_img.shape[0]):
                    for j in range(patches_img.shape[1]):
                        single_patch_img = patches_img[i, j, :, :]
                        # Use MinMaxScaler to normalize
                        single_patch_img = scaler.fit_transform(single_patch_img.reshape(-1, single_patch_img.shape[-1])).reshape(single_patch_img.shape)
                        single_patch_img = single_patch_img[0]  # Drop extra dimension added by patchify
                        image_dataset.append(single_patch_img)

# Now for masks
mask_dataset = []
for path, subdirs, files in os.walk(root_directory):
    dirname = path.split(os.path.sep)[-1]
    if dirname == 'masks':   # Find all 'masks' directories
        masks = os.listdir(path)
        for mask_name in masks:
            if mask_name.endswith(".png"):
                mask = cv2.imread(os.path.join(path, mask_name), 1)
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
                SIZE_X = (mask.shape[1] // patch_size) * patch_size
                SIZE_Y = (mask.shape[0] // patch_size) * patch_size
                mask = Image.fromarray(mask)
                mask = mask.crop((0, 0, SIZE_X, SIZE_Y))
                mask = np.array(mask)
                print("Now patchifying mask:", os.path.join(path, mask_name))
                patches_mask = patchify(mask, (patch_size, patch_size, 3), step=patch_size)
                for i in range(patches_mask.shape[0]):
                    for j in range(patches_mask.shape[1]):
                        single_patch_mask = patches_mask[i, j, :, :]
                        single_patch_mask = single_patch_mask[0]  # Drop extra dimension
                        mask_dataset.append(single_patch_mask)

image_dataset = np.array(image_dataset)
mask_dataset = np.array(mask_dataset)

# Convert HEX colors to RGB arrays for label mapping
a = int('3C', 16)
print(a)
Building = '#3C1098'.lstrip('#')
Building = np.array(tuple(int(Building[i:i+2], 16) for i in (0, 2, 4)))
Land = '#8429F6'.lstrip('#')
Land = np.array(tuple(int(Land[i:i+2], 16) for i in (0, 2, 4)))
Road = '#6EC1E4'.lstrip('#')
Road = np.array(tuple(int(Road[i:i+2], 16) for i in (0, 2, 4)))
Vegetation = 'FEDD3A'.lstrip('#')
Vegetation = np.array(tuple(int(Vegetation[i:i+2], 16) for i in (0, 2, 4)))
Water = 'E2A929'.lstrip('#')
Water = np.array(tuple(int(Water[i:i+2], 16) for i in (0, 2, 4)))
Unlabeled = '#9B9B9B'.lstrip('#')
Unlabeled = np.array(tuple(int(Unlabeled[i:i+2], 16) for i in (0, 2, 4)))

def rgb_to_2D_label(label):
    """
    Replace pixels with specific RGB values with integer labels.
    """
    label_seg = np.zeros(label.shape, dtype=np.uint8)
    label_seg[np.all(label == Building, axis=-1)] = 0
    label_seg[np.all(label == Land, axis=-1)] = 1
    label_seg[np.all(label == Road, axis=-1)] = 2
    label_seg[np.all(label == Vegetation, axis=-1)] = 3
    label_seg[np.all(label == Water, axis=-1)] = 4
    label_seg[np.all(label == Unlabeled, axis=-1)] = 5
    label_seg = label_seg[:, :, 0]  # Use one channel only
    return label_seg

labels = []
for i in range(mask_dataset.shape[0]):
    label = rgb_to_2D_label(mask_dataset[i])
    labels.append(label)
labels = np.array(labels)
labels = np.expand_dims(labels, axis=3)
print("Unique labels in label dataset are: ", np.unique(labels))

# Sanity check - display one example (commented out for non-verbose execution)
# import random
# image_number = random.randint(0, len(image_dataset)-1)
# plt.figure(figsize=(12, 12))
# plt.subplot(221)
# plt.imshow(np.reshape(image_dataset[image_number], (patch_size, patch_size, 3)))
# plt.title("Original Image")
# plt.subplot(222)
# plt.imshow(np.reshape(mask_dataset[image_number], (patch_size, patch_size, 3)))
# plt.title("Original Mask")
# plt.subplot(223)
# plt.imshow(image_dataset[image_number])
# plt.title("Processed Image")
# plt.subplot(224)
# plt.imshow(labels[image_number][:, :, 0])
# plt.title("Converted Label")
# plt.show()

# Split the dataset (no extra expansion here, labels are already (samples, H, W, 1))
X_train, X_test, y_train, y_test = train_test_split(image_dataset, labels, test_size=0.20, random_state=42)

n_classes = len(np.unique(labels))
IMG_HEIGHT = X_train.shape[1]
IMG_WIDTH  = X_train.shape[2]
IMG_CHANNELS = X_train.shape[3]

# Define custom Jaccard (IoU) metric for sparse labels
def jacard_coef(y_true, y_pred, smooth=1e-6):
    # y_true: (batch, H, W, 1) with integer labels; y_pred: (batch, H, W, n_classes) probabilities.
    # Remove the last dim from y_true
    y_true = tf.cast(tf.squeeze(y_true, axis=-1), tf.int32)
    y_pred = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
    
    iou = 0.0
    for i in range(n_classes):
        true_class = tf.cast(tf.equal(y_true, i), tf.float32)
        pred_class = tf.cast(tf.equal(y_pred, i), tf.float32)
        intersection = tf.reduce_sum(true_class * pred_class)
        union = tf.reduce_sum(true_class) + tf.reduce_sum(pred_class) - intersection
        iou += (intersection + smooth) / (union + smooth)
    return iou / n_classes

from tensorflow.keras.layers import Lambda, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
import tensorflow as tf

def spp_module(x):
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    pool2 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(x)
    pool3 = MaxPooling2D(pool_size=(8, 8), strides=(8, 8), padding='same')(x)
    
    pool1 = tf.image.resize(pool1, (x.shape[1], x.shape[2]))
    pool2 = tf.image.resize(pool2, (x.shape[1], x.shape[2]))
    pool3 = tf.image.resize(pool3, (x.shape[1], x.shape[2]))
    
    spp_out = concatenate([x, pool1, pool2, pool3], axis=-1)
    return spp_out

def multi_unet_model(n_classes=6, IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH, IMG_CHANNELS=IMG_CHANNELS):
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    s = inputs

    # Contraction path
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
    c1 = Dropout(0.2)(c1)
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.2)(c2)
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    c5 = spp_module(c5)
    
    # Expansive path 
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
    
    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
    
    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.2)(c8)
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
    
    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.2)(c9)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
    
    outputs = Conv2D(n_classes, (1, 1), activation='softmax')(c9)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    return model

metrics = ['accuracy', jacard_coef]

def get_model():
    return multi_unet_model(n_classes=6, IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH, IMG_CHANNELS=IMG_CHANNELS)

model = get_model()
model.compile(optimizer=Adam(learning_rate=1e-3),
              loss='sparse_categorical_crossentropy',
              metrics=metrics)
model.summary()

# Fit the model (without extra output)
history = model.fit(X_train, y_train,
                    batch_size=8,
                    epochs=5,
                    verbose=1,
                    validation_data=(X_test, y_test),
                    shuffle=False)

# (Optional) After training, you can use history.history to plot the performance metrics.
import matplotlib.pyplot as plt

# Assuming 'history' is the training history object returned by model.fit()
# Plotting the training & validation loss and accuracy (if available)

plt.figure(figsize=(12, 6))

# Plot loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot accuracy (only if accuracy is available)
if 'accuracy' in history.history:
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

plt.tight_layout()
plt.show()


# VISUALIZATION CODE
#######################################################################

# Function to convert prediction to colored segmentation map
def prediction_to_rgb(prediction):
    """
    Converts the predicted labels to an RGB image for visualization.
    """
    segmented_img = np.zeros((prediction.shape[0], prediction.shape[1], 3), dtype=np.uint8)
    segmented_img[prediction == 0] = Building
    segmented_img[prediction == 1] = Land
    segmented_img[prediction == 2] = Road
    segmented_img[prediction == 3] = Vegetation
    segmented_img[prediction == 4] = Water
    segmented_img[prediction == 5] = Unlabeled
    return segmented_img

# Get a few test images
n_samples = 5  # Number of samples to visualize
test_indices = np.random.choice(X_test.shape[0], n_samples, replace=False)

for i in test_indices:
    # Get the i'th test image and ground truth
    test_img = X_test[i]
    ground_truth = y_test[i]

    # Predict the segmentation map
    predicted_mask = model.predict(np.expand_dims(test_img, axis=0))
    predicted_mask = np.argmax(predicted_mask, axis=-1)[0]  # Get class index

    # Convert integer predictions and ground truth to RGB for visualization
    segmented_img = prediction_to_rgb(predicted_mask)
    ground_truth_img = prediction_to_rgb(np.squeeze(ground_truth, axis=-1))

    # Display side by side
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(test_img)
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(ground_truth_img)
    plt.title("Ground Truth Mask")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(segmented_img)
    plt.title("Predicted Mask")
    plt.axis('off')

    plt.show()


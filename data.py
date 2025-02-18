import os
import cv2
import numpy as np

dataset_path = "images"

# Get list of folders, filter out non-numeric names (e.g., .DS_Store)
class_folders = [f for f in os.listdir(dataset_path) if f.isdigit()]
class_folders = sorted(class_folders, key=lambda x: int(x))  # Sort numerically 1-84

images = []
labels = []

for class_folder in class_folders:
    class_path = os.path.join(dataset_path, class_folder)
    class_label = int(class_folder)  # Folder name is the true label (1-84)
    
    for img_name in os.listdir(class_path):
        # Skip non-image files
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        
        # Load and preprocess image
        img_path = os.path.join(class_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (32, 32))
        
        images.append(img)
        labels.append(class_label - 1)  # Convert to 0-based labels (0-83)
        

# Convert to arrays
images = np.array(images)
labels = np.array(labels)

print("Labels shape:", labels.shape)
print("Images shape:", images.shape)

from keras.utils import to_categorical

# Normalize pixel values to [0, 1]
images = images.astype('float32') / 255.0

# Reshape for CNN input (add channel dimension for grayscale)
images = np.expand_dims(images, axis=-1)  # Shape: (164124, 32, 32, 1)

# One-hot encode labels
num_classes = 84  # Since folders are 1-84 (converted to 0-83)
labels = to_categorical(labels, num_classes=num_classes)


from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    images, labels, 
    test_size=0.2,  # 80% training, 20% validation
    random_state=42
)

from keras import layers, models

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')  # 84 classes
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(
    X_train, y_train,
    epochs=20,
    validation_data=(X_val, y_val)
)

import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

test_loss, test_acc = model.evaluate(X_val, y_val)
print(f"Test Accuracy: {test_acc:.4f}")
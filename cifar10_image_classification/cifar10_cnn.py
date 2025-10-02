import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# -------------------------------
# 1. Load CIFAR-10 dataset
# -------------------------------
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Normalize pixel values [0, 1]
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# One-hot encoding
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

print(f"Training data: {X_train.shape}, Labels: {y_train_cat.shape}")
print(f"Testing data: {X_test.shape}, Labels: {y_test_cat.shape}")

# CIFAR-10 class names
class_names = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck"
]

# -------------------------------
# 2. Data Augmentation
# -------------------------------
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
datagen.fit(X_train)

# -------------------------------
# 3A. Baseline Model
# -------------------------------
baseline_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

baseline_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("\n--- Baseline Model Summary ---")
baseline_model.summary()

# Train baseline
history_base = baseline_model.fit(
    X_train, y_train_cat,
    validation_data=(X_test, y_test_cat),
    epochs=10,
    batch_size=64,
    verbose=1
)

# Evaluate baseline
base_loss, base_acc = baseline_model.evaluate(X_test, y_test_cat, verbose=0)
print(f"\n✅ Baseline Test Accuracy: {base_acc:.4f}")

# -------------------------------
# 3B. Improved Model
# -------------------------------
improved_model = Sequential([
    Conv2D(64, (5, 5), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(128, (5, 5), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
improved_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
print("\n--- Improved Model Summary ---")
improved_model.summary()

# Train improved
history_improved = improved_model.fit(
    datagen.flow(X_train, y_train_cat, batch_size=64),
    validation_data=(X_test, y_test_cat),
    epochs=20,
    verbose=1
)

# Evaluate improved
imp_loss, imp_acc = improved_model.evaluate(X_test, y_test_cat, verbose=0)
print(f"\n🔥 Improved Model Test Accuracy: {imp_acc:.4f}")

# -------------------------------
# 4. Save Best Model
# -------------------------------
os.makedirs("models", exist_ok=True)
improved_model.save("models/cifar10_cnn.h5")  # Save improved version
print("💾 Improved Model saved at models/cifar10_cnn.h5")

# -------------------------------
# 5. Plot Comparison
# -------------------------------
plt.figure(figsize=(8, 6))
plt.plot(history_base.history['accuracy'], label='Baseline Train Acc')
plt.plot(history_base.history['val_accuracy'], label='Baseline Val Acc')
plt.plot(history_improved.history['accuracy'], label='Improved Train Acc')
plt.plot(history_improved.history['val_accuracy'], label='Improved Val Acc')
plt.title("Baseline vs Improved Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# -------------------------------
# 6. Confusion Matrix (Improved Model)
# -------------------------------
y_pred = improved_model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = y_test.flatten()

cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix - Improved Model")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.show()

# Classification Report
print("\nClassification Report (Improved Model):\n")
print(classification_report(y_true, y_pred_classes, target_names=class_names))

# ===============================================
# FUNCTIONAL DEEP LEARNING MODEL WITH VISUALIZATIONS
# Task: Image Classification using CNN (MNIST)
# Framework: TensorFlow + Keras
# ===============================================

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# ============ Step 1: Load Dataset ============
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# ============ Step 2: Preprocessing ============
X_train = X_train / 255.0
X_test = X_test / 255.0
X_train = X_train[..., tf.newaxis]  # shape: (num_samples, 28, 28, 1)
X_test = X_test[..., tf.newaxis]

print("✅ Data loaded and normalized.")
print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)

# Visualize sample images
plt.figure(figsize=(10, 2))
for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.imshow(X_train[i].squeeze(), cmap='gray')
    plt.title(y_train[i])
    plt.axis('off')
plt.suptitle("🔍 Sample Training Images")
plt.show()

# ============ Step 3: Build CNN Model ============
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.summary()  # Show model architecture

# ============ Step 4: Compile Model ============
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# ============ Step 5: Train the Model ============
early_stop = EarlyStopping(monitor='val_loss', patience=2)

history = model.fit(X_train, y_train,
                    validation_split=0.1,
                    epochs=10,
                    callbacks=[early_stop],
                    verbose=1)

# ============ Step 6: Evaluate Model ============
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n✅ Test Accuracy: {test_acc:.4f}")

# ============ Step 7: Plot Training Results ============
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('📈 Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('📉 Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()

# ============ Step 8: Classification Report ============
y_pred_probs = model.predict(X_test)
y_pred_classes = np.argmax(y_pred_probs, axis=1)

print("\n📋 Classification Report:\n")
print(classification_report(y_test, y_pred_classes))

# ============ Step 9: Confusion Matrix ============
cm = confusion_matrix(y_test, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(10))
disp.plot(cmap='Blues')
plt.title("📊 Confusion Matrix")
plt.show()

# ============ Step 10: Save Model ============
model.save("mnist_cnn_model.h5")
print("💾 Model saved as 'mnist_cnn_model.h5'")
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import numpy as np
import matplotlib.pyplot as plt
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load and Preprocess the MNIST Dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Step 2: Build the Neural Network Model
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=3, validation_data=(x_test, y_test))
model.save("mnist_model.h5")

# Step 3: Generate Adversarial Examples
# Load the trained model
print("Generating adversarial examples...")
model = tf.keras.models.load_model("mnist_model.h5")

# Generate adversarial examples using FGSM
epsilon = 0.2  # Amount of perturbation
x_adv = fast_gradient_method(model, x_test, epsilon, np.inf)  # FGSM attack

# Visualize original and adversarial examples
def visualize_adversarial_examples(original, adversarial, labels, num_examples=5):
    plt.figure(figsize=(10, 5))
    for i in range(num_examples):
        # Original image
        plt.subplot(2, num_examples, i + 1)
        plt.imshow(original[i], cmap='gray')
        plt.title(f"Original: {labels[i]}")
        plt.axis('off')

        # Adversarial image
        plt.subplot(2, num_examples, i + 1 + num_examples)
        plt.imshow(adversarial[i], cmap='gray')
        plt.title("Adversarial")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Visualize 5 examples
visualize_adversarial_examples(x_test, x_adv, y_test)

# step 4 Prepare dataset for adversarial detection
print("Preparing dataset for adversarial detection...")

# Flatten the image data for use in the detector
x_test_flat = x_test.reshape(len(x_test), -1)  # Flatten normal test images
x_adv_flat = x_adv.numpy().reshape(len(x_adv), -1)  # Convert TensorFlow tensor to NumPy and flatten

# Combine original and adversarial data
x_combined = np.vstack((x_test_flat, x_adv_flat))  # Stack normal and adversarial images
y_combined = np.hstack((np.zeros(len(x_test_flat)), np.ones(len(x_adv_flat))))  # Labels: 0 for normal, 1 for adversarial


# Split into training and testing sets
x_train_combined, x_val_combined, y_train_combined, y_val_combined = train_test_split(
    x_combined, y_combined, test_size=0.3, random_state=42
)

# Train a Simple Detector (Random Forest Classifier)
print("Training adversarial detector...")
detector = RandomForestClassifier()
detector.fit(x_train_combined, y_train_combined)

# Evaluate the Detector
print("Evaluating adversarial detector...")
y_pred_combined = detector.predict(x_val_combined)
accuracy = accuracy_score(y_val_combined, y_pred_combined)
print(f"Detection Accuracy: {accuracy:.2f}")

# Detailed Classification Report
print("\nClassification Report:")
print(classification_report(y_val_combined, y_pred_combined))

# Simulate Real-Time Detection
import random

print("\nSimulating real-time detection...")

# Randomly select a few examples from the combined dataset
num_samples = 5  # Number of samples to simulate
random_indices = random.sample(range(len(x_combined)), num_samples)

# Loop through selected samples and display predictions
for idx in random_indices:
    sample = x_combined[idx].reshape(1, -1)  # Reshape sample for prediction
    label = y_combined[idx]  # True label (0: normal, 1: adversarial)
    prediction = detector.predict(sample)  # Detector's prediction
    
    # Print result for each sample
    print(f"Sample {idx}: True Label = {'Normal' if label == 0 else 'Adversarial'}, "
          f"Prediction = {'Normal' if prediction[0] == 0 else 'Adversarial'}")

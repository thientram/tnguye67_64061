# Step 1: Import modules for this project
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# Step 2: Create a new simulated classification dataset
# Adjusting informative and redundant features to be consistent with n_features
centers = [[2, 4], [6, 6], [1, 9]]
n_classes = len(centers)
data, labels = make_blobs(n_samples=150, centers=np.array(centers), random_state=1)

# Step 3: Training and testing split
train_data, test_data, train_labels, test_labels = train_test_split(data, labels,
                                                                    train_size=0.8,
                                                                    test_size=0.2,
                                                                    random_state=12)

# Step 4: Create and fit a nearest-neighbor classifier
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(train_data, train_labels)

# Step 5: Make predictions on training data and report accuracy

train_data_predicted = knn.predict(train_data)
print("Predictions from the classifier on training data:")
print(train_data_predicted)
print("Target values (Training):")
print(train_labels)
train_accuracy = accuracy_score(train_data_predicted, train_labels)
print(f"Training accuracy: {train_accuracy}")

# Step 6: Make predictions on test data and report accuracy
test_data_predicted = knn.predict(test_data)
print("\nPredictions from the classifier on test data:")
print(test_data_predicted)
print("Target values (Test):")
print(test_labels)
test_accuracy = accuracy_score(test_data_predicted, test_labels)
print(f"Test accuracy: {test_accuracy}")

# Step 7: Plot the results
plt.figure(figsize=(10, 6))

# Plot the training data with true labels
plt.subplot(1, 2, 1)
plt.scatter(train_data[:, 0], train_data[:, 1D], c=train_labels, cmap='viridis', edgecolor='k', s=100)
plt.title("Training Data (True Labels)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

# Plot the test data with predictions
plt.subplot(1, 2, 2)
plt.scatter(test_data[:, 0], test_data[:, 1], c=test_data_predicted, cmap='viridis', edgecolor='k', s=100)
plt.title("Test Data (Predicted Labels)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.tight_layout()
plt.show()

#IMDB MOVIE REVIEW CLASSIFICATION'S PERFORMANCE WITH 32 HIDDEN UNITS

from tensorflow.keras.datasets import imdb
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

# Load IMDB data
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# Function to vectorize sequences
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

# Vectorizing train and test data
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

# Model definition with 3 layers, 32 units, tanh activation, regularization, and dropout
from tensorflow.keras import layers
model = keras.Sequential([
    layers.Dense(32, activation="tanh", kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.Dropout(0.5),  # Dropout layer to reduce overfitting
    layers.Dense(32, activation="tanh", kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.Dropout(0.5),
    layers.Dense(1, activation="sigmoid")
])

# Model compilation with mse loss function
model.compile(optimizer='rmsprop',
              loss='mean_squared_error',
              metrics=['accuracy'])

# Splitting the data for validation
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

# Training the model
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))

# Evaluating the model
results = model.evaluate(x_test, test_labels)
# Making predictions
predictions = model.predict(x_test)

# Plotting Training and Validation Loss
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plotting Accuracy
plt.plot(epochs, acc_values, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc_values, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()



# #IMDB MOVIE REVIEW CLASSIFICATION'S PERFORMANCE WITH 64 HIDDEN UNITS
# from tensorflow.keras.datasets import imdb
# import numpy as np
# from tensorflow import keras
# import matplotlib.pyplot as plt

# # Load IMDB data
# (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# # Function to vectorize sequences
# def vectorize_sequences(sequences, dimension=10000):
#     results = np.zeros((len(sequences), dimension))
#     for i, sequence in enumerate(sequences):
#         results[i, sequence] = 1.
#     return results

# # Vectorizing train and test data
# x_train = vectorize_sequences(train_data)
# x_test = vectorize_sequences(test_data)

# # Model definition with 3 layers, 64 units, tanh activation, regularization, and dropout
# from tensorflow.keras import layers
# model = keras.Sequential([
#     layers.Dense(64, activation="tanh", kernel_regularizer=keras.regularizers.l2(0.001)),  # Regularization
#     layers.Dropout(0.5),  # Dropout layer to reduce overfitting
#     layers.Dense(64, activation="tanh", kernel_regularizer=keras.regularizers.l2(0.001)),
#     layers.Dropout(0.5),
#     layers.Dense(1, activation="sigmoid")
# ])

# # Model compilation with mse loss function
# model.compile(optimizer='rmsprop',
#               loss='mean_squared_error',
#               metrics=['accuracy'])

# # Splitting the data for validation
# x_val = x_train[:10000]
# partial_x_train = x_train[10000:]
# y_val = train_labels[:10000]
# partial_y_train = train_labels[10000:]

# # Training the model
# history = model.fit(partial_x_train,
#                     partial_y_train,
#                     epochs=20,
#                     batch_size=512,
#                     validation_data=(x_val, y_val))

# # Evaluating the model
# results = model.evaluate(x_test, test_labels)
# # Making predictions
# predictions = model.predict(x_test)

# # Plotting Training and Validation Loss
# history_dict = history.history
# loss_values = history_dict['loss']
# val_loss_values = history_dict['val_loss']
# acc_values = history_dict['accuracy']
# val_acc_values = history_dict['val_accuracy']
# epochs = range(1, len(loss_values) + 1)

# # Plotting Loss
# plt.plot(epochs, loss_values, 'bo', label='Training loss')
# plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()

# # Plotting Accuracy
# plt.plot(epochs, acc_values, 'bo', label='Training accuracy')
# plt.plot(epochs, val_acc_values, 'b', label='Validation accuracy')
# plt.title('Training and validation accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()

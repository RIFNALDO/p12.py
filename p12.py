import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# 1. Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize the data (scale pixel values between 0 and 1)
x_train, x_test = x_train / 255.0, x_test / 255.0

# Reshape the data to fit the fully connected layers (28x28 -> 784)
x_train_reshaped = x_train.reshape(-1, 28 * 28)
x_test_reshaped = x_test.reshape(-1, 28 * 28)

# 2. Data Augmentation setup (reshape training images to 28x28x1 for ImageDataGenerator)
x_train_img = x_train.reshape(-1, 28, 28, 1)
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1
)
datagen.fit(x_train_img)

# 3. Build the Deeper Neural Network Model with LeakyReLU
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    # First hidden layer using LeakyReLU
    layers.Dense(128, activation=tf.keras.layers.LeakyReLU(alpha=0.1)),
    # Additional hidden layer for a deeper network
    layers.Dense(64, activation=tf.keras.layers.LeakyReLU(alpha=0.1)),
    # Output layer with softmax for 10 digit classes
    layers.Dense(10, activation='softmax')
])

# 4. Compile the Model using the RMSprop optimizer
model.compile(
    optimizer='rmsprop',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 5. Train the Model with Augmented Data for at least 10 epochs
model.fit(
    datagen.flow(x_train_img, y_train, batch_size=32),
    epochs=10
)

# 6. Evaluate Model Performance on the Test Set
test_loss, test_acc = model.evaluate(x_test_reshaped, y_test)
print(f"Test accuracy: {test_acc}")

# 7. Make Predictions and Visualize Using Matplotlib
predictions = model.predict(x_test_reshaped)

plt.imshow(x_test[0], cmap=plt.cm.binary)
plt.title(f"Predicted: {predictions[0].argmax()}")
plt.show()
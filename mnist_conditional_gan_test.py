import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import numpy as np
import imageio


# configurations
latent_dim = 128
num_classes = 10
alpha = 5

# Loading generator model
generator_model = load_model('models/conditional_generator_model_500.h5')
print('Generator Model Loaded')

# Making one hot encoding
label = keras.utils.to_categorical([alpha], num_classes)[0]
labels = tf.cast(label, tf.float32)
labels = tf.reshape(labels, (1, num_classes))
# print("Labels:",labels)

# Sample noise
noise = tf.random.normal(shape=(1, latent_dim))
noise = tf.reshape(noise, (1, latent_dim))
# print("Noise:",noise)

noise_and_labels = tf.concat([noise, labels], 1)
# print("Noise and label:",noise_and_labels)
generated_image = generator_model.predict(noise_and_labels)

generated_image *= 255.0
converted_images = generated_image.astype(np.uint8)
converted_images = tf.image.resize(converted_images, (96, 96)).numpy().astype(np.uint8)
converted_images = converted_images.reshape(96, 96)

imageio.imwrite('plots/Generated_image.png', converted_images)
print('Image saved successfully!')

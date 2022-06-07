###########################################################################################################################################################
import cv2 as cv
from flask import Flask, request, Response,send_file,render_template,send_from_directory
import tensorflow as tf
from tensorflow import keras
import os, time, imageio
import numpy as np
from matplotlib import pyplot
import warnings
warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings

from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
import random
###########################################################################################################################################################
app = Flask(__name__)
UPLOAD_FOLDER = os.path.basename('/static/uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
counter = len(os.listdir("static/uploads"))
predicted_label = -1
query_image_path = ""
###########################################################################################################################################################
print('------------------------------------------------------------------------------------------------')
start_time = time.time()
# Loading MNIST Generation Model
generator_model = load_model('models/generator_model_500.h5')
print("MNIST generation model loaded...",end='')

# Loading MNIST classification model
classification_model = load_model('models/mnist_classification_model.h5')
print("MNIST classification model loaded...",end='')

# Loading MNIST Conditional Generation Model
conditional_generator_model = load_model('models/conditional_generator_model_500.h5')
print('MNIST conditional generation Model...', end='')

end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))

# Loading MNIST Dataset
print("Loading MNIST Dataset...",end='')
start_time = time.time()
(trainX, trainY), (testX, testY) = mnist.load_data()
testX = testX.reshape((testX.shape[0], 28, 28, 1))
test_norm = testX.astype('float32')
test_norm = test_norm / 255.0

end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))
print("Number of test samples:",test_norm.shape)
print('------------------------------------------------------------------------------------------------')
###########################################################################################################################################################
@app.route('/')
def home():
    global counter
    global predicted_label
    global classification_model
    global query_image 

    counter = counter + 1
    random_index= random.randint(0, test_norm.shape[0])
    test_image = test_norm[random_index].reshape(1, 28, 28, 1)
    test_image_gt = testY[random_index]
    print("Test image:", test_image.shape, test_image_gt)

    test_image *= 255.0
    save_img = tf.image.resize(test_image.astype(np.uint8), (256, 256)).numpy().astype(np.uint8)
    save_img = save_img.reshape(256, 256)
    converted_image = tf.image.resize(test_image.astype(np.uint8), (28, 28)).numpy().astype(np.uint8)
    print('Image Shape:',converted_image.shape)

    prediction = classification_model.predict(converted_image.reshape(1, 28, 28, 1))
    # print("Prediction:", prediction)
    predicted_label = prediction.argmax()
    print("Predicted Class:",predicted_label)
    
    # Saving the image
    query_image_path = "static/uploads/"+str(counter) + "_" + str(test_image_gt) + ".png"
    imageio.imwrite(query_image_path, save_img)

    html_body = '<input type="text" id="input_number" name="input_number" value="2" size="2">'
    return render_template('index.html', query_image=query_image_path, generated_image=query_image_path,
                                html_text = html_body)
###########################################################################################################################################################    
@app.route('/processImage',methods=['POST'])
def processImage():
    global predicted_label
    global counter

    counter =  counter + 1
    input_class = int(request.form['input_number'])
    query_image_path = request.form['query_image_path']
    num_classes = 10
    latent_dim = 128
    class_label_sum = predicted_label + input_class
    print('-----------------------------------------------------------------------------')
    
    label = keras.utils.to_categorical([class_label_sum], num_classes)[0]
    labels = tf.cast(label, tf.float32)
    labels = tf.reshape(labels, (1, num_classes))
    # print("Labels:",labels)

    # Sample noise
    noise = tf.random.normal(shape=(1, latent_dim))
    noise = tf.reshape(noise, (1, latent_dim))
    # print("Noise:",noise)

    noise_and_labels = tf.concat([noise, labels], 1)
    # print("Noise and label:",noise_and_labels)
    generated_image = conditional_generator_model.predict(noise_and_labels)

    generated_image *= 255.0
    generated_image = tf.image.resize(generated_image.astype(np.uint8), (256, 256)).numpy().astype(np.uint8)
    generated_image = generated_image.reshape(256, 256)
    

    
    # Saving the image
    generated_image_path = "static/uploads/"+str(counter) + "_" + str(class_label_sum) + ".png"
    imageio.imwrite(generated_image_path, generated_image)

    html_body = '<input type="text" id="input_number" name="input_number" value="'+str(input_class)+'" size="2">'
    return render_template('index.html', query_image=query_image_path, generated_image=generated_image_path,
                            html_text = html_body)
###########################################################################################################################################################
@app.route('/generateImage',methods=['POST'])
def generateImage():
    global counter
    counter =  counter + 1

    n_samples = int(request.form['input_number']) # number of samples to be genrated
    
    # Genrated latent vector
    latent_dim = 100 # the dimension of the latent vector
    # generate points in the latent space
    x_input = np.random.randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    latent_points = x_input.reshape(n_samples, latent_dim)
    print("Latent points:",latent_points.shape)


    # generate images
    images = generator_model.predict(latent_points)
    print("Image Dimension:",images.shape)
    n = int(np.sqrt(n_samples))
    print('Number of samples to be generated:',n_samples)
    print('N:',n)

    # Define the height and width of the figure
    f = pyplot.figure()
    f.set_figwidth(100)
    f.set_figheight(100)

    # plot images
    for i in range(n * n):
        # define subplot
        pyplot.subplot(n, n, 1 + i)
        # turn off axis
        pyplot.axis('off')
        # plot raw pixel data
        pyplot.imshow(images[i, :, :, 0], cmap='gray_r')
    # pyplot.show()
    # save plot to file
    query_image_path = 'static/uploads/generated_plot_counter_%03d_%03d.png' % (counter,n)
    print('Image Path:',query_image_path)
    pyplot.savefig(query_image_path)
    pyplot.close()

    html_body = '<input type="text" id="input_number" name="input_number" value="'+str(n_samples)+'" size="2">'
    return render_template('mnist_digit_generation.html', query_image=query_image_path, html_text = html_body)
###########################################################################################################################################################
@app.route('/mnist')
def mnist():
    global counter
    counter =  counter + 1

    # Genrated latent vector
    n_samples = 25 # number of samples to be genrated
    latent_dim = 100 # the dimension of the latent vector
    # generate points in the latent space
    x_input = np.random.randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    latent_points = x_input.reshape(n_samples, latent_dim)
    print("Latent points:",latent_points.shape)


    # generate images
    images = generator_model.predict(latent_points)
    print("Image Dimension:",images.shape)
    n = int(np.sqrt(n_samples))
    print('N:',n)

    # Define the height and width of the figure
    f = pyplot.figure()
    f.set_figwidth(100)
    f.set_figheight(100)

    # plot images
    for i in range(n * n):
        # define subplot
        pyplot.subplot(n, n, 1 + i)
        # turn off axis
        pyplot.axis('off')
        # plot raw pixel data
        pyplot.imshow(images[i, :, :, 0], cmap='gray_r')
    # pyplot.show()
    # save plot to file
    query_image_path = 'static/uploads/generated_plot_counter_%03d_%03d.png' % (counter,n)
    print('Image Path:',query_image_path)
    pyplot.savefig(query_image_path)
    pyplot.close()

    html_body = '<input type="text" id="input_number" name="input_number" value="25" size="3">'
    return render_template('mnist_digit_generation.html', query_image=query_image_path, html_text = html_body)
###########################################################################################################################################################
if __name__ == '__main__':
   app.run(host='0.0.0.0' , port=5000)
###########################################################################################################################################################
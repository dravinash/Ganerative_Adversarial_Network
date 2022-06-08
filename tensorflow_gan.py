# example of generating random samples from X^2
from numpy.random import rand
from numpy.random import randn
from numpy import hstack, ones, zeros
from matplotlib import pyplot

from keras.utils.vis_utils import plot_model
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm

# generate randoms sample from x^2
def generate_real_samples(n):
    X1 = rand(n) - 0.5
    X2 = X1 * X1
    X = hstack((X1.reshape(n, 1), X2.reshape(n, 1)))
    return X, ones((n, 1))

def generate_latent_points(latent_dim, n):
	# generate points in the latent space
	x_input = randn(latent_dim * n)
	# reshape into a batch of inputs for the network
	x_input = x_input.reshape(n, latent_dim)
	return x_input

# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n):
	# generate points in latent space
	x_input = generate_latent_points(latent_dim, n)
	# predict outputs
	X = generator.predict(x_input)
	return X, zeros((n, 1))

# def generate_fake_samples(n):
#     # generate inputs in [-1, 1]
#     X1 = -1 + rand(n) * 2
#     # generate outputs in [-1, 1]
#     X2 = -1 + rand(n) * 2
#     X = hstack((X1.reshape(n, 1), X2.reshape(n, 1)))
#     return X, zeros((n, 1))

def discriminator_network(n_inputs,h_dim):
	model = keras.Sequential()
	model.add(layers.Dense(h_dim, activation=layers.LeakyReLU(0.1),kernel_initializer='he_uniform', input_dim=n_inputs))
	model.add(layers.Dense(1, activation='sigmoid'))
	# compile model
	model.compile(loss='binary_crossentropy', optimizer=Adam(0.001), metrics=['accuracy'])
	return model

def generator_network(latent_dim, h_dim, n_outputs):
	model = keras.Sequential()
	model.add(layers.Dense(h_dim, activation=layers.LeakyReLU(0.1), kernel_initializer='he_uniform',input_dim=latent_dim))
	model.add(layers.Dense(n_outputs, activation='linear'))
	return model

def gan(generator, discriminator):
	# make weights in the discriminator not trainable
	discriminator.trainable = False
	# connect them
	model = keras.Sequential()
	# add generator
	model.add(generator)
	# add the discriminator
	model.add(discriminator)
	# compile model
	model.compile(loss='binary_crossentropy', optimizer='adam')
	return model

def summarize_performance(epoch, generator, discriminator, latent_dim, n=100):
    # prepare real samples
    x_real, y_real = generate_real_samples(n)
    # evaluate discriminator on real examples
    _, acc_real = discriminator.evaluate(x_real, y_real, verbose=0)
    # prepare fake examples
    x_fake, y_fake = generate_fake_samples(generator, latent_dim, n)
    # evaluate discriminator on fake examples
    _, acc_fake = discriminator.evaluate(x_fake, y_fake, verbose=0)
    # summarize discriminator performance
    print(epoch, acc_real, acc_fake)
    # scatter plot real and fake data points
    pyplot.scatter(x_real[:, 0], x_real[:, 1], color='red')
    pyplot.scatter(x_fake[:, 0], x_fake[:, 1], color='blue')
    # pyplot.show()
    pyplot.savefig('plots/gan_%d.png' % epoch)
    pyplot.close()

# train the generator and discriminator
def train(g_model, d_model, gan_model, latent_dim, n_epochs=10000, n_batch=128, n_eval=2000):
	# determine half the size of one batch, for updating the discriminator
	half_batch = int(n_batch / 2)
	# manually enumerate epochs
	for i in tqdm(range(n_epochs)):
		# prepare real samples
		x_real, y_real = generate_real_samples(half_batch)
		# prepare fake examples
		x_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
		# update discriminator
		d_model.train_on_batch(x_real, y_real)
		d_model.train_on_batch(x_fake, y_fake)
		# prepare points in latent space as input for the generator
		x_gan = generate_latent_points(latent_dim, n_batch)
		# create inverted labels for the fake samples
		y_gan = ones((n_batch, 1))
		# update the generator via the discriminator's error
		gan_model.train_on_batch(x_gan, y_gan)
		# evaluate the model every n_eval epochs
		if (i+1) % n_eval == 0:
			summarize_performance(i+1, g_model, d_model, latent_dim)


n_inputs = 2
latent_dim = 16
h_dim = 32

# define the discriminator model
discriminator_model = discriminator_network(n_inputs, h_dim)
# summarize the model
discriminator_model.summary()
# plot_model(discriminator_model, to_file='discriminator_model.png', show_shapes=True, show_layer_names=True)

# define the generator model
generator_model = generator_network(latent_dim, h_dim, n_inputs)
# summarize the model
generator_model.summary()
# plot_model(generator_model, to_file='generator_model.png', show_shapes=True, show_layer_names=True)

# create the gan
gan_model = gan(generator_model, discriminator_model)
# summarize the model
gan_model.summary()
# plot_model(gan_model, to_file='gan.png', show_shapes=True, show_layer_names=True)

# train model
train(generator_model, discriminator_model, gan_model, latent_dim)

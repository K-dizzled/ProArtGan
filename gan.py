import math
import numpy as np
from keras import Input
import tensorflow as tf
from keras import backend
from matplotlib import pyplot
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras import layers
from skimage.transform import resize
from tensorflow.keras.layers import Input
from tensorflow.keras.utils import Progbar
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.constraints import max_norm

lat_dim = 100


class WeightedSum(layers.Add):
    def __init__(self, alpha=0.0, **kwargs):
        super(WeightedSum, self).__init__(**kwargs)
        self.alpha = keras.backend.variable(alpha, name='ws_alpha')

    def _merge_function(self, inputs):
        # only supports a weighted sum of two inputs
        assert (len(inputs) == 2)
        # ((1-a) * input1) + (a * input2)
        output = ((1.0 - self.alpha) * inputs[0]) + (self.alpha * inputs[1])
        return output

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'alpha': self.alpha.numpy()
        })

        return config


class MinibatchStdev(layers.Layer):
    # initialize the layer
    def __init__(self, **kwargs):
        super(MinibatchStdev, self).__init__(**kwargs)

    def call(self, inputs, *args, **kwargs):
        mean = keras.backend.mean(inputs, axis=0, keepdims=True)
        squ_diffs = keras.backend.square(inputs - mean)
        mean_sq_diff = keras.backend.mean(squ_diffs, axis=0, keepdims=True)
        # add a small value to avoid a blow-up when we calculate stdev
        mean_sq_diff += 1e-8

        stdev = keras.backend.sqrt(mean_sq_diff)
        mean_pix = keras.backend.mean(stdev, keepdims=True)

        shape = keras.backend.shape(inputs)
        output = keras.backend.tile(mean_pix, (shape[0], shape[1], shape[2], 1))

        combined = keras.backend.concatenate([inputs, output], axis=-1)
        return combined

    def compute_output_shape(self, input_shape):
        input_shape = list(input_shape)
        # add one to the channel dimension
        input_shape[-1] += 1

        return tuple(input_shape)


class PixelNormalization(layers.Layer):
    def __init__(self, **kwargs):
        super(PixelNormalization, self).__init__(**kwargs)

    def call(self, inputs, *args, **kwargs):
        values = inputs ** 2.0
        mean_values = keras.backend.mean(values, axis=-1, keepdims=True)
        # ensure the mean is not zero
        mean_values += 1.0e-8

        l2 = keras.backend.sqrt(mean_values)
        normalized = inputs / l2
        return normalized

    def compute_output_shape(self, input_shape):
        return input_shape


def wasserstein_loss(y_true, y_pred):
    return keras.backend.mean(y_true * y_pred)


def add_discriminator_block(model, layers_to_skip=3):
    # Weights of the layers are initialized with random
    # numbers with std = 0.02
    init = RandomNormal(stddev=0.02)
    # weight constraint
    const = max_norm(1.0)

    old_inp_shape = list(model.input.shape)
    new_inp_shape = (old_inp_shape[-2] * 2, old_inp_shape[-2] * 2, old_inp_shape[-1])

    inp = Input(shape=new_inp_shape)
    x = layers.Conv2D(128, kernel_size=(1, 1), padding='same',
                      kernel_initializer=init,
                      kernel_constraint=const)(inp)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Conv2D(128, kernel_size=(3, 3), padding='same',
                      kernel_initializer=init,
                      kernel_constraint=const)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Conv2D(128, kernel_size=(3, 3), padding='same',
                      kernel_initializer=init,
                      kernel_constraint=const)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.AveragePooling2D()(x)
    block_new = x

    for i in range(layers_to_skip, len(model.layers)):
        x = model.layers[i](x)

    model1 = models.Model(inp, x)
    model1.compile(loss=wasserstein_loss,
                   optimizer=Adam(lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8))

    downsample = layers.AveragePooling2D()(inp)

    block_old = model.layers[1](downsample)
    block_old = model.layers[2](block_old)

    x = WeightedSum()([block_old, block_new])

    for i in range(layers_to_skip, len(model.layers)):
        x = model.layers[i](x)

    model2 = models.Model(inp, x)
    model2.compile(loss=wasserstein_loss,
                   optimizer=Adam(lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8))
    return [model1, model2]


def define_discriminator(n_blocks, initial_shape=(4, 4, 3)):
    init = RandomNormal(stddev=0.02)
    const = max_norm(1.0)

    model_list = list()
    inp = Input(shape=initial_shape)
    x = layers.Conv2D(128, kernel_size=(1, 1), padding='same',
                      kernel_initializer=init,
                      kernel_constraint=const)(inp)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = MinibatchStdev()(x)
    x = layers.Conv2D(128, kernel_size=(3, 3), padding='same',
                      kernel_initializer=init,
                      kernel_constraint=const)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Conv2D(128, kernel_size=(4, 4), padding='same',
                      kernel_initializer=init,
                      kernel_constraint=const)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Flatten()(x)
    output = layers.Dense(1)(x)

    model = models.Model(inp, output)
    model.compile(loss=wasserstein_loss,
                  optimizer=Adam(lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8))

    model_list.append([model, model])
    for i in range(1, n_blocks):
        old_model = model_list[i - 1][0]
        pair = add_discriminator_block(old_model)
        model_list.append(pair)
    return model_list


def add_generator_block(model):
    init = RandomNormal(stddev=0.02)
    const = max_norm(1.0)

    # We need to continue the sequence of layers, but we don't need
    # the last 1 by 1 con layer, cause it produces the output image
    previous = model.layers[-2].output
    u = layers.UpSampling2D()(previous)
    x = layers.Conv2D(128, kernel_size=(3, 3), padding='same',
                      kernel_initializer=init,
                      kernel_constraint=const)(u)
    x = PixelNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Conv2D(128, kernel_size=(3, 3), padding='same',
                      kernel_initializer=init,
                      kernel_constraint=const)(x)
    x = PixelNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    output = layers.Conv2D(3, kernel_size=(1, 1), padding='same',
                           kernel_initializer=init,
                           kernel_constraint=const)(x)

    model1 = models.Model(model.input, output)

    previously_generated_image = model.layers[-1](u)
    merged = WeightedSum()([previously_generated_image, output])
    model2 = models.Model(model.input, merged)

    return [model1, model2]


def define_generator(latent_dim, n_blocks, in_dim=4):
    init = RandomNormal(stddev=0.02)
    const = max_norm(1.0)

    model_list = list()

    inp = Input(shape=(latent_dim,))
    g = layers.Dense(128 * in_dim * in_dim,
                     kernel_initializer=init,
                     kernel_constraint=const)(inp)
    g = layers.Reshape((in_dim, in_dim, 128))(g)
    g = layers.Conv2D(128, kernel_size=(3, 3), padding='same',
                      kernel_initializer=init,
                      kernel_constraint=const)(g)
    g = PixelNormalization()(g)
    g = layers.LeakyReLU(alpha=0.2)(g)
    g = layers.Conv2D(128, kernel_size=(3, 3), padding='same',
                      kernel_initializer=init,
                      kernel_constraint=const)(g)
    g = PixelNormalization()(g)
    g = layers.LeakyReLU(alpha=0.2)(g)
    out_image = layers.Conv2D(3, kernel_size=(1, 1), padding='same',
                              kernel_initializer=init,
                              kernel_constraint=const)(g)

    model = models.Model(inp, out_image)

    model_list.append([model, model])
    for i in range(1, n_blocks):
        old_model = model_list[i - 1][0]
        pair = add_generator_block(old_model)
        model_list.append(pair)
    return model_list


def define_composite(discriminators, generators):
    model_list = list()

    for i in range(len(discriminators)):
        g_models, d_models = generators[i], discriminators[i]

        d_models[0].trainable = False
        model1 = models.Sequential()
        model1.add(g_models[0])
        model1.add(d_models[0])
        model1.compile(loss=wasserstein_loss,
                       optimizer=Adam(lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8))

        d_models[1].trainable = False
        model2 = models.Sequential()
        model2.add(g_models[1])
        model2.add(d_models[1])
        model2.compile(loss=wasserstein_loss,
                       optimizer=Adam(lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8))

        model_list.append([model1, model2])
    return model_list


# load dataset
def load_real_samples(filename):
    # load dataset
    dt = np.load(filename)
    # extract numpy array
    X = dt['arr_0']
    # convert from ints to floats
    X = X.astype('float32')
    # scale from [0,255] to [-1,1]
    X = (X - 127.5) / 127.5
    return X


def generate_latent_points(latent_dim, n_samples):
    # generate points in the latent space
    x_input = np.random.randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input


def generate_fake_samples(generator, latent_dim, n_samples):
    # generate points in latent space
    x_input = generate_latent_points(latent_dim, n_samples)
    # predict outputs
    X = generator.predict(x_input)
    return X


def update_fadein(model_list, step, n_steps):
    # calculate current alpha (linear from 0 to 1)
    alpha = step / float(n_steps - 1)
    # update the alpha for each model
    for model in model_list:
        for layer in model.layers:
            if isinstance(layer, WeightedSum):
                backend.set_value(layer.alpha, alpha)


def scale_dataset(images, new_shape):
    images_list = list()
    for image in images:
        # resize with nearest neighbor interpolation
        new_image = resize(image, new_shape, 0)
        # store
        images_list.append(new_image)
    return np.asarray(images_list)


# generate samples and save as a plot and save the model
def summarize_performance(status, g_model, latent_dim, n_samples=25):
    # devise name
    gen_shape = g_model.output_shape
    name = '%03dx%03d-%s' % (gen_shape[1], gen_shape[2], status)
    # generate images
    X = generate_fake_samples(g_model, latent_dim, n_samples)
    # normalize pixel values to the range [0,1]
    X = (X - X.min()) / (X.max() - X.min())
    # plot real images
    square = int(math.sqrt(n_samples))
    for i in range(n_samples):
        pyplot.subplot(square, square, 1 + i)
        pyplot.axis('off')
        pyplot.imshow(X[i])
    # save plot to file
    filename1 = 'plot_%s.png' % name
    pyplot.savefig(filename1)
    pyplot.close()
    # save the generator model
    filename2 = 'model_%s.h5' % name
    g_model.save(filename2)
    print('>Saved: %s and %s' % (filename1, filename2))


class GAN(keras.Model):
    def __init__(self, gan_models_list,
                 generators_list,
                 discriminators_list,
                 latent_dim=100):
        super().__init__()
        self.model_list = gan_models_list
        self.generators_list = generators_list
        self.discriminators_list = discriminators_list

        self.latent_dim = latent_dim
        self.training_phase = 0
        self.fade_in = 0

        # Callbacks
        self.callbacks = []

    def compile(self):
        super(GAN, self).compile()

    def train_step(self, dataset, number_of_epochs, batch_size):
        # Samples random points in the latent space
        generator = self.generators_list[self.training_phase][self.fade_in]
        discriminator = self.discriminators_list[self.training_phase][self.fade_in]
        gan_model = self.model_list[self.training_phase][self.fade_in]

        amount_of_batches_per_epoch = int(dataset.shape[0] / batch_size)
        number_of_batches_at_all = amount_of_batches_per_epoch * number_of_epochs

        half_batch = int(batch_size / 2)

        # Activate callbacks
        for callback in self.callbacks:
            if getattr(callback, 'on_phase_begin'):
                callback.on_phase_begin()

        # Show progress bar
        p_bar = Progbar(target=number_of_batches_at_all)

        for i in range(number_of_batches_at_all):
            # Proportionally updates fade-in if in correct phase
            if self.fade_in:
                update_fadein([generator, discriminator, gan_model], i, number_of_batches_at_all)

            # # Generate real images
            ix = np.random.randint(0, dataset.shape[0], half_batch)
            X_real = dataset[ix]

            # Generate fake images
            X_fake = generate_fake_samples(generator, lat_dim, half_batch)

            # Labels
            y_real = tf.ones((half_batch, 1))
            y_fake = -tf.ones((half_batch, 1))

            # Trick to normalize labels
            y_real += 0.05 * tf.random.uniform(tf.shape(y_real))
            y_fake += 0.05 * tf.random.uniform(tf.shape(y_fake))

            # Update discriminator model
            d_loss1 = discriminator.train_on_batch(X_real, y_real)
            d_loss2 = discriminator.train_on_batch(X_fake, y_fake)
            discriminator_loss = abs(d_loss1) + abs(d_loss2)

            # Here we feed random points into the whole model
            # containing a generator and discriminator. We want it to
            # take random points of the latent space, generate images,
            # then pass them through discriminator. And here we want
            # them to be 1. We make a gradient descent for this.
            # And as the discriminator layer in the gan_model is frozen,
            # we update only the weights of the generator.
            z_input = generate_latent_points(lat_dim, batch_size)
            y_real2 = np.ones((batch_size, 1))
            generator_loss = gan_model.train_on_batch(z_input, y_real2)

            p_bar.update(i + 1, values=[('d_loss', discriminator_loss), ('g_loss', abs(generator_loss))])

        # Activate callbacks
        for callback in self.callbacks:
            if getattr(callback, 'on_phase_end'):
                callback.on_phase_end(self.fade_in, generator)

    def fit(self, dataset, epochs, batch_sizes, callbacks=None):
        if callbacks:
            self.callbacks = callbacks

        generator = self.generators_list[0][0]
        phase_image_scale = generator.output_shape
        rescaled_data = scale_dataset(dataset, phase_image_scale[1:])

        # First run for 4 by 4 images. Already "tuned"
        self.fade_in = 1
        self.train_step(rescaled_data, epochs[0], batch_sizes[0])
        self.fade_in = 0

        # Loop for fading and tuning phases of training
        for i in range(1, len(self.model_list)):
            self.training_phase += 1

            # Rescale data to appropriate size
            generator = self.generators_list[self.training_phase][0]
            phase_image_scale = generator.output_shape
            rescaled_data = scale_dataset(dataset, phase_image_scale[1:])

            # Run regular phase training
            self.train_step(rescaled_data, epochs[0], batch_sizes[0])
            # Run fade-in phase of training
            self.fade_in = 1
            self.train_step(rescaled_data, epochs[0], batch_sizes[0])
            self.fade_in = 0


class ProGANMonitor(keras.callbacks.Callback):
    def __init__(self, phase_amount, latent_dim):
        self.phase_num = 0
        self.phase_amount = phase_amount
        self.latent_dim = latent_dim

    def on_phase_begin(self):
        self.phase_num += 1
        print(f'Phase {self.phase_num}/{self.phase_amount}')

    def on_phase_end(self, fade_in, generator):
        faded = 'tuned' if fade_in else 'faded'
        summarize_performance(faded, generator, self.latent_dim)


num_blocks = 7

models_d = define_discriminator(num_blocks)
models_g = define_generator(lat_dim, num_blocks)
gan_models_ = define_composite(models_d, models_g)

data = load_real_samples('landscapes_256by256_dataset.npz')
print('Loaded', data.shape)

num_batch = [16, 16, 16, 8, 4, 4, 4]
num_epochs = [8, 12, 12, 16, 16, 16, 20]

pro_gan = GAN(gan_models_, models_g, models_d)
pro_gan.compile()
pro_gan.fit(
    dataset=data,
    epochs=num_epochs,
    batch_sizes=num_batch,
    callbacks=[ProGANMonitor(phase_amount=7, latent_dim=100)]
)

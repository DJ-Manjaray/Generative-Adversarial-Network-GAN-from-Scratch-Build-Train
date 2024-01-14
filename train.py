import numpy as np

from utils import X_train, plot_generated_images
from keras.layers import Input
from model import build_generator, build_discriminator
from keras.models import Model
from keras.optimizers import Adam


# Input shape
img_shape = (28,28,1)
channels = 1
latent_dim = 100


optimizer = Adam(0.0002, 0.5)

# Build and compile the discriminator
discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Build the generator
generator = build_generator()

# The generator takes noise as input and generates imgs
z = Input(shape=(latent_dim,))
img = generator(z)

# For the combined model we will only train the generator
discriminator.trainable = False

# The discriminator takes generated images as input and determines validity
valid = discriminator(img)

# The combined model  (stacked generator and discriminator)
# Trains the generator to fool the discriminator
combined = Model(z, valid)
combined.compile(loss='binary_crossentropy', optimizer=optimizer)



def train(epochs, batch_size=128, save_interval=50):

    """
    Purpose: Trains a Generative Adversarial Network (GAN) for a specified number of epochs.

    Key Parameters:
     - epochs: The number of training iterations.
     - batch_size: The number of samples processed in each iteration.
     - save_interval: The frequency of saving generated images.
    """
    # Define labels for real and fake samples
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    # Training loop
    for epoch in range(epochs):  # Train Discriminator network
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        imgs = X_train[idx]
        
        noise = np.random.normal(0, 1, (batch_size, 100))
        gen_imgs = generator.predict(noise)
        
        #  Train the Discriminator on real and fake samples
        d_loss_real = discriminator.train_on_batch(imgs, valid)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        # Train the Generator to fool the Discriminator
        g_loss = combined.train_on_batch(noise, valid)
        
        # printing progress
        print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" %(epoch, d_loss[0], 100*d_loss[1], g_loss))
        
        if epoch % save_interval == 0:
            plot_generated_images(epoch, generator)


train(epochs=10000, batch_size=32, save_interval=50)




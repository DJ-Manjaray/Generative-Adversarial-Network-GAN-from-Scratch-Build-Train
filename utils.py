from __future__ import print_function, division
from keras.datasets import fashion_mnist

import numpy as np
import matplotlib.pyplot as plt

from model import build_generator, build_discriminator
from keras.layers import Input



# Input shape
img_shape = (28,28,1)
channels = 1
latent_dim = 100


(training_data, _), (_, _) = fashion_mnist.load_data()
X_train = training_data / 127.5 - 1.
X_train = np.expand_dims(X_train, axis=3)

def visualize_input(img, ax):
    """    
    The visualize_input function takes an image (img) and an axis object (ax) as input.
    It visualizes the image in a grayscale colormap and annotates each pixel with its value, making it easier to inspect pixel-level details.

    Parameters:
    - img: The input 2D image (assumed to be grayscale).
    - ax: The matplotlib Axes object to use for visualization.
    
    """
     # Display the input image using a grayscale colormap
    ax.imshow(img, cmap='gray')
    
    width, height = img.shape
    # Threshold for determining text color
    thresh = img.max()/2.5

    # Annotate each pixel in the image with its value
    for x in range(width):
        for y in range(height):
            ax.annotate(str(round(img[x][y],2)), xy=(y,x),
                        horizontalalignment='center',
                        verticalalignment='center',
                        color='white' if img[x][y]<thresh else 'black')        

# Create a figure and add a subplot
    
fig = plt.figure(figsize = (12,12))

ax = fig.add_subplot(111)
# Visualize the input image at index 3343 in the training_data array
visualize_input(training_data[3343], ax)


def plot_generated_images(epoch, generator, examples=100, dim=(10, 10),figsize=(10, 10)):
    """
    Generates a grid of images using a trained generator model.
    Visualizes the progress of the generator's image synthesis capabilities at a given epoch.

    Parameters:
    - epoch: The current epoch or iteration number in the training process.
    - generator: The trained generator model.
    - examples: The number of images to generate and display in the grid (default is 100).
    - dim: A tuple specifying the dimensions of the grid (default is (10, 10)).
    - figsize: A tuple specifying the size of the entire figure (default is (10, 10)).
    
    """
    # Generate random noise as input to the generator
    noise = np.random.normal(0, 1, size=[examples, latent_dim])
    # Use the generator to produce synthetic images based on the random noise
    generated_images = generator.predict(noise)
    # Reshape the generated images for visualization
    generated_images = generated_images.reshape(examples, 28, 28)

    # Set up the plotting figure
    plt.figure(figsize=figsize)

    # Plot each generated image in the grid
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_images[i], interpolation='nearest', cmap='gray_r')
        plt.axis('off')

    # Adjust layout for better visualization
    plt.tight_layout()
    # Save the generated image grid with a filename indicating the epoch
    plt.savefig('gan_generated_image_epoch_%d.png' % epoch)



# optimizer = Adam(0.0002, 0.5)

# # Build and compile the discriminator
# discriminator = build_discriminator()
# discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# # Build the generator
# generator = build_generator()

# # The generator takes noise as input and generates imgs
# z = Input(shape=(latent_dim,))
# img = generator(z)

# # For the combined model we will only train the generator
# discriminator.trainable = False

# # The discriminator takes generated images as input and determines validity
# valid = discriminator(img)

# # The combined model  (stacked generator and discriminator)
# # Trains the generator to fool the discriminator
# combined = Model(z, valid)
# combined.compile(loss='binary_crossentropy', optimizer=optimizer)



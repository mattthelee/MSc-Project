from keras.layers import Dense, Conv2D, Flatten, Dropout, Input, Lambda, Reshape, Conv2DTranspose
from keras.models import Model
from keras.losses import mse, binary_crossentropy

from keras import backend as K
from sklearn.model_selection import train_test_split
import numpy as np
import minerl
import pdb
from PIL import Image
from keras.models import model_from_json
import argparse
import random
from keras.utils.vis_utils import plot_model

'''This code was based on the Keras VAE tutorial available here:
Variational autoencoder deconv - Keras Documentation. https://keras.io/examples/variational_autoencoder_deconv/ (accessed 01 Jul2019).
In particular the createModel and sampling functions features code taken from the example.
Adaptations were made for efficiency and to tailor to project requirements
'''

def createModel(latentVDim):
    """ Creates a VAE model that uses latent vector of dimensionality set by latentVDim.
    The encoder network is convolutional and the decoder network is deconvolutional.
    Once trained the encoder can be used to reduce the dimensionality of images while minimising information loss.
    """
    # Create Encoder
    input = Input(shape=(64,64,3))
    model = Conv2D(64,3,activation='relu',strides=2,padding='same')(input)
    model = Conv2D(64,3,activation='relu',strides=2,padding='same')(model)
    model = Conv2D(64,3,activation='relu',strides=2,padding='same')(model)
    model = Conv2D(64,3,activation='relu',strides=2,padding='same')(model)

    shape = K.int_shape(model) # used later to ensure output shape is correct

    # Flatten required to get flat output
    model = Flatten()(model)
    model = Dense(1024,activation='relu')(model)
    model = Dense(latentVDim,activation='relu')(model)
    z_mean = Dense(latentVDim, name='z_mean')(model)
    z_log_var = Dense(latentVDim, name='z_log_var')(model)
    z = Lambda(sampling, output_shape=(latentVDim,), name='z')([z_mean, z_log_var])

    # Conver encoder to model
    encoder = Model(input, [z_mean, z_log_var, z], name='encoder')
    # Save model structure
    plot_model(encoder, to_file='encoder.png', show_shapes=True, show_layer_names=False)
    encoder.summary()

    # Create decoder
    decoderinput = Input(shape=(latentVDim,))
    decodermodel = Dense(1024,activation='relu')(decoderinput)

    decodermodel = Dense(shape[1] * shape[2] * shape[3], activation='relu')(decodermodel)
    decodermodel = Reshape((shape[1], shape[2], shape[3]))(decodermodel)
    decodermodel = Conv2DTranspose(64,3,activation='relu',strides=2,padding='same')(decodermodel)
    decodermodel = Conv2DTranspose(64,3,activation='relu',strides=2,padding='same')(decodermodel)
    decodermodel = Conv2DTranspose(64,3,activation='relu',strides=2,padding='same')(decodermodel)
    output = Conv2DTranspose(3,3,activation='sigmoid',strides=2,padding='same')(decodermodel)
    decoder = Model(decoderinput, output, name='decoder')
    decoder.summary()
    plot_model(decoder, to_file='decoder.png', show_shapes=True, show_layer_names=False)

    # Create VAE
    output = decoder(encoder(input)[2])
    vae = Model(input, output, name='vae')

    # Use either MSE or BCE loss
    #reconstructionLoss = binary_crossentropy(K.flatten(input),K.flatten(output))
    reconstructionLoss = mse(K.flatten(input),K.flatten(output))

    # Use kielbak-lieber loss
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    # combine kullback lieber loss with reconstructionLoss
    vae_loss = K.mean(reconstructionLoss + kl_loss)
    vae.add_loss(reconstructionLoss)
    vae.compile(optimizer='rmsprop')
    vae.summary()
    return vae

def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.

    # Arguments
        args (tensor): mean and log of variance of Q(z|X)

    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

def getData(limit = 10000, images = [], env = 'MineRLNavigateDense-v0' ):
    """
    Extract images from given environment. Limit prevents out-of-memory errors.
    Images are appended to the passed in images parameter to allow construction of large dataset from multiple envs
    """
    count = 0
    initialisations = 0
    # Initialise datasource iterable
    dataSource = minerl.data.make(env)
    # Get max 10 images from each trajectory to ensure diversity
    for obs, action, reward, next_state, done in dataSource.sarsd_iter(num_epochs = 1, max_sequence_len =  10):
        initialisations += 1
        if limit != -1 and count >= limit:
            break

        # Divide images by 255 to match utils image processing and append to images list
        for step in range(len(obs['pov'])):
            if limit != -1 and count >= limit:
                break
            count += 1
            image = obs['pov'][step]
            images.append(image/255.)
            if done[step]:
                break
    print(f"Datasize: {len(images)}, initilisations: {initialisations}")
    return images

def showArray(array):
    """ Shows image using numpy array, used to compare input and ouptput resutls
    """
    img = Image.fromarray(array, 'RGB')
    img.show()
    return img

def showExamples(vaeModel):
    """ Selects random images from dataset and shows both the input and the output
    This allows for qualitative evaluation of information loss """
    images = getData(10000)
    index = random.randint(0,10000)
    inputArray = images[index]*255.
    showArray(inputArray.astype('uint8'))
    predArray = vaeModel.predict(images[index].reshape(-1,64,64,3))[0]*255
    showArray(predArray.astype('uint8'))
    del images

def loadModel(jsonFile = 'vae.json', weightsFile = 'vae.h5'):
    """ Loads vae model from file
    """
    # Loads vae structure from json file
    json_file = open(jsonFile, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    vaeModel = model_from_json(loaded_model_json)
    # load weights into new model
    vaeModel.load_weights(weightsFile)
    return vaeModel


def train(load = False, latentVDim = 256, showImages= False, epochs = 100):
    """ Trains VAE model. If load = True, then simply loads model from file. Can set latent vector size with latentVDim.
    If showImages = True, then it will show example input and output images for qualitative assessment. Epochs sets max number of training epochs
    """
    if load:
        vaeModel = loadModel()
    else:
        images = []
        # Iterate over tasks to ensure data taken from each task
        for env in ['MineRLNavigateDense-v0','MineRLNavigateExtremeDense-v0','MineRLNavigateExtreme-v0','MineRLNavigate-v0','MineRLObtainDiamondDense-v0','MineRLObtainDiamond-v0','MineRLObtainIronPickaxeDense-v0','MineRLObtainIronPickaxe-v0','MineRLTreechop-v0']:
            images = getData(images=images,env=env)

        # Split data into train and test to get validtion scores
        x_train, x_test = train_test_split(images)

        # Reshaping required for Keras input
        x_train = np.reshape(x_train, [-1, 64, 64, 3])
        x_test = np.reshape(x_test, [-1, 64, 64, 3])

        # Get model then fit it
        vaeModel = createModel(latentVDim)
        vaeModel.fit(x_train,epochs=epochs,batch_size= 256) #, validation_data=(X_test,None))

        # Save model weights
        vaeModel.save_weights('vae.h5')
        # Save model structure
        model_json = vaeModel.to_json()
        with open("vae.json", "w") as json_file:
            json_file.write(model_json)

    if showImages:
        showExamples(vaeModel)

    # remove decoder layer and return only the encoder as that's needed for image preprocessing
    encoder = Model(inputs=vaeModel.input, outputs = vaeModel.get_layer('encoder').get_output_at(-1))
    return encoder


if __name__ == '__main__':
    # Take in args
    parser = argparse.ArgumentParser()
    parser.add_argument("-l","--load", help="Load the weights and model from local vae.h5 and vae.json files", action='store_true')
    parser.add_argument("-z","--zdim", help="Dimensions of latent vector", default = 256, type=int)

    args = parser.parse_args()
    train(args.load, args.zdim)

from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.initializers import RandomUniform
from keras.layers import BatchNormalization
from keras.layers import Input, Dense, Flatten, Activation
from keras.layers.convolutional import Conv2D
from keras.models import Model
from keras.optimizers import RMSprop
import numpy as np
import matplotlib.pyplot as plt


# RDF ------------------------------------------------------------------------------------------------------------------


class RDF:
    """ Auto Encoder class.
    """

    def __init__(self, input_shape, latent_dim, learning_rate=0.01):
        self.input_shape = input_shape  # (w,h,c)
        self.latent_dim = latent_dim  # n
        self.input_shared = Input(shape=self.input_shape)

        # RND Target
        encoder_target = self._build_encoder(bn_on=False)
        h_target = encoder_target(self.input_shared)
        h_target = Dense(512, activation='relu',
                         kernel_initializer=RandomUniform(-1, 1), bias_initializer=RandomUniform(-1, 1))(h_target)  # ID
        h_target = Dense(512, activation='relu',
                         kernel_initializer=RandomUniform(-1, 1), bias_initializer=RandomUniform(-1, 1))(h_target)
        h_target = Dense(512, activation='relu',
                         kernel_initializer=RandomUniform(-1, 1), bias_initializer=RandomUniform(-1, 1))(h_target)
        h_target = Dense(512, activation='relu',
                         kernel_initializer=RandomUniform(-1, 1), bias_initializer=RandomUniform(-1, 1))(h_target)
        self.target_output = Dense(latent_dim)(h_target)
        self.target_model = Model(self.input_shared, self.target_output)
        self.target_model.trainable = False

        # RND prediction
        encoder_prediction = self._build_encoder()
        h_prediction = encoder_prediction(self.input_shared)
        h_prediction = Dense(512, activation='relu')(h_prediction)
        self.prediction_output = Dense(latent_dim)(h_prediction)
        self.prediction_model = Model(self.input_shared, self.prediction_output)
        self.prediction_model.trainable = True

        # Compile
        loss = self._build_loss()
        self.prediction_model.compile(optimizer=RMSprop(lr=learning_rate), loss=loss)

        # Compute discriminator score (by means of the distance)
        self.compute_score = K.function([self.input_shared], [loss(None, None)])

    def _build_encoder(self, bn_on=True):
        # Input
        encoder_input = Input(shape=self.input_shape)

        # Encoder
        h = Conv2D(16, 3, strides=2, padding='same')(encoder_input)
        h = Activation('relu')(h)
        h = Conv2D(32, 3, strides=2, padding='same')(h)
        if bn_on: h = BatchNormalization()(h)
        h = Activation('relu')(h)
        h = Conv2D(64, 3, strides=2, padding='same')(h)
        if bn_on: h = BatchNormalization()(h)
        h = Activation('relu')(h)
        encoder_output = Flatten()(h)

        # Model
        return Model(encoder_input, encoder_output)

    def _build_loss(self):
        loss = lambda y, y_pred: K.mean(K.square(self.prediction_output - self.target_output),
                                        axis=-1)  # Dummy function
        return loss

    def train(self, train_dir, val_dir, epochs=10, batch_size=128):
        # Generators
        color_mode = 'rgb' if self.input_shape[-1] > 1 else 'grayscale'
        datagen = ImageDataGenerator(rescale=1. / 255, fill_mode='constant')
        train_gen = datagen.flow_from_directory(train_dir, target_size=self.input_shape[:2], interpolation='bilinear',
                                                color_mode=color_mode, class_mode='categorical', batch_size=batch_size)
        val_gen = datagen.flow_from_directory(val_dir, target_size=self.input_shape[:2], interpolation='bilinear',
                                              color_mode=color_mode, class_mode='categorical', batch_size=batch_size)
        # Fit
        steps_per_epoch = (np.ceil(train_gen.n / batch_size)).astype('int')
        steps_per_val = (np.ceil(val_gen.n / batch_size)).astype('int')
        self.prediction_model.fit_generator(train_gen, validation_data=val_gen, steps_per_epoch=steps_per_epoch,
                                            validation_steps=steps_per_val, epochs=epochs)

        # Save weights
        self.target_model.save_weights('./target_model.h5')
        self.prediction_model.save_weights('./prediction_model.h5')

    def restore_weights(self):
        self.target_model.load_weights('./target_model.h5')
        self.prediction_model.load_weights('./prediction_model.h5')

    def compute_distance(self, dir, vis_id=0):
        color_mode = 'rgb' if self.input_shape[-1] > 1 else 'grayscale'
        datagen = ImageDataGenerator(rescale=1. / 255, fill_mode='constant')
        gen = datagen.flow_from_directory(dir, target_size=self.input_shape[:2], interpolation='bilinear',
                                          color_mode=color_mode, class_mode='categorical', batch_size=25)

        x, _ = gen.next()
        dist = self.compute_score([x])[0]

        f = plt.figure()
        plt.clf()
        for i in range(min(x.shape[0], 25)):
            plt.subplot(5, 5, i + 1)
            plt.imshow(x[i]) if x.shape[-1] > 1 else plt.imshow(np.squeeze(x[i]), cmap='gray')
            d = (dist[i] - np.min(dist)) / (np.max(dist) - np.min(dist))
            plt.title('d_%.2f' % d, fontdict={'fontsize': 4})
            plt.axis('off')
        f.canvas.draw()
        plt.savefig('distance_samples_e%i.eps' % vis_id)
        plt.close()

        print(np.mean(dist))

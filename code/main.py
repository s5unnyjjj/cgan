import tensorflow
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt
import numpy as np

size = 10000
loss_array = np.zeros((size, 2))

class CGAN():
    def __init__(self):
        self.channels = 1
        self.img_shape = (28, 28, self.channels)
        self.num_classes = 10
        self.latent_dim = 100

        optimizer = Adam(0.0002, 0.5)

        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=['binary_crossentropy'], optimizer=optimizer, metrics=['accuracy'])

        self.generator = self.build_generator()

        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,))
        img = self.generator([noise, label])

        self.discriminator.trainable = False

        valid = self.discriminator([img, label])

        self.combined = Model([noise, label], valid)
        self.combined.compile(loss=['binary_crossentropy'], optimizer=optimizer)


    def build_generator(self):
        model = Sequential()

        model.add(Dense(7*7*256, input_dim=self.latent_dim))
        model.add(BatchNormalization(momentum=0.9))
        model.add(ReLU())
        model.add(Reshape((7, 7, 256)))
        model.add(Dropout(0.4))
        model.add(UpSampling2D())

        model.add(Conv2DTranspose(128, kernel_size=(5, 5), padding='same'))
        model.add(BatchNormalization(momentum=0.9))
        model.add(ReLU())
        model.add(UpSampling2D())

        model.add(Conv2DTranspose(64, kernel_size=(5, 5), padding='same'))
        model.add(BatchNormalization(momentum=0.9))
        model.add(ReLU())

        model.add(Conv2DTranspose(32, kernel_size=(5, 5), padding='same'))
        model.add(BatchNormalization(momentum=0.9))
        model.add(ReLU())

        model.add(Conv2DTranspose(1, kernel_size=(5, 5), padding='same'))
        model.add(Activation('tanh'))
        model.add(Reshape(self.img_shape))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,), dtype='int32')

        label_embedding = Flatten()(Embedding(self.num_classes, self.latent_dim)(label))

        model_input = multiply([noise, label_embedding])

        img = model(model_input)

        return Model([noise, label], img)


    def build_discriminator(self):
        model = Sequential()

        model.add(Conv2D(64, kernel_size=(5, 5), strides=(2, 2), activation=LeakyReLU(alpha=0.2), padding='same', input_shape=self.img_shape))
        model.add(Dropout(0.4))
        model.add(Conv2D(128, kernel_size=(5, 5), strides=(2, 2), activation=LeakyReLU(alpha=0.2), padding='same'))
        model.add(Dropout(0.4))
        model.add(Conv2D(256, kernel_size=(5, 5), strides=(2, 2), activation=LeakyReLU(alpha=0.2), padding='same'))
        model.add(Dropout(0.4))
        model.add(Conv2D(512, kernel_size=(5, 5), strides=(1, 1), activation=LeakyReLU(alpha=0.2), padding='same'))
        model.add(Dropout(0.4))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        img = Input(shape=self.img_shape)
        label = Input(shape=(1,), dtype='int32')

        label_embedding = Flatten()(Embedding(self.num_classes, np.prod(self.img_shape))(label))
        flat_img = Flatten()(img)

        model_input = multiply([flat_img, label_embedding])

        validity = model(tensorflow.reshape(model_input, [-1, 28, 28, 1]))

        return Model([img, label], validity)


    def train(self, epochs, batch_size=128, sample_interval=50):

        (X_train, y_train), (_, _) = mnist.load_data()

        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)
        y_train = y_train.reshape(-1, 1)

        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs, labels = X_train[idx], y_train[idx]

            noise = np.random.normal(0, 1, (batch_size, 100))

            gen_imgs = self.generator.predict([noise, labels])

            d_loss_real = self.discriminator.train_on_batch([imgs, labels], valid)
            d_loss_fake = self.discriminator.train_on_batch([gen_imgs, labels], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            sampled_labels = np.random.randint(0, 10, batch_size).reshape(-1, 1)
            g_loss = self.combined.train_on_batch([noise, sampled_labels], valid)

            print ("%d [D loss: %f] [G loss: %f]" % (epoch, d_loss[0], g_loss))
            loss_array[epoch][0] = "%0.2f" % g_loss
            loss_array[epoch][1] = "%0.2f" % d_loss[0]

            if epoch % sample_interval == 0:
                self.sample_images(epoch)


    def sample_images(self, epoch):
        r, c = 2, 5
        noise = np.random.normal(0, 1, (r * c, 100))
        sampled_labels = np.arange(0, 10).reshape(-1, 1)

        gen_imgs = self.generator.predict([noise, sampled_labels])

        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt,:,:,0], cmap='gray')
                axs[i,j].set_title("Digit: %d" % sampled_labels[cnt])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    cgan = CGAN()
    cgan.train(epochs=size, batch_size=32, sample_interval=200)

    X = range(size)
    Y_G = [loss_array[value][0] for value in X]
    Y_D = [loss_array[value][1] for value in X]

    fig = plt.figure()
    fig.set_size_inches(10, 5)
    graph_1 = fig.add_subplot(1, 2, 1)

    graph_1.plot(X, Y_G, c="b")
    plt.yticks(np.arange(0, 8, 0.5))

    graph_2 = fig.add_subplot(1, 2, 2)
    graph_2.plot(X, Y_D, c="r")
    plt.yticks(np.arange(-0.05, 1.8, 0.1))

    plt.show()

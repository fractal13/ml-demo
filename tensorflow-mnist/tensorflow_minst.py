import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras as keras
import numpy as np

class MNIST:

    def __init__(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = self.load_data()
        return

    def get_train_data_shape(self):
        return self.x_train.shape, self.y_train.shape

    def get_test_data_shape(self):
        return self.x_test.shape, self.y_test.shape

    def get_train_image(self,image_index):
        if image_index >= 0 and image_index < len(self.x_train):
            return self.x_train[image_index]
        return None

    def show_train_image(self,image_index):
        if image_index >= 0 and image_index < len(self.x_train):
            plt.figure(image_index)
            plt.imshow(self.x_train[image_index], cmap='Greys')
        return

    def show_train_images(self,image_index_list, rows):
        cols = int(len(image_index_list) / rows)
        f, axarr = plt.subplots(rows,cols)
        i = 0
        while i < len(image_index_list):
            r = int(i / cols)
            c = int(i % cols)
            axarr[r,c].imshow(self.x_train[image_index_list[i]], cmap='Greys')
            i += 1
        plt.show()
        return

    def get_train_value(self,image_index):
        if image_index >= 0 and image_index < len(self.x_train):
            return self.y_train[image_index]
        return -1

    def load_data(self):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        return (x_train, y_train), (x_test, y_test)

    def prepare_data(self):
        # add the 4th dimension (of size 1) to appease the network structure
        self.x_train = self.x_train.reshape(self.x_train.shape[0], self.x_train.shape[1], self.x_train.shape[2], 1)
        self.x_test = self.x_test.reshape(self.x_test.shape[0], self.x_test.shape[1], self.x_test.shape[2], 1)

        # convert from integer to floating point
        self.x_train = self.x_train.astype('float32')
        self.x_test = self.x_test.astype('float32')

        # normalize to 0.0-1.0 instead of 0-255.
        self.x_train /= 255.0
        self.x_test /= 255.0
        return

    def build_network(self):
        input_shape = (self.x_train.shape[1],self.x_train.shape[2],self.x_train.shape[3])
        self.model = keras.models.Sequential()
        self.model.add(keras.layers.Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
        self.model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        self.model.add(keras.layers.Flatten()) # Flattening the 2D arrays for fully connected layers
        self.model.add(keras.layers.Dense(128, activation=tf.nn.relu))
        self.model.add(keras.layers.Dropout(0.2))
        self.model.add(keras.layers.Dense(10,activation=tf.nn.softmax))
        return

    def compile_network(self):
        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])
        return

    def train_network(self):
        self.model.fit(x=self.x_train, y=self.y_train, epochs=5)
        return

    def evaluate_network(self):
        return self.model.evaluate(self.x_test, self.y_test)

    def save_network(self,filename):
        return self.model.save(filename)

    def load_network(self,filename):
        self.model = keras.models.load_model(filename)
        return
    
def main():
    np.set_printoptions(linewidth=120)
    mnist = MNIST()

    # reshape and normalize the data
    prepare_data = True
    # create network
    create_network = False
    # train network
    train_network = False
    # evaluate network
    evaluate_network = True
    # save network
    save_network = False
    # load network
    load_network = True

    # images and answers
    show_training_data = False
    # data as array
    show_one_image_data = False
    # numpy data structure
    show_data_shape = False

    if prepare_data:
        mnist.prepare_data()

    if create_network:
        mnist.build_network()
        mnist.compile_network()

    if train_network:
        mnist.train_network()

    if save_network:
        mnist.save_network('mnist-model.h5')

    if load_network:
        mnist.load_network('mnist-model.h5')
    
    if evaluate_network:
        loss, accuracy = mnist.evaluate_network()
        print("loss: ", loss)
        print("accuracy: ", accuracy)

    
    if show_training_data:
        image_index_list = [51, 3, 16, 50, 53, 47, 13, 52, 46, 45]+[108, 6, 109, 107, 115, 100, 106, 101, 137, 110]
        rows = 2
        for i in image_index_list:
            print(i, mnist.get_train_value(i))
        mnist.show_train_images(image_index_list, rows)

    if show_one_image_data:
        np.set_printoptions(precision=1)

        image_index = 45
        image_data = mnist.get_train_image(image_index)
        if len(image_data.shape) == 3:
            image_data = image_data.reshape(image_data.shape[0],image_data.shape[1])
        print(image_data)

    if show_data_shape:
        print("train:",mnist.get_train_data_shape())
        print("test:",mnist.get_test_data_shape())

        
    return

if __name__ == "__main__":
    main()
    

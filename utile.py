from mp1 import *
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten
import time

def test_show_generate():
    im1 = generate_a_rectangle(10, True)
    plt.imshow(im1.reshape(100,100), cmap='gray')

    plt.show()

    im2 = generate_a_disk(10, True)
    plt.imshow(im2.reshape(100,100), cmap='gray')

    plt.show()

    [im3, v] = generate_a_triangle(20, True)
    plt.imshow(im3.reshape(100,100), cmap='gray')

    plt.show()

def linear_classifier(X_train = None, Y_train= None, nb_data = 500, data_noise = 0.0, data_free_location = False, activation = 'softmax',
                      loss = 'categorical_crossentropy', optimizer = 'adam', epochs = 15):
    if X_train is None and Y_train is None:
        [X_train, Y_train] = generate_dataset_classification(nb_data, data_noise, data_free_location)

    model = Sequential()
    model.add(Dense(3, input_shape=(generate_a_rectangle().shape[0],)))
    model.add(Activation(activation))

    if optimizer == 'sgd':
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9,  nesterov=True)
        model.compile(loss=loss, optimizer='sgd', metrics=['accuracy'])
    else:
        model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

    model.fit(X_train, Y_train, epochs=epochs, batch_size=32)
    return model

def test_linear_classifier(model):
    err = 0

    print("TEST MODEL: ")
    X_test1 = generate_a_disk()
    X_test1 = X_test1.reshape(1, X_test1.shape[0])
    res = model.predict(X_test1)
    print(str(res) + "disk")
    if res[0][1] >= 0.99 and res[0][2] <= 0.1 and res[0][0] <= 0.1:
        print("\t\t- ok")
    else:
        print("\t\t-error disk res:" + str(res))
        err+=1


    X_test2 = generate_a_rectangle()
    X_test2 = X_test2.reshape(1, X_test2.shape[0])
    res = model.predict(X_test2)
    print(str(res) + "rectangle")
    if res[0][0] >= 0.99 and res[0][1] <= 0.1 and res[0][2] <= 0.1:
        print("\t\t- ok")
    else:
        print("\t\t-error rectangle res:" + str(res))
        err+=1

    X_test3, vect = generate_a_triangle()
    X_test3 = X_test3.reshape(1, X_test3.shape[0])
    res = model.predict(X_test3)
    print(str(res) + "triangle")
    if res[0][2] >= 0.99 and res[0][1] <= 0.1 and res[0][0] <= 0.1:
        print("\t\t- ok")
    else:
        print("\t\t-error triangle res:" + str(res))
        err+=1
    return err

def visualize_column(model):
    c1 = model.get_weights()[0][:,0]
    plt.imshow(c1.reshape(100,100), cmap='gray')
    plt.show()

    c2 = model.get_weights()[0][:,1]
    plt.imshow(c2.reshape(100,100), cmap='gray')
    plt.show()

    c3 = model.get_weights()[0][:,2]
    plt.imshow(c3.reshape(100,100), cmap='gray')
    plt.show()


def deep_network(X_train = None, Y_train= None, nb_data = 500, data_noise = 0.0, data_free_location = False,
                 activation1 = 'relu', activation2 = 'softmax', loss = 'categorical_crossentropy', optimizer = 'adam', epochs = 15):
    if X_train is None and Y_train is None:
        [X_train, Y_train] = generate_dataset_classification(nb_data, data_noise, data_free_location)

    X_train = X_train.reshape(X_train.shape[0], 100, 100, 1)
    X_train = X_train.astype('float32')

    model = Sequential()
    model.add(Conv2D(16, (5,5), activation=activation1, input_shape=(100, 100, 1)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(3,activation=activation2))
    if optimizer == 'sgd':
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9,  nesterov=True)
        model.compile(loss=loss ,optimizer='sgd', metrics=['accuracy'])
    else:
        model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

    model.fit(X_train, Y_train, epochs=epochs, batch_size=32)

    return model

def deep_network_regression(X_train = None, Y_train= None, nb_data = 500, data_noise = 0.0, data_free_location = False,
                            activation1 = 'relu', activation2 = 'softmax',loss = 'mean_squared_error', optimizer = 'adam', epochs = 15):
    if X_train is None and Y_train is None:
        [X_train, Y_train] = generate_dataset_classification(nb_data, data_noise, data_free_location)

    X_train = X_train.reshape(X_train.shape[0], 100, 100, 1)
    X_train = X_train.astype('float32')

    model = Sequential()
    model.add(Conv2D(16, (5,5), activation=activation1, input_shape=(100, 100, 1)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(6))
    if optimizer == 'sgd':
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9,  nesterov=True)
        model.compile(loss=loss ,optimizer='sgd', metrics=['accuracy'])
    else:
        model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

    model.fit(X_train, Y_train, epochs=epochs, batch_size=32)

    return model

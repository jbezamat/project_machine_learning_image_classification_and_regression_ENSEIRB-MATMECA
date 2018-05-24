from mp1 import *
import utile
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD

# utile.test_show_generate()

nb_data=300
data_noise=0.0
data_free_location=False

[X_train, Y_train] = generate_dataset_classification(nb_data, data_noise, data_free_location)

model = Sequential()
nb_neurons = 20
model.add(Dense(nb_neurons, input_shape=(generate_a_rectangle().shape[0],)))
model.add(Activation('relu'))
model.add(Dense(1))

sgd = SGD(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True)
model.compile(loss='mean_squared_error', optimizer = sgd)

model.fit(X_train, Y_train, epochs = 10, batch_size = 32)

print("coucou\n")
X_test = generate_a_rectangle()
X_test = X_test.reshape(1, X_test.shape[0])
print(model.predict(X_test))

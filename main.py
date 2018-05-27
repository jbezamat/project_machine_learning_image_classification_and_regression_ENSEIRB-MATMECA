from mp1 import *
import utile


# utile.test_show_generate()

#############################
### Simple Classification ###
#############################

[X_train, Y_train] = generate_dataset_classification(600, 0.0, False)
[X_test, Y_test] = generate_test_set_classification(300, 0.0, False)

#SGD
print("\n\n\nLinear Classifier with SGD activation\n")
model = utile.linear_classifier(X_train, Y_train, optimizer = 'sgd', epochs = 1000)
utile.test_linear_classifier(model)
print("model evaluate: ", model.evaluate(X_test, Y_test),"\n")

#ADAM
print("\n\n\nLinear Classifier with ADAM activation\n")
model = utile.linear_classifier(X_train, Y_train, optimizer = 'adam', epochs = 1000)
utile.test_linear_classifier(model)
print("model evaluate: ", model.evaluate(X_test, Y_test),"\n")

utile.visualize_column(model)

###############################################
### A More Difficult Classification Problem ###
###############################################

[X_train, Y_train] = generate_dataset_classification(600, 20, True)
[X_test, Y_test] = generate_test_set_classification(300)

#Linear Classifier
print("\n\n\nLinear Classifier\n")
model = utile.linear_classifier(X_train, Y_train, optimizer = 'adam', epochs = 1000)
print("model evaluate: ", model.evaluate(X_test, Y_test),"\n")


X_test = X_test.reshape(X_test.shape[0], 100, 100, 1)
X_test = X_test.astype('float32')

#Deep Network
print("\n\n\nDeep Network\n")
model = utile.deep_network(X_train, Y_train, optimizer = 'adam', epochs = 1000)
print("model evaluate: ", model.evaluate(X_test, Y_test),"\n")

#############################
### A Regression Problem ###
############################

[X_train, Y_train] = generate_dataset_regression(600, 20)
visualize_prediction(X_train[0], Y_train[0])

[X_test, Y_test] = generate_test_set_regression()
X_test = X_test.reshape(X_test.shape[0], 100, 100, 1)
X_test = X_test.astype('float32')

model = utile.deep_network_regression(X_train, Y_train, optimizer = 'adam', epochs = 1000)
print("model evaluate: ", model.evaluate(X_test, Y_test),"\n")

for i in range(0, 5):
    visualize_prediction(X_test[i], model.predict(X_test[i].reshape(1, 100, 100, 1)))

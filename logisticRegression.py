# numpy is the only needed library to create Logistic regression from scratch
import numpy as np

# sigmoid (logistic) function
# the function that responsible for converting the continues output into 0/1 class
def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s

# initialize the weights and the bias parameters for the logistic regression
def initialize_parameters(dim):
    w = np.zeros((dim, 1))
    b = 0
    return w, b

# execute the forward propagation and backward propagation to calculate the activation and the gradients respectively
def propagate(w, b, X, Y):
    # forward propagation
    Z = np.dot(w.T, X) + b
    A = sigmoid(Z)
    # compute the cost function
    cost = (-1/X.shape[1])*np.sum(np.multiply(Y,np.log(A))+np.multiply((1-Y),np.log(1-A)))
    # backward propagation
    dz = A - Y
    dw = np.dot(X, dz.T) / X.shape[1]
    db = np.sum(dz)

    return dw, db, cost

# the gradient descent in the function that is responsible for improving the weights and bisa values using the calculated gradients
def gradient_descent(w, b, X, Y, num_iterations, learning_rate):
    costs = [] # to save the cost during the process and show the improvement
    for i in range(num_iterations):
        dw, db, cost = propagate(w, b, X, Y)
        # change the parameters values using the gradients and the pre defined learning rate
        w = w - (learning_rate * dw)
        b = b - (learning_rate * db)
        if i % 100 == 0:
            costs.append(cost)
            print("Cost after iteration %i: %f" % (i, cost))

    return w, b, costs

# predict function for calculating the final prediction for training and testing sets using the final parameters
def predict(w, b, X):
    # create empty array to save the results into
    y_predicted = np.zeros((1, X.shape[1]))
    w = w.reshape(X.shape[0], 1)
    # calculate the prediction using linear function followed by logistic function(sigmoid)
    Z = np.dot(w.T, X) + b
    A = sigmoid(Z)
    # convert to 0/1 classes based on the threshold 0.5 in this case
    for i in range(A.shape[1]):
        if A[0,i] >= 0.5:
            y_predicted[0, i] = 1
        else:
            y_predicted[0, i] = 0
    return y_predicted

# logistic regression function
def logistic_regression(X_train, Y_train, X_test, Y_test, num_iterations = 3000, learning_rate = 0.01):
    # initialize weights and bias
    w, b = initialize_parameters(X_train.shape[0])
    # calculate the gradient and use them to have the final parameters
    w, b, costs = gradient_descent(w, b, X_train, Y_train, num_iterations, learning_rate)
    # use the final parameters to calculate the predictions
    y_predicted_train = predict(w, b, X_train)
    y_predicted_test = predict(w, b, X_test)
    # print out the results
    print("train accuracy: {} %".format(100 - np.mean(np.abs(y_predicted_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_predicted_test - Y_test)) * 100))


##########################################################
#################       EXAMPLE      #####################
##########################################################
# import the sci-kit learn library to use it datasets
from sklearn import datasets
# import data split model from sci-kit learn
from sklearn.model_selection import train_test_split
# load the dataset
iris = datasets.load_iris()
# divide into features and labels
X = iris.data[:, :2]
y = (iris.target != 0) * 1
# reshape to change from 1 rank array (x,) shape
y = y.reshape(y.shape[0], 1)
# split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# run the logistic regression model
logistic_regression(X_train.T, y_train.T, X_test.T, y_test.T, num_iterations=3000, learning_rate=0.01)

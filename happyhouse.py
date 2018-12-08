import numpy as np 
import matplotlib.pyplot as plt
import scipy
import h5py
import math
import tensorflow as tf 
from PIL import Image
from scipy import ndimage

"""
Creates a list of random minibatches from (X, Y)

Arguments:
X -- input data, of shape (input size, number of examples) (m, Hi, Wi, Ci)
Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples) (m, n_y)
mini_batch_size - size of the mini-batches, integer
seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.

Returns:
mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
"""
def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0) :
    m = X.shape[0]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:,:,:]
    shuffled_Y = Y[permutation,:]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:,:,:]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m,:,:,:]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches


"""
load the data from file
"""
def load_dataset():
    train_dataset = h5py.File('datasets/train_happy.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('datasets/test_happy.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


"""
data pre-process for better using
shape of X_train_orig : (600, 64, 64, 3)
shape of Y_train_orig : (1, 600)
shape of X_test_orig  : (150, 64, 64, 3)
shape of Y_test_orig  : (1, 150) 
"""
def data_preprocess(X_train_orig, Y_train_orig, X_test_orig, Y_test_orig) :
    X_train = X_train_orig / 255
    X_test = X_test_orig / 255
    Y_train = Y_train_orig.T 
    Y_test = Y_test_orig.T

    print("number of training examples: " + str(X_train.shape[0]))
    print("number of test examples: " + str(X_test.shape[0]))
    return X_train, X_test, Y_train, Y_test


"""
create placeholders for X and Y
shape of X_train : (600, 64, 64, 3)
shape of X_test  : (150, 64, 64, 3)
shape of Y_train : (600, 1)
shape of Y_test  : (150, 1)
"""
def create_XY(height, width, channels, dim_y) : 
    X = tf.placeholder(dtype = tf.float32, shape = (None, height, width, channels), name = "X")
    Y = tf.placeholder(dtype = tf.float32, shape = (None, dim_y), name = "Y")
    return X, Y


"""
initialize the parameters used by the neural network
"""
def initialize_para() : 
    
    # initialize the Weight parameters for conv layer
    # in tensorflow, no need to initialize the bias parameters for conv layer since tf does it
    # in tensorflow, no need to initialize all the parameters of fully connected layers
    Weight1 = tf.get_variable(name = "Weight1", shape = (3, 3, 3, 8), dtype = tf.float32,
                              initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    Weight2 = tf.get_variable(name = "Weight2", shape = (3, 3, 8, 16), dtype = tf.float32,
                              initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    Weight3 = tf.get_variable(name = "Weight3", shape = (3, 3, 16, 32), dtype = tf.float32,
                              initializer = tf.contrib.layers.xavier_initializer(seed = 0))

    parameters = {"Weight1" : Weight1,
                  "Weight2" : Weight2,
                  "Weight3" : Weight3}
    
    return parameters


"""
implement the forward propagation
the structure of the conv nn is :
    conv1-->relu-->pool-->conv2-->relu-->pool-->conv3-->relu-->pool-->fc
shape of X : (None, 64, 64, 3)
shape of Z4 : (None, 1) 
"""
def forward_propagation(X, parameters) :

    # get the weight parameters
    Weight1 = parameters["Weight1"]
    Weight2 = parameters["Weight2"]
    Weight3 = parameters["Weight3"]

    # conv layers
    Z1 = tf.nn.conv2d(input = X, filter = Weight1, strides = (1, 1, 1, 1), padding = "SAME")
    A1 = tf.nn.relu(Z1)
    A1_pool = tf.nn.max_pool(value = A1, ksize = (1, 2, 2, 1), strides = (1, 2, 2, 1), padding = "VALID")
    
    Z2 = tf.nn.conv2d(input = A1_pool, filter = Weight2, strides = (1, 1, 1, 1), padding = "SAME")
    A2 = tf.nn.relu(Z2)
    A2_pool = tf.nn.max_pool(value = A2, ksize = (1, 2, 2, 1), strides = (1, 2, 2, 1), padding = "VALID")

    Z3 = tf.nn.conv2d(input = A2_pool, filter = Weight3, strides = (1, 1, 1, 1), padding = "SAME")
    A3 = tf.nn.relu(Z3)
    A3_pool = tf.nn.max_pool(value = A3, ksize = (1, 2, 2, 1), strides = (1, 2, 2, 1), padding = "VALID")

    # FC layers
    A3_pool = tf.contrib.layers.flatten(inputs = A3_pool)

    # apply a FC layer, the parameters of FC is initialized by the tensorflow system
    Z4 = tf.contrib.layers.fully_connected(A3_pool, 1, activation_fn = None)
    return Z4


"""
implement the cost computation
shape of Y : (None, 1)
shape of Z : (None, 1)  
"""
def compute_cost(Z, Y) : 
    print("the shape of Z is " + str(Z.shape))
    print("the shape of Y is " + str(Y.shape))
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Z, labels = Y))
    return cost


"""
implement the happy detect model
"""
def happy_detect_model(X_train, X_test, Y_train, Y_test, learning_rate = 0.002,
                       num_epoches = 200, minibatch_size = 60, print_cost = True) :
    
    tf.reset_default_graph()
    
    # this seed is used to generate the minibatch randomly
    assert(X_train.shape[0] == Y_train.shape[0])
    (num, height, width, channels) = X_train.shape
    (num, dims_y) = Y_train.shape
    num_minibatches = num / minibatch_size
    costs = []
    seed = 1

    # build the computation graph
    X, Y = create_XY(height, width, channels, dims_y)
    parameters = initialize_para()
    Z = forward_propagation(X, parameters)
    cost = compute_cost(Z, Y)
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

    # all variables need to be initialized
    init = tf.global_variables_initializer()

    # run the evalution of specific tensors
    sess = tf.Session()
    sess.run(init)
    
    for i in range(num_epoches) : 
        minibatch_cost = 0
        seed = seed + 1
        minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)
        
        for minibatch in minibatches : 
            (minibatch_X, minibatch_Y) = minibatch
            _, temp_cost = sess.run([optimizer, cost], feed_dict = {X : minibatch_X, Y : minibatch_Y})
            minibatch_cost += (temp_cost / num_minibatches)

        if (print_cost == True) and (i % 5 == 0) : 
                print("After epoch %d %f" %(i, minibatch_cost))
        if (print_cost == True) and (i % 1 == 0) : 
                costs.append(minibatch_cost)
    
    plt.plot(costs)
    plt.ylabel("cost")
    plt.xlabel("epoches")
    plt.title("learning rate = " + str(learning_rate))
    plt.show()

    # predict on the test set
    predict = sess.run(Z, feed_dict = {X : X_test})
    assert(predict.shape == Y_test.shape)
    print(predict.shape)
    predict = np.where(predict > 0, 1, 0)
    accuracy = np.mean((np.equal(predict, Y_test)).astype(np.float))

    # add two new picture of me
    fname_01 = "images/happy.jpg"
    image_01 = np.array(ndimage.imread(fname_01, flatten = False))
    my_image_01 = scipy.misc.imresize(image_01, size = (64,64))

    fname_02 = "images/nohappy.jpg"
    image_02 = np.array(ndimage.imread(fname_02, flatten = False))
    my_image_02 = scipy.misc.imresize(image_02, size = (64,64))
    
    X_new = np.zeros((2, 64, 64, 3))
    X_new[0] = my_image_01
    X_new[1] = my_image_02
    
    predict_me = sess.run(Z, feed_dict = {X : X_new})
    predict_me = np.where(predict_me > 0, 1, 0)

    sess.close()
    return accuracy, predict_me


"""############
begin of main function
#######################"""

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, _ = load_dataset()
X_train, X_test, Y_train, Y_test = data_preprocess(X_train_orig, Y_train_orig, X_test_orig, Y_test_orig)
predict_accuracy, predict_me = happy_detect_model(X_train, X_test, Y_train, Y_test)
print("the predict accuracy is: " + str(predict_accuracy))
print("for the fisrt image, i am happy " + str(predict_me[0]))
print("for the second image, i am happy " + str(predict_me[1]))
print(predict_me)
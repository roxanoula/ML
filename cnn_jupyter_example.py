# example of loading the mnist dataset from keras
from keras.datasets import mnist
from matplotlib import pyplot
from keras.utils import to_categorical

# load dataset
def load_mnist_dataset():
    (trainX, trainY), (testX, testY) = mnist.load_data()
    # reshape dataset to have a single channel (1 color)
    trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
    testX = testX.reshape((testX.shape[0], 28, 28, 1))
    
    # convert the vector trainY and testY in a one hot encoding (an array of 0s and 1)
    trainY = to_categorical(trainY, num_classes=None, dtype='float32')
    testY = to_categorical(testY, num_classes=None, dtype='float32')
    
    return trainX, trainY, testX, testY

# summarize loaded dataset
def summarize_mnist_dataset(trainX, trainY, testX, testY): 
    print('Train: X=%s, y=%s' % (trainX.shape, trainY.shape))
    print('Test: X=%s, y=%s' % (testX.shape, testY.shape))

    # plot first few images
    for i in range(9):
        # define subplot
        pyplot.subplot(330 + 1 + i)
        # plot raw pixel data
        pyplot.imshow(trainX[i], cmap=pyplot.get_cmap('gray'))

    # show the figure
    pyplot.show()

trainX, trainY, testX, testY = load_mnist_dataset()
summarize_mnist_dataset(trainX, trainY, testX, testY)

# scale pixels
def prep_pixels(train, test):
    # Modifying the values of each pixel such that they range from 0 to 1 will improve the rate at which our model learns.
    
    #convert into to float
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')
    # normalize to range 0-1
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0

    return train_norm, test_norm

train_norm, test_norm = prep_pixels(trainX, trainY)

# define cnn model
def define_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    # compile model
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

from keras.models import Sequential
from sklearn.model_selection import KFold
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.optimizers import SGD

# evaluate a model using k-fold cross-validation
def evaluate_model(dataX, dataY, n_folds=5):
    scores, histories = list(), list()
    # prepare cross validation
    kfold = KFold(n_folds, shuffle=True, random_state=1)
    # enumerate splits
    for train_ix, test_ix in kfold.split(dataX):
        # define model
        model = define_model()
        # select rows for train and test
        trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]
        # fit model
        # The batch size must match the number of images going into our first convolutional layer.
        history = model.fit(trainX, trainY, epochs=5, batch_size=32, validation_data=(testX, testY), verbose=0)
        # evaluate model
        loss , acc = model.evaluate(testX, testY, verbose=0)
        print('Test loss', loss)
        print('Test accuracy', acc)
        print('> %.3f' % (acc * 100.0))
        
        # stores scores
        scores.append(acc)
        histories.append(history)
        
        return scores, histories

scores, histories = evaluate_model(train_norm, test_norm)
# from neuralnetwork import make_prediction, load_parameters
# from tensorflow.keras.datasets import mnist
import pandas as pd
import numpy as np

if __name__ == "__main__":
    '''
    Take sample of MNIST and test NN
    '''
    # (X_train, y_train), _ = mnist.load_data()
    # test_img = X_train[0].reshape(784, 1) / 255.0
    # W1, b1, W2, b2 = load_parameters()
    # prediction, probs = make_prediction(test_img, W1, b1, W2, b2)
    # print(f"Known {y_train[0]} classified as: {prediction[0]}")

    data = pd.read_csv('./dataset/train.csv')
    data.head()

    data = np.array(data)
    print(data)
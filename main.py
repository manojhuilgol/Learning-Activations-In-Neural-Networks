from sklearn import datasets, preprocessing
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from neuralnetwork import NeuralNetwork
import numpy as np
import pandas as pd


def main():
    # This is used to check which dataset is used to run the experiments
    hyperparams = {
        "MNIST": {"input_dims": 784, "hidden_dims": 25, "output_dims": 10, "lr_rate": 0.001, "epochs": 3000},
        "IRIS": {"input_dims": 4, "hidden_dims": 16, "output_dims": 3, "lr_rate": 0.001, "epochs": 30000},
        "BANK_NOTE": {"input_dims": 4, "hidden_dims": 16, "output_dims": 2, "lr_rate": 0.001, "epochs": 1000}
    }

    # Two cases as mentioned in the report.
    case = {"case_1": {"mean": 1, "std_dev": 0.5},
            "case_2": {"mean": -2, "std_dev": 1}}

    # Training IRIS DATASET
    iris = datasets.load_iris()
    x, y = iris.data, iris.target

    # Normalize features
    x_norm = preprocessing.normalize(x)
    y = to_categorical(y)
    permutation = np.random.permutation(y.shape[0])
    x_norm = x_norm[permutation]
    y = y[permutation]

    x_train, x_val = x_norm[:120], x_norm[120:]
    y_train, y_val = y[:120], y[120:]

    model = NeuralNetwork(input_dims=hyperparams["IRIS"]["input_dims"], hidden_dims=hyperparams["IRIS"]["hidden_dims"],
                          output_dims=hyperparams["IRIS"]["output_dims"], lr_rate=hyperparams["IRIS"]["lr_rate"])
    model.initialize_params(trainX=x_train, trainY=y_train, case=case["case_1"])

    for epochs in range(hyperparams["IRIS"]["epochs"]):
        model.forward_propogation()
        model.backward_propogation()

        if epochs % 200 == 0:
            print("Epochs : {}".format(epochs))

    # Save the metrics and parameters as a pickle file
    model.pickle_parameters(case["case_1"])

    # Training BANK NOTE DATASET
    download_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"
    banknote_df = pd.read_csv(download_url, names=["wavelet_transformed_variance", "wavelet_transformed_skewness",
                                                   "wavelet_transformed_curtosis",
                                                   "image_entropy", "counterfeit"])
    x = banknote_df[["wavelet_transformed_variance", "wavelet_transformed_skewness", "wavelet_transformed_curtosis",
                     "image_entropy"]].values
    y = banknote_df.counterfeit.values

    # Normalize features
    x_norm = preprocessing.normalize(x)
    y = to_categorical(y)

    permutation = np.random.permutation(y.shape[0])

    x_norm = x_norm[permutation]
    y = y[permutation]

    x_train, x_val = x_norm[:1000], x_norm[1000:]
    y_train, y_val = y[:1000], y[1000:]

    model = NeuralNetwork(input_dims=hyperparams["BANK_NOTE"]["input_dims"],
                          hidden_dims=hyperparams["BANK_NOTE"]["hidden_dims"],
                          output_dims=hyperparams["BANK_NOTE"]["output_dims"],
                          lr_rate=hyperparams["BANK_NOTE"]["lr_rate"])
    model.initialize_params(trainX=x_train, trainY=y_train, case=case["case_1"])

    for epochs in range(hyperparams["BANK_NOTE"]["epochs"]):
        model.forward_propogation()
        model.backward_propogation()

        if epochs % 200 == 0:
            print("Epochs : {}".format(epochs))

    # Save the metrics and parameters as a pickle file
    model.pickle_parameters(case["case_1"])

    # Training MNIST DATASET
    x, y = datasets.fetch_openml('mnist_784', version=1, return_X_y=True)
    x = (x / 255).astype('float32')
    y = to_categorical(y)

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.15, random_state=42)

    model = NeuralNetwork(input_dims=hyperparams["MNIST"]["input_dims"],
                          hidden_dims=hyperparams["MNIST"]["hidden_dims"],
                          output_dims=hyperparams["MNIST"]["output_dims"], lr_rate=hyperparams["MNIST"]["lr_rate"])
    model.initialize_params(trainX=x_train, trainY=y_train, case=case["case_1"])

    for epochs in range(hyperparams["MNIST"]["epochs"]):
        model.forward_propogation()
        model.backward_propogation()

        if epochs % 200 == 0:
            print("Epochs : {}".format(epochs))

    # Save the metrics and parameters as a pickle file
    model.pickle_parameters(case["case_1"])


if __name__ == '__main__':
    main()

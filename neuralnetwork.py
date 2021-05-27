import numpy as np
from utils import ada_act_function, ada_act_function_arc, softmax, get_accuracy, get_cross_entropy_loss
import collections
import os
from sklearn.metrics import f1_score
import pickle


class NeuralNetwork():
    """
    Methods
    -------
    _nested_dict():
            This is used to initialize and store the metric in a nested dictionary format
    _init_weights():
            Initialization of weights before training
    _init_bias():
            Initialize bias
    _save_trained_metrics():
            Based on train or test mode, the computed acc, loss and f1-score are saved in the dict
    pickle_parameters():
            This saves the metrics as a pickled file in the current project path folder
    initialize_params():
            a function call from main function to initialize all the network params
    forward_propogation():
            Forward prop for the neural network weight parameters
    backward_propogation():
            Backward prop and update the gradients with W1, W2, W3, B1, B2, B3, K
    predict():
            This is used for classifying the samples based on Test data

    """
    def __init__(self, input_dims, hidden_dims, output_dims, lr_rate, seed=0):
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.output_dims = output_dims
        self.weight_params = dict()
        self.bias = dict()
        self.K_history = {key: [] for key in ["train", "test"]}
        self.network = self._nested_dict()
        self.lr_rate = lr_rate
        self.acc = {key: [] for key in ["train", "test"]}
        self.loss = {key: [] for key in ["train", "test"]}
        self.f_measure = {key: [] for key in ["train", "test"]}
        self.train_y = None
        self.test_y = None
        self.K = None
        self.seed = seed
        self.initial_weights = dict()
        self.initial_bias = dict()
        self.initial_K = dict()

    def _nested_dict(self):
        return collections.defaultdict(self._nested_dict)

    def _init_weights(self):
        np.random.seed(self.seed)
        self.weight_params["w1"] = np.random.randn(self.input_dims, self.hidden_dims) * np.sqrt(2.0 / self.input_dims)
        np.random.seed(self.seed)
        self.weight_params["w2"] = np.random.randn(self.hidden_dims, self.hidden_dims) * np.sqrt(2.0 / self.hidden_dims)
        np.random.seed(self.seed)
        self.weight_params["w3"] = np.random.randn(self.hidden_dims, self.output_dims) * np.sqrt(2.0 / self.hidden_dims)

        self.initial_weights = {"w1": self.weight_params["w1"], "w2": self.weight_params["w2"],
                                "w3": self.weight_params["w3"]}

    def _init_bias(self):
        np.random.seed(self.seed)
        self.bias["b1"] = np.full((1, self.hidden_dims), 0.1)
        np.random.seed(self.seed)
        self.bias["b2"] = np.full((1, self.hidden_dims), 0.1)
        np.random.seed(self.seed)
        self.bias["b3"] = np.full((1, self.output_dims), 0.1)

        self.initial_bias = {"b1": self.bias["b1"], "b2": self.bias["b2"], "b3": self.bias["b3"]}

    def _save_trained_metrics(self, mode):
        if mode == "train":
            y_labels = self.train_y
        else:
            y_labels = self.test_y

        self.acc[mode].append(get_accuracy(y_labels, self.network[mode]["output_layer"]["a3"]))
        self.loss[mode].append(get_cross_entropy_loss(y_labels, self.network[mode]["output_layer"]["a3"]))
        self.f_measure[mode].append(
            f1_score(y_labels.argmax(1), self.network[mode]["output_layer"]["a3"].argmax(1), average="micro"))
        self.K_history[mode].append(self.K)

    def pickle_parameters(self, case):

        pickle_file_name = "train_and_test_metrics_norm_distribution_mean_{}_std_{}.pkl".format(case["mean"],
                                                                                                case["std_dev"])

        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), pickle_file_name)

        train_and_test_metrics = {"accuracy": self.acc, "loss": self.loss, "f_measure": self.f_measure,
                                  "initial_weights": self.initial_weights,
                                  "initial_bias": self.initial_bias, "initial_K": self.initial_K, "K": self.K_history,
                                  "seed": self.seed}
        print("Saved file path {}".format(path))
        with open(path, 'wb') as handle:
            pickle.dump(train_and_test_metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def initialize_params(self, trainX, trainY, case):
        self._init_weights()
        self._init_bias()
        mean = case["mean"]
        std = case["std_dev"]

        print("K-params: Mean={}, Standard deviation={}".format(mean, std))

        np.random.seed(self.seed)
        self.K = np.random.normal(loc=mean, scale=std, size=(3, 1))
        self.initial_K = {"K": self.K}

        self.network["train"]["input_layer"]["a0"] = trainX
        self.train_y = trainY

    def forward_propogation(self, mode="train"):
        self.network[mode]["input_layer"]["z1"] = np.dot(self.network[mode]["input_layer"]["a0"],
                                                         self.weight_params["w1"]) + self.bias["b1"]
        self.network[mode]["input_layer"]["a1"] = ada_act_function(self.network[mode]["input_layer"]["z1"], self.K)

        self.network[mode]["hidden_layer"]["z2"] = np.dot(self.network[mode]["input_layer"]["a1"],
                                                          self.weight_params["w2"]) + self.bias["b2"]
        self.network[mode]["hidden_layer"]["a2"] = ada_act_function(self.network[mode]["hidden_layer"]["z2"], self.K)

        self.network[mode]["output_layer"]["z3"] = np.dot(self.network[mode]["hidden_layer"]["a2"],
                                                          self.weight_params["w3"]) + self.bias["b3"]
        self.network[mode]["output_layer"]["a3"] = softmax(self.network[mode]["output_layer"]["z3"])

    def backward_propogation(self):
        mode = "train"
        m = self.train_y.shape[0]

        dz3 = self.network[mode]["output_layer"]["a3"] - self.train_y
        dw3 = np.dot((1 / m) * self.network[mode]["hidden_layer"]["a2"].T, dz3)
        db3 = np.mean(dz3, axis=0)
        da2 = np.dot(dz3, self.weight_params["w3"].T)
        dz2 = np.multiply(ada_act_function_arc(self.network[mode]["hidden_layer"]["z2"], self.K), da2)
        dw2 = np.dot((1 / m) * self.network[mode]["input_layer"]["a1"].T, dz2)
        db2 = np.mean(dz2, axis=0)
        da1 = np.dot(dz2, self.weight_params["w2"].T)
        dz1 = np.multiply(ada_act_function(self.network[mode]["input_layer"]["z1"], self.K), da1)
        dw1 = np.dot((1 / m) * self.network[mode]["input_layer"]["a0"].T, dz1)
        db1 = np.mean(dz1, axis=0)
        dK = np.array([[np.mean(da1)], [np.mean(np.multiply(da1, self.network[mode]["input_layer"]["z1"]))],
                       [np.mean(np.multiply(da1, np.square(self.network[mode]["input_layer"]["z1"])))]])

        # Update learnt Weights and Bias
        self.weight_params["w1"] = self.weight_params["w1"] - self.lr_rate * dw1
        self.weight_params["w2"] = self.weight_params["w2"] - self.lr_rate * dw2
        self.weight_params["w3"] = self.weight_params["w3"] - self.lr_rate * dw3
        self.bias["b1"] = self.bias["b1"] - self.lr_rate * db1
        self.bias["b2"] = self.bias["b2"] - self.lr_rate * db2
        self.bias["b3"] = self.bias["b3"] - self.lr_rate * db3
        self.K = self.K - self.lr_rate * dK

        self._save_trained_metrics(mode="train")

    def predict(self, testX, testY):
        mode = "test"

        self.test_y = testY
        self.network[mode]["input_layer"]["a0"] = testX
        self.forward_propogation(mode="test")
        self._save_trained_metrics(mode="test")

        return self.acc[mode], self.loss[mode], self.K_history[mode]

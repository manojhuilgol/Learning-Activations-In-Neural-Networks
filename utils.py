import numpy as np


def ada_act_function(x, K):
    return K[0] + K[1] * x


def ada_act_function_arc(x, K):
    return K[1]


def softmax(x):
    return np.exp(x) / np.exp(x).sum(axis=1, keepdims=True)


def get_accuracy(actual_one_hot, pred_one_hot):
    correct = 0
    actual = actual_one_hot.argmax(1)
    pred = pred_one_hot.argmax(1)

    for i in range(len(actual)):
        if actual[i] == pred[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


def get_cross_entropy_loss(actual, pred):
    n_samples = actual.shape[0]
    return -(1. / n_samples) * np.sum(np.multiply(actual, np.log(pred)))

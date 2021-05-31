import numpy as np

def accuracy(predict, label):
    assert len(predict) == len(label)
    return np.sum(predict == label) / len(label)

def precision(predict, label):
    assert len(predict) == len(label)
    if not predict.dtype == int:
        predict = predict.astype(int)
    return np.sum(predict & label) / np.sum(predict)

def recall(predict, label):
    assert len(predict) == len(label)
    if not predict.dtype == int:
        predict = predict.astype(int)
    return np.sum(predict & label) / np.sum(label)

def f1(predict, label):
    assert len(predict) == len(label)
    if not predict.dtype == int:
        predict = predict.astype(int)
    return 2 * np.sum(predict & label) / (np.sum(label) + np.sum(predict))


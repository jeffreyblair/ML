import numpy as np


def cross_validation(data, num_folds, lambda_seq):
    """
    Cross validation function
    :param data: data
    :param num_folds: number of partitions
    :param lambda_seq: sequence of values
    :return: vector of errors
    """
    data_shf = shuffle_data(data)
    er = []
    for i in range(len(lambda_seq)):
        lambd = lambda_seq[i]
        cv_loss_lmd = 0
        for fold in range(1, num_folds + 1): 
            val_cv, train_cv = split_data(data_shf, num_folds, fold-1)
            model = train_model(train_cv, lambd)
            cv_loss_lmd += loss(val_cv, model)
        er.append(cv_loss_lmd/num_folds)
    return er


def split_data(data, num_folds, fold):
    """
    Partitions and returns selected partition
    :param data: data
    :param num_folds: num folds to make
    :param fold: selected fold
    :return: data_fold, data_rest
    """
    X = data['X']
    t = data['t']

    xJoinT = np.c_[X, t]
    rows = np.size(xJoinT, 0)
    ends = rows%num_folds
    removed = []
    if ends != 0:
        for i in range(ends):
            removed.append(xJoinT[i])
            xJoinT = np.delete(xJoinT, i, axis=0)

    folds = np.split(xJoinT, num_folds)
    for r in removed:
        folds[i] = np.vstack((folds[i], r))

    data_fold = folds.pop(fold)
    data_rest = folds[0]

    for i in range(1, len(folds)):
        data_rest = np.vstack((data_rest, folds[i]))

    last_fold = np.shape(data_fold)[1] - 1
    last_rest = np.shape(data_rest)[1] - 1

    t_fold = data_fold[:,last_fold].flatten()
    t_rest = data_rest[:,last_rest].flatten()

    X_fold = np.delete(data_fold, last_fold, 1)
    X_rest = np.delete(data_rest, last_rest, 1)

    df = {'X': X_fold, 't': t_fold}
    dr = {'X': X_rest, 't': t_rest}

    return df, dr

def predict(data, model):
    """
    Fitted values for data
    :param data: X vals
    :param model: fitted regression param
    :return: fitted y vals
    """

    return np.dot(data['X'], model)


def loss(data, model):
    """
    Computes Squared Error Loss of data on weights in model
    :param data: Data
    :param model: vector of weights
    :return: squared error loss
    """
    w = model
    X = data['X']
    t = data['t']
    n = np.size(t)
    return (np.linalg.norm(t - predict(data, model)) ** 2)/n


def train_model(data, lambd):
    """
    Fits a model to the data based on lambda
    :param data: data to be fitted
    :param lambd: value
    :return: weight vector
    """
    X = data['X']
    t = data['t']
    XTX = np.linalg.inv((np.dot(np.transpose(X), X) + lambd*np.identity(np.shape(X)[1])))
    Xt = np.dot(np.transpose(X), t)
    return np.dot(XTX, Xt)


def shuffle_data(data):
    """
    Shuffle data keeping pairs in place
    :param data: dict{'X': data 't': targets}
    :return: shuffled data
    """
    X = data['X']
    t = data['t']
    xJoinT = np.c_[X, t]
    shuffled = np.random.permutation(xJoinT)
    last_col = np.shape(shuffled)[1] - 1
    t = shuffled[:,last_col]
    X = np.delete(shuffled, last_col, 1)
    return {'X': X, 't': t}


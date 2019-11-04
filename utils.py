import numpy as np

def sigmoid(x):
    """Computes the element wise logistic sigmoid of x.

    Inputs:
        x: Either a row vector or a column vector.
    """
    return 1.0 / (1.0 + np.exp(-x))

def load_train(filePath):
    """Loads training data."""
    with open(filePath, 'rb') as f:
        train_set = np.load(f)
        train_inputs = train_set['train_inputs']
        train_targets = train_set['train_targets']
    return train_inputs, train_targets 

def load_valid(filePath):
    """Loads validation data."""
    with open(filePath, 'rb') as f:
        valid_set = np.load(f)
        valid_inputs = valid_set['valid_inputs']
        valid_targets = valid_set['valid_targets']
    
    return valid_inputs, valid_targets 

def load_test(filePath):
    """Loads test data."""
    with open(filePath, 'rb') as f:
        test_set = np.load(f)
        test_inputs = test_set['test_inputs']
        test_targets = test_set['test_targets']

    return test_inputs, test_targets 

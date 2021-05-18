#!/usr/bin/python
#
# CIS 472/572 - Logistic Regression Template Code
#
# Author: Daniel Lowd <lowd@cs.uoregon.edu>
# Date:   2/9/2018
#
# Please use this code as the template for your solution.
#
# Submit: Yiran Liu
import sys
import re
from math import log
from math import exp
from math import sqrt

MAX_ITERS = 100


# Load data from a file
def read_data(filename):
    f = open(filename, 'r')
    p = re.compile(',')
    data = []
    header = f.readline().strip()
    varnames = p.split(header)
    namehash = {}
    for l in f:
        example = [int(x) for x in p.split(l.strip())]
        x = example[0:-1]
        y = example[-1]
        data.append((x, y))
    return (data, varnames)


# Train a logistic regression model using batch gradient descent
def train_lr(data, eta, l2_reg_weight):
    numvars = len(data[0][0])
    w = [0.0] * numvars
    b = 0.0

    #
    # YOUR CODE HERE
    #
    for _ in range(MAX_ITERS):
        wrong = 0
        cached_w = [0.] * numvars
        cached_b = 0.
        for x, y in data:
            act = predict_lr((w, b), x)
            if (y * act - .5) <= 0:
                wrong += 1
            g = y * _sigmoids(-y * (sum(w_d * x_d for (w_d, x_d) in zip(w, x)) + b))
            cached_w = [w_d + g * x_d for (w_d, x_d) in zip(cached_w, x)]
            cached_b += g
        #Stop if convergence
        if wrong == 0:
            break
        w = [w_d + (-2 * eta * l2_reg_weight) * x_d for (w_d, x_d) in zip(w, w)]
        w = [w_d + eta * x_d for (w_d, x_d) in zip(w, cached_w)]
        b += eta * (cached_b - 2 * l2_reg_weight * b)

    return (w, b)


def _sigmoids(wxb) -> float:
    #           1
    # --------------------
    # 1 + exp(-(w * x + b))
    '''
    y = 1. + exp(-wxb)
    if y == 0.:
        return 0.
    return (1 / y)
    '''
    try:
        y = 1. + exp(-wxb)
        return 1 / y
    except OverflowError:
        return 0.


# Predict the probability of the positive label (y=+1) given the
# attributes, x.
def predict_lr(model, x):
    (w, b) = model

    #
    # YOUR CODE HERE
    #
    wxb = sum(w_d * x_d for (w_d, x_d) in zip(w, x)) + b

    return _sigmoids(wxb) # This is an random probability, fix this according to your solution


# Load train and test data.  Learn model.  Report accuracy.
def main(argv):
    if (len(argv) != 5):
        print('Usage: lr.py <train> <test> <eta> <lambda> <model>')
        sys.exit(2)
    (train, varnames) = read_data(argv[0])
    (test, testvarnames) = read_data(argv[1])
    eta = float(argv[2])
    lam = float(argv[3])
    modelfile = argv[4]

    # Train model
    (w, b) = train_lr(train, eta, lam)

    # Write model file
    f = open(modelfile, "w+")
    f.write('%f\n' % b)
    for i in range(len(w)):
        f.write('%s %f\n' % (varnames[i], w[i]))

    # Make predictions, compute accuracy
    correct = 0
    for (x, y) in test:
        prob = predict_lr((w, b), x)
        print(prob)
        if (prob - 0.5) * y > 0:
            correct += 1
    acc = float(correct) / len(test)
    print("Accuracy: ", acc)


if __name__ == "__main__":
    main(sys.argv[1:])

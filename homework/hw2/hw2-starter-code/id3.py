#!/usr/bin/python3
#
# CIS 472/572 -- Programming Homework #1
#
# Starter code provided by Daniel Lowd, 1/25/2018
#
#Submit: Yiran Liu
#
from os import rename
import sys
import re
from typing import Counter
# Node class for the decision tree
import node

#My import
import math

train = None
varnames = None
test = None
testvarnames = None
root = None


# Helper function computes entropy of Bernoulli distribution with
# parameter p
def entropy(p) -> float:
    # >>>> YOUR CODE GOES HERE <<<<
    # For now, always return "0":
    if p == 0 or p == 1.0: return 0
    return (-p * math.log2(p) - (1 - p) * math.log2(1 - p))


# Compute information gain for a particular split, given the counts
# py_pxi : number of occurences of y=1 with x_i=1 for all i=1 to n
# pxi : number of occurrences of x_i=1
# py : number of ocurrences of y=1
# total : total length of the data
def infogain(py_pxi, pxi, py, total) -> float:
    # >>>> YOUR CODE GOES HERE <<<<
    # For now, always return "0":
    '''
    entropy_main = entropy(py / total) #root E
    if pxi == 0: #positive leaf number / total
        pos = 0
    else:
        pos = float(py_pxi / pxi)
    if total == pxi: #negative number / total
        neg = 0
    else:
        neg = float((py - py_pxi) / (total - pxi))

    entropy_pos = float(pxi / total) * entropy(pos) #positive leaf E
    entropy_neg = float((total - pxi) / total) * entropy(neg) #negative leaf E

    return entropy_main - entropy_pos - entropy_neg #gain = rootE - positiveE - negativeE
    '''
    entropy_main = entropy(py / total) #root E

    if isinstance(pxi, int): pxi = [pxi, total - pxi]
    if isinstance(py_pxi, int): py_pxi = [py_pxi, py - py_pxi]

    entropy_leaf = 0
    for y, n in zip(py_pxi, pxi):
        if n == 0: continue
        entropy_leaf += n / total * entropy(y / n)
    
    return entropy_main - entropy_leaf


# OTHER SUGGESTED HELPER FUNCTIONS:
# - collect counts for each variable value with each class label
# - find the best variable to split on, according to mutual information
# - partition data based on a given variable


# Load data from a file
def read_data(filename):
    f = open(filename, 'r')
    p = re.compile(',')
    data = []
    header = f.readline().strip()
    varnames = p.split(header)
    namehash = {}
    for l in f:
        data.append([int(x) for x in p.split(l.strip())])
    return (data, varnames)


# Saves the model to a file.  Most of the work here is done in the
# node class.  This should work as-is with no changes needed.
def print_model(root, modelfile):
    f = open(modelfile, 'w+')
    root.write(f, 0)


#to find the most common value's leaf
def _common_val_leaf(varname, data):
    count = Counter([x[-1] for x in data])
    #most_common(top n)
    #[('top n name', count), ...]
    return node.Leaf(varname, count.most_common(1)[0][0])
    #return leaf(name, leaf count)


# Build tree in a top-down manner, selecting splits until we hit a
# pure leaf or all splits look bad.
def build_tree(data, varnames, remains = None):
    # >>>> YOUR CODE GOES HERE <<<<
    # For now, always return a leaf predicting "1":
    if remains is None: remains = list(range(len(varnames) - 1))
    
    #check if all data in this set are the same lable
    this_y = data[0][-1]
    if all([this_y == x[-1] for x in data]): return node.Leaf(varnames, this_y)

    #if no remain varnames
    if not remains: _common_val_leaf(varnames, data)

    best_gain = 0.0
    best_var = None
    best_x = None
    #total, pxi, py, py_pxi
    total = len(data)
    py = sum(1 for x in data if x[-1] == 1)
    #split
    for id in remains:
        data_x = [[x for x in data if x[id] == i]for i in range(2)]
        pxi = [len(grp) for grp in data_x]
        if any(l == 0 for l in pxi): continue
        py_pxi = [sum(1 for x in data_x[i] if x[-1] == i) for i in range(len(data_x))]

        #update
        this_info_gain = infogain(py_pxi, pxi, py, total)
        if this_info_gain > best_gain:
            best_gain = this_info_gain
            best_var = id
            best_x = data_x
        
    #if not worth to split further
    if best_gain == 0: return _common_val_leaf(varnames, data)

    #else, split to left and right
    remains = [id for id in remains if id != best_var]
    return node.Split(varnames, best_var, build_tree(best_x[0], varnames, remains), build_tree(best_x[1], varnames, remains))


# "varnames" is a list of names, one for each variable
# "train" and "test" are lists of examples.
# Each example is a list of attribute values, where the last element in
# the list is the class value.
def loadAndTrain(trainS, testS, modelS):
    global train
    global varnames
    global test
    global testvarnames
    global root
    (train, varnames) = read_data(trainS)
    (test, testvarnames) = read_data(testS)
    modelfile = modelS

    # build_tree is the main function you'll have to implement, along with
    # any helper functions needed.  It should return the root node of the
    # decision tree.
    root = build_tree(train, varnames)
    print_model(root, modelfile)


def runTest():
    correct = 0
    # The position of the class label is the last element in the list.
    yi = len(test[0]) - 1
    for x in test:
        # Classification is done recursively by the node class.
        # This should work as-is.
        pred = root.classify(x)
        if pred == x[yi]:
            correct += 1
    acc = float(correct) / len(test)
    return acc


# Load train and test data.  Learn model.  Report accuracy.
def main(argv):
    if (len(argv) != 3):
        print('Usage: python3 id3.py <train> <test> <model>')
        sys.exit(2)
    loadAndTrain(argv[0], argv[1], argv[2])

    acc = runTest()
    print("Accuracy: ", acc)


if __name__ == "__main__":
    main(sys.argv[1:])

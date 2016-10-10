import numpy as np
import os
import sys
from sklearn import preprocessing
from sklearn.cross_validation import StratifiedKFold
from six.moves import cPickle as pickle
from sklearn import svm

# csv file.
TRAIN_DATAFILE ='./original_data/train.csv'
# test file.
TEST_DATAFILE ='./original_data/train.csv'
# pickle file(python object dumped into the file)
OUTPUT_DATAFILE ='./preprocessed_data/train'

# Read the data and normalize all the data so that the mean is zero.
def read_data(filename):
  data = np.genfromtxt(TRAIN_DATAFILE, delimiter=',', skip_header=1)
  return data[:,1:-1], data[:, -1]

def read_testdata(filename):
  data = np.genfromtxt(TEST_DATAFILE, delimiter=',', skip_header=1)
  #return data[:,1:], data[:, 0]
  return data[:,1:-1], data[:, 0]

X, y = read_data(TRAIN_DATAFILE)
#scaler = preprocessing.StandardScaler().fit(X)
#X = scaler.transform(X)
X = preprocessing.scale(X)
print "preprocessing is done..."
clf = svm.SVC(kernel='poly',probability=True, max_iter=10, class_weight="balanced")
clf.fit(X, y)
print "svm fitting is done..."
test_X, ids = read_testdata(TEST_DATAFILE)
#test_X = scaler.transform(test_X)
test_X = X

# We test the model over the training data itself.
prediction = clf.predict_proba(test_X)
# Calculate average probablity for 1.
sum_probability = 0.0
one_cnt = 0
for i in range(y.shape[0]):
  if y[i] != 0:
    sum_probability += prediction[i][1]
    one_cnt += 1

print "avg probabliities for one:%f" % (sum_probability/one_cnt)

# Print output.
#print "ID,TARGET"
#for i in range(len(ids)):
#  print "%d,%d,%f" % (ids[i], y[i], prediction[i][1])


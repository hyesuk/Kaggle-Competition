import numpy as np
import os
import sys
import copy
import math
import random
import xgboost as xgb
from sklearn.grid_search import GridSearchCV,RandomizedSearchCV
from sklearn import svm
from sklearn import cross_validation
from sklearn import preprocessing
from sklearn.cross_validation import StratifiedKFold
from six.moves import cPickle as pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from unbalanced_dataset import SMOTE, SMOTEENN, UnderSampler
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

IS_DEBUG = 'debug' in sys.argv
IS_LIGHT_TRAIN = 'light' in sys.argv
IS_RESULT_GEN = 'test' in sys.argv

DEBUG_TRAIN_DATAFILE ='./original_data/train_artificial.csv'
DEBUG_TEST_DATAFILE ='./original_data/test_artificial.csv'

TRAIN_DATAFILE ='./original_data/train.csv'
LIGHT_TRAIN_DATAFILE ='./original_data/light_train.csv'
TEST_DATAFILE ='./original_data/test.csv'

# pickle file(python object dumped into the file)
PICKLED_FILE ='pickled_file.p'

def log(s):
  sys.stderr.write(s)
  sys.stderr.write("\n")

class Scaler:
  def __init__(self, X, y):
    self.scaler = preprocessing.StandardScaler().fit(X)
    X = self.scaler.transform(X)
    self.selector = VarianceThreshold(threshold=0.001).fit(X)
    X = self.selector.transform(X)
    self.pca = PCA(n_components=25).fit(X)

  def transform(self, X):
    log("before scaling: %s" % str(X.shape))
    X = self.scaler.transform(X)
    X = self.selector.transform(X)
    X_projected = self.pca.transform(X)
    X = np.append(X, X_projected, axis=1)
    zeros = (X == 0).astype(int).sum(axis=1)
    zeros = zeros.reshape((X.shape[0], 1))
    X = np.append(X, zeros, axis=1)
    #X = self.picker.transform(X)
    log("after scaling: %s" % str(X.shape))
    return X
    
# Read the data and normalize all the data so that the mean is zero.
def read_data():
  filename = ''
  if IS_DEBUG:
    filename = DEBUG_TRAIN_DATAFILE
  elif IS_LIGHT_TRAIN:
    filename = LIGHT_TRAIN_DATAFILE
  else:
    filename = TRAIN_DATAFILE
  data = np.genfromtxt(filename, delimiter=',', skip_header=1)
  X = data[:,1:-1]
  y = data[:, -1]
  np.place(X, X == -999999, 2)
  return X, y

def read_testdata():
  filename = TEST_DATAFILE if not IS_DEBUG else DEBUG_TEST_DATAFILE
  data = np.genfromtxt(filename, delimiter=',', skip_header=1)
  return data[:,1:], data[:, 0]

def randomize(X, y):
  # make a column vector.
  y = np.array([y[:]]).T
  Xy = np.concatenate((X, y), 1)
  np.random.shuffle(Xy)
  return Xy[:,0:-1], Xy[:,-1]

def update_param(P):
  log("current parametervalue:%s" % str(P))
  log("update parameter for the next run")
  while True:
    try:
      log("type param name or type 'run' to finish")
      line  = sys.stdin.readline().strip()
      if line == "run":
        return
      if line in P:
        log("current value:" + str(P[line]))
      else:
        log("creating new field")
      value  = sys.stdin.readline().strip()
      value = eval(value)
      if value == None:
        del GRID_PARAM[line]
      else:
        GRID_PARAM[line] = value
    except:
      log("error occured..")
      pass

P = {}
# Fixed ones from parameter tuning
P['max_depth']=3
P['n_estimators']= 700
P['min_child_weight']=18
P['subsample']=0.4
P['colsample_bytree']=0.4
P['colsample_bylevel']=0.7
P['reg_alpha']=100
P['reg_lambda']=0.01
P['scale_pos_weight']=24.0
P['learning_rate']=0.07
P['max_delta_step']=4

P['gamma']=0.0
P['base_score']=0.5
P['silent']=True
P['objective']="binary:logistic"
P['nthread']=1
P['seed']=0
P['missing']=None

log("START:preprocessing: (read data, scaling, remove constant columns")
X, y = read_data()
scaler = Scaler(X, y)
X = scaler.transform(X)


GRID_PARAMS = {
  'max_depth': [3],
  'min_child_weight': [15,21,24,33],
  'subsample': [0.5,0.7,0.8],
  'colsample_bytree': [0.6, 0.8],
  'colsample_bylevel': [0.5, 0.6, 0.8],
  'reg_alpha': [90,110,120],
  'reg_lambda': [0.1, 1, 5],
  'learning_rate': [0.05],
  'max_delta_step': [2, 4, 6]
}

'''
GRID_PARAMS = {
  'max_depth': range(3, 5),
  'min_child_weight': range(18, 24, 3),
  'subsample': [i/10.0 for i in range(5, 8)],
  'colsample_bytree': [i/10.0 for i in range(8, 10)],
  'colsample_bylevel': [i/10.0 for i in range(4, 6)],
  'reg_alpha': range(100, 121, 10),
  'reg_lambda': [10, 50, 100],
  'learning_rate': [0.025, 0.05],
  'max_delta_step': [5]
}
'''
gsearch = RandomizedSearchCV(estimator = xgb.XGBClassifier(**P),
                             param_distributions = GRID_PARAMS,
                             n_iter = 10,
                             scoring='roc_auc',
                             n_jobs=8,
                             iid=False,
                             cv=4,
                             verbose=3)
gsearch.fit(X, y)
log("\n".join([ str(score) for score in gsearch.grid_scores_]))
log("best score:%f" % gsearch.best_score_)
log("best param:%s" % str(gsearch.best_params_))

import numpy as np
import os
import sys
import copy
import math
import random
import xgboost as xgb
from sklearn.neural_network import MLPClassifier
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
    #self.scaler = preprocessing.RobustScaler().fit(X)
    #self.scaler = preprocessing.MinMaxScaler().fit(X)
    self.scaler = preprocessing.StandardScaler().fit(X)
    X = self.scaler.transform(X)
    self.selector = VarianceThreshold(threshold=0.05).fit(X)
    X = self.selector.transform(X)
    self.transformed = X

  def getOriginalTransformedData(self):
    return self.transformed

  def transform(self, X):
    log("before scaling: %s" % str(X.shape))
    X = self.scaler.transform(X)
    X = self.selector.transform(X)
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
  return data[:,1:-1], data[:, -1]

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
    log("type param name or type 'run' to finish")
    line  = sys.stdin.readline().strip()
    if line == "run":
      return
    if line not in P:
      log("file not found")
      continue
    log("current value:" + str(P[line]))
    value  = sys.stdin.readline().strip()
    if value == 'None':
      P[line] = None
    else:
      P[line] = eval(value)

#***result score average:0.834149, variance:0.000002
#current parametervalue:{'ratio': 5, 'eval_metric': 'auc', 'colsample_bylevel': 1, 'k': 50, 'm': 50, 'max_depth': 4, 'n_estimators': 64, 'subsample': 0.7, 'learning_rate': 0.15}

#***result score average:0.834196, variance:0.000009
#current parametervalue:{'ratio': 3, 'eval_metric': 'auc', 'colsample_bylevel': 1, 'k': 50, 'm': 50, 'max_depth': 4, 'n_estimators': 64, 'subsample': 0.7, 'learning_rate': 0.125}


P = {}

P['is_smote'] = True
P['k'] = 100
P['m'] = 100
P['ratio'] = 5

P['activation'] = 'relu'
P['algorithm'] = 'l-bfgs'
P['alpha'] = 1e-5
P['learning_rate'] = 'invscaling'
P['tol'] = 1e-4

P['layer'] = (10,10,10,10,)

if IS_LIGHT_TRAIN:
  P['k'] = 5
  P['m'] = 5

log("START:preprocessing: (read data, scaling, remove constant columns")
orig_X, orig_y = read_data()
skf = StratifiedKFold(orig_y, n_folds=4, shuffle=True)

while True:
  scores = []
  for train_index, test_index in skf:
    X, X_cv = orig_X[train_index], orig_X[test_index]
    y, y_cv = orig_y[train_index], orig_y[test_index]
    
    # Fraction of majority samples to draw with respect to samples of
    # minority class.
    sampled_X,sampled_y = X,y
    # Oversample data from the minority class.

    if P['is_smote']:
      sampled_X, sampled_y = SMOTE(k=P['k'], m=P['m'], ratio=P['ratio'], verbose=False, kind='regular').fit_transform(sampled_X, sampled_y)
      # Undersample samples from the majority class.
      sampled_X, sampled_y = UnderSampler(1.0).fit_transform(sampled_X, sampled_y)
    
    # Fit a scaler only for the sampled data.
    scaler = Scaler(sampled_X, sampled_y)
    sampled_X = scaler.getOriginalTransformedData()
    #model = RandomForestClassifier(n_estimators=100).fit(sampled_X, sampled_y)
    #model = RandomForestClassifier(n_estimators=P['n_estimators'], criterion=P['criterion'], max_depth=P['max_depth'], min_samples_split=P['min_samples_split'], min_samples_leaf=P['min_samples_leaf'], min_weight_fraction_leaf=P['min_weight_fraction_leaf'], max_features=P['max_features'], max_leaf_nodes=P['max_leaf_nodes'], bootstrap=P['bootstrap'], oob_score=P['oob_score'], n_jobs=8, random_state=None, verbose=0, warm_start=False, class_weight=None).fit(sampled_X, sampled_y)
    model = MLPClassifier(activation=P['activation'], algorithm=P['algorithm'], alpha=P['alpha'], hidden_layer_sizes=P['layer'], learning_rate=P['learning_rate'], tol=P['tol'], random_state=1).fit(sampled_X, sampled_y)
    #model = xgb.XGBClassifier(max_depth=P['max_depth'], n_estimators=P['n_estimators'], learning_rate=P['learning_rate'], nthread=8, subsample=P['subsample'], colsample_bylevel=P['colsample_bylevel']).fit(sampled_X, sampled_y, eval_metric=P['eval_metric'])
    prediction_cv = model.predict_proba(scaler.transform(X_cv))
    auc_score = roc_auc_score(y_cv, prediction_cv[:,1])
    scores.append(auc_score)
    log("***roc_auc_score:%f" % auc_score)
  
  avg = np.average(scores)
  var = np.var(scores)
  log("***result score average:%f, variance:%f" % (avg, var))
  update_param(P)

  

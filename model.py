"""

__author__ = 'amortized'
"""

import numpy  as np;
from sklearn.preprocessing import Imputer;
from sklearn.grid_search import ParameterGrid;
from multiprocessing import Pool;
import copy;
import random;
import sys;
import warnings;
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix;
import matplotlib.pyplot as plt;
import numpy as np
import scipy.sparse
import pickle
import xgboost as xgb
import copy
from sklearn.preprocessing import OneHotEncoder
from random import randint
from random import shuffle
import math
import copy

def generateParams():
    # Set the parameters by cross-validation
    paramaters_grid    = {'eta': [0.01], 'min_child_weight' : [6],  'colsample_bytree' : [0.95], 'subsample' : [0.95], 'gamma' : [0], 'max_depth' : [5]};

    paramaters_search  = list(ParameterGrid(paramaters_grid));

    parameters_to_try  = [];
    for ps in paramaters_search:
        params           = {'eval_metric' : 'auc', 'objective' : 'binary:logistic', 'nthread' : 4};
        for param in ps.keys():
            params[str(param)] = ps[param];
        parameters_to_try.append(copy.copy(params));

    return parameters_to_try;     



def train(train_X, train_Y):
    imp     = Imputer(missing_values='NaN', strategy='median', axis=0);
    imp.fit(train_X);

    train_X = imp.transform(train_X);
    train_Y = np.array(train_Y);

    dtrain      = xgb.DMatrix( train_X, label=train_Y);

    parameters_to_try = generateParams();

    best_params          = None;
    overall_best_auc     = 0;
    overall_best_nrounds = 0;

    for i in range(0, len(parameters_to_try)):
        param     = parameters_to_try[i]
        num_round = 3000

        bst_cv    = xgb.cv(param, dtrain, num_round, nfold=20, metrics={'auc'}, show_stdv=False, seed=0)
        
        best_iteration = 0;
        best_auc = 0;
        for i in range(0, len(bst_cv)):
            eval_result = bst_cv[i].split("\t");
            val_auc     = float(eval_result[1].split(":")[1]);
            if val_auc > best_auc:
                best_auc = val_auc;
                best_iteration = int(eval_result[0].replace("[","").replace("]",""));

        print("\n Best AUC : " + str(best_auc) + " for Params " + str(param) + " occurs at " + str(best_iteration));

        if best_auc > overall_best_auc:
            overall_best_auc     = best_auc;
            best_params          = copy.copy(param);
            overall_best_nrounds = best_iteration;

    print("\n Training the model on the entire training set with the best params")

    bst = xgb.train(best_params, dtrain, overall_best_nrounds);
    print("\n\n Overall Best AUC : " + str(overall_best_auc) + " for Params " + str(best_params) + " occurs at " + str(best_iteration));

    return bst, imp;

def predict_and_write(best_model, test_X, test_ids, test_bidders_ids_without_bids, imp):

    test_X = imp.transform(test_X);
    test_X = xgb.DMatrix(test_X);
    Y_hat  = best_model.predict( test_X );

    f = open("./data/submission.csv", "w");
    f.write("bidder_id,prediction\n");

    for i in range(0, len(Y_hat)):
       f.write(str(test_ids[i]) + "," + str(Y_hat[i]) + "\n");

    #Predict humans for bidders without activities
    for i in range(0, len(test_bidders_ids_without_bids)):
       f.write(str(test_bidders_ids_without_bids[i]) + "," + str(0.0) + "\n"); 

    f.close();

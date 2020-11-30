import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix
import itertools
import os
import sys
from sys import argv
import six
sys.modules['sklearn.externals.six'] = six
from resampling.resampler import resampling_assigner
from imblearn.metrics import geometric_mean_score
from collections import Counter

data_dir = "/home/jongchan/BPIC2013_closed/window_3_closed_problems_preprocessed" + ".csv"
data = pd.read_csv(data_dir, encoding='cp437')
X = data[['ACT_COMB_1', 'ACT_COMB_2', 'ACT_COMB_3','duration_in_days']]
y = data[['ACT_COMB_4']]

imb_technique = argv[1] # Baseline / ADASYN / ALLKNN / CNN / ENN / IHT / NCR / NM / OSS / RENN / ROS / RUS / SMOTE / BSMOTE / SMOTEENN / SMOTETOMEK / TOMEK

# Dummification
X_dummy = pd.get_dummies(X)
X_dummy.iloc[:, 0] = (X_dummy.iloc[:, 0] - X_dummy.iloc[:, 0].mean()) / X_dummy.iloc[:, 0].std()

# X and y here will be used for hyperparameter tuning using random search
X_randomsearch = X.replace(regex=True, to_replace=["Accepted/Assigned","Accepted/In Progress","Accepted/Wait","Completed/Closed","Queued/Awaiting Assignment","Completed/Cancelled","Unmatched/Unmatched"], value=[1,2,3,4,5,6,7])
y_randomsearch = y.replace(regex=True, to_replace=["Accepted/Assigned","Accepted/In Progress","Accepted/Wait","Completed/Closed","Queued/Awaiting Assignment"], value=[1,2,3,4,5])

nsplits = 5
kf = KFold(n_splits=nsplits)
kf.get_n_splits(X_dummy)

dnn_f1_score_pen_1_kfoldcv, dnn_f1_score_pen_5_kfoldcv, dnn_ovr_accuracy_kfoldcv, dnn_auc_kfoldcv, dnn_gmean_kfoldcv = [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2)
dnn_params_hls_AA, dnn_params_hls_AI, dnn_params_hls_AW, dnn_params_hls_CC, dnn_params_hls_QA = [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2)
dnn_params_lri_AA, dnn_params_lri_AI, dnn_params_lri_AW, dnn_params_lri_CC, dnn_params_lri_QA = [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2)

lr_f1_score_pen_1_kfoldcv, lr_f1_score_pen_5_kfoldcv, lr_ovr_accuracy_kfoldcv, lr_auc_kfoldcv, lr_gmean_kfoldcv = [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2)
lr_params_solver_AA, lr_params_solver_AI, lr_params_solver_AW, lr_params_solver_CC, lr_params_solver_QA = [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2)
lr_params_tol_AA, lr_params_tol_AI, lr_params_tol_AW, lr_params_tol_CC, lr_params_tol_QA = [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2)
lr_params_C_AA, lr_params_C_AI, lr_params_C_AW, lr_params_C_CC, lr_params_C_QA = [None] * (nsplits+2)

nb_f1_score_pen_1_kfoldcv, nb_f1_score_pen_5_kfoldcv, nb_ovr_accuracy_kfoldcv, nb_auc_kfoldcv, nb_gmean_kfoldcv = [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2)
nb_params_vs_AA, nb_params_vs_AI, nb_params_vs_AW, nb_params_vs_CC, nb_params_vs_QA = [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2)

rf_f1_score_pen_1_kfoldcv, rf_f1_score_pen_5_kfoldcv, rf_ovr_accuracy_kfoldcv, rf_auc_kfoldcv, rf_gmean_kfoldcv = [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2)
rf_params_est_AA, rf_params_est_AI, rf_params_est_AW, rf_params_est_CC, rf_params_est_QA = [None] * (nsplits+2)
rf_params_md_AA, rf_params_md_AI, rf_params_md_AW, rf_params_md_CC, rf_params_md_QA = [None] * (nsplits+2)
rf_params_mss_AA, rf_params_mss_AI, rf_params_mss_AW, rf_params_mss_CC, rf_params_mss_QA = [None] * (nsplits+2)

svm_f1_score_pen_1_kfoldcv, svm_f1_score_pen_5_kfoldcv, svm_ovr_accuracy_kfoldcv, svm_auc_kfoldcv, svm_gmean_kfoldcv = [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2)
svm_params_tol_AA, svm_params_tol_AI, svm_params_tol_AW, svm_params_tol_CC, svm_params_tol_QA = [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2)

outfile = os.path.join("/home/jongchan/BPIC2013_closed", "performance_results_%s_%s.csv" % ("BPIC2013", imb_technique))
outfile_param_AA = os.path.join("/home/jongchan/BPIC2013_closed", "parameters_AA_%s_%s.csv" % ("BPIC2013", imb_technique))
outfile_param_AI = os.path.join("/home/jongchan/BPIC2013_closed", "parameters_AI_%s_%s.csv" % ("BPIC2013", imb_technique))
outfile_param_AW = os.path.join("/home/jongchan/BPIC2013_closed", "parameters_AW_%s_%s.csv" % ("BPIC2013", imb_technique))
outfile_param_CC = os.path.join("/home/jongchan/BPIC2013_closed", "parameters_CC_%s_%s.csv" % ("BPIC2013", imb_technique))
outfile_param_QA = os.path.join("/home/jongchan/BPIC2013_closed", "parameters_QA_%s_%s.csv" % ("BPIC2013", imb_technique))

repeat = 0
for train_index, test_index in kf.split(X_dummy):
    X_train, X_test = X_dummy.iloc[train_index], X_dummy.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    y_test_AA = pd.DataFrame([1 if i == "Accepted/Assigned" else 0 for i in y_test['ACT_COMB_4']])
    y_test_AI = pd.DataFrame([1 if i == "Accepted/In Progress" else 0 for i in y_test['ACT_COMB_4']])
    y_test_AW = pd.DataFrame([1 if i == "Accepted/Wait" else 0 for i in y_test['ACT_COMB_4']])
    y_test_CC = pd.DataFrame([1 if i == "Completed/Closed" else 0 for i in y_test['ACT_COMB_4']])
    y_test_QA = pd.DataFrame([1 if i == "Queued/Awaiting Assignment" else 0 for i in y_test['ACT_COMB_4']])
    train = pd.concat([X_train, y_train], axis=1)
    ACT_COMB_4_index = np.unique(data['ACT_COMB_1']).size + np.unique(data['ACT_COMB_2']).size + np.unique(data['ACT_COMB_3']).size + 1

    def train_and_test(train, ACT_COMB_4_index, act_name):
        NN = train[train.ACT_COMB_4 == act_name]
        NN_rest = train[train.ACT_COMB_4 != act_name]
        NN_rest = NN_rest.copy()
        NN_rest.iloc[:, ACT_COMB_4_index] = "Others"
        NN = NN.reset_index()
        NN_rest = NN_rest.reset_index()
        NN = NN.iloc[:, 1:ACT_COMB_4_index+2]        
        NN_rest = NN_rest.iloc[:, 1:ACT_COMB_4_index+2]        
        NN_ova = pd.concat([NN, NN_rest])
        NN_ova_X_train = NN_ova.iloc[:, 0:ACT_COMB_4_index]
        NN_ova_y_train = NN_ova.iloc[:, ACT_COMB_4_index]
        NN_X_res = NN_ova_X_train
        NN_y_res = NN_ova_y_train
        Counter(NN_ova_y_train)
        return NN, NN_rest, NN_ova, NN_ova_X_train, NN_ova_y_train, NN_X_res, NN_y_res

    AA, AA_rest, AA_ova, AA_ova_X_train, AA_ova_y_train, AA_X_res, AA_y_res = train_and_test(train, ACT_COMB_4_index, "Accepted/Assigned")
    AI, AI_rest, AI_ova, AI_ova_X_train, AI_ova_y_train, AI_X_res, AI_y_res = train_and_test(train, ACT_COMB_4_index, "Accepted/In Progress")
    AW, AW_rest, AW_ova, AW_ova_X_train, AW_ova_y_train, AW_X_res, AW_y_res = train_and_test(train, ACT_COMB_4_index, "Accepted/Wait")
    CC, CC_rest, CC_ova, CC_ova_X_train, CC_ova_y_train, CC_X_res, CC_y_res = train_and_test(train, ACT_COMB_4_index, "Completed/Closed")
    QA, QA_rest, QA_ova, QA_ova_X_train, QA_ova_y_train, QA_X_res, QA_y_res = train_and_test(train, ACT_COMB_4_index, "Queued/Awaiting Assignment")

    AA_X_res, AA_y_res, AI_X_res, AI_y_res, AW_X_res, AW_y_res, CC_X_res, CC_y_res, QA_X_res, QA_y_res = resampling_assigner(imb_technique, AA_ova_X_train, AA_ova_y_train, AI_ova_X_train, AI_ova_y_train, AW_ova_X_train, AW_ova_y_train, CC_ova_X_train, CC_ova_y_train, QA_ova_X_train, QA_ova_y_train)

    first_digit_parameters = [x for x in itertools.product((5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100), repeat=1)]
    all_digit_parameters = first_digit_parameters
    learning_rate_init_parameters = [0.1, 0.01, 0.001]
    parameters = {'hidden_layer_sizes': all_digit_parameters,
                  'learning_rate_init': learning_rate_init_parameters}
    dnn_AA, dnn_AI, dnn_AW, dnn_CC, dnn_QA = MLPClassifier(max_iter=10000, activation='relu'),MLPClassifier(max_iter=10000, activation='relu'),MLPClassifier(max_iter=10000, activation='relu'),MLPClassifier(max_iter=10000, activation='relu'),MLPClassifier(max_iter=10000, activation='relu')
    dnn_AA_clf = RandomizedSearchCV(dnn_AA, parameters, n_jobs=-1, cv=5)
    dnn_AA_clf.fit(AA_X_res, AA_y_res)
    dnn_AI_clf = RandomizedSearchCV(dnn_AI, parameters, n_jobs=-1, cv=5)
    dnn_AI_clf.fit(AI_X_res, AI_y_res)
    dnn_AW_clf = RandomizedSearchCV(dnn_AW, parameters, n_jobs=-1, cv=5)
    dnn_AW_clf.fit(AW_X_res, AW_y_res)
    dnn_CC_clf = RandomizedSearchCV(dnn_CC, parameters, n_jobs=-1, cv=5)
    dnn_CC_clf.fit(CC_X_res, CC_y_res)
    dnn_QA_clf = RandomizedSearchCV(dnn_QA, parameters, n_jobs=-1, cv=5)
    dnn_QA_clf.fit(QA_X_res, QA_y_res)

    dnn_params_hls_AA[repeat] = dnn_AA_clf.best_params_['hidden_layer_sizes']
    dnn_params_hls_AI[repeat] = dnn_AI_clf.best_params_['hidden_layer_sizes']
    dnn_params_hls_AW[repeat] = dnn_AW_clf.best_params_['hidden_layer_sizes']
    dnn_params_hls_CC[repeat] = dnn_CC_clf.best_params_['hidden_layer_sizes'] 
    dnn_params_hls_QA[repeat] = dnn_QA_clf.best_params_['hidden_layer_sizes'] 
    dnn_params_lri_AA[repeat] = dnn_AA_clf.best_params_['learning_rate_init']
    dnn_params_lri_AI[repeat] = dnn_AI_clf.best_params_['learning_rate_init']
    dnn_params_lri_AW[repeat] = dnn_AW_clf.best_params_['learning_rate_init']
    dnn_params_lri_CC[repeat] = dnn_CC_clf.best_params_['learning_rate_init'] 
    dnn_params_lri_QA[repeat] = dnn_QA_clf.best_params_['learning_rate_init'] 

    solver = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
    tol = [1e-2, 1e-3, 1e-4, 1e-5]
    reg_strength = [0.5, 1.0, 1.5]
    parameters = {'solver': solver,
	          'tol': tol,
	          'C': reg_strength}
    lr_AA, lr_AI, lr_AW, lr_CC, lr_QA = LogisticRegression(),LogisticRegression(),LogisticRegression(),LogisticRegression(),LogisticRegression()
    lr_AA_clf = RandomizedSearchCV(lr_AA, parameters, n_jobs = -1, cv = 5)
    lr_AA_clf.fit(AA_X_res, AA_y_res)
    lr_AI_clf = RandomizedSearchCV(lr_AI, parameters, n_jobs = -1, cv = 5)
    lr_AI_clf.fit(AI_X_res, AI_y_res)
    lr_AW_clf = RandomizedSearchCV(lr_AW, parameters, n_jobs = -1, cv = 5)
    lr_AW_clf.fit(AW_X_res, AW_y_res)
    lr_CC_clf = RandomizedSearchCV(lr_CC, parameters, n_jobs = -1, cv = 5)
    lr_CC_clf.fit(CC_X_res, CC_y_res)
    lr_QA_clf = RandomizedSearchCV(lr_QA, parameters, n_jobs = -1, cv = 5)
    lr_QA_clf.fit(QA_X_res, QA_y_res)

    lr_params_solver_AA[repeat] = lr_AA_clf.best_params_['solver']
    lr_params_solver_AI[repeat] = lr_AI_clf.best_params_['solver']
    lr_params_solver_AW[repeat] = lr_AW_clf.best_params_['solver']
    lr_params_solver_CC[repeat] = lr_CC_clf.best_params_['solver'] 
    lr_params_solver_QA[repeat] = lr_QA_clf.best_params_['solver'] 
    lr_params_tol_AA[repeat] = lr_AA_clf.best_params_['tol']
    lr_params_tol_AI[repeat] = lr_AI_clf.best_params_['tol']
    lr_params_tol_AW[repeat] = lr_AW_clf.best_params_['tol']
    lr_params_tol_CC[repeat] = lr_CC_clf.best_params_['tol'] 
    lr_params_tol_QA[repeat] = lr_QA_clf.best_params_['tol'] 
    lr_params_C_AA[repeat] = lr_AA_clf.best_params_['C']
    lr_params_C_AI[repeat] = lr_AI_clf.best_params_['C']
    lr_params_C_AW[repeat] = lr_AW_clf.best_params_['C']
    lr_params_C_CC[repeat] = lr_CC_clf.best_params_['C'] 
    lr_params_C_QA[repeat] = lr_QA_clf.best_params_['C'] 

    # Below codes are for the implementation of Gaussian Naive Bayes training
    #In Gaussian NB, 'var_smoothing' parameter optimization makes convergence errors
    var_smoothing = [1e-07, 1e-08, 1e-09]
    parameters = {'var_smoothing': var_smoothing}
    nb_AA, nb_AI, nb_AW, nb_CC, nb_QA = GaussianNB(),GaussianNB(),GaussianNB(),GaussianNB(),GaussianNB()
    nb_AA_clf = RandomizedSearchCV(nb_AA, parameters, n_jobs = -1, cv = 5)
    nb_AA_clf.fit(AA_X_res, AA_y_res)
    nb_AI_clf = RandomizedSearchCV(nb_AI, parameters, n_jobs = -1, cv = 5)
    nb_AI_clf.fit(AI_X_res, AI_y_res)
    nb_AW_clf = RandomizedSearchCV(nb_AW, parameters, n_jobs = -1, cv = 5)
    nb_AW_clf.fit(AW_X_res, AW_y_res)
    nb_CC_clf = RandomizedSearchCV(nb_CC, parameters, n_jobs = -1, cv = 5)
    nb_CC_clf.fit(CC_X_res, CC_y_res)
    nb_QA_clf = RandomizedSearchCV(nb_QA, parameters, n_jobs = -1, cv = 5)
    nb_QA_clf.fit(QA_X_res, QA_y_res)

    nb_params_vs_AA[repeat] = nb_AA_clf.best_params_['var_smoothing']
    nb_params_vs_AI[repeat] = nb_AI_clf.best_params_['var_smoothing']
    nb_params_vs_AW[repeat] = nb_AW_clf.best_params_['var_smoothing']
    nb_params_vs_CC[repeat] = nb_CC_clf.best_params_['var_smoothing'] 
    nb_params_vs_QA[repeat] = nb_QA_clf.best_params_['var_smoothing'] 

    # Below codes are for the implementation of random forest training
    n_tree = [50, 100, 200, 300, 400, 500, 600, 700]
    max_depth = [10, 20, 30, 40, 50, 60, 70]
    min_samples_split = [5, 10, 15, 20, 25, 30]
    parameters = {'n_estimators': n_tree,
		  'max_depth': max_depth,
		  'min_samples_split': min_samples_split}
    rf_AA, rf_AI, rf_AW, rf_CC, rf_QA = RandomForestClassifier(),RandomForestClassifier(),RandomForestClassifier(),RandomForestClassifier(),RandomForestClassifier()
    rf_AA_clf = RandomizedSearchCV(rf_AA, parameters, n_jobs = -1, cv = 5)
    rf_AA_clf.fit(AA_X_res, AA_y_res)
    rf_AI_clf = RandomizedSearchCV(rf_AI, parameters, n_jobs = -1, cv = 5)
    rf_AI_clf.fit(AI_X_res, AI_y_res)
    rf_AW_clf = RandomizedSearchCV(rf_AW, parameters, n_jobs = -1, cv=5)
    rf_AW_clf.fit(AW_X_res, AW_y_res)
    rf_CC_clf = RandomizedSearchCV(rf_CC, parameters, n_jobs = -1, cv=5)
    rf_CC_clf.fit(CC_X_res, CC_y_res)
    rf_QA_clf = RandomizedSearchCV(rf_QA, parameters, n_jobs = -1, cv=5)
    rf_QA_clf.fit(QA_X_res, QA_y_res)

    rf_params_est_AA[repeat] = rf_AA_clf.best_params_['n_estimators']
    rf_params_est_AI[repeat] = rf_AI_clf.best_params_['n_estimators']
    rf_params_est_AW[repeat] = rf_AW_clf.best_params_['n_estimators']
    rf_params_est_CC[repeat] = rf_CC_clf.best_params_['n_estimators'] 
    rf_params_est_QA[repeat] = rf_QA_clf.best_params_['n_estimators'] 
    rf_params_md_AA[repeat] = rf_AA_clf.best_params_['max_depth']
    rf_params_md_AI[repeat] = rf_AI_clf.best_params_['max_depth']
    rf_params_md_AW[repeat] = rf_AW_clf.best_params_['max_depth']
    rf_params_md_CC[repeat] = rf_CC_clf.best_params_['max_depth'] 
    rf_params_md_QA[repeat] = rf_QA_clf.best_params_['max_depth'] 
    rf_params_mss_AA[repeat] = rf_AA_clf.best_params_['min_samples_split']
    rf_params_mss_AI[repeat] = rf_AI_clf.best_params_['min_samples_split']
    rf_params_mss_AW[repeat] = rf_AW_clf.best_params_['min_samples_split']
    rf_params_mss_CC[repeat] = rf_CC_clf.best_params_['min_samples_split'] 
    rf_params_mss_QA[repeat] = rf_QA_clf.best_params_['min_samples_split'] 

    # Below codes are for the implementation of support vector machine training
    tol = [1e-2, 1e-3, 1e-4]
    parameters = {'tol': tol,
                  'kernel': ['linear'],
                  'probability': [True]}
    svm_AA, svm_AI, svm_AW, svm_CC, svm_QA = SVC(),SVC(),SVC(),SVC(),SVC()
    svm_AA_clf = RandomizedSearchCV(svm_AA, parameters, n_jobs = -1, cv = 5)
    svm_AA_clf.fit(AA_X_res, AA_y_res)
    svm_AI_clf = RandomizedSearchCV(svm_AI, parameters, n_jobs = -1, cv = 5)
    svm_AI_clf.fit(AI_X_res, AI_y_res)
    svm_AW_clf = RandomizedSearchCV(svm_AW, parameters, n_jobs = -1, cv = 5)
    svm_AW_clf.fit(AW_X_res, AW_y_res)
    svm_CC_clf = RandomizedSearchCV(svm_CC, parameters, n_jobs = -1, cv = 5)
    svm_CC_clf.fit(CC_X_res, CC_y_res)
    svm_QA_clf = RandomizedSearchCV(svm_QA, parameters, n_jobs = -1, cv = 5)
    svm_QA_clf.fit(QA_X_res, QA_y_res)

    svm_params_tol_AA[repeat] = svm_AA_clf.best_params_['tol']
    svm_params_tol_AI[repeat] = svm_AI_clf.best_params_['tol']
    svm_params_tol_AW[repeat] = svm_AW_clf.best_params_['tol']
    svm_params_tol_CC[repeat] = svm_CC_clf.best_params_['tol'] 
    svm_params_tol_QA[repeat] = svm_QA_clf.best_params_['tol'] 

    def prediction(AA_clf, AI_clf, AW_clf, CC_clf, QA_clf, X_test):
        pred_class_AA = AA_clf.predict(X_test)
        pred_prob_AA = AA_clf.predict_proba(X_test)
        pred_class_AI = AI_clf.predict(X_test)
        pred_prob_AI = AI_clf.predict_proba(X_test)
        pred_class_AW = AW_clf.predict(X_test)
        pred_prob_AW = AW_clf.predict_proba(X_test)
        pred_class_CC = CC_clf.predict(X_test)
        pred_prob_CC = CC_clf.predict_proba(X_test)
        pred_class_QA = QA_clf.predict(X_test)
        pred_prob_QA = QA_clf.predict_proba(X_test)
        return pred_class_AA, pred_prob_AA, pred_class_AI, pred_prob_AI, pred_class_AW, pred_prob_AW, pred_class_CC, pred_prob_CC, pred_class_QA, pred_prob_QA

    dnn_pred_class_AA, dnn_pred_prob_AA, dnn_pred_class_AI, dnn_pred_prob_AI, dnn_pred_class_AW, dnn_pred_prob_AW, dnn_pred_class_CC, dnn_pred_prob_CC, dnn_pred_class_QA, dnn_pred_prob_QA = prediction(dnn_AA_clf, dnn_AI_clf, dnn_AW_clf, dnn_CC_clf, dnn_QA_clf, X_test)
    lr_pred_class_AA, lr_pred_prob_AA, lr_pred_class_AI, lr_pred_prob_AI, lr_pred_class_AW, lr_pred_prob_AW, lr_pred_class_CC, lr_pred_prob_CC, lr_pred_class_QA, lr_pred_prob_QA = prediction(lr_AA_clf, lr_AI_clf, lr_AW_clf, lr_CC_clf, lr_QA_clf, X_test)
    nb_pred_class_AA, nb_pred_prob_AA, nb_pred_class_AI, nb_pred_prob_AI, nb_pred_class_AW, nb_pred_prob_AW, nb_pred_class_CC, nb_pred_prob_CC, nb_pred_class_QA, nb_pred_prob_QA = prediction(nb_AA_clf, nb_AI_clf, nb_AW_clf, nb_CC_clf, nb_QA_clf, X_test)
    rf_pred_class_AA, rf_pred_prob_AA, rf_pred_class_AI, rf_pred_prob_AI, rf_pred_class_AW, rf_pred_prob_AW, rf_pred_class_CC, rf_pred_prob_CC, rf_pred_class_QA, rf_pred_prob_QA = prediction(rf_AA_clf, rf_AI_clf, rf_AW_clf, rf_CC_clf, rf_QA_clf, X_test)
    svm_pred_class_AA, svm_pred_prob_AA, svm_pred_class_AI, svm_pred_prob_AI, svm_pred_class_AW, svm_pred_prob_AW, svm_pred_class_CC, svm_pred_prob_CC, svm_pred_class_QA, svm_pred_prob_QA = prediction(svm_AA_clf, svm_AI_clf, svm_AW_clf, svm_CC_clf, svm_QA_clf, X_test)

    dnn_prediction, lr_prediction, nb_prediction, rf_prediction, svm_prediction = pd.DataFrame(columns=['Prediction']),pd.DataFrame(columns=['Prediction']),pd.DataFrame(columns=['Prediction']),pd.DataFrame(columns=['Prediction']),pd.DataFrame(columns=['Prediction'])

    def label_assigner(y_test, prediction, pred_class_AA, pred_class_AI, pred_class_AW, pred_class_CC, pred_class_QA, pred_prob_AA, pred_prob_AI, pred_prob_AW, pred_prob_CC, pred_prob_QA):
        for i in range(0, len(y_test)):
            AA_index, AI_index, AW_index, CC_index, QA_index = 0, 0, 0, 0, 0
            if pred_class_AA[i] == "Accepted/Assigned":
                AA_index = 0 if pred_prob_AA[i][0] >= 0.5 else 1
            elif pred_class_AA[i] == "Others":
                AA_index = 0 if pred_prob_AA[i][0] < 0.5 else 1
            if pred_class_AI[i] == "Accepted/In Progress":
                AI_index = 0 if pred_prob_AI[i][0] >= 0.5 else 1
            elif pred_class_AI[i] == "Others":
                AI_index = 0 if pred_prob_AI[i][0] < 0.5 else 1
            if pred_class_AW[i] == "Accepted/Wait":
                AW_index = 0 if pred_prob_AW[i][0] >= 0.5 else 1
            elif pred_class_AW[i] == "Others":
                AW_index = 0 if pred_prob_AW[i][0] < 0.5 else 1
            if pred_class_CC[i] == "Completed/Closed":
                CC_index = 0 if pred_prob_CC[i][0] >= 0.5 else 1
            elif pred_class_CC[i] == "Others":
                CC_index = 0 if pred_prob_CC[i][0] < 0.5 else 1
            if pred_class_QA[i] == "Queued/Awaiting Assignment":
                QA_index = 0 if pred_prob_QA[i][0] >= 0.5 else 1
            elif pred_class_QA[i] == "Others":
                QA_index = 0 if pred_prob_QA[i][0] < 0.5 else 1
            if pred_prob_AA[i][AA_index] == max(pred_prob_AA[i][AA_index], pred_prob_AI[i][AI_index],
                                                        pred_prob_AW[i][AW_index],
                                                        pred_prob_CC[i][CC_index], pred_prob_QA[i][QA_index]):
                prediction.loc[i] = "Accepted/Assigned"
            elif pred_prob_AI[i][AI_index] == max(pred_prob_AA[i][AA_index], pred_prob_AI[i][AI_index],
                                                          pred_prob_AW[i][AW_index], pred_prob_CC[i][CC_index],
                                                          pred_prob_QA[i][QA_index]):
                prediction.loc[i] = "Accepted/In Progress"
            elif pred_prob_AW[i][AW_index] == max(pred_prob_AA[i][AA_index], pred_prob_AI[i][AI_index],
                                                          pred_prob_AW[i][AW_index], pred_prob_CC[i][CC_index],
                                                          pred_prob_QA[i][QA_index]):
                prediction.loc[i] = "Accepted/Wait"
            elif pred_prob_CC[i][CC_index] == max(pred_prob_AA[i][AA_index], pred_prob_AI[i][AI_index],
                                                          pred_prob_AW[i][AW_index], pred_prob_CC[i][CC_index],
                                                          pred_prob_QA[i][QA_index]):
                prediction.loc[i] = "Completed/Closed"
            elif pred_prob_QA[i][QA_index] == max(pred_prob_AA[i][AA_index], pred_prob_AI[i][AI_index],
                                                          pred_prob_AW[i][AW_index], pred_prob_CC[i][CC_index],
                                                          pred_prob_QA[i][QA_index]):
                prediction.loc[i] = "Queued/Awaiting Assignment"
        return pred_prob_AA, pred_prob_AI, pred_prob_AW, pred_prob_CC, pred_prob_QA, prediction

    def get_precision(conf_matrix):
        tp_1 = conf_matrix[0][0]
        tp_2 = conf_matrix[1][1]
        tp_3 = conf_matrix[2][2]
        tp_4 = conf_matrix[3][3]
        tp_5 = conf_matrix[4][4]
        fp_1 = conf_matrix[1][0] + conf_matrix[2][0] + conf_matrix[3][0] + conf_matrix[4][0]
        fp_2 = conf_matrix[0][1] + conf_matrix[2][1] + conf_matrix[3][1] + conf_matrix[4][1]
        fp_3 = conf_matrix[0][2] + conf_matrix[1][2] + conf_matrix[3][2] + conf_matrix[4][2]
        fp_4 = conf_matrix[0][3] + conf_matrix[1][3] + conf_matrix[2][3] + conf_matrix[4][3]
        fp_5 = conf_matrix[0][4] + conf_matrix[1][4] + conf_matrix[2][4] + conf_matrix[3][4]
        precision_1 = 0 if tp_1 + fp_1 == 0 else tp_1 / (tp_1 + fp_1)
        precision_2 = 0 if tp_2 + fp_2 == 0 else tp_2 / (tp_2 + fp_2)
        precision_3 = 0 if tp_3 + fp_3 == 0 else tp_3 / (tp_3 + fp_3)
        precision_4 = 0 if tp_4 + fp_4 == 0 else tp_4 / (tp_4 + fp_4)
        precision_5 = 0 if tp_5 + fp_5 == 0 else tp_5 / (tp_5 + fp_5)
        precision_avg = (precision_1 + precision_2 + precision_3 + precision_4 + precision_5) / 5
        return precision_avg

    def get_recall_pen_1(conf_matrix):
        tp_1 = conf_matrix[0][0]
        tp_2 = conf_matrix[1][1]
        tp_3 = conf_matrix[2][2]
        tp_4 = conf_matrix[3][3]
        tp_5 = conf_matrix[4][4]
        fn_1 = conf_matrix[0][1] + conf_matrix[0][2] + conf_matrix[0][3] + conf_matrix[0][4]
        fn_2 = conf_matrix[1][0] + conf_matrix[1][2] + conf_matrix[1][3] + conf_matrix[1][4]
        fn_3 = conf_matrix[2][0] + conf_matrix[2][1] + conf_matrix[2][3] + conf_matrix[2][4]
        fn_4 = conf_matrix[3][0] + conf_matrix[3][1] + conf_matrix[3][2] + conf_matrix[3][4]
        fn_5 = conf_matrix[4][0] + conf_matrix[4][1] + conf_matrix[4][2] + conf_matrix[4][3]
        recall_1 = 0 if tp_1 + fn_1 == 0 else tp_1 / (tp_1 + fn_1)
        recall_2 = 0 if tp_2 + fn_2 == 0 else tp_2 / (tp_2 + fn_2)
        recall_3 = 0 if tp_3 + fn_3 == 0 else tp_3 / (tp_3 + fn_3)
        recall_4 = 0 if tp_4 + fn_4 == 0 else tp_4 / (tp_4 + fn_4)
        recall_5 = 0 if tp_5 + fn_5 == 0 else tp_5 / (tp_5 + fn_5)
        recall_avg_pen_1 = (recall_1 + recall_2 + recall_3 +recall_4 + recall_5) / (5+1-1)
        return recall_avg_pen_1

    def get_recall_pen_5(conf_matrix):
        tp_1 = conf_matrix[0][0]
        tp_2 = conf_matrix[1][1]
        tp_3 = conf_matrix[2][2]
        tp_4 = conf_matrix[3][3]
        tp_5 = conf_matrix[4][4]
        fn_1 = conf_matrix[0][1] + conf_matrix[0][2] + conf_matrix[0][3] + conf_matrix[0][4]
        fn_2 = conf_matrix[1][0] + conf_matrix[1][2] + conf_matrix[1][3] + conf_matrix[1][4]
        fn_3 = conf_matrix[2][0] + conf_matrix[2][1] + conf_matrix[2][3] + conf_matrix[2][4]
        fn_4 = conf_matrix[3][0] + conf_matrix[3][1] + conf_matrix[3][2] + conf_matrix[3][4]
        fn_5 = conf_matrix[4][0] + conf_matrix[4][1] + conf_matrix[4][2] + conf_matrix[4][3]
        recall_1 = 0 if  tp_1 + fn_1 == 0 else tp_1 / (tp_1 + fn_1)
        recall_2 = 0 if  tp_2 + fn_2 == 0 else tp_2 / (tp_2 + fn_2)
        recall_3 = 0 if  tp_3 + fn_3 == 0 else tp_3 / (tp_3 + fn_3)
        recall_4 = 0 if  tp_4 + fn_4 == 0 else tp_4 / (tp_4 + fn_4)
        recall_5 = 0 if  tp_5 + fn_5 == 0 else tp_5 / (tp_5 + fn_5)
        recall_avg_pen_5 = (recall_1 + recall_2 + (5*recall_3) +recall_4 + recall_5) / (5+5-1)
        return recall_avg_pen_5

    dnn_pred_prob_AA, dnn_pred_prob_AI, dnn_pred_prob_AW, dnn_pred_prob_CC, dnn_pred_prob_QA, dnn_prediction = label_assigner(y_test, dnn_prediction, dnn_pred_class_AA, dnn_pred_class_AI, dnn_pred_class_AW, dnn_pred_class_CC, dnn_pred_class_QA, dnn_pred_prob_AA, dnn_pred_prob_AI, dnn_pred_prob_AW, dnn_pred_prob_CC, dnn_pred_prob_QA)
    lr_pred_prob_AA, lr_pred_prob_AI, lr_pred_prob_AW, lr_pred_prob_CC, lr_pred_prob_QA, lr_prediction = label_assigner(y_test, lr_prediction, lr_pred_class_AA, lr_pred_class_AI, lr_pred_class_AW, lr_pred_class_CC, lr_pred_class_QA, lr_pred_prob_AA, lr_pred_prob_AI, lr_pred_prob_AW, lr_pred_prob_CC, lr_pred_prob_QA)
    nb_pred_prob_AA, nb_pred_prob_AI, nb_pred_prob_AW, nb_pred_prob_CC, nb_pred_prob_QA, nb_prediction = label_assigner(y_test, nb_prediction, nb_pred_class_AA, nb_pred_class_AI, nb_pred_class_AW, nb_pred_class_CC, nb_pred_class_QA, nb_pred_prob_AA, nb_pred_prob_AI, nb_pred_prob_AW, nb_pred_prob_CC, nb_pred_prob_QA)
    rf_pred_prob_AA, rf_pred_prob_AI, rf_pred_prob_AW, rf_pred_prob_CC, rf_pred_prob_QA, rf_prediction = label_assigner(y_test, rf_prediction, rf_pred_class_AA, rf_pred_class_AI, rf_pred_class_AW, rf_pred_class_CC, rf_pred_class_QA, rf_pred_prob_AA, rf_pred_prob_AI, rf_pred_prob_AW, rf_pred_prob_CC, rf_pred_prob_QA)
    svm_pred_prob_AA, svm_pred_prob_AI, svm_pred_prob_AW, svm_pred_prob_CC, svm_pred_prob_QA, svm_prediction = label_assigner(y_test, svm_prediction, svm_pred_class_AA, svm_pred_class_AI, svm_pred_class_AW, svm_pred_class_CC, svm_pred_class_QA, svm_pred_prob_AA, svm_pred_prob_AI, svm_pred_prob_AW, svm_pred_prob_CC, svm_pred_prob_QA)

    def perf_evaluator(y_test, pred_prob_AA, pred_prob_AI, pred_prob_AW, pred_prob_CC, pred_prob_QA, prediction, f1_score_pen_1_kfoldcv, f1_score_pen_5_kfoldcv, ovr_accuracy_kfoldcv, auc_kfoldcv, gmean_kfoldcv, clf, repeat):
        conf_matrix = confusion_matrix(y_test, prediction)
        precision = get_precision(conf_matrix)
        recall_pen_1 = get_recall_pen_1(conf_matrix)
        recall_pen_5 = get_recall_pen_5(conf_matrix)
        f1_score_pen_1 = 2 * (precision * recall_pen_1) / (precision + recall_pen_1)
        f1_score_pen_5 = 2 * (precision * recall_pen_5) / (precision + recall_pen_5)
        ovr_accuracy = (conf_matrix[0][0] + conf_matrix[1][1] + conf_matrix[2][2] + conf_matrix[3][3] + conf_matrix[4][4]) / (sum(conf_matrix[0]) + sum(conf_matrix[1]) + sum(conf_matrix[2]) + sum(conf_matrix[3]) + sum(conf_matrix[4]))
        auc_AA = roc_auc_score(y_true = y_test_AA, y_score = pd.DataFrame(pred_prob_AA).iloc[:,0])
        auc_AI = roc_auc_score(y_true = y_test_AI, y_score = pd.DataFrame(pred_prob_AI).iloc[:,0])
        auc_AW = roc_auc_score(y_true = y_test_AW, y_score = pd.DataFrame(pred_prob_AW).iloc[:,0])
        auc_CC = roc_auc_score(y_true = y_test_CC, y_score = pd.DataFrame(pred_prob_CC).iloc[:,0])
        auc_QA = roc_auc_score(y_true = y_test_QA, y_score = pd.DataFrame(pred_prob_QA).iloc[:,0])
        gmean = geometric_mean_score(y_true = y_test, y_pred = prediction, average = 'macro')
        conf_matrix = pd.DataFrame(conf_matrix)               
        conf_matrix.to_csv('conf_matrix_'+imb_technique+'_' + clf + '_bpic2013_closed_'+ str(nsplits) +'foldcv_' + str(repeat+1)+'.csv',header=False,index=False) #First repetition
        #conf_matrix.to_csv('conf_matrix_'+imb_technique+'_penalty_' + str(penalty) + '_' + clf + '_bpic2013_closed_'+ str(nsplits) +'foldcv_' + str(repeat+6)+'.csv',header=False,index=False) #Second repetition
        f1_score_pen_1_kfoldcv[repeat] = f1_score_pen_1
        f1_score_pen_5_kfoldcv[repeat] = f1_score_pen_5
        ovr_accuracy_kfoldcv[repeat] = ovr_accuracy
        auc_kfoldcv[repeat] = (auc_AA + auc_AI + auc_AW + auc_CC + auc_QA)/5
        gmean_kfoldcv[repeat] = gmean
        return conf_matrix, f1_score_pen_1_kfoldcv, f1_score_pen_5_kfoldcv, ovr_accuracy_kfoldcv, auc_kfoldcv, gmean_kfoldcv

    dnn_conf_matrix, dnn_f1_score_pen_1_kfoldcv, dnn_f1_score_pen_5_kfoldcv, dnn_ovr_accuracy_kfoldcv, dnn_auc_kfoldcv, dnn_gmean_kfoldcv = perf_evaluator(y_test, dnn_pred_prob_AA, dnn_pred_prob_AI, dnn_pred_prob_AW, dnn_pred_prob_CC, dnn_pred_prob_QA, dnn_prediction, dnn_f1_score_pen_1_kfoldcv, dnn_f1_score_pen_5_kfoldcv, dnn_ovr_accuracy_kfoldcv, dnn_auc_kfoldcv, dnn_gmean_kfoldcv, "dnn", repeat)
    lr_conf_matrix, lr_f1_score_pen_1_kfoldcv, lr_f1_score_pen_5_kfoldcv, lr_ovr_accuracy_kfoldcv, lr_auc_kfoldcv, lr_gmean_kfoldcv = perf_evaluator(y_test, lr_pred_prob_AA, lr_pred_prob_AI, lr_pred_prob_AW, lr_pred_prob_CC, lr_pred_prob_QA, lr_prediction, lr_f1_score_pen_1_kfoldcv, lr_f1_score_pen_5_kfoldcv, lr_ovr_accuracy_kfoldcv, lr_auc_kfoldcv, lr_gmean_kfoldcv, "lr", repeat)
    nb_conf_matrix, nb_f1_score_pen_1_kfoldcv, nb_f1_score_pen_5_kfoldcv, nb_ovr_accuracy_kfoldcv, nb_auc_kfoldcv, nb_gmean_kfoldcv = perf_evaluator(y_test, nb_pred_prob_AA, nb_pred_prob_AI, nb_pred_prob_AW, nb_pred_prob_CC, nb_pred_prob_QA, nb_prediction, nb_f1_score_pen_1_kfoldcv, nb_f1_score_pen_5_kfoldcv, nb_ovr_accuracy_kfoldcv, nb_auc_kfoldcv, nb_gmean_kfoldcv, "nb", repeat)
    rf_conf_matrix, rf_f1_score_pen_1_kfoldcv, rf_f1_score_pen_5_kfoldcv, rf_ovr_accuracy_kfoldcv, rf_auc_kfoldcv, rf_gmean_kfoldcv = perf_evaluator(y_test, rf_pred_prob_AA, rf_pred_prob_AI, rf_pred_prob_AW, rf_pred_prob_CC, rf_pred_prob_QA, rf_prediction, rf_f1_score_pen_1_kfoldcv, rf_f1_score_pen_5_kfoldcv, rf_ovr_accuracy_kfoldcv, rf_auc_kfoldcv, rf_gmean_kfoldcv, "rf", repeat)
    svm_conf_matrix, svm_f1_score_pen_1_kfoldcv, svm_f1_score_pen_5_kfoldcv, svm_ovr_accuracy_kfoldcv, svm_auc_kfoldcv, svm_gmean_kfoldcv = perf_evaluator(y_test, svm_pred_prob_AA, svm_pred_prob_AI, svm_pred_prob_AW, svm_pred_prob_CC, svm_pred_prob_QA, svm_prediction, svm_f1_score_pen_1_kfoldcv, svm_f1_score_pen_5_kfoldcv, svm_ovr_accuracy_kfoldcv, svm_auc_kfoldcv, svm_gmean_kfoldcv, "svm", repeat)

    repeat = repeat + 1

with open(outfile, 'w') as fout:
    fout.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % ("dnn_f1_pen1", "dnn_f1_pen5", "dnn_ovr_acc", "dnn_auc", "dnn_gmean","lr_f1_pen1", "lr_f1_pen5", "lr_ovr_acc", "lr_auc", "lr_gmean","nb_f1_pen1", "nb_f1_pen5", "nb_ovr_acc", "nb_auc", "nb_gmean", "rf_f1_pen1", "rf_f1_pen5", "rf_ovr_acc", "rf_auc", "rf_gmean","svm_f1_pen1", "svm_f1_pen5", "svm_ovr_acc", "svm_auc", "svm_gmean"))
    fout.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_f1_score_pen_1_kfoldcv[0], dnn_f1_score_pen_5_kfoldcv[0], dnn_ovr_accuracy_kfoldcv[0], dnn_auc_kfoldcv[0], dnn_gmean_kfoldcv[0], lr_f1_score_pen_1_kfoldcv[0], lr_f1_score_pen_5_kfoldcv[0], lr_ovr_accuracy_kfoldcv[0], lr_auc_kfoldcv[0], lr_gmean_kfoldcv[0], nb_f1_score_pen_1_kfoldcv[0], nb_f1_score_pen_5_kfoldcv[0], nb_ovr_accuracy_kfoldcv[0], nb_auc_kfoldcv[0], nb_gmean_kfoldcv[0], rf_f1_score_pen_1_kfoldcv[0], rf_f1_score_pen_5_kfoldcv[0], rf_ovr_accuracy_kfoldcv[0], rf_auc_kfoldcv[0], rf_gmean_kfoldcv[0], svm_f1_score_pen_1_kfoldcv[0], svm_f1_score_pen_5_kfoldcv[0], svm_ovr_accuracy_kfoldcv[0], svm_auc_kfoldcv[0], svm_gmean_kfoldcv[0]))
    fout.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_f1_score_pen_1_kfoldcv[1], dnn_f1_score_pen_5_kfoldcv[1], dnn_ovr_accuracy_kfoldcv[1], dnn_auc_kfoldcv[1], dnn_gmean_kfoldcv[1], lr_f1_score_pen_1_kfoldcv[1], lr_f1_score_pen_5_kfoldcv[1], lr_ovr_accuracy_kfoldcv[1], lr_auc_kfoldcv[1], lr_gmean_kfoldcv[1], nb_f1_score_pen_1_kfoldcv[1], nb_f1_score_pen_5_kfoldcv[1], nb_ovr_accuracy_kfoldcv[1], nb_auc_kfoldcv[1], nb_gmean_kfoldcv[1], rf_f1_score_pen_1_kfoldcv[1], rf_f1_score_pen_5_kfoldcv[1], rf_ovr_accuracy_kfoldcv[1], rf_auc_kfoldcv[1], rf_gmean_kfoldcv[1], svm_f1_score_pen_1_kfoldcv[1], svm_f1_score_pen_5_kfoldcv[1], svm_ovr_accuracy_kfoldcv[1], svm_auc_kfoldcv[1], svm_gmean_kfoldcv[1]))
    fout.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_f1_score_pen_1_kfoldcv[2], dnn_f1_score_pen_5_kfoldcv[2], dnn_ovr_accuracy_kfoldcv[2], dnn_auc_kfoldcv[2], dnn_gmean_kfoldcv[2], lr_f1_score_pen_1_kfoldcv[2], lr_f1_score_pen_5_kfoldcv[2], lr_ovr_accuracy_kfoldcv[2], lr_auc_kfoldcv[2], lr_gmean_kfoldcv[2], nb_f1_score_pen_1_kfoldcv[2], nb_f1_score_pen_5_kfoldcv[2], nb_ovr_accuracy_kfoldcv[2], nb_auc_kfoldcv[2], nb_gmean_kfoldcv[2], rf_f1_score_pen_1_kfoldcv[2], rf_f1_score_pen_5_kfoldcv[2], rf_ovr_accuracy_kfoldcv[2], rf_auc_kfoldcv[2], rf_gmean_kfoldcv[2], svm_f1_score_pen_1_kfoldcv[2], svm_f1_score_pen_5_kfoldcv[2], svm_ovr_accuracy_kfoldcv[2], svm_auc_kfoldcv[2], svm_gmean_kfoldcv[2]))
    fout.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_f1_score_pen_1_kfoldcv[3], dnn_f1_score_pen_5_kfoldcv[3], dnn_ovr_accuracy_kfoldcv[3], dnn_auc_kfoldcv[3], dnn_gmean_kfoldcv[3], lr_f1_score_pen_1_kfoldcv[3], lr_f1_score_pen_5_kfoldcv[3], lr_ovr_accuracy_kfoldcv[3], lr_auc_kfoldcv[3], lr_gmean_kfoldcv[3], nb_f1_score_pen_1_kfoldcv[3], nb_f1_score_pen_5_kfoldcv[3], nb_ovr_accuracy_kfoldcv[3], nb_auc_kfoldcv[3], nb_gmean_kfoldcv[3], rf_f1_score_pen_1_kfoldcv[3], rf_f1_score_pen_5_kfoldcv[3], rf_ovr_accuracy_kfoldcv[3], rf_auc_kfoldcv[3], rf_gmean_kfoldcv[3], svm_f1_score_pen_1_kfoldcv[3], svm_f1_score_pen_5_kfoldcv[3], svm_ovr_accuracy_kfoldcv[3], svm_auc_kfoldcv[3], svm_gmean_kfoldcv[3]))
    fout.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_f1_score_pen_1_kfoldcv[4], dnn_f1_score_pen_5_kfoldcv[4], dnn_ovr_accuracy_kfoldcv[4], dnn_auc_kfoldcv[4], dnn_gmean_kfoldcv[4], lr_f1_score_pen_1_kfoldcv[4], lr_f1_score_pen_5_kfoldcv[4], lr_ovr_accuracy_kfoldcv[4], lr_auc_kfoldcv[4], lr_gmean_kfoldcv[4], nb_f1_score_pen_1_kfoldcv[4], nb_f1_score_pen_5_kfoldcv[4], nb_ovr_accuracy_kfoldcv[4], nb_auc_kfoldcv[4], nb_gmean_kfoldcv[4], rf_f1_score_pen_1_kfoldcv[4], rf_f1_score_pen_5_kfoldcv[4], rf_ovr_accuracy_kfoldcv[4], rf_auc_kfoldcv[4], rf_gmean_kfoldcv[4], svm_f1_score_pen_1_kfoldcv[4], svm_f1_score_pen_5_kfoldcv[4], svm_ovr_accuracy_kfoldcv[4], svm_auc_kfoldcv[4], svm_gmean_kfoldcv[4]))
with open(outfile_param_AA, 'w') as fout_param_AA:
    fout_param_AA.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % ("dnn_hidden_layer_sizes", "dnn_learning_rate_init", "lr_solver", "lr_tol", "lr_C", "nb_var_smoothing", "rf_n_estimator", "rf_max_depth", "rf_min_sample_split", "svm_tol"))
    fout_param_AA.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_AA[0], dnn_params_lri_AA[0], lr_params_solver_AA[0], lr_params_tol_AA[0], lr_params_C_AA[0], nb_params_vs_AA[0], rf_params_est_AA[0], rf_params_md_AA[0], rf_params_mss_AA[0], svm_params_tol_AA[0]))
    fout_param_AA.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_AA[1], dnn_params_lri_AA[1], lr_params_solver_AA[1], lr_params_tol_AA[1], lr_params_C_AA[1], nb_params_vs_AA[1], rf_params_est_AA[1], rf_params_md_AA[1], rf_params_mss_AA[1], svm_params_tol_AA[1]))
    fout_param_AA.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_AA[2], dnn_params_lri_AA[2], lr_params_solver_AA[2], lr_params_tol_AA[2], lr_params_C_AA[2], nb_params_vs_AA[2], rf_params_est_AA[2], rf_params_md_AA[2], rf_params_mss_AA[2], svm_params_tol_AA[2]))
    fout_param_AA.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_AA[3], dnn_params_lri_AA[3], lr_params_solver_AA[3], lr_params_tol_AA[3], lr_params_C_AA[3], nb_params_vs_AA[3], rf_params_est_AA[3], rf_params_md_AA[3], rf_params_mss_AA[3], svm_params_tol_AA[3]))
    fout_param_AA.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_AA[4], dnn_params_lri_AA[4], lr_params_solver_AA[4], lr_params_tol_AA[4], lr_params_C_AA[4], nb_params_vs_AA[4], rf_params_est_AA[4], rf_params_md_AA[4], rf_params_mss_AA[4], svm_params_tol_AA[4]))
with open(outfile_param_AI, 'w') as fout_param_AI:
    fout_param_AI.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % ("dnn_hidden_layer_sizes", "dnn_learning_rate_init", "lr_solver", "lr_tol", "lr_C", "nb_var_smoothing", "rf_n_estimator", "rf_max_depth", "rf_min_sample_split", "svm_tol"))
    fout_param_AI.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_AI[0], dnn_params_lri_AI[0], lr_params_solver_AI[0], lr_params_tol_AI[0], lr_params_C_AI[0], nb_params_vs_AI[0], rf_params_est_AI[0], rf_params_md_AI[0], rf_params_mss_AI[0], svm_params_tol_AI[0]))
    fout_param_AI.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_AI[1], dnn_params_lri_AI[1], lr_params_solver_AI[1], lr_params_tol_AI[1], lr_params_C_AI[1], nb_params_vs_AI[1], rf_params_est_AI[1], rf_params_md_AI[1], rf_params_mss_AI[1], svm_params_tol_AI[1]))
    fout_param_AI.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_AI[2], dnn_params_lri_AI[2], lr_params_solver_AI[2], lr_params_tol_AI[2], lr_params_C_AI[2], nb_params_vs_AI[2], rf_params_est_AI[2], rf_params_md_AI[2], rf_params_mss_AI[2], svm_params_tol_AI[2]))
    fout_param_AI.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_AI[3], dnn_params_lri_AI[3], lr_params_solver_AI[3], lr_params_tol_AI[3], lr_params_C_AI[3], nb_params_vs_AI[3], rf_params_est_AI[3], rf_params_md_AI[3], rf_params_mss_AI[3], svm_params_tol_AI[3]))
    fout_param_AI.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_AI[4], dnn_params_lri_AI[4], lr_params_solver_AI[4], lr_params_tol_AI[4], lr_params_C_AI[4], nb_params_vs_AI[4], rf_params_est_AI[4], rf_params_md_AI[4], rf_params_mss_AI[4], svm_params_tol_AI[4]))
with open(outfile_param_AW, 'w') as fout_param_AW:
    fout_param_AW.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % ("dnn_hidden_layer_sizes", "dnn_learning_rate_init", "lr_solver", "lr_tol", "lr_C", "nb_var_smoothing", "rf_n_estimator", "rf_max_depth", "rf_min_sample_split", "svm_tol"))
    fout_param_AW.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_AW[0], dnn_params_lri_AW[0], lr_params_solver_AW[0], lr_params_tol_AW[0], lr_params_C_AW[0], nb_params_vs_AW[0], rf_params_est_AW[0], rf_params_md_AW[0], rf_params_mss_AW[0], svm_params_tol_AW[0]))
    fout_param_AW.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_AW[1], dnn_params_lri_AW[1], lr_params_solver_AW[1], lr_params_tol_AW[1], lr_params_C_AW[1], nb_params_vs_AW[1], rf_params_est_AW[1], rf_params_md_AW[1], rf_params_mss_AW[1], svm_params_tol_AW[1]))
    fout_param_AW.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_AW[2], dnn_params_lri_AW[2], lr_params_solver_AW[2], lr_params_tol_AW[2], lr_params_C_AW[2], nb_params_vs_AW[2], rf_params_est_AW[2], rf_params_md_AW[2], rf_params_mss_AW[2], svm_params_tol_AW[2]))
    fout_param_AW.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_AW[3], dnn_params_lri_AW[3], lr_params_solver_AW[3], lr_params_tol_AW[3], lr_params_C_AW[3], nb_params_vs_AW[3], rf_params_est_AW[3], rf_params_md_AW[3], rf_params_mss_AW[3], svm_params_tol_AW[3]))
    fout_param_AW.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_AW[4], dnn_params_lri_AW[4], lr_params_solver_AW[4], lr_params_tol_AW[4], lr_params_C_AW[4], nb_params_vs_AW[4], rf_params_est_AW[4], rf_params_md_AW[4], rf_params_mss_AW[4], svm_params_tol_AW[4]))
with open(outfile_param_CC, 'w') as fout_param_CC:
    fout_param_CC.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % ("dnn_hidden_layer_sizes", "dnn_learning_rate_init", "lr_solver", "lr_tol", "lr_C", "nb_var_smoothing", "rf_n_estimator", "rf_max_depth", "rf_min_sample_split", "svm_tol"))
    fout_param_CC.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_CC[0], dnn_params_lri_CC[0], lr_params_solver_CC[0], lr_params_tol_CC[0], lr_params_C_CC[0], nb_params_vs_CC[0], rf_params_est_CC[0], rf_params_md_CC[0], rf_params_mss_CC[0], svm_params_tol_CC[0]))
    fout_param_CC.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_CC[1], dnn_params_lri_CC[1], lr_params_solver_CC[1], lr_params_tol_CC[1], lr_params_C_CC[1], nb_params_vs_CC[1], rf_params_est_CC[1], rf_params_md_CC[1], rf_params_mss_CC[1], svm_params_tol_CC[1]))
    fout_param_CC.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_CC[2], dnn_params_lri_CC[2], lr_params_solver_CC[2], lr_params_tol_CC[2], lr_params_C_CC[2], nb_params_vs_CC[2], rf_params_est_CC[2], rf_params_md_CC[2], rf_params_mss_CC[2], svm_params_tol_CC[2]))
    fout_param_CC.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_CC[3], dnn_params_lri_CC[3], lr_params_solver_CC[3], lr_params_tol_CC[3], lr_params_C_CC[3], nb_params_vs_CC[3], rf_params_est_CC[3], rf_params_md_CC[3], rf_params_mss_CC[3], svm_params_tol_CC[3]))
    fout_param_CC.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_CC[4], dnn_params_lri_CC[4], lr_params_solver_CC[4], lr_params_tol_CC[4], lr_params_C_CC[4], nb_params_vs_CC[4], rf_params_est_CC[4], rf_params_md_CC[4], rf_params_mss_CC[4], svm_params_tol_CC[4]))
with open(outfile_param_QA, 'w') as fout_param_QA:
    fout_param_QA.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % ("dnn_hidden_layer_sizes", "dnn_learning_rate_init", "lr_solver", "lr_tol", "lr_C", "nb_var_smoothing", "rf_n_estimator", "rf_max_depth", "rf_min_sample_split", "svm_tol"))
    fout_param_QA.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_QA[0], dnn_params_lri_QA[0], lr_params_solver_QA[0], lr_params_tol_QA[0], lr_params_C_QA[0], nb_params_vs_QA[0], rf_params_est_QA[0], rf_params_md_QA[0], rf_params_mss_QA[0], svm_params_tol_QA[0]))
    fout_param_QA.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_QA[1], dnn_params_lri_QA[1], lr_params_solver_QA[1], lr_params_tol_QA[1], lr_params_C_QA[1], nb_params_vs_QA[1], rf_params_est_QA[1], rf_params_md_QA[1], rf_params_mss_QA[1], svm_params_tol_QA[1]))
    fout_param_QA.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_QA[2], dnn_params_lri_QA[2], lr_params_solver_QA[2], lr_params_tol_QA[2], lr_params_C_QA[2], nb_params_vs_QA[2], rf_params_est_QA[2], rf_params_md_QA[2], rf_params_mss_QA[2], svm_params_tol_QA[2]))
    fout_param_QA.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_QA[3], dnn_params_lri_QA[3], lr_params_solver_QA[3], lr_params_tol_QA[3], lr_params_C_QA[3], nb_params_vs_QA[3], rf_params_est_QA[3], rf_params_md_QA[3], rf_params_mss_QA[3], svm_params_tol_QA[3]))
    fout_param_QA.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_QA[4], dnn_params_lri_QA[4], lr_params_solver_QA[4], lr_params_tol_QA[4], lr_params_C_QA[4], nb_params_vs_QA[4], rf_params_est_QA[4], rf_params_md_QA[4], rf_params_mss_QA[4], svm_params_tol_QA[4]))
dnn_f1_score_pen_1_kfoldcv[5] = "Average" #To let users figure out that f1_score_kfoldcv[6] value is the average when seeing the csv file
dnn_f1_score_pen_1_kfoldcv[6] = (dnn_f1_score_pen_1_kfoldcv[0]+dnn_f1_score_pen_1_kfoldcv[1]+dnn_f1_score_pen_1_kfoldcv[2]+dnn_f1_score_pen_1_kfoldcv[3]+dnn_f1_score_pen_1_kfoldcv[4])/5
dnn_f1_score_pen_1_kfoldcv = pd.DataFrame(dnn_f1_score_pen_1_kfoldcv)
dnn_f1_score_pen_1_kfoldcv.to_csv('f1_score_'+imb_technique+'_penalty_1_dnn_bpic2013_closed_' + str(nsplits) + 'foldcv_1~5.csv',header=False,index=False) #First repetition
dnn_f1_score_pen_5_kfoldcv[5] = "Average" #To let users figure out that f1_score_kfoldcv[6] value is the average when seeing the csv file
dnn_f1_score_pen_5_kfoldcv[6] = (dnn_f1_score_pen_5_kfoldcv[0]+dnn_f1_score_pen_5_kfoldcv[1]+dnn_f1_score_pen_5_kfoldcv[2]+dnn_f1_score_pen_5_kfoldcv[3]+dnn_f1_score_pen_5_kfoldcv[4])/5
dnn_f1_score_pen_5_kfoldcv = pd.DataFrame(dnn_f1_score_pen_5_kfoldcv)
dnn_f1_score_pen_5_kfoldcv.to_csv('f1_score_'+imb_technique+'_penalty_5_dnn_bpic2013_closed_' + str(nsplits) + 'foldcv_1~5.csv',header=False,index=False) #First repetition
#dnn_f1_score_kfoldcv.to_csv('f1_score_'+imb_technique+'_penalty_' + str(penalty) + '_dnn_bpic2013_closed_' + str(nsplits) + 'foldcv_6~10.csv',header=False,index=False) #Second repetition
dnn_ovr_accuracy_kfoldcv[5] = "Average" #To let users figure out that f1_score_kfoldcv[6] value is the average when seeing the csv file
dnn_ovr_accuracy_kfoldcv[6] = (dnn_ovr_accuracy_kfoldcv[0]+dnn_ovr_accuracy_kfoldcv[1]+dnn_ovr_accuracy_kfoldcv[2]+dnn_ovr_accuracy_kfoldcv[3]+dnn_ovr_accuracy_kfoldcv[4])/5
dnn_ovr_accuracy_kfoldcv = pd.DataFrame(dnn_ovr_accuracy_kfoldcv)
dnn_ovr_accuracy_kfoldcv.to_csv('ovr_accuracy_'+imb_technique + '_dnn_bpic2013_closed_' + str(nsplits) + 'foldcv_1~5.csv',header=False,index=False) #First repetition
#dnn_ovr_accuracy_kfoldcv.to_csv('ovr_accuracy_'+imb_technique+'_penalty_' + str(penalty) + '_dnn_bpic2013_closed_' + str(nsplits) + 'foldcv_6~10.csv',header=False,index=False) #Second repetition
dnn_auc_kfoldcv[5] = "Average" #To let users figure out that f1_score_kfoldcv[6] value is the average when seeing the csv file
dnn_auc_kfoldcv[6] = (dnn_auc_kfoldcv[0]+dnn_auc_kfoldcv[1]+dnn_auc_kfoldcv[2]+dnn_auc_kfoldcv[3]+dnn_auc_kfoldcv[4])/5
dnn_auc_kfoldcv = pd.DataFrame(dnn_auc_kfoldcv)
dnn_auc_kfoldcv.to_csv('auc_'+imb_technique+'_dnn_bpic2013_closed_' + str(nsplits) + 'foldcv_1~5.csv',header=False,index=False)
dnn_gmean_kfoldcv[5] = "Average" #To let users figure out that f1_score_kfoldcv[6] value is the average when seeing the csv file
dnn_gmean_kfoldcv[6] = (dnn_gmean_kfoldcv[0]+dnn_gmean_kfoldcv[1]+dnn_gmean_kfoldcv[2]+dnn_gmean_kfoldcv[3]+dnn_gmean_kfoldcv[4])/5
dnn_gmean_kfoldcv = pd.DataFrame(dnn_gmean_kfoldcv)
dnn_gmean_kfoldcv.to_csv('gmean_'+imb_technique+'_dnn_bpic2013_closed_' + str(nsplits) + 'foldcv_1~5.csv',header=False,index=False)

lr_f1_score_pen_1_kfoldcv[5] = "Average" #To let users figure out that f1_score_kfoldcv[6] value is the average when seeing the csv file
lr_f1_score_pen_1_kfoldcv[6] = (lr_f1_score_pen_1_kfoldcv[0]+lr_f1_score_pen_1_kfoldcv[1]+lr_f1_score_pen_1_kfoldcv[2]+lr_f1_score_pen_1_kfoldcv[3]+lr_f1_score_pen_1_kfoldcv[4])/5
lr_f1_score_pen_1_kfoldcv = pd.DataFrame(lr_f1_score_pen_1_kfoldcv)
lr_f1_score_pen_1_kfoldcv.to_csv('f1_score_'+imb_technique+'_penalty_1_lr_bpic2013_closed_' + str(nsplits) + 'foldcv_1~5.csv',header=False,index=False) #First repetition
lr_f1_score_pen_5_kfoldcv[5] = "Average" #To let users figure out that f1_score_kfoldcv[6] value is the average when seeing the csv file
lr_f1_score_pen_5_kfoldcv[6] = (lr_f1_score_pen_5_kfoldcv[0]+lr_f1_score_pen_5_kfoldcv[1]+lr_f1_score_pen_5_kfoldcv[2]+lr_f1_score_pen_5_kfoldcv[3]+lr_f1_score_pen_5_kfoldcv[4])/5
lr_f1_score_pen_5_kfoldcv = pd.DataFrame(lr_f1_score_pen_5_kfoldcv)
lr_f1_score_pen_5_kfoldcv.to_csv('f1_score_'+imb_technique+'_penalty_5_lr_bpic2013_closed_' + str(nsplits) + 'foldcv_1~5.csv',header=False,index=False) #First repetition
#lr_f1_score_kfoldcv.to_csv('f1_score_'+imb_technique+'_penalty_' + str(penalty) + '_lr_bpic2013_closed_' + str(nsplits) + 'foldcv_6~10.csv',header=False,index=False) #Second repetition
lr_ovr_accuracy_kfoldcv[5] = "Average" #To let users figure out that f1_score_kfoldcv[6] value is the average when seeing the csv file
lr_ovr_accuracy_kfoldcv[6] = (lr_ovr_accuracy_kfoldcv[0]+lr_ovr_accuracy_kfoldcv[1]+lr_ovr_accuracy_kfoldcv[2]+lr_ovr_accuracy_kfoldcv[3]+lr_ovr_accuracy_kfoldcv[4])/5
lr_ovr_accuracy_kfoldcv = pd.DataFrame(lr_ovr_accuracy_kfoldcv)
lr_ovr_accuracy_kfoldcv.to_csv('ovr_accuracy_' + imb_technique + '_lr_bpic2013_closed_' + str(nsplits) + 'foldcv_1~5.csv',header=False,index=False) #First repetition
#lr_ovr_accuracy_kfoldcv.to_csv('ovr_accuracy_'+imb_technique+'_penalty_' + str(penalty) + '_lr_bpic2013_closed_' + str(nsplits) + 'foldcv_6~10.csv',header=False,index=False) #Second repetition
lr_auc_kfoldcv[5] = "Average" #To let users figure out that f1_score_kfoldcv[6] value is the average when seeing the csv file
lr_auc_kfoldcv[6] = (lr_auc_kfoldcv[0]+lr_auc_kfoldcv[1]+lr_auc_kfoldcv[2]+lr_auc_kfoldcv[3]+lr_auc_kfoldcv[4])/5
lr_auc_kfoldcv = pd.DataFrame(lr_auc_kfoldcv)
lr_auc_kfoldcv.to_csv('auc_'+imb_technique+'_lr_bpic2013_closed_' + str(nsplits) + 'foldcv_1~5.csv',header=False,index=False)
lr_gmean_kfoldcv[5] = "Average" #To let users figure out that f1_score_kfoldcv[6] value is the average when seeing the csv file
lr_gmean_kfoldcv[6] = (lr_gmean_kfoldcv[0]+lr_gmean_kfoldcv[1]+lr_gmean_kfoldcv[2]+lr_gmean_kfoldcv[3]+lr_gmean_kfoldcv[4])/5
lr_gmean_kfoldcv = pd.DataFrame(lr_gmean_kfoldcv)
lr_gmean_kfoldcv.to_csv('gmean_'+imb_technique+'_lr_bpic2013_closed_' + str(nsplits) + 'foldcv_1~5.csv',header=False,index=False)

nb_f1_score_pen_1_kfoldcv[5] = "Average" #To let users figure out that f1_score_kfoldcv[6] value is the average when seeing the csv file
nb_f1_score_pen_1_kfoldcv[6] = (nb_f1_score_pen_1_kfoldcv[0]+nb_f1_score_pen_1_kfoldcv[1]+nb_f1_score_pen_1_kfoldcv[2]+nb_f1_score_pen_1_kfoldcv[3]+nb_f1_score_pen_1_kfoldcv[4])/5
nb_f1_score_pen_1_kfoldcv = pd.DataFrame(nb_f1_score_pen_1_kfoldcv)
nb_f1_score_pen_1_kfoldcv.to_csv('f1_score_'+imb_technique+'_penalty_1_nb_bpic2013_closed_' + str(nsplits) + 'foldcv_1~5.csv',header=False,index=False) #First repetition
nb_f1_score_pen_5_kfoldcv[5] = "Average" #To let users figure out that f1_score_kfoldcv[6] value is the average when seeing the csv file
nb_f1_score_pen_5_kfoldcv[6] = (nb_f1_score_pen_5_kfoldcv[0]+nb_f1_score_pen_5_kfoldcv[1]+nb_f1_score_pen_5_kfoldcv[2]+nb_f1_score_pen_5_kfoldcv[3]+nb_f1_score_pen_5_kfoldcv[4])/5
nb_f1_score_pen_5_kfoldcv = pd.DataFrame(nb_f1_score_pen_5_kfoldcv)
nb_f1_score_pen_5_kfoldcv.to_csv('f1_score_'+imb_technique+'_penalty_5_nb_bpic2013_closed_' + str(nsplits) + 'foldcv_1~5.csv',header=False,index=False) #First repetition
#nb_f1_score_kfoldcv.to_csv('f1_score_'+imb_technique+'_penalty_' + str(penalty) + '_nb_bpic2013_closed_' + str(nsplits) + 'foldcv_6~10.csv',header=False,index=False) #Second repetition
nb_ovr_accuracy_kfoldcv[5] = "Average" #To let users figure out that f1_score_kfoldcv[6] value is the average when seeing the csv file
nb_ovr_accuracy_kfoldcv[6] = (nb_ovr_accuracy_kfoldcv[0]+nb_ovr_accuracy_kfoldcv[1]+nb_ovr_accuracy_kfoldcv[2]+nb_ovr_accuracy_kfoldcv[3]+nb_ovr_accuracy_kfoldcv[4])/5
nb_ovr_accuracy_kfoldcv = pd.DataFrame(nb_ovr_accuracy_kfoldcv)
nb_ovr_accuracy_kfoldcv.to_csv('ovr_accuracy_' + imb_technique + '_nb_bpic2013_closed_' + str(nsplits) + 'foldcv_1~5.csv',header=False,index=False) #First repetition
#nb_ovr_accuracy_kfoldcv.to_csv('ovr_accuracy_'+imb_technique+'_penalty_' + str(penalty) + '_nb_bpic2013_closed_' + str(nsplits) + 'foldcv_6~10.csv',header=False,index=False) #Second repetition
nb_auc_kfoldcv[5] = "Average" #To let users figure out that f1_score_kfoldcv[6] value is the average when seeing the csv file
nb_auc_kfoldcv[6] = (nb_auc_kfoldcv[0]+nb_auc_kfoldcv[1]+nb_auc_kfoldcv[2]+nb_auc_kfoldcv[3]+nb_auc_kfoldcv[4])/5
nb_auc_kfoldcv = pd.DataFrame(nb_auc_kfoldcv)
nb_auc_kfoldcv.to_csv('auc_'+imb_technique+'_nb_bpic2013_closed_' + str(nsplits) + 'foldcv_1~5.csv',header=False,index=False)
nb_gmean_kfoldcv[5] = "Average" #To let users figure out that f1_score_kfoldcv[6] value is the average when seeing the csv file
nb_gmean_kfoldcv[6] = (nb_gmean_kfoldcv[0]+nb_gmean_kfoldcv[1]+nb_gmean_kfoldcv[2]+nb_gmean_kfoldcv[3]+nb_gmean_kfoldcv[4])/5
nb_gmean_kfoldcv = pd.DataFrame(nb_gmean_kfoldcv)
nb_gmean_kfoldcv.to_csv('gmean_'+imb_technique+'_nb_bpic2013_closed_' + str(nsplits) + 'foldcv_1~5.csv',header=False,index=False)

rf_f1_score_pen_1_kfoldcv[5] = "Average" #To let users figure out that f1_score_kfoldcv[6] value is the average when seeing the csv file
rf_f1_score_pen_1_kfoldcv[6] = (rf_f1_score_pen_1_kfoldcv[0]+rf_f1_score_pen_1_kfoldcv[1]+rf_f1_score_pen_1_kfoldcv[2]+rf_f1_score_pen_1_kfoldcv[3]+rf_f1_score_pen_1_kfoldcv[4])/5
rf_f1_score_pen_1_kfoldcv = pd.DataFrame(rf_f1_score_pen_1_kfoldcv)
rf_f1_score_pen_1_kfoldcv.to_csv('f1_score_'+imb_technique+'_penalty_1_rf_bpic2013_closed_' + str(nsplits) + 'foldcv_1~5.csv',header=False,index=False) #First repetition
rf_f1_score_pen_5_kfoldcv[5] = "Average" #To let users figure out that f1_score_kfoldcv[6] value is the average when seeing the csv file
rf_f1_score_pen_5_kfoldcv[6] = (rf_f1_score_pen_5_kfoldcv[0]+rf_f1_score_pen_5_kfoldcv[1]+rf_f1_score_pen_5_kfoldcv[2]+rf_f1_score_pen_5_kfoldcv[3]+rf_f1_score_pen_5_kfoldcv[4])/5
rf_f1_score_pen_5_kfoldcv = pd.DataFrame(rf_f1_score_pen_5_kfoldcv)
rf_f1_score_pen_5_kfoldcv.to_csv('f1_score_'+imb_technique+'_penalty_5_rf_bpic2013_closed_' + str(nsplits) + 'foldcv_1~5.csv',header=False,index=False) #First repetition
#rf_f1_score_kfoldcv.to_csv('f1_score_'+imb_technique+'_penalty_' + str(penalty) + '_rf_bpic2013_closed_' + str(nsplits) + 'foldcv_6~10.csv',header=False,index=False) #Second repetition
rf_ovr_accuracy_kfoldcv[5] = "Average" #To let users figure out that f1_score_kfoldcv[6] value is the average when seeing the csv file
rf_ovr_accuracy_kfoldcv[6] = (rf_ovr_accuracy_kfoldcv[0]+rf_ovr_accuracy_kfoldcv[1]+rf_ovr_accuracy_kfoldcv[2]+rf_ovr_accuracy_kfoldcv[3]+rf_ovr_accuracy_kfoldcv[4])/5
rf_ovr_accuracy_kfoldcv = pd.DataFrame(rf_ovr_accuracy_kfoldcv)
rf_ovr_accuracy_kfoldcv.to_csv('ovr_accuracy_' + imb_technique + '_rf_bpic2013_closed_' + str(nsplits) + 'foldcv_1~5.csv',header=False,index=False) #First repetition
#rf_ovr_accuracy_kfoldcv.to_csv('ovr_accuracy_'+imb_technique+'_penalty_' + str(penalty) + '_rf_bpic2013_closed_' + str(nsplits) + 'foldcv_6~10.csv',header=False,index=False) #Second repetition
rf_auc_kfoldcv[5] = "Average" #To let users figure out that f1_score_kfoldcv[6] value is the average when seeing the csv file
rf_auc_kfoldcv[6] = (rf_auc_kfoldcv[0]+rf_auc_kfoldcv[1]+rf_auc_kfoldcv[2]+rf_auc_kfoldcv[3]+rf_auc_kfoldcv[4])/5
rf_auc_kfoldcv = pd.DataFrame(rf_auc_kfoldcv)
rf_auc_kfoldcv.to_csv('auc_'+imb_technique+'_rf_bpic2013_closed_' + str(nsplits) + 'foldcv_1~5.csv',header=False,index=False)
rf_gmean_kfoldcv[5] = "Average" #To let users figure out that f1_score_kfoldcv[6] value is the average when seeing the csv file
rf_gmean_kfoldcv[6] = (rf_gmean_kfoldcv[0]+rf_gmean_kfoldcv[1]+rf_gmean_kfoldcv[2]+rf_gmean_kfoldcv[3]+rf_gmean_kfoldcv[4])/5
rf_gmean_kfoldcv = pd.DataFrame(rf_gmean_kfoldcv)
rf_gmean_kfoldcv.to_csv('gmean_'+imb_technique+'_rf_bpic2013_closed_' + str(nsplits) + 'foldcv_1~5.csv',header=False,index=False)

svm_f1_score_pen_1_kfoldcv[5] = "Average" #To let users figure out that f1_score_kfoldcv[6] value is the average when seeing the csv file
svm_f1_score_pen_1_kfoldcv[6] = (svm_f1_score_pen_1_kfoldcv[0]+svm_f1_score_pen_1_kfoldcv[1]+svm_f1_score_pen_1_kfoldcv[2]+svm_f1_score_pen_1_kfoldcv[3]+svm_f1_score_pen_1_kfoldcv[4])/5
svm_f1_score_pen_1_kfoldcv = pd.DataFrame(svm_f1_score_pen_1_kfoldcv)
svm_f1_score_pen_1_kfoldcv.to_csv('f1_score_'+imb_technique+'_penalty_1_svm_bpic2013_closed_' + str(nsplits) + 'foldcv_1~5.csv',header=False,index=False) #First repetition
svm_f1_score_pen_5_kfoldcv[5] = "Average" #To let users figure out that f1_score_kfoldcv[6] value is the average when seeing the csv file
svm_f1_score_pen_5_kfoldcv[6] = (svm_f1_score_pen_5_kfoldcv[0]+svm_f1_score_pen_5_kfoldcv[1]+svm_f1_score_pen_5_kfoldcv[2]+svm_f1_score_pen_5_kfoldcv[3]+svm_f1_score_pen_5_kfoldcv[4])/5
svm_f1_score_pen_5_kfoldcv = pd.DataFrame(svm_f1_score_pen_5_kfoldcv)
svm_f1_score_pen_5_kfoldcv.to_csv('f1_score_'+imb_technique+'_penalty_5_svm_bpic2013_closed_' + str(nsplits) + 'foldcv_1~5.csv',header=False,index=False) #First repetition
#svm_f1_score_kfoldcv.to_csv('f1_score_'+imb_technique+'_penalty_' + str(penalty) + '_svm_bpic2013_closed_' + str(nsplits) + 'foldcv_6~10.csv',header=False,index=False) #Second repetition
svm_ovr_accuracy_kfoldcv[5] = "Average" #To let users figure out that f1_score_kfoldcv[6] value is the average when seeing the csv file
svm_ovr_accuracy_kfoldcv[6] = (svm_ovr_accuracy_kfoldcv[0]+svm_ovr_accuracy_kfoldcv[1]+svm_ovr_accuracy_kfoldcv[2]+svm_ovr_accuracy_kfoldcv[3]+svm_ovr_accuracy_kfoldcv[4])/5
svm_ovr_accuracy_kfoldcv = pd.DataFrame(svm_ovr_accuracy_kfoldcv)
svm_ovr_accuracy_kfoldcv.to_csv('ovr_accuracy_'+ imb_technique + '_svm_bpic2013_closed_' + str(nsplits) + 'foldcv_1~5.csv',header=False,index=False) #First repetition
#svm_ovr_accuracy_kfoldcv.to_csv('ovr_accuracy_'+imb_technique+'_penalty_' + str(penalty) + '_svm_bpic2013_closed_' + str(nsplits) + 'foldcv_6~10.csv',header=False,index=False) #Second repetition
svm_auc_kfoldcv[5] = "Average" #To let users figure out that f1_score_kfoldcv[6] value is the average when seeing the csv file
svm_auc_kfoldcv[6] = (svm_auc_kfoldcv[0]+svm_auc_kfoldcv[1]+svm_auc_kfoldcv[2]+svm_auc_kfoldcv[3]+svm_auc_kfoldcv[4])/5
svm_auc_kfoldcv = pd.DataFrame(svm_auc_kfoldcv)
svm_auc_kfoldcv.to_csv('auc_'+imb_technique+'_svm_bpic2013_closed_' + str(nsplits) + 'foldcv_1~5.csv',header=False,index=False)
svm_gmean_kfoldcv[5] = "Average" #To let users figure out that f1_score_kfoldcv[6] value is the average when seeing the csv file
svm_gmean_kfoldcv[6] = (svm_gmean_kfoldcv[0]+svm_gmean_kfoldcv[1]+svm_gmean_kfoldcv[2]+svm_gmean_kfoldcv[3]+svm_gmean_kfoldcv[4])/5
svm_gmean_kfoldcv = pd.DataFrame(svm_gmean_kfoldcv)
svm_gmean_kfoldcv.to_csv('gmean_'+imb_technique+'_svm_bpic2013_closed_' + str(nsplits) + 'foldcv_1~5.csv',header=False,index=False)

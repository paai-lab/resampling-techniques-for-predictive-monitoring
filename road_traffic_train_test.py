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
from resampling.resampler_road_traffic import resampling_assigner
from imblearn.metrics import geometric_mean_score
from collections import Counter

data_dir = "/home/jongchan/Road_traffic/window_3_road_traffic_reduced_preprocessed.csv"
data = pd.read_csv(data_dir, encoding='cp437')
X = data[['ACT_1', 'ACT_2', 'ACT_3','duration_in_days']]
y = data[['ACT_4']]

imb_technique = argv[1] # Baseline / ADASYN / ALLKNN / CNN / ENN / IHT / NCR / NM / OSS / RENN / ROS / RUS / SMOTE / BSMOTE / SMOTEENN / SMOTETOMEK / TOMEK

# Dummification
X_dummy = pd.get_dummies(X)
X_dummy.iloc[:, 0] = (X_dummy.iloc[:, 0] - X_dummy.iloc[:, 0].mean()) / X_dummy.iloc[:, 0].std()

# X and y here will be used for hyperparameter tuning using random search
X_randomsearch = X.replace(regex=True, to_replace=["Add penalty","Payment","Send for Credit Collection","Insert Fine Notification","Send Fine","Create Fine"], value=[1,2,3,4,5,6])
y_randomsearch = y.replace(regex=True, to_replace=["Add penalty","Payment","Send for Credit Collection","Insert Fine Notification","Send Fine","Create Fine"], value=[1,2,3,4,5,6])

nsplits = 5
kf = KFold(n_splits=nsplits)
kf.get_n_splits(X_dummy)

dnn_f1_score_pen_1_kfoldcv, dnn_f1_score_pen_5_kfoldcv, dnn_ovr_accuracy_kfoldcv, dnn_auc_kfoldcv, dnn_gmean_kfoldcv = [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2)
dnn_params_hls_AP, dnn_params_hls_PM, dnn_params_hls_SC = [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2)
dnn_params_lri_AP, dnn_params_lri_PM, dnn_params_lri_SC = [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2)

lr_f1_score_pen_1_kfoldcv, lr_f1_score_pen_5_kfoldcv, lr_ovr_accuracy_kfoldcv, lr_auc_kfoldcv, lr_gmean_kfoldcv = [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2)
lr_params_solver_AP, lr_params_solver_PM, lr_params_solver_SC = [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2)
lr_params_tol_AP, lr_params_tol_PM, lr_params_tol_SC = [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2)
lr_params_C_AP, lr_params_C_PM, lr_params_C_SC = [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2)

nb_f1_score_pen_1_kfoldcv, nb_f1_score_pen_5_kfoldcv, nb_ovr_accuracy_kfoldcv, nb_auc_kfoldcv, nb_gmean_kfoldcv = [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2)
nb_params_vs_AP, nb_params_vs_PM, nb_params_vs_SC = [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2)

rf_f1_score_pen_1_kfoldcv, rf_f1_score_pen_5_kfoldcv, rf_ovr_accuracy_kfoldcv, rf_auc_kfoldcv, rf_gmean_kfoldcv = [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2)
rf_params_est_AP, rf_params_est_PM, rf_params_est_SC = [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2)
rf_params_md_AP, rf_params_md_PM, rf_params_md_SC = [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2)
rf_params_mss_AP, rf_params_mss_PM, rf_params_mss_SC = [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2)

svm_f1_score_pen_1_kfoldcv, svm_f1_score_pen_5_kfoldcv, svm_ovr_accuracy_kfoldcv, svm_auc_kfoldcv, svm_gmean_kfoldcv = [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2)
svm_params_tol_AP, svm_params_tol_PM, svm_params_tol_SC = [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2)

outfile = os.path.join("/home/jongchan/Road_traffic", "performance_results_%s_%s.csv" % ("Road_traffic", imb_technique))
outfile_param_AP = os.path.join("/home/jongchan/Road_traffic", "parameters_AP_%s_%s.csv" % ("Road_traffic", imb_technique))
outfile_param_PM = os.path.join("/home/jongchan/Road_traffic", "parameters_PM_%s_%s.csv" % ("Road_traffic", imb_technique))
outfile_param_SC = os.path.join("/home/jongchan/Road_traffic", "parameters_SC_%s_%s.csv" % ("Road_traffic", imb_technique))

repeat = 0
for train_index, test_index in kf.split(X_dummy):
    X_train, X_test = X_dummy.iloc[train_index], X_dummy.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    y_test_AP = pd.DataFrame([1 if i == "Add penalty" else 0 for i in y_test['ACT_4']])
    y_test_PM = pd.DataFrame([1 if i == "Payment" else 0 for i in y_test['ACT_4']])
    y_test_SC = pd.DataFrame([1 if i == "Send for Credit Collection" else 0 for i in y_test['ACT_4']])
    train = pd.concat([X_train, y_train], axis=1)
    ACT_4_index = np.unique(data['ACT_1']).size + np.unique(data['ACT_2']).size + np.unique(data['ACT_3']).size + 1

    def train_and_test(train, ACT_4_index, act_name):
        NN = train[train.ACT_4 == act_name]
        NN_rest = train[train.ACT_4 != act_name]
        NN_rest = NN_rest.copy()
        NN_rest.iloc[:, ACT_4_index] = "Others"
        NN = NN.reset_index()
        NN_rest = NN_rest.reset_index()
        NN = NN.iloc[:, 1:ACT_4_index+2]        
        NN_rest = NN_rest.iloc[:, 1:ACT_4_index+2]        
        NN_ova = pd.concat([NN, NN_rest])
        NN_ova_X_train = NN_ova.iloc[:, 0:ACT_4_index]
        NN_ova_y_train = NN_ova.iloc[:, ACT_4_index]
        NN_X_res = NN_ova_X_train
        NN_y_res = NN_ova_y_train
        Counter(NN_ova_y_train)
        return NN, NN_rest, NN_ova, NN_ova_X_train, NN_ova_y_train, NN_X_res, NN_y_res

    AP, AP_rest, AP_ova, AP_ova_X_train, AP_ova_y_train, AP_X_res, AP_y_res = train_and_test(train, ACT_4_index, "Add penalty")
    PM, PM_rest, PM_ova, PM_ova_X_train, PM_ova_y_train, PM_X_res, PM_y_res = train_and_test(train, ACT_4_index, "Payment")
    SC, SC_rest, SC_ova, SC_ova_X_train, SC_ova_y_train, SC_X_res, SC_y_res = train_and_test(train, ACT_4_index, "Send for Credit Collection")

    AP_X_res, AP_y_res, PM_X_res, PM_y_res, SC_X_res, SC_y_res = resampling_assigner(imb_technique, AP_ova_X_train, AP_ova_y_train, PM_ova_X_train, PM_ova_y_train, SC_ova_X_train, SC_ova_y_train)

    first_digit_parameters = [x for x in itertools.product((5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100), repeat=1)]
    all_digit_parameters = first_digit_parameters
    learning_rate_init_parameters = [0.1, 0.01, 0.001]
    parameters = {'hidden_layer_sizes': all_digit_parameters,
                  'learning_rate_init': learning_rate_init_parameters}
    dnn_AP, dnn_PM, dnn_SC = MLPClassifier(max_iter=10000, activation='relu'), MLPClassifier(max_iter=10000, activation='relu'), MLPClassifier(max_iter=10000, activation='relu')
    dnn_AP_clf = RandomizedSearchCV(dnn_AP, parameters, n_jobs=-1, cv=5)
    dnn_AP_clf.fit(AP_X_res, AP_y_res)
    dnn_PM_clf = RandomizedSearchCV(dnn_PM, parameters, n_jobs=-1, cv=5)
    dnn_PM_clf.fit(PM_X_res, PM_y_res)
    dnn_SC_clf = RandomizedSearchCV(dnn_SC, parameters, n_jobs=-1, cv=5)
    dnn_SC_clf.fit(SC_X_res, SC_y_res)

    dnn_params_hls_AP[repeat] = dnn_AP_clf.best_params_['hidden_layer_sizes']
    dnn_params_hls_PM[repeat] = dnn_PM_clf.best_params_['hidden_layer_sizes']
    dnn_params_hls_SC[repeat] = dnn_SC_clf.best_params_['hidden_layer_sizes']
    dnn_params_lri_AP[repeat] = dnn_AP_clf.best_params_['learning_rate_init']
    dnn_params_lri_PM[repeat] = dnn_PM_clf.best_params_['learning_rate_init']
    dnn_params_lri_SC[repeat] = dnn_SC_clf.best_params_['learning_rate_init']

    solver = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
    tol = [1e-2, 1e-3, 1e-4, 1e-5]
    reg_strength = [0.5, 1.0, 1.5]
    parameters = {'solver': solver,
	          'tol': tol,
	          'C': reg_strength}
    lr_AP, lr_PM, lr_SC = LogisticRegression(), LogisticRegression(), LogisticRegression()
    lr_AP_clf = RandomizedSearchCV(lr_AP, parameters, n_jobs = -1, cv = 5)
    lr_AP_clf.fit(AP_X_res, AP_y_res)
    lr_PM_clf = RandomizedSearchCV(lr_PM, parameters, n_jobs = -1, cv = 5)
    lr_PM_clf.fit(PM_X_res, PM_y_res)
    lr_SC_clf = RandomizedSearchCV(lr_SC, parameters, n_jobs = -1, cv = 5)
    lr_SC_clf.fit(SC_X_res, SC_y_res)

    lr_params_solver_AP[repeat] = lr_AP_clf.best_params_['solver']
    lr_params_solver_PM[repeat] = lr_PM_clf.best_params_['solver']
    lr_params_solver_SC[repeat] = lr_SC_clf.best_params_['solver']
    lr_params_tol_AP[repeat] = lr_AP_clf.best_params_['tol']
    lr_params_tol_PM[repeat] = lr_PM_clf.best_params_['tol']
    lr_params_tol_SC[repeat] = lr_SC_clf.best_params_['tol']
    lr_params_C_AP[repeat] = lr_AP_clf.best_params_['C']
    lr_params_C_PM[repeat] = lr_PM_clf.best_params_['C']
    lr_params_C_SC[repeat] = lr_SC_clf.best_params_['C']

    var_smoothing = [1e-07, 1e-08, 1e-09]
    parameters = {'var_smoothing': var_smoothing}
    nb_AP, nb_PM, nb_SC = GaussianNB(), GaussianNB(), GaussianNB()
    nb_AP_clf = RandomizedSearchCV(nb_AP, parameters, n_jobs = -1, cv = 5)
    nb_AP_clf.fit(AP_X_res, AP_y_res)
    nb_PM_clf = RandomizedSearchCV(nb_PM, parameters, n_jobs = -1, cv = 5)
    nb_PM_clf.fit(PM_X_res, PM_y_res)
    nb_SC_clf = RandomizedSearchCV(nb_SC, parameters, n_jobs = -1, cv = 5)
    nb_SC_clf.fit(SC_X_res, SC_y_res)

    nb_params_vs_AP[repeat] = nb_AP_clf.best_params_['var_smoothing']
    nb_params_vs_PM[repeat] = nb_PM_clf.best_params_['var_smoothing']
    nb_params_vs_SC[repeat] = nb_SC_clf.best_params_['var_smoothing']

    n_tree = [50, 100, 200, 300, 400, 500, 600, 700]
    max_depth = [10, 20, 30, 40, 50, 60, 70]
    min_samples_split = [5, 10, 15, 20, 25, 30]
    parameters = {'n_estimators': n_tree,
		  'max_depth': max_depth,
		  'min_samples_split': min_samples_split}
    rf_AP, rf_PM, rf_SC = RandomForestClassifier(), RandomForestClassifier(), RandomForestClassifier()
    rf_AP_clf = RandomizedSearchCV(rf_AP, parameters, n_jobs = -1, cv=5)
    rf_AP_clf.fit(AP_X_res, AP_y_res)
    rf_PM_clf = RandomizedSearchCV(rf_PM, parameters, n_jobs = -1, cv=5)
    rf_PM_clf.fit(PM_X_res, PM_y_res)
    rf_SC_clf = RandomizedSearchCV(rf_SC, parameters, n_jobs = -1, cv=5)
    rf_SC_clf.fit(SC_X_res, SC_y_res)

    rf_params_est_AP[repeat] = rf_AP_clf.best_params_['n_estimators']
    rf_params_est_PM[repeat] = rf_PM_clf.best_params_['n_estimators']
    rf_params_est_SC[repeat] = rf_SC_clf.best_params_['n_estimators']
    rf_params_md_AP[repeat] = rf_AP_clf.best_params_['max_depth']
    rf_params_md_PM[repeat] = rf_PM_clf.best_params_['max_depth']
    rf_params_md_SC[repeat] = rf_SC_clf.best_params_['max_depth']
    rf_params_mss_AP[repeat] = rf_AP_clf.best_params_['min_samples_split']
    rf_params_mss_PM[repeat] = rf_PM_clf.best_params_['min_samples_split']
    rf_params_mss_SC[repeat] = rf_SC_clf.best_params_['min_samples_split']

    tol = [1e-2, 1e-3, 1e-4]
    parameters = {'tol': tol,
                  'kernel': ['linear'],
                  'probability': [True]}
    svm_AP, svm_PM, svm_SC = SVC(), SVC(), SVC()
    svm_AP_clf = RandomizedSearchCV(svm_AP, parameters, n_jobs = -1, cv = 5)
    svm_AP_clf.fit(AP_X_res, AP_y_res)
    svm_PM_clf = RandomizedSearchCV(svm_PM, parameters, n_jobs = -1, cv = 5)
    svm_PM_clf.fit(PM_X_res, PM_y_res)
    svm_SC_clf = RandomizedSearchCV(svm_SC, parameters, n_jobs = -1, cv = 5)
    svm_SC_clf.fit(SC_X_res, SC_y_res)

    svm_params_tol_AP[repeat] = svm_AP_clf.best_params_['tol']
    svm_params_tol_PM[repeat] = svm_PM_clf.best_params_['tol']
    svm_params_tol_SC[repeat] = svm_SC_clf.best_params_['tol']

    def prediction(AP_clf, PM_clf, SC_clf, X_test):
        pred_class_AP = AP_clf.predict(X_test)
        pred_prob_AP = AP_clf.predict_proba(X_test)
        pred_class_PM = PM_clf.predict(X_test)
        pred_prob_PM = PM_clf.predict_proba(X_test)
        pred_class_SC = SC_clf.predict(X_test)
        pred_prob_SC = SC_clf.predict_proba(X_test)
        return pred_class_AP, pred_prob_AP, pred_class_PM, pred_prob_PM, pred_class_SC, pred_prob_SC

    dnn_pred_class_AP, dnn_pred_prob_AP, dnn_pred_class_PM, dnn_pred_prob_PM, dnn_pred_class_SC, dnn_pred_prob_SC = prediction(dnn_AP_clf, dnn_PM_clf, dnn_SC_clf, X_test)
    lr_pred_class_AP, lr_pred_prob_AP, lr_pred_class_PM, lr_pred_prob_PM, lr_pred_class_SC, lr_pred_prob_SC = prediction(lr_AP_clf, lr_PM_clf, lr_SC_clf, X_test)
    nb_pred_class_AP, nb_pred_prob_AP, nb_pred_class_PM, nb_pred_prob_PM, nb_pred_class_SC, nb_pred_prob_SC = prediction(nb_AP_clf, nb_PM_clf, nb_SC_clf, X_test)
    rf_pred_class_AP, rf_pred_prob_AP, rf_pred_class_PM, rf_pred_prob_PM, rf_pred_class_SC, rf_pred_prob_SC = prediction(rf_AP_clf, rf_PM_clf, rf_SC_clf, X_test)
    svm_pred_class_AP, svm_pred_prob_AP, svm_pred_class_PM, svm_pred_prob_PM, svm_pred_class_SC, svm_pred_prob_SC = prediction(svm_AP_clf, svm_PM_clf, svm_SC_clf, X_test)

    dnn_prediction, lr_prediction, nb_prediction, rf_prediction, svm_prediction = pd.DataFrame(columns=['Prediction']), pd.DataFrame(columns=['Prediction']), pd.DataFrame(columns=['Prediction']), pd.DataFrame(columns=['Prediction']), pd.DataFrame(columns=['Prediction'])

    def label_assigner(y_test, prediction, pred_class_AP, pred_class_PM, pred_class_SC, pred_prob_AP, pred_prob_PM, pred_prob_SC):
        for i in range(0, len(y_test)):
            AP_index, PM_index, SC_index = 0, 0, 0
            if pred_class_AP[i] == "Add penalty":
                AP_index = 0 if pred_prob_AP[i][0] >= 0.5 else 1
            elif pred_class_AP[i] == "Others":
                AP_index = 0 if pred_prob_AP[i][0] < 0.5 else 1
            if pred_class_PM[i] == "Payment":
                PM_index = 0 if pred_prob_PM[i][0] >= 0.5 else 1
            elif pred_class_PM[i] == "Others":
                PM_index = 0 if pred_prob_PM[i][0] < 0.5 else 1
            if pred_class_SC[i] == "Accepted/Wait":
                SC_index = 0 if pred_prob_SC[i][0] >= 0.5 else 1
            elif pred_class_SC[i] == "Others":
                SC_index = 0 if pred_prob_SC[i][0] < 0.5 else 1
            if pred_prob_AP[i][AP_index] == max(pred_prob_AP[i][AP_index], pred_prob_PM[i][PM_index],
                                                        pred_prob_SC[i][SC_index]):
                prediction.loc[i] = "Add penalty"
            elif pred_prob_PM[i][PM_index] == max(pred_prob_AP[i][AP_index], pred_prob_PM[i][PM_index],
                                                          pred_prob_SC[i][SC_index]):
                prediction.loc[i] = "Payment"
            elif pred_prob_SC[i][SC_index] == max(pred_prob_AP[i][AP_index], pred_prob_PM[i][PM_index],
                                                          pred_prob_SC[i][SC_index]):
                prediction.loc[i] = "Send for Credit Collection"
        return pred_prob_AP, pred_prob_PM, pred_prob_SC, prediction

    def get_precision(conf_matrix):
        tp_1 = conf_matrix[0][0]
        tp_2 = conf_matrix[1][1]
        tp_3 = conf_matrix[2][2]
        fp_1 = conf_matrix[1][0] + conf_matrix[2][0]
        fp_2 = conf_matrix[0][1] + conf_matrix[2][1]
        fp_3 = conf_matrix[0][2] + conf_matrix[1][2]
        precision_1 = 0 if tp_1 + fp_1 == 0 else tp_1 / (tp_1 + fp_1)
        precision_2 = 0 if tp_2 + fp_2 == 0 else tp_2 / (tp_2 + fp_2)
        precision_3 = 0 if tp_3 + fp_3 == 0 else tp_3 / (tp_3 + fp_3)
        precision_avg = (precision_1 + precision_2 + precision_3) / 3
        return precision_avg

    def get_recall_pen_1(conf_matrix):
        tp_1 = conf_matrix[0][0]
        tp_2 = conf_matrix[1][1]
        tp_3 = conf_matrix[2][2]
        fn_1 = conf_matrix[0][1] + conf_matrix[0][2] + conf_matrix[0][3] + conf_matrix[0][4]
        fn_2 = conf_matrix[1][0] + conf_matrix[1][2] + conf_matrix[1][3] + conf_matrix[1][4]
        fn_3 = conf_matrix[2][0] + conf_matrix[2][1] + conf_matrix[2][3] + conf_matrix[2][4]
        recall_1 = 0 if tp_1 + fn_1 == 0 else tp_1 / (tp_1 + fn_1)
        recall_2 = 0 if tp_2 + fn_2 == 0 else tp_2 / (tp_2 + fn_2)
        recall_3 = 0 if tp_3 + fn_3 == 0 else tp_3 / (tp_3 + fn_3)
        recall_avg_pen_1 = (recall_1 + recall_2 + recall_3) / (3+1-1)
        return recall_avg_pen_1

    def get_recall_pen_5(conf_matrix):
        tp_1 = conf_matrix[0][0]
        tp_2 = conf_matrix[1][1]
        tp_3 = conf_matrix[2][2]
        fn_1 = conf_matrix[0][1] + conf_matrix[0][2] + conf_matrix[0][3] + conf_matrix[0][4]
        fn_2 = conf_matrix[1][0] + conf_matrix[1][2] + conf_matrix[1][3] + conf_matrix[1][4]
        fn_3 = conf_matrix[2][0] + conf_matrix[2][1] + conf_matrix[2][3] + conf_matrix[2][4]
        recall_1 = 0 if  tp_1 + fn_1 == 0 else tp_1 / (tp_1 + fn_1)
        recall_2 = 0 if  tp_2 + fn_2 == 0 else tp_2 / (tp_2 + fn_2)
        recall_3 = 0 if  tp_3 + fn_3 == 0 else tp_3 / (tp_3 + fn_3)
        recall_avg_pen_5 = (recall_1 + (5*recall_2) + recall_3) / (3+5-1)
        return recall_avg_pen_5

    dnn_pred_prob_AP, dnn_pred_prob_PM, dnn_pred_prob_SC, dnn_prediction = label_assigner(y_test, dnn_prediction, dnn_pred_class_AP, dnn_pred_class_PM, dnn_pred_class_SC, dnn_pred_prob_AP, dnn_pred_prob_PM, dnn_pred_prob_SC)
    lr_pred_prob_AP, lr_pred_prob_PM, lr_pred_prob_SC, lr_prediction = label_assigner(y_test, lr_prediction, lr_pred_class_AP, lr_pred_class_PM, lr_pred_class_SC, lr_pred_prob_AP, lr_pred_prob_PM, lr_pred_prob_SC)
    nb_pred_prob_AP, nb_pred_prob_PM, nb_pred_prob_SC, nb_prediction = label_assigner(y_test, nb_prediction, nb_pred_class_AP, nb_pred_class_PM, nb_pred_class_SC, nb_pred_prob_AP, nb_pred_prob_PM, nb_pred_prob_SC)
    rf_pred_prob_AP, rf_pred_prob_PM, rf_pred_prob_SC, rf_prediction = label_assigner(y_test, rf_prediction, rf_pred_class_AP, rf_pred_class_PM, rf_pred_class_SC, rf_pred_prob_AP, rf_pred_prob_PM, rf_pred_prob_SC)
    svm_pred_prob_AP, svm_pred_prob_PM, svm_pred_prob_SC, svm_prediction = label_assigner(y_test, svm_prediction, svm_pred_class_AP, svm_pred_class_PM, svm_pred_class_SC, svm_pred_prob_AP, svm_pred_prob_PM, svm_pred_prob_SC)

    def perf_evaluator(y_test, pred_prob_AP, pred_prob_PM, pred_prob_SC, prediction, f1_score_pen_1_kfoldcv, f1_score_pen_5_kfoldcv, ovr_accuracy_kfoldcv, auc_kfoldcv, gmean_kfoldcv, clf, repeat):
        conf_matrix = confusion_matrix(y_test, prediction)
        precision = get_precision(conf_matrix)
        recall_pen_1 = get_recall_pen_1(conf_matrix)
        recall_pen_5 = get_recall_pen_5(conf_matrix)
        f1_score_pen_1 = 2 * (precision * recall_pen_1) / (precision + recall_pen_1)
        f1_score_pen_5 = 2 * (precision * recall_pen_5) / (precision + recall_pen_5)
        ovr_accuracy = (conf_matrix[0][0] + conf_matrix[1][1] + conf_matrix[2][2]) / (sum(conf_matrix[0]) + sum(conf_matrix[1]) + sum(conf_matrix[2]))
        auc_AP = roc_auc_score(y_true = y_test_AP, y_score = pd.DataFrame(pred_prob_AP).iloc[:,0])
        auc_PM = roc_auc_score(y_true = y_test_PM, y_score = pd.DataFrame(pred_prob_PM).iloc[:,0])
        auc_SC = roc_auc_score(y_true = y_test_SC, y_score = pd.DataFrame(pred_prob_SC).iloc[:,0])
        gmean = geometric_mean_score(y_true = y_test, y_pred = prediction, average = 'macro')
        conf_matrix = pd.DataFrame(conf_matrix)               
        conf_matrix.to_csv('conf_matrix_'+imb_technique+'_' + clf + '_bpic2013_closed_'+ str(nsplits) +'foldcv_' + str(repeat+1)+'.csv',header=False,index=False) #First repetition
        #conf_matrix.to_csv('conf_matrix_'+imb_technique+'_penalty_' + str(penalty) + '_' + clf + '_bpic2013_closed_'+ str(nsplits) +'foldcv_' + str(repeat+6)+'.csv',header=False,index=False) #Second repetition
        f1_score_pen_1_kfoldcv[repeat] = f1_score_pen_1
        f1_score_pen_5_kfoldcv[repeat] = f1_score_pen_5
        ovr_accuracy_kfoldcv[repeat] = ovr_accuracy
        auc_kfoldcv[repeat] = (auc_AP + auc_PM + auc_SC)/3
        gmean_kfoldcv[repeat] = gmean
        return conf_matrix, f1_score_pen_1_kfoldcv, f1_score_pen_5_kfoldcv, ovr_accuracy_kfoldcv, auc_kfoldcv, gmean_kfoldcv

    dnn_conf_matrix, dnn_f1_score_pen_1_kfoldcv, dnn_f1_score_pen_5_kfoldcv, dnn_ovr_accuracy_kfoldcv, dnn_auc_kfoldcv, dnn_gmean_kfoldcv = perf_evaluator(y_test, dnn_pred_prob_AP, dnn_pred_prob_PM, dnn_pred_prob_SC, dnn_prediction, dnn_f1_score_pen_1_kfoldcv, dnn_f1_score_pen_5_kfoldcv, dnn_ovr_accuracy_kfoldcv, dnn_auc_kfoldcv, dnn_gmean_kfoldcv, "dnn", repeat)
    lr_conf_matrix, lr_f1_score_pen_1_kfoldcv, lr_f1_score_pen_5_kfoldcv, lr_ovr_accuracy_kfoldcv, lr_auc_kfoldcv, lr_gmean_kfoldcv = perf_evaluator(y_test, lr_pred_prob_AP, lr_pred_prob_PM, lr_pred_prob_SC, lr_prediction, lr_f1_score_pen_1_kfoldcv, lr_f1_score_pen_5_kfoldcv, lr_ovr_accuracy_kfoldcv, lr_auc_kfoldcv, lr_gmean_kfoldcv, "lr", repeat)
    nb_conf_matrix, nb_f1_score_pen_1_kfoldcv, nb_f1_score_pen_5_kfoldcv, nb_ovr_accuracy_kfoldcv, nb_auc_kfoldcv, nb_gmean_kfoldcv = perf_evaluator(y_test, nb_pred_prob_AP, nb_pred_prob_PM, nb_pred_prob_SC, nb_prediction, nb_f1_score_pen_1_kfoldcv, nb_f1_score_pen_5_kfoldcv, nb_ovr_accuracy_kfoldcv, nb_auc_kfoldcv, nb_gmean_kfoldcv, "nb", repeat)
    rf_conf_matrix, rf_f1_score_pen_1_kfoldcv, rf_f1_score_pen_5_kfoldcv, rf_ovr_accuracy_kfoldcv, rf_auc_kfoldcv, rf_gmean_kfoldcv = perf_evaluator(y_test, rf_pred_prob_AP, rf_pred_prob_PM, rf_pred_prob_SC, rf_prediction, rf_f1_score_pen_1_kfoldcv, rf_f1_score_pen_5_kfoldcv, rf_ovr_accuracy_kfoldcv, rf_auc_kfoldcv, rf_gmean_kfoldcv, "rf", repeat)
    svm_conf_matrix, svm_f1_score_pen_1_kfoldcv, svm_f1_score_pen_5_kfoldcv, svm_ovr_accuracy_kfoldcv, svm_auc_kfoldcv, svm_gmean_kfoldcv = perf_evaluator(y_test, svm_pred_prob_AP, svm_pred_prob_PM, svm_pred_prob_SC, svm_prediction, svm_f1_score_pen_1_kfoldcv, svm_f1_score_pen_5_kfoldcv, svm_ovr_accuracy_kfoldcv, svm_auc_kfoldcv, svm_gmean_kfoldcv, "svm", repeat)

    repeat = repeat + 1
with open(outfile, 'w') as fout:
    fout.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % ("dnn_f1_pen1", "dnn_f1_pen5", "dnn_ovr_acc", "dnn_auc", "dnn_gmean","lr_f1_pen1", "lr_f1_pen5", "lr_ovr_acc", "lr_auc", "lr_gmean","nb_f1_pen1", "nb_f1_pen5", "nb_ovr_acc", "nb_auc", "nb_gmean", "rf_f1_pen1", "rf_f1_pen5", "rf_ovr_acc", "rf_auc", "rf_gmean","svm_f1_pen1", "svm_f1_pen5", "svm_ovr_acc", "svm_auc", "svm_gmean"))
    fout.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_f1_score_pen_1_kfoldcv[0], dnn_f1_score_pen_5_kfoldcv[0], dnn_ovr_accuracy_kfoldcv[0], dnn_auc_kfoldcv[0], dnn_gmean_kfoldcv[0], lr_f1_score_pen_1_kfoldcv[0], lr_f1_score_pen_5_kfoldcv[0], lr_ovr_accuracy_kfoldcv[0], lr_auc_kfoldcv[0], lr_gmean_kfoldcv[0], nb_f1_score_pen_1_kfoldcv[0], nb_f1_score_pen_5_kfoldcv[0], nb_ovr_accuracy_kfoldcv[0], nb_auc_kfoldcv[0], nb_gmean_kfoldcv[0], rf_f1_score_pen_1_kfoldcv[0], rf_f1_score_pen_5_kfoldcv[0], rf_ovr_accuracy_kfoldcv[0], rf_auc_kfoldcv[0], rf_gmean_kfoldcv[0], svm_f1_score_pen_1_kfoldcv[0], svm_f1_score_pen_5_kfoldcv[0], svm_ovr_accuracy_kfoldcv[0], svm_auc_kfoldcv[0], svm_gmean_kfoldcv[0]))
    fout.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_f1_score_pen_1_kfoldcv[1], dnn_f1_score_pen_5_kfoldcv[1], dnn_ovr_accuracy_kfoldcv[1], dnn_auc_kfoldcv[1], dnn_gmean_kfoldcv[1], lr_f1_score_pen_1_kfoldcv[1], lr_f1_score_pen_5_kfoldcv[1], lr_ovr_accuracy_kfoldcv[1], lr_auc_kfoldcv[1], lr_gmean_kfoldcv[1], nb_f1_score_pen_1_kfoldcv[1], nb_f1_score_pen_5_kfoldcv[1], nb_ovr_accuracy_kfoldcv[1], nb_auc_kfoldcv[1], nb_gmean_kfoldcv[1], rf_f1_score_pen_1_kfoldcv[1], rf_f1_score_pen_5_kfoldcv[1], rf_ovr_accuracy_kfoldcv[1], rf_auc_kfoldcv[1], rf_gmean_kfoldcv[1], svm_f1_score_pen_1_kfoldcv[1], svm_f1_score_pen_5_kfoldcv[1], svm_ovr_accuracy_kfoldcv[1], svm_auc_kfoldcv[1], svm_gmean_kfoldcv[1]))
    fout.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_f1_score_pen_1_kfoldcv[2], dnn_f1_score_pen_5_kfoldcv[2], dnn_ovr_accuracy_kfoldcv[2], dnn_auc_kfoldcv[2], dnn_gmean_kfoldcv[2], lr_f1_score_pen_1_kfoldcv[2], lr_f1_score_pen_5_kfoldcv[2], lr_ovr_accuracy_kfoldcv[2], lr_auc_kfoldcv[2], lr_gmean_kfoldcv[2], nb_f1_score_pen_1_kfoldcv[2], nb_f1_score_pen_5_kfoldcv[2], nb_ovr_accuracy_kfoldcv[2], nb_auc_kfoldcv[2], nb_gmean_kfoldcv[2], rf_f1_score_pen_1_kfoldcv[2], rf_f1_score_pen_5_kfoldcv[2], rf_ovr_accuracy_kfoldcv[2], rf_auc_kfoldcv[2], rf_gmean_kfoldcv[2], svm_f1_score_pen_1_kfoldcv[2], svm_f1_score_pen_5_kfoldcv[2], svm_ovr_accuracy_kfoldcv[2], svm_auc_kfoldcv[2], svm_gmean_kfoldcv[2]))
    fout.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_f1_score_pen_1_kfoldcv[3], dnn_f1_score_pen_5_kfoldcv[3], dnn_ovr_accuracy_kfoldcv[3], dnn_auc_kfoldcv[3], dnn_gmean_kfoldcv[3], lr_f1_score_pen_1_kfoldcv[3], lr_f1_score_pen_5_kfoldcv[3], lr_ovr_accuracy_kfoldcv[3], lr_auc_kfoldcv[3], lr_gmean_kfoldcv[3], nb_f1_score_pen_1_kfoldcv[3], nb_f1_score_pen_5_kfoldcv[3], nb_ovr_accuracy_kfoldcv[3], nb_auc_kfoldcv[3], nb_gmean_kfoldcv[3], rf_f1_score_pen_1_kfoldcv[3], rf_f1_score_pen_5_kfoldcv[3], rf_ovr_accuracy_kfoldcv[3], rf_auc_kfoldcv[3], rf_gmean_kfoldcv[3], svm_f1_score_pen_1_kfoldcv[3], svm_f1_score_pen_5_kfoldcv[3], svm_ovr_accuracy_kfoldcv[3], svm_auc_kfoldcv[3], svm_gmean_kfoldcv[3]))
    fout.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_f1_score_pen_1_kfoldcv[4], dnn_f1_score_pen_5_kfoldcv[4], dnn_ovr_accuracy_kfoldcv[4], dnn_auc_kfoldcv[4], dnn_gmean_kfoldcv[4], lr_f1_score_pen_1_kfoldcv[4], lr_f1_score_pen_5_kfoldcv[4], lr_ovr_accuracy_kfoldcv[4], lr_auc_kfoldcv[4], lr_gmean_kfoldcv[4], nb_f1_score_pen_1_kfoldcv[4], nb_f1_score_pen_5_kfoldcv[4], nb_ovr_accuracy_kfoldcv[4], nb_auc_kfoldcv[4], nb_gmean_kfoldcv[4], rf_f1_score_pen_1_kfoldcv[4], rf_f1_score_pen_5_kfoldcv[4], rf_ovr_accuracy_kfoldcv[4], rf_auc_kfoldcv[4], rf_gmean_kfoldcv[4], svm_f1_score_pen_1_kfoldcv[4], svm_f1_score_pen_5_kfoldcv[4], svm_ovr_accuracy_kfoldcv[4], svm_auc_kfoldcv[4], svm_gmean_kfoldcv[4]))
with open(outfile_param_AP, 'w') as fout_param_AP:
    fout_param_AP.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % ("dnn_hidden_layer_sizes", "dnn_learning_rate_init", "lr_solver", "lr_tol", "lr_C", "nb_var_smoothing", "rf_n_estimator", "rf_max_depth", "rf_min_sample_split", "svm_tol"))
    fout_param_AP.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_AP[0], dnn_params_lri_AP[0], lr_params_solver_AP[0], lr_params_tol_AP[0], lr_params_C_AP[0], nb_params_vs_AP[0], rf_params_est_AP[0], rf_params_md_AP[0], rf_params_mss_AP[0], svm_params_tol_AP[0]))
    fout_param_AP.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_AP[1], dnn_params_lri_AP[1], lr_params_solver_AP[1], lr_params_tol_AP[1], lr_params_C_AP[1], nb_params_vs_AP[1], rf_params_est_AP[1], rf_params_md_AP[1], rf_params_mss_AP[1], svm_params_tol_AP[1]))
    fout_param_AP.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_AP[2], dnn_params_lri_AP[2], lr_params_solver_AP[2], lr_params_tol_AP[2], lr_params_C_AP[2], nb_params_vs_AP[2], rf_params_est_AP[2], rf_params_md_AP[2], rf_params_mss_AP[2], svm_params_tol_AP[2]))
    fout_param_AP.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_AP[3], dnn_params_lri_AP[3], lr_params_solver_AP[3], lr_params_tol_AP[3], lr_params_C_AP[3], nb_params_vs_AP[3], rf_params_est_AP[3], rf_params_md_AP[3], rf_params_mss_AP[3], svm_params_tol_AP[3]))
    fout_param_AP.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_AP[4], dnn_params_lri_AP[4], lr_params_solver_AP[4], lr_params_tol_AP[4], lr_params_C_AP[4], nb_params_vs_AP[4], rf_params_est_AP[4], rf_params_md_AP[4], rf_params_mss_AP[4], svm_params_tol_AP[4]))
with open(outfile_param_PM, 'w') as fout_param_PM:
    fout_param_PM.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % ("dnn_hidden_layer_sizes", "dnn_learning_rate_init", "lr_solver", "lr_tol", "lr_C", "nb_var_smoothing", "rf_n_estimator", "rf_max_depth", "rf_min_sample_split", "svm_tol"))
    fout_param_PM.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_PM[0], dnn_params_lri_PM[0], lr_params_solver_PM[0], lr_params_tol_PM[0], lr_params_C_PM[0], nb_params_vs_PM[0], rf_params_est_PM[0], rf_params_md_PM[0], rf_params_mss_PM[0], svm_params_tol_PM[0]))
    fout_param_PM.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_PM[1], dnn_params_lri_PM[1], lr_params_solver_PM[1], lr_params_tol_PM[1], lr_params_C_PM[1], nb_params_vs_PM[1], rf_params_est_PM[1], rf_params_md_PM[1], rf_params_mss_PM[1], svm_params_tol_PM[1]))
    fout_param_PM.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_PM[2], dnn_params_lri_PM[2], lr_params_solver_PM[2], lr_params_tol_PM[2], lr_params_C_PM[2], nb_params_vs_PM[2], rf_params_est_PM[2], rf_params_md_PM[2], rf_params_mss_PM[2], svm_params_tol_PM[2]))
    fout_param_PM.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_PM[3], dnn_params_lri_PM[3], lr_params_solver_PM[3], lr_params_tol_PM[3], lr_params_C_PM[3], nb_params_vs_PM[3], rf_params_est_PM[3], rf_params_md_PM[3], rf_params_mss_PM[3], svm_params_tol_PM[3]))
    fout_param_PM.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_PM[4], dnn_params_lri_PM[4], lr_params_solver_PM[4], lr_params_tol_PM[4], lr_params_C_PM[4], nb_params_vs_PM[4], rf_params_est_PM[4], rf_params_md_PM[4], rf_params_mss_PM[4], svm_params_tol_PM[4]))
with open(outfile_param_SC, 'w') as fout_param_SC:
    fout_param_SC.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % ("dnn_hidden_layer_sizes", "dnn_learning_rate_init", "lr_solver", "lr_tol", "lr_C", "nb_var_smoothing", "rf_n_estimator", "rf_max_depth", "rf_min_sample_split", "svm_tol"))
    fout_param_SC.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_SC[0], dnn_params_lri_SC[0], lr_params_solver_SC[0], lr_params_tol_SC[0], lr_params_C_SC[0], nb_params_vs_SC[0], rf_params_est_SC[0], rf_params_md_SC[0], rf_params_mss_SC[0], svm_params_tol_SC[0]))
    fout_param_SC.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_SC[1], dnn_params_lri_SC[1], lr_params_solver_SC[1], lr_params_tol_SC[1], lr_params_C_SC[1], nb_params_vs_SC[1], rf_params_est_SC[1], rf_params_md_SC[1], rf_params_mss_SC[1], svm_params_tol_SC[1]))
    fout_param_SC.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_SC[2], dnn_params_lri_SC[2], lr_params_solver_SC[2], lr_params_tol_SC[2], lr_params_C_SC[2], nb_params_vs_SC[2], rf_params_est_SC[2], rf_params_md_SC[2], rf_params_mss_SC[2], svm_params_tol_SC[2]))
    fout_param_SC.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_SC[3], dnn_params_lri_SC[3], lr_params_solver_SC[3], lr_params_tol_SC[3], lr_params_C_SC[3], nb_params_vs_SC[3], rf_params_est_SC[3], rf_params_md_SC[3], rf_params_mss_SC[3], svm_params_tol_SC[3]))
    fout_param_SC.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_SC[4], dnn_params_lri_SC[4], lr_params_solver_SC[4], lr_params_tol_SC[4], lr_params_C_SC[4], nb_params_vs_SC[4], rf_params_est_SC[4], rf_params_md_SC[4], rf_params_mss_SC[4], svm_params_tol_SC[4]))
dnn_f1_score_pen_1_kfoldcv[5] = "Average" #To let users figure out that f1_score_kfoldcv[6] value is the average when seeing the csv file
dnn_f1_score_pen_1_kfoldcv[6] = (dnn_f1_score_pen_1_kfoldcv[0]+dnn_f1_score_pen_1_kfoldcv[1]+dnn_f1_score_pen_1_kfoldcv[2]+dnn_f1_score_pen_1_kfoldcv[3]+dnn_f1_score_pen_1_kfoldcv[4])/5
dnn_f1_score_pen_1_kfoldcv = pd.DataFrame(dnn_f1_score_pen_1_kfoldcv)
dnn_f1_score_pen_1_kfoldcv.to_csv('f1_score_'+imb_technique+'_penalty_1_dnn_road_traffic_' + str(nsplits) + 'foldcv_1~5.csv',header=False,index=False) #First repetition
dnn_f1_score_pen_5_kfoldcv[5] = "Average" #To let users figure out that f1_score_kfoldcv[6] value is the average when seeing the csv file
dnn_f1_score_pen_5_kfoldcv[6] = (dnn_f1_score_pen_5_kfoldcv[0]+dnn_f1_score_pen_5_kfoldcv[1]+dnn_f1_score_pen_5_kfoldcv[2]+dnn_f1_score_pen_5_kfoldcv[3]+dnn_f1_score_pen_5_kfoldcv[4])/5
dnn_f1_score_pen_5_kfoldcv = pd.DataFrame(dnn_f1_score_pen_5_kfoldcv)
dnn_f1_score_pen_5_kfoldcv.to_csv('f1_score_'+imb_technique+'_penalty_5_dnn_road_traffic_' + str(nsplits) + 'foldcv_1~5.csv',header=False,index=False) #First repetition
#dnn_f1_score_kfoldcv.to_csv('f1_score_'+imb_technique+'_penalty_' + str(penalty) + '_dnn_road_traffic_' + str(nsplits) + 'foldcv_6~10.csv',header=False,index=False) #Second repetition
dnn_ovr_accuracy_kfoldcv[5] = "Average" #To let users figure out that f1_score_kfoldcv[6] value is the average when seeing the csv file
dnn_ovr_accuracy_kfoldcv[6] = (dnn_ovr_accuracy_kfoldcv[0]+dnn_ovr_accuracy_kfoldcv[1]+dnn_ovr_accuracy_kfoldcv[2]+dnn_ovr_accuracy_kfoldcv[3]+dnn_ovr_accuracy_kfoldcv[4])/5
dnn_ovr_accuracy_kfoldcv = pd.DataFrame(dnn_ovr_accuracy_kfoldcv)
dnn_ovr_accuracy_kfoldcv.to_csv('ovr_accuracy_'+imb_technique + '_dnn_road_traffic_' + str(nsplits) + 'foldcv_1~5.csv',header=False,index=False) #First repetition
#dnn_ovr_accuracy_kfoldcv.to_csv('ovr_accuracy_'+imb_technique+'_penalty_' + str(penalty) + '_dnn_road_traffic_' + str(nsplits) + 'foldcv_6~10.csv',header=False,index=False) #Second repetition
dnn_auc_kfoldcv[5] = "Average" #To let users figure out that f1_score_kfoldcv[6] value is the average when seeing the csv file
dnn_auc_kfoldcv[6] = (dnn_auc_kfoldcv[0]+dnn_auc_kfoldcv[1]+dnn_auc_kfoldcv[2]+dnn_auc_kfoldcv[3]+dnn_auc_kfoldcv[4])/5
dnn_auc_kfoldcv = pd.DataFrame(dnn_auc_kfoldcv)
dnn_auc_kfoldcv.to_csv('auc_'+imb_technique+'_dnn_road_traffic_' + str(nsplits) + 'foldcv_1~5.csv',header=False,index=False)
dnn_gmean_kfoldcv[5] = "Average" #To let users figure out that f1_score_kfoldcv[6] value is the average when seeing the csv file
dnn_gmean_kfoldcv[6] = (dnn_gmean_kfoldcv[0]+dnn_gmean_kfoldcv[1]+dnn_gmean_kfoldcv[2]+dnn_gmean_kfoldcv[3]+dnn_gmean_kfoldcv[4])/5
dnn_gmean_kfoldcv = pd.DataFrame(dnn_gmean_kfoldcv)
dnn_gmean_kfoldcv.to_csv('gmean_'+imb_technique+'_dnn_road_traffic_' + str(nsplits) + 'foldcv_1~5.csv',header=False,index=False)

lr_f1_score_pen_1_kfoldcv[5] = "Average" #To let users figure out that f1_score_kfoldcv[6] value is the average when seeing the csv file
lr_f1_score_pen_1_kfoldcv[6] = (lr_f1_score_pen_1_kfoldcv[0]+lr_f1_score_pen_1_kfoldcv[1]+lr_f1_score_pen_1_kfoldcv[2]+lr_f1_score_pen_1_kfoldcv[3]+lr_f1_score_pen_1_kfoldcv[4])/5
lr_f1_score_pen_1_kfoldcv = pd.DataFrame(lr_f1_score_pen_1_kfoldcv)
lr_f1_score_pen_1_kfoldcv.to_csv('f1_score_'+imb_technique+'_penalty_1_lr_road_traffic_' + str(nsplits) + 'foldcv_1~5.csv',header=False,index=False) #First repetition
lr_f1_score_pen_5_kfoldcv[5] = "Average" #To let users figure out that f1_score_kfoldcv[6] value is the average when seeing the csv file
lr_f1_score_pen_5_kfoldcv[6] = (lr_f1_score_pen_5_kfoldcv[0]+lr_f1_score_pen_5_kfoldcv[1]+lr_f1_score_pen_5_kfoldcv[2]+lr_f1_score_pen_5_kfoldcv[3]+lr_f1_score_pen_5_kfoldcv[4])/5
lr_f1_score_pen_5_kfoldcv = pd.DataFrame(lr_f1_score_pen_5_kfoldcv)
lr_f1_score_pen_5_kfoldcv.to_csv('f1_score_'+imb_technique+'_penalty_5_lr_road_traffic_' + str(nsplits) + 'foldcv_1~5.csv',header=False,index=False) #First repetition
#lr_f1_score_kfoldcv.to_csv('f1_score_'+imb_technique+'_penalty_' + str(penalty) + '_lr_road_traffic_' + str(nsplits) + 'foldcv_6~10.csv',header=False,index=False) #Second repetition
lr_ovr_accuracy_kfoldcv[5] = "Average" #To let users figure out that f1_score_kfoldcv[6] value is the average when seeing the csv file
lr_ovr_accuracy_kfoldcv[6] = (lr_ovr_accuracy_kfoldcv[0]+lr_ovr_accuracy_kfoldcv[1]+lr_ovr_accuracy_kfoldcv[2]+lr_ovr_accuracy_kfoldcv[3]+lr_ovr_accuracy_kfoldcv[4])/5
lr_ovr_accuracy_kfoldcv = pd.DataFrame(lr_ovr_accuracy_kfoldcv)
lr_ovr_accuracy_kfoldcv.to_csv('ovr_accuracy_' + imb_technique + '_lr_road_traffic_' + str(nsplits) + 'foldcv_1~5.csv',header=False,index=False) #First repetition
#lr_ovr_accuracy_kfoldcv.to_csv('ovr_accuracy_'+imb_technique+'_penalty_' + str(penalty) + '_lr_road_traffic_' + str(nsplits) + 'foldcv_6~10.csv',header=False,index=False) #Second repetition
lr_auc_kfoldcv[5] = "Average" #To let users figure out that f1_score_kfoldcv[6] value is the average when seeing the csv file
lr_auc_kfoldcv[6] = (lr_auc_kfoldcv[0]+lr_auc_kfoldcv[1]+lr_auc_kfoldcv[2]+lr_auc_kfoldcv[3]+lr_auc_kfoldcv[4])/5
lr_auc_kfoldcv = pd.DataFrame(lr_auc_kfoldcv)
lr_auc_kfoldcv.to_csv('auc_'+imb_technique+'_lr_road_traffic_' + str(nsplits) + 'foldcv_1~5.csv',header=False,index=False)
lr_gmean_kfoldcv[5] = "Average" #To let users figure out that f1_score_kfoldcv[6] value is the average when seeing the csv file
lr_gmean_kfoldcv[6] = (lr_gmean_kfoldcv[0]+lr_gmean_kfoldcv[1]+lr_gmean_kfoldcv[2]+lr_gmean_kfoldcv[3]+lr_gmean_kfoldcv[4])/5
lr_gmean_kfoldcv = pd.DataFrame(lr_gmean_kfoldcv)
lr_gmean_kfoldcv.to_csv('gmean_'+imb_technique+'_lr_road_traffic_' + str(nsplits) + 'foldcv_1~5.csv',header=False,index=False)

nb_f1_score_pen_1_kfoldcv[5] = "Average" #To let users figure out that f1_score_kfoldcv[6] value is the average when seeing the csv file
nb_f1_score_pen_1_kfoldcv[6] = (nb_f1_score_pen_1_kfoldcv[0]+nb_f1_score_pen_1_kfoldcv[1]+nb_f1_score_pen_1_kfoldcv[2]+nb_f1_score_pen_1_kfoldcv[3]+nb_f1_score_pen_1_kfoldcv[4])/5
nb_f1_score_pen_1_kfoldcv = pd.DataFrame(nb_f1_score_pen_1_kfoldcv)
nb_f1_score_pen_1_kfoldcv.to_csv('f1_score_'+imb_technique+'_penalty_1_nb_road_traffic_' + str(nsplits) + 'foldcv_1~5.csv',header=False,index=False) #First repetition
nb_f1_score_pen_5_kfoldcv[5] = "Average" #To let users figure out that f1_score_kfoldcv[6] value is the average when seeing the csv file
nb_f1_score_pen_5_kfoldcv[6] = (nb_f1_score_pen_5_kfoldcv[0]+nb_f1_score_pen_5_kfoldcv[1]+nb_f1_score_pen_5_kfoldcv[2]+nb_f1_score_pen_5_kfoldcv[3]+nb_f1_score_pen_5_kfoldcv[4])/5
nb_f1_score_pen_5_kfoldcv = pd.DataFrame(nb_f1_score_pen_5_kfoldcv)
nb_f1_score_pen_5_kfoldcv.to_csv('f1_score_'+imb_technique+'_penalty_5_nb_road_traffic_' + str(nsplits) + 'foldcv_1~5.csv',header=False,index=False) #First repetition
#nb_f1_score_kfoldcv.to_csv('f1_score_'+imb_technique+'_penalty_' + str(penalty) + '_nb_road_traffic_' + str(nsplits) + 'foldcv_6~10.csv',header=False,index=False) #Second repetition
nb_ovr_accuracy_kfoldcv[5] = "Average" #To let users figure out that f1_score_kfoldcv[6] value is the average when seeing the csv file
nb_ovr_accuracy_kfoldcv[6] = (nb_ovr_accuracy_kfoldcv[0]+nb_ovr_accuracy_kfoldcv[1]+nb_ovr_accuracy_kfoldcv[2]+nb_ovr_accuracy_kfoldcv[3]+nb_ovr_accuracy_kfoldcv[4])/5
nb_ovr_accuracy_kfoldcv = pd.DataFrame(nb_ovr_accuracy_kfoldcv)
nb_ovr_accuracy_kfoldcv.to_csv('ovr_accuracy_' + imb_technique + '_nb_road_traffic_' + str(nsplits) + 'foldcv_1~5.csv',header=False,index=False) #First repetition
#nb_ovr_accuracy_kfoldcv.to_csv('ovr_accuracy_'+imb_technique+'_penalty_' + str(penalty) + '_nb_road_traffic_' + str(nsplits) + 'foldcv_6~10.csv',header=False,index=False) #Second repetition
nb_auc_kfoldcv[5] = "Average" #To let users figure out that f1_score_kfoldcv[6] value is the average when seeing the csv file
nb_auc_kfoldcv[6] = (nb_auc_kfoldcv[0]+nb_auc_kfoldcv[1]+nb_auc_kfoldcv[2]+nb_auc_kfoldcv[3]+nb_auc_kfoldcv[4])/5
nb_auc_kfoldcv = pd.DataFrame(nb_auc_kfoldcv)
nb_auc_kfoldcv.to_csv('auc_'+imb_technique+'_nb_road_traffic_' + str(nsplits) + 'foldcv_1~5.csv',header=False,index=False)
nb_gmean_kfoldcv[5] = "Average" #To let users figure out that f1_score_kfoldcv[6] value is the average when seeing the csv file
nb_gmean_kfoldcv[6] = (nb_gmean_kfoldcv[0]+nb_gmean_kfoldcv[1]+nb_gmean_kfoldcv[2]+nb_gmean_kfoldcv[3]+nb_gmean_kfoldcv[4])/5
nb_gmean_kfoldcv = pd.DataFrame(nb_gmean_kfoldcv)
nb_gmean_kfoldcv.to_csv('gmean_'+imb_technique+'_nb_road_traffic_' + str(nsplits) + 'foldcv_1~5.csv',header=False,index=False)

rf_f1_score_pen_1_kfoldcv[5] = "Average" #To let users figure out that f1_score_kfoldcv[6] value is the average when seeing the csv file
rf_f1_score_pen_1_kfoldcv[6] = (rf_f1_score_pen_1_kfoldcv[0]+rf_f1_score_pen_1_kfoldcv[1]+rf_f1_score_pen_1_kfoldcv[2]+rf_f1_score_pen_1_kfoldcv[3]+rf_f1_score_pen_1_kfoldcv[4])/5
rf_f1_score_pen_1_kfoldcv = pd.DataFrame(rf_f1_score_pen_1_kfoldcv)
rf_f1_score_pen_1_kfoldcv.to_csv('f1_score_'+imb_technique+'_penalty_1_rf_road_traffic_' + str(nsplits) + 'foldcv_1~5.csv',header=False,index=False) #First repetition
rf_f1_score_pen_5_kfoldcv[5] = "Average" #To let users figure out that f1_score_kfoldcv[6] value is the average when seeing the csv file
rf_f1_score_pen_5_kfoldcv[6] = (rf_f1_score_pen_5_kfoldcv[0]+rf_f1_score_pen_5_kfoldcv[1]+rf_f1_score_pen_5_kfoldcv[2]+rf_f1_score_pen_5_kfoldcv[3]+rf_f1_score_pen_5_kfoldcv[4])/5
rf_f1_score_pen_5_kfoldcv = pd.DataFrame(rf_f1_score_pen_5_kfoldcv)
rf_f1_score_pen_5_kfoldcv.to_csv('f1_score_'+imb_technique+'_penalty_5_rf_road_traffic_' + str(nsplits) + 'foldcv_1~5.csv',header=False,index=False) #First repetition
#rf_f1_score_kfoldcv.to_csv('f1_score_'+imb_technique+'_penalty_' + str(penalty) + '_rf_road_traffic_' + str(nsplits) + 'foldcv_6~10.csv',header=False,index=False) #Second repetition
rf_ovr_accuracy_kfoldcv[5] = "Average" #To let users figure out that f1_score_kfoldcv[6] value is the average when seeing the csv file
rf_ovr_accuracy_kfoldcv[6] = (rf_ovr_accuracy_kfoldcv[0]+rf_ovr_accuracy_kfoldcv[1]+rf_ovr_accuracy_kfoldcv[2]+rf_ovr_accuracy_kfoldcv[3]+rf_ovr_accuracy_kfoldcv[4])/5
rf_ovr_accuracy_kfoldcv = pd.DataFrame(rf_ovr_accuracy_kfoldcv)
rf_ovr_accuracy_kfoldcv.to_csv('ovr_accuracy_' + imb_technique + '_rf_road_traffic_' + str(nsplits) + 'foldcv_1~5.csv',header=False,index=False) #First repetition
#rf_ovr_accuracy_kfoldcv.to_csv('ovr_accuracy_'+imb_technique+'_penalty_' + str(penalty) + '_rf_road_traffic_' + str(nsplits) + 'foldcv_6~10.csv',header=False,index=False) #Second repetition
rf_auc_kfoldcv[5] = "Average" #To let users figure out that f1_score_kfoldcv[6] value is the average when seeing the csv file
rf_auc_kfoldcv[6] = (rf_auc_kfoldcv[0]+rf_auc_kfoldcv[1]+rf_auc_kfoldcv[2]+rf_auc_kfoldcv[3]+rf_auc_kfoldcv[4])/5
rf_auc_kfoldcv = pd.DataFrame(rf_auc_kfoldcv)
rf_auc_kfoldcv.to_csv('auc_'+imb_technique+'_rf_road_traffic_' + str(nsplits) + 'foldcv_1~5.csv',header=False,index=False)
rf_gmean_kfoldcv[5] = "Average" #To let users figure out that f1_score_kfoldcv[6] value is the average when seeing the csv file
rf_gmean_kfoldcv[6] = (rf_gmean_kfoldcv[0]+rf_gmean_kfoldcv[1]+rf_gmean_kfoldcv[2]+rf_gmean_kfoldcv[3]+rf_gmean_kfoldcv[4])/5
rf_gmean_kfoldcv = pd.DataFrame(rf_gmean_kfoldcv)
rf_gmean_kfoldcv.to_csv('gmean_'+imb_technique+'_rf_road_traffic_' + str(nsplits) + 'foldcv_1~5.csv',header=False,index=False)

svm_f1_score_pen_1_kfoldcv[5] = "Average" #To let users figure out that f1_score_kfoldcv[6] value is the average when seeing the csv file
svm_f1_score_pen_1_kfoldcv[6] = (svm_f1_score_pen_1_kfoldcv[0]+svm_f1_score_pen_1_kfoldcv[1]+svm_f1_score_pen_1_kfoldcv[2]+svm_f1_score_pen_1_kfoldcv[3]+svm_f1_score_pen_1_kfoldcv[4])/5
svm_f1_score_pen_1_kfoldcv = pd.DataFrame(svm_f1_score_pen_1_kfoldcv)
svm_f1_score_pen_1_kfoldcv.to_csv('f1_score_'+imb_technique+'_penalty_1_svm_road_traffic_' + str(nsplits) + 'foldcv_1~5.csv',header=False,index=False) #First repetition
svm_f1_score_pen_5_kfoldcv[5] = "Average" #To let users figure out that f1_score_kfoldcv[6] value is the average when seeing the csv file
svm_f1_score_pen_5_kfoldcv[6] = (svm_f1_score_pen_5_kfoldcv[0]+svm_f1_score_pen_5_kfoldcv[1]+svm_f1_score_pen_5_kfoldcv[2]+svm_f1_score_pen_5_kfoldcv[3]+svm_f1_score_pen_5_kfoldcv[4])/5
svm_f1_score_pen_5_kfoldcv = pd.DataFrame(svm_f1_score_pen_5_kfoldcv)
svm_f1_score_pen_5_kfoldcv.to_csv('f1_score_'+imb_technique+'_penalty_5_svm_road_traffic_' + str(nsplits) + 'foldcv_1~5.csv',header=False,index=False) #First repetition
#svm_f1_score_kfoldcv.to_csv('f1_score_'+imb_technique+'_penalty_' + str(penalty) + '_svm_road_traffic_' + str(nsplits) + 'foldcv_6~10.csv',header=False,index=False) #Second repetition
svm_ovr_accuracy_kfoldcv[5] = "Average" #To let users figure out that f1_score_kfoldcv[6] value is the average when seeing the csv file
svm_ovr_accuracy_kfoldcv[6] = (svm_ovr_accuracy_kfoldcv[0]+svm_ovr_accuracy_kfoldcv[1]+svm_ovr_accuracy_kfoldcv[2]+svm_ovr_accuracy_kfoldcv[3]+svm_ovr_accuracy_kfoldcv[4])/5
svm_ovr_accuracy_kfoldcv = pd.DataFrame(svm_ovr_accuracy_kfoldcv)
svm_ovr_accuracy_kfoldcv.to_csv('ovr_accuracy_'+ imb_technique + '_svm_road_traffic_' + str(nsplits) + 'foldcv_1~5.csv',header=False,index=False) #First repetition
#svm_ovr_accuracy_kfoldcv.to_csv('ovr_accuracy_'+imb_technique+'_penalty_' + str(penalty) + '_svm_road_traffic_' + str(nsplits) + 'foldcv_6~10.csv',header=False,index=False) #Second repetition
svm_auc_kfoldcv[5] = "Average" #To let users figure out that f1_score_kfoldcv[6] value is the average when seeing the csv file
svm_auc_kfoldcv[6] = (svm_auc_kfoldcv[0]+svm_auc_kfoldcv[1]+svm_auc_kfoldcv[2]+svm_auc_kfoldcv[3]+svm_auc_kfoldcv[4])/5
svm_auc_kfoldcv = pd.DataFrame(svm_auc_kfoldcv)
svm_auc_kfoldcv.to_csv('auc_'+imb_technique+'_svm_road_traffic_' + str(nsplits) + 'foldcv_1~5.csv',header=False,index=False)
svm_gmean_kfoldcv[5] = "Average" #To let users figure out that f1_score_kfoldcv[6] value is the average when seeing the csv file
svm_gmean_kfoldcv[6] = (svm_gmean_kfoldcv[0]+svm_gmean_kfoldcv[1]+svm_gmean_kfoldcv[2]+svm_gmean_kfoldcv[3]+svm_gmean_kfoldcv[4])/5
svm_gmean_kfoldcv = pd.DataFrame(svm_gmean_kfoldcv)
svm_gmean_kfoldcv.to_csv('gmean_'+imb_technique+'_svm_road_traffic_' + str(nsplits) + 'foldcv_1~5.csv',header=False,index=False)

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

dnn_f1_score_pen_1_kfoldcv = dnn_f1_score_pen_5_kfoldcv = dnn_ovr_accuracy_kfoldcv = dnn_auc_kfoldcv = dnn_gmean_kfoldcv = [None] * (nsplits+2)
dnn_params_hls_AA = dnn_params_hls_AI = dnn_params_hls_AW = dnn_params_hls_CC = dnn_params_hls_QA = [None] * (nsplits+2)
dnn_params_lri_AA = dnn_params_lri_AI = dnn_params_lri_AW = dnn_params_lri_CC = dnn_params_lri_QA = [None] * (nsplits+2)

lr_f1_score_pen_1_kfoldcv = lr_f1_score_pen_5_kfoldcv = lr_ovr_accuracy_kfoldcv = lr_auc_kfoldcv = lr_gmean_kfoldcv = [None] * (nsplits+2)
lr_params_solver_AA = lr_params_solver_AI = lr_params_solver_AW = lr_params_solver_CC = lr_params_solver_QA = [None] * (nsplits+2)
lr_params_tol_AA = lr_params_tol_AI = lr_params_tol_AW = lr_params_tol_CC = lr_params_tol_QA = [None] * (nsplits+2)
lr_params_C_AA = lr_params_C_AI = lr_params_C_AW = lr_params_C_CC = lr_params_C_QA = [None] * (nsplits+2)

nb_f1_score_pen_1_kfoldcv = nb_f1_score_pen_5_kfoldcv = nb_ovr_accuracy_kfoldcv = nb_auc_kfoldcv = nb_gmean_kfoldcv = [None] * (nsplits+2)
nb_params_vs_AA = nb_params_vs_AI = nb_params_vs_AW = nb_params_vs_CC = nb_params_vs_QA = [None] * (nsplits+2)

rf_f1_score_pen_1_kfoldcv = rf_f1_score_pen_5_kfoldcv = rf_ovr_accuracy_kfoldcv = rf_auc_kfoldcv = rf_gmean_kfoldcv = [None] * (nsplits+2)
rf_params_est_AA = rf_params_est_AI = rf_params_est_AW = rf_params_est_CC = rf_params_est_QA = [None] * (nsplits+2)
rf_params_md_AA = rf_params_md_AI = rf_params_md_AW = rf_params_md_CC = rf_params_md_QA = [None] * (nsplits+2)
rf_params_mss_AA = rf_params_mss_AI = rf_params_mss_AW = rf_params_mss_CC = rf_params_mss_QA = [None] * (nsplits+2)

svm_f1_score_pen_1_kfoldcv = svm_f1_score_pen_5_kfoldcv = svm_ovr_accuracy_kfoldcv = svm_auc_kfoldcv = svm_gmean_kfoldcv = [None] * (nsplits+2)
svm_params_tol_AA = svm_params_tol_AI = svm_params_tol_AW = svm_params_tol_CC = svm_params_tol_QA = [None] * (nsplits+2)

outfile = os.path.join("/home/jongchan/BPIC2013_closed", "performance_results_%s_%s.csv" % ("BPIC2013", imb_technique))
outfile_param_AA = os.path.join("/home/jongchan/BPIC2013_closed", "parameters_AA_%s_%s.csv" % ("BPIC2013", imb_technique))
outfile_param_AI = os.path.join("/home/jongchan/BPIC2013_closed", "parameters_AI_%s_%s.csv" % ("BPIC2013", imb_technique))
outfile_param_AW = os.path.join("/home/jongchan/BPIC2013_closed", "parameters_AW_%s_%s.csv" % ("BPIC2013", imb_technique))
outfile_param_CC = os.path.join("/home/jongchan/BPIC2013_closed", "parameters_CC_%s_%s.csv" % ("BPIC2013", imb_technique))
outfile_param_QA = os.path.join("/home/jongchan/BPIC2013_closed", "parameters_QA_%s_%s.csv" % ("BPIC2013", imb_technique))

repeat = 0
for train_index, test_index in kf.split(X_dummy):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X_dummy.iloc[train_index], X_dummy.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    y_test_AA = pd.DataFrame([1 if i == "Accepted/Assigned" else 0 for i in y_test['ACT_COMB_4']])
    y_test_AI = pd.DataFrame([1 if i == "Accepted/In Progress" else 0 for i in y_test['ACT_COMB_4']])
    y_test_AW = pd.DataFrame([1 if i == "Accepted/Wait" else 0 for i in y_test['ACT_COMB_4']])
    y_test_CC = pd.DataFrame([1 if i == "Completed/Closed" else 0 for i in y_test['ACT_COMB_4']])
    y_test_QA = pd.DataFrame([1 if i == "Queued/Awaiting Assignment" else 0 for i in y_test['ACT_COMB_4']])
    train = pd.concat([X_train, y_train], axis=1)
    ACT_COMB_4_index = np.unique(data['ACT_COMB_1']).size + np.unique(data['ACT_COMB_2']).size + np.unique(data['ACT_COMB_3']).size + 1

    AA = train[train.ACT_COMB_4 == "Accepted/Assigned"].reset_index()
    AA_rest = train[train.ACT_COMB_4 != "Accepted/Assigned"].copy()
    AA_rest.iloc[:, ACT_COMB_4_index] = "Others"
    AA_rest = AA_rest.reset_index()
    AA = AA.iloc[:, 1:ACT_COMB_4_index+2]
    AA_rest = AA_rest.iloc[:, 1:ACT_COMB_4_index+2]
    AA_ova = pd.concat([AA, AA_rest])
    AA_ova_X_train = AA_ova.iloc[:, 0:ACT_COMB_4_index]
    AA_ova_y_train = AA_ova.iloc[:, ACT_COMB_4_index]
    AA_X_res, AA_y_res = AA_ova_X_train, AA_ova_y_train
    Counter(AA_ova_y_train)

    AI = train[train.ACT_COMB_4 == "Accepted/In Progress"].reset_index()
    AI_rest = train[train.ACT_COMB_4 != "Accepted/In Progress"].copy()
    AI_rest.iloc[:, ACT_COMB_4_index] = "Others"
    AI_rest = AI_rest.reset_index()
    AI = AI.iloc[:, 1:ACT_COMB_4_index+2]
    AI_rest = AI_rest.iloc[:, 1:ACT_COMB_4_index+2]
    AI_ova = pd.concat([AI, AI_rest])
    AI_ova_X_train = AI_ova.iloc[:, 0:ACT_COMB_4_index]
    AI_ova_y_train = AI_ova.iloc[:, ACT_COMB_4_index]
    AI_X_res, AI_y_res = AI_ova_X_train, AI_ova_y_train
    Counter(AI_ova_y_train)

    AW = train[train.ACT_COMB_4 == "Accepted/Wait"].reset_index()
    AW_rest = train[train.ACT_COMB_4 != "Accepted/Wait"].copy()
    AW_rest.iloc[:, ACT_COMB_4_index] = "Others"
    AW_rest = AW_rest.reset_index()
    AW = AW.iloc[:, 1:ACT_COMB_4_index+2]
    AW_rest = AW_rest.iloc[:, 1:ACT_COMB_4_index+2]
    AW_ova = pd.concat([AW, AW_rest])
    AW_ova_X_train = AW_ova.iloc[:, 0:ACT_COMB_4_index]
    AW_ova_y_train = AW_ova.iloc[:, ACT_COMB_4_index]
    AW_X_res, AW_y_res = AW_ova_X_train, AW_ova_y_train
    Counter(AW_ova_y_train)

    CC = train[train.ACT_COMB_4 == "Completed/Closed"].reset_index()
    CC_rest = train[train.ACT_COMB_4 != "Completed/Closed"].copy()
    CC_rest.iloc[:, ACT_COMB_4_index] = "Others"
    CC = CC.reset_index()
    CC_rest = CC_rest.reset_index()
    CC = CC.iloc[:, 1:ACT_COMB_4_index+2]
    CC_rest = CC_rest.iloc[:, 1:ACT_COMB_4_index+2]
    CC_ova = pd.concat([CC, CC_rest])
    CC_ova_X_train = CC_ova.iloc[:, 0:ACT_COMB_4_index]
    CC_ova_y_train = CC_ova.iloc[:, ACT_COMB_4_index]
    CC_X_res, CC_y_res = CC_ova_X_train, CC_ova_y_train
    Counter(CC_ova_y_train)

    QA = train[train.ACT_COMB_4 == "Queued/Awaiting Assignment"].reset_index()
    QA_rest = train[train.ACT_COMB_4 != "Queued/Awaiting Assignment"].copy()
    QA_rest.iloc[:, ACT_COMB_4_index] = "Others"
    QA = QA.reset_index()
    QA_rest = QA_rest.reset_index()
    QA = QA.iloc[:, 1:ACT_COMB_4_index+2]
    QA_rest = QA_rest.iloc[:, 1:ACT_COMB_4_index+2]
    QA_ova = pd.concat([QA, QA_rest])
    QA_ova_X_train = QA_ova.iloc[:, 0:ACT_COMB_4_index]
    QA_ova_y_train = QA_ova.iloc[:, ACT_COMB_4_index]
    QA_X_res, QA_y_res = QA_ova_X_train, QA_ova_y_train
    Counter(QA_ova_y_train)

    if imb_technique == "ADASYN":
        from imblearn.over_sampling import ADASYN
        AA_ada, AI_ada, AW_ada, CC_ada, QA_ada = ADASYN(), ADASYN(), ADASYN(), ADASYN(), ADASYN()
        AA_X_res, AA_y_res = AA_ada.fit_resample(AA_ova_X_train, AA_ova_y_train)
        AI_X_res, AI_y_res = AI_ada.fit_resample(AI_ova_X_train, AI_ova_y_train)
        AW_X_res, AW_y_res = AW_ada.fit_resample(AW_ova_X_train, AW_ova_y_train)
        CC_X_res, CC_y_res = CC_ada.fit_resample(CC_ova_X_train, CC_ova_y_train)
        QA_X_res, QA_y_res = QA_ada.fit_resample(QA_ova_X_train, QA_ova_y_train)

    if imb_technique == "ALLKNN":
        from imblearn.under_sampling import AllKNN
        AA_allknn, AI_allknn, AW_allknn, CC_allknn, QA_allknn = AllKNN(), AllKNN(), AllKNN(), AllKNN(), AllKNN()
        AA_X_res, AA_y_res = AA_allknn.fit_resample(AA_ova_X_train, AA_ova_y_train)
        AI_X_res, AI_y_res = AI_allknn.fit_resample(AI_ova_X_train, AI_ova_y_train)
        AW_X_res, AW_y_res = AW_allknn.fit_resample(AW_ova_X_train, AW_ova_y_train)
        CC_X_res, CC_y_res = CC_allknn.fit_resample(CC_ova_X_train, CC_ova_y_train)
        QA_X_res, QA_y_res = QA_allknn.fit_resample(QA_ova_X_train, QA_ova_y_train)

    if imb_technique == "CNN":
        from imblearn.under_sampling import CondensedNearestNeighbour
        AA_cnn, AI_cnn, AW_cnn, CC_cnn, QA_cnn = CondensedNearestNeighbour(), CondensedNearestNeighbour(), CondensedNearestNeighbour(), CondensedNearestNeighbour(), CondensedNearestNeighbour()
        AA_X_res, AA_y_res = AA_cnn.fit_resample(AA_ova_X_train, AA_ova_y_train)
        AI_X_res, AI_y_res = AI_cnn.fit_resample(AI_ova_X_train, AI_ova_y_train)
        AW_X_res, AW_y_res = AW_cnn.fit_resample(AW_ova_X_train, AW_ova_y_train)
        CC_X_res, CC_y_res = CC_cnn.fit_resample(CC_ova_X_train, CC_ova_y_train)
        QA_X_res, QA_y_res = QA_cnn.fit_resample(QA_ova_X_train, QA_ova_y_train)

    if imb_technique == "ENN":
        from imblearn.under_sampling import EditedNearestNeighbours
        AA_enn, AI_enn, AW_enn, CC_enn, QA_enn = EditedNearestNeighbours(),EditedNearestNeighbours(),EditedNearestNeighbours(),EditedNearestNeighbours(),EditedNearestNeighbours()
        AA_X_res, AA_y_res = AA_enn.fit_resample(AA_ova_X_train, AA_ova_y_train)
        AI_X_res, AI_y_res = AI_enn.fit_resample(AI_ova_X_train, AI_ova_y_train)
        AW_X_res, AW_y_res = AW_enn.fit_resample(AW_ova_X_train, AW_ova_y_train)
        CC_X_res, CC_y_res = CC_enn.fit_resample(CC_ova_X_train, CC_ova_y_train)
        QA_X_res, QA_y_res = QA_enn.fit_resample(QA_ova_X_train, QA_ova_y_train)

    if imb_technique == "IHT":
        from imblearn.under_sampling import InstanceHardnessThreshold
        AA_iht, AI_iht, AW_iht, CC_iht, QA_iht = InstanceHardnessThreshold(),InstanceHardnessThreshold(),InstanceHardnessThreshold(),InstanceHardnessThreshold(),InstanceHardnessThreshold()
        AA_X_res, AA_y_res = AA_iht.fit_resample(AA_ova_X_train, AA_ova_y_train)
        AI_X_res, AI_y_res = AI_iht.fit_resample(AI_ova_X_train, AI_ova_y_train)
        AW_X_res, AW_y_res = AW_iht.fit_resample(AW_ova_X_train, AW_ova_y_train)
        CC_X_res, CC_y_res = CC_iht.fit_resample(CC_ova_X_train, CC_ova_y_train)
        QA_X_res, QA_y_res = QA_iht.fit_resample(QA_ova_X_train, QA_ova_y_train)

    if imb_technique == "NCR":
        from imblearn.under_sampling import NeighbourhoodCleaningRule
        AA_ncr, AI_ncr, AW_ncr, CC_ncr, QA_ncr = NeighbourhoodCleaningRule(),NeighbourhoodCleaningRule(),NeighbourhoodCleaningRule(),NeighbourhoodCleaningRule(),NeighbourhoodCleaningRule()
        AA_ova_y_train = [0 if i == "Accepted/Assigned" else 1 for i in AA_ova_y_train]
        AA_X_res, AA_y_res = AA_ncr.fit_resample(AA_ova_X_train, AA_ova_y_train)
        AI_ova_y_train = [0 if i == "Accepted/In Progress" else 1 for i in AI_ova_y_train]
        AI_X_res, AI_y_res = AI_ncr.fit_resample(AI_ova_X_train, AI_ova_y_train)
        AW_ova_y_train = [0 if i == "Accepted/Wait" else 1 for i in AW_ova_y_train]
        AW_X_res, AW_y_res = AW_ncr.fit_resample(AW_ova_X_train, AW_ova_y_train)
        CC_ova_y_train = [0 if i == "Completed/Closed" else 1 for i in CC_ova_y_train]
        CC_X_res, CC_y_res = CC_ncr.fit_resample(CC_ova_X_train, CC_ova_y_train)
        QA_ova_y_train = [0 if i == "Queued/Awaiting Assignment" else 1 for i in QA_ova_y_train]
        QA_X_res, QA_y_res = QA_ncr.fit_resample(QA_ova_X_train, QA_ova_y_train)

    if imb_technique == "NM":
        from imblearn.under_sampling import NearMiss
        AA_nm, AI_nm, AW_nm, CC_nm, QA_nm = NearMiss(),NearMiss(),NearMiss(),NearMiss(),NearMiss()
        AA_X_res, AA_y_res = AA_nm.fit_resample(AA_ova_X_train, AA_ova_y_train)
        AI_X_res, AI_y_res = AI_nm.fit_resample(AI_ova_X_train, AI_ova_y_train)
        AW_X_res, AW_y_res = AW_nm.fit_resample(AW_ova_X_train, AW_ova_y_train)
        CC_X_res, CC_y_res = CC_nm.fit_resample(CC_ova_X_train, CC_ova_y_train)
        QA_X_res, QA_y_res = QA_nm.fit_resample(QA_ova_X_train, QA_ova_y_train)

    if imb_technique == "OSS":
        from imblearn.under_sampling import OneSidedSelection
        AA_oss, AI_oss, AW_oss, CC_oss, QA_oss = OneSidedSelection(),OneSidedSelection(),OneSidedSelection(),OneSidedSelection(),OneSidedSelection()
        AA_X_res, AA_y_res = AA_oss.fit_resample(AA_ova_X_train, AA_ova_y_train)
        AI_X_res, AI_y_res = AI_oss.fit_resample(AI_ova_X_train, AI_ova_y_train)
        AW_X_res, AW_y_res = AW_oss.fit_resample(AW_ova_X_train, AW_ova_y_train)
        CC_X_res, CC_y_res = CC_oss.fit_resample(CC_ova_X_train, CC_ova_y_train)
        QA_X_res, QA_y_res = QA_oss.fit_resample(QA_ova_X_train, QA_ova_y_train)

    if imb_technique == "RENN":
        from imblearn.under_sampling import RepeatedEditedNearestNeighbours
        AA_renn, AI_renn, AW_renn, CC_renn, QA_renn = RepeatedEditedNearestNeighbours(),RepeatedEditedNearestNeighbours(),RepeatedEditedNearestNeighbours(),RepeatedEditedNearestNeighbours(),RepeatedEditedNearestNeighbours()
        AA_X_res, AA_y_res = AA_renn.fit_resample(AA_ova_X_train, AA_ova_y_train)
        AI_X_res, AI_y_res = AI_renn.fit_resample(AI_ova_X_train, AI_ova_y_train)
        AW_X_res, AW_y_res = AW_renn.fit_resample(AW_ova_X_train, AW_ova_y_train)
        CC_X_res, CC_y_res = CC_renn.fit_resample(CC_ova_X_train, CC_ova_y_train)
        QA_X_res, QA_y_res = QA_renn.fit_resample(QA_ova_X_train, QA_ova_y_train)

    if imb_technique == "SMOTE":
        from imblearn.over_sampling import SMOTE
        AA_sm, AI_sm, AW_sm, CC_sm, QA_sm = SMOTE(),SMOTE(),SMOTE(),SMOTE(),SMOTE()
        AA_X_res, AA_y_res = AA_sm.fit_resample(AA_ova_X_train, AA_ova_y_train)
        AI_X_res, AI_y_res = AI_sm.fit_resample(AI_ova_X_train, AI_ova_y_train)
        AW_X_res, AW_y_res = AW_sm.fit_resample(AW_ova_X_train, AW_ova_y_train)
        CC_X_res, CC_y_res = CC_sm.fit_resample(CC_ova_X_train, CC_ova_y_train)
        QA_X_res, QA_y_res = QA_sm.fit_resample(QA_ova_X_train, QA_ova_y_train)

    if imb_technique == "BSMOTE":
        from imblearn.over_sampling import BorderlineSMOTE
        AA_bsm, AI_bsm, AW_bsm, CC_bsm, QA_bsm = BorderlineSMOTE(),BorderlineSMOTE(),BorderlineSMOTE()BorderlineSMOTE()BorderlineSMOTE()
        AA_X_res, AA_y_res = AA_bsm.fit_resample(AA_ova_X_train, AA_ova_y_train)
        AI_X_res, AI_y_res = AI_bsm.fit_resample(AI_ova_X_train, AI_ova_y_train)
        AW_X_res, AW_y_res = AW_bsm.fit_resample(AW_ova_X_train, AW_ova_y_train)
        CC_X_res, CC_y_res = CC_bsm.fit_resample(CC_ova_X_train, CC_ova_y_train)
        QA_X_res, QA_y_res = QA_bsm.fit_resample(QA_ova_X_train, QA_ova_y_train)

    if imb_technique == "SMOTEENN":
        from imblearn.combine import SMOTEENN
        AA_smenn, AI_smenn, AW_smenn, CC_smenn, QA_smenn = SMOTEENN(),SMOTEENN(),SMOTEENN()SMOTEENN()SMOTEENN()
        AA_X_res, AA_y_res = AA_smenn.fit_resample(AA_ova_X_train, AA_ova_y_train)
        AI_X_res, AI_y_res = AI_smenn.fit_resample(AI_ova_X_train, AI_ova_y_train)
        AW_X_res, AW_y_res = AW_smenn.fit_resample(AW_ova_X_train, AW_ova_y_train)
        CC_X_res, CC_y_res = CC_smenn.fit_resample(CC_ova_X_train, CC_ova_y_train)
        QA_X_res, QA_y_res = QA_smenn.fit_resample(QA_ova_X_train, QA_ova_y_train)

    if imb_technique == "SMOTETOMEK":
        from imblearn.combine import SMOTETomek
        AA_smtm, AI_smtm, AW_smtm, CC_smtm, QA_smtm = SMOTETomek(),SMOTETomek(),SMOTETomek(),SMOTETomek()SMOTETomek()
        AA_X_res, AA_y_res = AA_smtm.fit_resample(AA_ova_X_train, AA_ova_y_train)
        AI_X_res, AI_y_res = AI_smtm.fit_resample(AI_ova_X_train, AI_ova_y_train)
        AW_X_res, AW_y_res = AW_smtm.fit_resample(AW_ova_X_train, AW_ova_y_train)
        CC_X_res, CC_y_res = CC_smtm.fit_resample(CC_ova_X_train, CC_ova_y_train)
        QA_X_res, QA_y_res = QA_smtm.fit_resample(QA_ova_X_train, QA_ova_y_train)

    if imb_technique == "TOMEK":
        from imblearn.under_sampling import TomekLinks
        AA_tm, AI_tm, AW_tm, CC_tm, QA_tm = TomekLinks(),TomekLinks(),TomekLinks(),TomekLinks()TomekLinks()
        AA_X_res, AA_y_res = AA_tm.fit_resample(AA_ova_X_train, AA_ova_y_train)
        AI_X_res, AI_y_res = AI_tm.fit_resample(AI_ova_X_train, AI_ova_y_train)
        AW_X_res, AW_y_res = AW_tm.fit_resample(AW_ova_X_train, AW_ova_y_train)
        CC_X_res, CC_y_res = CC_tm.fit_resample(CC_ova_X_train, CC_ova_y_train)
        QA_X_res, QA_y_res = QA_tm.fit_resample(QA_ova_X_train, QA_ova_y_train)

    if imb_technique == "ROS":
        from imblearn.over_sampling import RandomOverSampler
        AA_ros, AI_ros, AW_ros, CC_ros, QA_ros = RandomOverSampler(),RandomOverSampler(),RandomOverSampler(),RandomOverSampler(),RandomOverSampler()
        AA_X_res, AA_y_res = AA_ros.fit_resample(AA_ova_X_train, AA_ova_y_train)
        AI_X_res, AI_y_res = AI_ros.fit_resample(AI_ova_X_train, AI_ova_y_train)
        AW_X_res, AW_y_res = AW_ros.fit_resample(AW_ova_X_train, AW_ova_y_train)
        CC_X_res, CC_y_res = CC_ros.fit_resample(CC_ova_X_train, CC_ova_y_train)
        QA_X_res, QA_y_res = QA_ros.fit_resample(QA_ova_X_train, QA_ova_y_train)

    if imb_technique == "RUS":
        from imblearn.under_sampling import RandomUnderSampler
        AA_rus, AI_rus, AW_rus, CC_rus, QA_rus = RandomUnderSampler(),RandomUnderSampler(),RandomUnderSampler(),RandomUnderSampler()RandomUnderSampler()
        AA_X_res, AA_y_res = AA_rus.fit_resample(AA_ova_X_train, AA_ova_y_train)
        AI_X_res, AI_y_res = AI_rus.fit_resample(AI_ova_X_train, AI_ova_y_train)
        AW_X_res, AW_y_res = AW_rus.fit_resample(AW_ova_X_train, AW_ova_y_train)
        CC_X_res, CC_y_res = CC_rus.fit_resample(CC_ova_X_train, CC_ova_y_train)
        QA_X_res, QA_y_res = QA_rus.fit_resample(QA_ova_X_train, QA_ova_y_train)

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

    dnn_pred_class_AA = dnn_AA_clf.predict(X_test)
    dnn_pred_prob_AA = dnn_AA_clf.predict_proba(X_test)
    dnn_pred_class_AI = dnn_AI_clf.predict(X_test)
    dnn_pred_prob_AI = dnn_AI_clf.predict_proba(X_test)
    dnn_pred_class_AW = dnn_AW_clf.predict(X_test)
    dnn_pred_prob_AW = dnn_AW_clf.predict_proba(X_test)
    dnn_pred_class_CC = dnn_CC_clf.predict(X_test)
    dnn_pred_prob_CC = dnn_CC_clf.predict_proba(X_test)
    dnn_pred_class_QA = dnn_QA_clf.predict(X_test)
    dnn_pred_prob_QA = dnn_QA_clf.predict_proba(X_test)

    lr_pred_class_AA = lr_AA_clf.predict(X_test)
    lr_pred_prob_AA = lr_AA_clf.predict_proba(X_test)
    lr_pred_class_AI = lr_AI_clf.predict(X_test)
    lr_pred_prob_AI = lr_AI_clf.predict_proba(X_test)
    lr_pred_class_AW = lr_AW_clf.predict(X_test)
    lr_pred_prob_AW = lr_AW_clf.predict_proba(X_test)
    lr_pred_class_CC = lr_CC_clf.predict(X_test)
    lr_pred_prob_CC = lr_CC_clf.predict_proba(X_test)
    lr_pred_class_QA = lr_QA_clf.predict(X_test)
    lr_pred_prob_QA = lr_QA_clf.predict_proba(X_test)

    nb_pred_class_AA = nb_AA_clf.predict(X_test)
    nb_pred_prob_AA = nb_AA_clf.predict_proba(X_test)
    nb_pred_class_AI = nb_AI_clf.predict(X_test)
    nb_pred_prob_AI = nb_AI_clf.predict_proba(X_test)
    nb_pred_class_AW = nb_AW_clf.predict(X_test)
    nb_pred_prob_AW = nb_AW_clf.predict_proba(X_test)
    nb_pred_class_CC = nb_CC_clf.predict(X_test)
    nb_pred_prob_CC = nb_CC_clf.predict_proba(X_test)
    nb_pred_class_QA = nb_QA_clf.predict(X_test)
    nb_pred_prob_QA = nb_QA_clf.predict_proba(X_test)

    rf_pred_class_AA = rf_AA_clf.predict(X_test)
    rf_pred_prob_AA = rf_AA_clf.predict_proba(X_test)
    rf_pred_class_AI = rf_AI_clf.predict(X_test)
    rf_pred_prob_AI = rf_AI_clf.predict_proba(X_test)
    rf_pred_class_AW = rf_AW_clf.predict(X_test)
    rf_pred_prob_AW = rf_AW_clf.predict_proba(X_test)
    rf_pred_class_CC = rf_CC_clf.predict(X_test)
    rf_pred_prob_CC = rf_CC_clf.predict_proba(X_test)
    rf_pred_class_QA = rf_QA_clf.predict(X_test)
    rf_pred_prob_QA = rf_QA_clf.predict_proba(X_test)

    svm_pred_class_AA = svm_AA_clf.predict(X_test)
    svm_pred_prob_AA = svm_AA_clf.predict_proba(X_test)
    svm_pred_class_AI = svm_AI_clf.predict(X_test)
    svm_pred_prob_AI = svm_AI_clf.predict_proba(X_test)
    svm_pred_class_AW = svm_AW_clf.predict(X_test)
    svm_pred_prob_AW = svm_AW_clf.predict_proba(X_test)
    svm_pred_class_CC = svm_CC_clf.predict(X_test)
    svm_pred_prob_CC = svm_CC_clf.predict_proba(X_test)
    svm_pred_class_QA = svm_QA_clf.predict(X_test)
    svm_pred_prob_QA = svm_QA_clf.predict_proba(X_test)

    dnn_prediction, lr_prediction, nb_prediction, rf_prediction, svm_prediction = pd.DataFrame(columns=['Prediction']),pd.DataFrame(columns=['Prediction']),pd.DataFrame(columns=['Prediction']),pd.DataFrame(columns=['Prediction']),pd.DataFrame(columns=['Prediction'])

    for i in range(0, len(y_test)):
        dnn_AA_index, dnn_AI_index, dnn_AW_index, dnn_CC_index, dnn_QA_index = 0, 0, 0, 0, 0
        if dnn_pred_class_AA[i] == "Accepted/Assigned":
            if dnn_pred_prob_AA[i][0] >= 0.5:
                dnn_AA_index = 0
            else:
                dnn_AA_index = 1
        elif dnn_pred_class_AA[i] == "Others":
            if dnn_pred_prob_AA[i][0] < 0.5:
                dnn_AA_index = 0
            else:
                dnn_AA_index = 1
        if dnn_pred_class_AI[i] == "Accepted/In Progress":
            if dnn_pred_prob_AI[i][0] >= 0.5:
                dnn_AI_index = 0
            else:
                dnn_AI_index = 1
        elif dnn_pred_class_AI[i] == "Others":
            if dnn_pred_prob_AI[i][0] < 0.5:
                dnn_AI_index = 0
            else:
                dnn_AI_index = 1
        if dnn_pred_class_AW[i] == "Accepted/Wait":
            if dnn_pred_prob_AW[i][0] >= 0.5:
                dnn_AW_index = 0
            else:
                dnn_AW_index = 1
        elif dnn_pred_class_AW[i] == "Others":
            if dnn_pred_prob_AW[i][0] < 0.5:
                dnn_AW_index = 0
            else:
                dnn_AW_index = 1
        if dnn_pred_class_CC[i] == "Completed/Closed":
            if dnn_pred_prob_CC[i][0] >= 0.5:
                dnn_CC_index = 0
            else:
                dnn_CC_index = 1
        elif dnn_pred_class_CC[i] == "Others":
            if dnn_pred_prob_CC[i][0] < 0.5:
                dnn_CC_index = 0
            else:
                dnn_CC_index = 1
        if dnn_pred_class_QA[i] == "Queued/Awaiting Assignment":
            if dnn_pred_prob_QA[i][0] >= 0.5:
                dnn_QA_index = 0
            else:
                dnn_QA_index = 1
        elif dnn_pred_class_QA[i] == "Others":
            if dnn_pred_prob_QA[i][0] < 0.5:
                dnn_QA_index = 0
            else:
                dnn_QA_index = 1
        if dnn_pred_prob_AA[i][dnn_AA_index] == max(dnn_pred_prob_AA[i][dnn_AA_index], dnn_pred_prob_AI[i][dnn_AI_index], dnn_pred_prob_AW[i][dnn_AW_index],
                                                    dnn_pred_prob_CC[i][dnn_CC_index], dnn_pred_prob_QA[i][dnn_QA_index]):
            dnn_prediction.loc[i] = "Accepted/Assigned"
        elif dnn_pred_prob_AI[i][dnn_AI_index] == max(dnn_pred_prob_AA[i][dnn_AA_index], dnn_pred_prob_AI[i][dnn_AI_index],
                                                      dnn_pred_prob_AW[i][dnn_AW_index], dnn_pred_prob_CC[i][dnn_CC_index],
                                                      dnn_pred_prob_QA[i][dnn_QA_index]):
            dnn_prediction.loc[i] = "Accepted/In Progress"
        elif dnn_pred_prob_AW[i][dnn_AW_index] == max(dnn_pred_prob_AA[i][dnn_AA_index], dnn_pred_prob_AI[i][dnn_AI_index],
                                                      dnn_pred_prob_AW[i][dnn_AW_index], dnn_pred_prob_CC[i][dnn_CC_index],
                                                      dnn_pred_prob_QA[i][dnn_QA_index]):
            dnn_prediction.loc[i] = "Accepted/Wait"
        elif dnn_pred_prob_CC[i][dnn_CC_index] == max(dnn_pred_prob_AA[i][dnn_AA_index], dnn_pred_prob_AI[i][dnn_AI_index],
                                                      dnn_pred_prob_AW[i][dnn_AW_index], dnn_pred_prob_CC[i][dnn_CC_index],
                                                      dnn_pred_prob_QA[i][dnn_QA_index]):
            dnn_prediction.loc[i] = "Completed/Closed"
        elif dnn_pred_prob_QA[i][dnn_QA_index] == max(dnn_pred_prob_AA[i][dnn_AA_index], dnn_pred_prob_AI[i][dnn_AI_index],
                                                      dnn_pred_prob_AW[i][dnn_AW_index], dnn_pred_prob_CC[i][dnn_CC_index],
                                                      dnn_pred_prob_QA[i][dnn_QA_index]):
            dnn_prediction.loc[i] = "Queued/Awaiting Assignment"

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

        if tp_1 + fp_1 == 0:
            precision_1 = 0
        else:
            precision_1 = tp_1 / (tp_1 + fp_1)
        if tp_2 + fp_2 == 0:
            precision_2 = 0
        else:
            precision_2 = tp_2 / (tp_2 + fp_2)
        if tp_3 + fp_3 == 0:
            precision_3 = 0
        else:
            precision_3 = tp_3 / (tp_3 + fp_3)
        if tp_4 + fp_4 == 0:
            precision_4 = 0
        else:
            precision_4 = tp_4 / (tp_4 + fp_4)
        if tp_5 + fp_5 == 0:
            precision_5 = 0
        else:
            precision_5 = tp_5 / (tp_5 + fp_5)
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
        if tp_1 + fn_1 == 0:
            recall_1 = 0
        else:
            recall_1 = tp_1 / (tp_1 + fn_1)
        if tp_2 + fn_2 == 0:
            recall_2 = 0
        else:
            recall_2 = tp_2 / (tp_2 + fn_2)
        if tp_3 + fn_3 == 0:
            recall_3 = 0
        else:
            recall_3 = tp_3 / (tp_3 + fn_3)
        if tp_4 + fn_4 == 0:
            recall_4 = 0
        else:
            recall_4 = tp_4 / (tp_4 + fn_4)
        if tp_5 + fn_5 == 0:
            recall_5 = 0
        else:
            recall_5 = tp_5 / (tp_5 +fn_5)
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
        if tp_1 + fn_1 == 0:
            recall_1 = 0
        else:
            recall_1 = tp_1 / (tp_1 + fn_1)
        if tp_2 + fn_2 == 0:
            recall_2 = 0
        else:
            recall_2 = tp_2 / (tp_2 + fn_2)
        if tp_3 + fn_3 == 0:
            recall_3 = 0
        else:
            recall_3 = tp_3 / (tp_3 + fn_3)
        if tp_4 + fn_4 == 0:
            recall_4 = 0
        else:
            recall_4 = tp_4 / (tp_4 + fn_4)
        if tp_5 + fn_5 == 0:
            recall_5 = 0
        else:
            recall_5 = tp_5 / (tp_5 +fn_5)
        recall_avg_pen_5 = (recall_1 + recall_2 + (5*recall_3) +recall_4 + recall_5) / (5+5-1)
        return recall_avg_pen_5

    dnn_conf_matrix = confusion_matrix(y_test, dnn_prediction)

    print("dnn_confusion matrix:")
    print(dnn_conf_matrix)
    dnn_precision = get_precision(dnn_conf_matrix)
    dnn_recall_pen_1 = get_recall_pen_1(dnn_conf_matrix)
    dnn_recall_pen_5 = get_recall_pen_5(dnn_conf_matrix)
    dnn_f1_score_pen_1 = 2 * (dnn_precision * dnn_recall_pen_1) / (dnn_precision + dnn_recall_pen_1)
    dnn_f1_score_pen_5 = 2 * (dnn_precision * dnn_recall_pen_5) / (dnn_precision + dnn_recall_pen_5)
    dnn_ovr_accuracy = (dnn_conf_matrix[0][0] + dnn_conf_matrix[1][1] + dnn_conf_matrix[2][2] + dnn_conf_matrix[3][3] + dnn_conf_matrix[4][4]) / (
                sum(dnn_conf_matrix[0]) + sum(dnn_conf_matrix[1]) + sum(dnn_conf_matrix[2]) + sum(dnn_conf_matrix[3]) + sum(dnn_conf_matrix[4]))
    dnn_auc_AA = roc_auc_score(y_true = y_test_AA, y_score = pd.DataFrame(dnn_pred_prob_AA).iloc[:,0])
    dnn_auc_AI = roc_auc_score(y_true = y_test_AI, y_score = pd.DataFrame(dnn_pred_prob_AI).iloc[:,0])
    dnn_auc_AW = roc_auc_score(y_true = y_test_AW, y_score = pd.DataFrame(dnn_pred_prob_AW).iloc[:,0])
    dnn_auc_CC = roc_auc_score(y_true = y_test_CC, y_score = pd.DataFrame(dnn_pred_prob_CC).iloc[:,0])
    dnn_auc_QA = roc_auc_score(y_true = y_test_QA, y_score = pd.DataFrame(dnn_pred_prob_QA).iloc[:,0])
    dnn_gmean = geometric_mean_score(y_true = y_test, y_pred = dnn_prediction, average = 'macro')
    print("dnn_f1 score of pen 1 is:")
    print(dnn_f1_score_pen_1)
    print("dnn_f1 score of pen 5 is:")
    print(dnn_f1_score_pen_5)
    print("dnn_overall accuracy is:")
    print(dnn_ovr_accuracy)
    print("dnn_auc is:")
    print((dnn_auc_AA + dnn_auc_AI + dnn_auc_AW + dnn_auc_CC + dnn_auc_QA)/5)
    print("dnn_gmean is:")
    print(dnn_gmean)
    dnn_conf_matrix = pd.DataFrame(dnn_conf_matrix)
    dnn_conf_matrix.to_csv('conf_matrix_'+imb_technique+'_dnn_bpic2013_closed_'+ str(nsplits) +'foldcv_' + str(repeat+1)+'.csv',header=False,index=False) #First repetition
    #dnn_conf_matrix.to_csv('conf_matrix_'+imb_technique+'_penalty_' + str(penalty) + '_dnn_bpic2013_closed_'+ str(nsplits) +'foldcv_' + str(repeat+6)+'.csv',header=False,index=False) #Second repetition
    dnn_f1_score_pen_1_kfoldcv[repeat] = dnn_f1_score_pen_1
    dnn_f1_score_pen_5_kfoldcv[repeat] = dnn_f1_score_pen_5
    dnn_ovr_accuracy_kfoldcv[repeat] = dnn_ovr_accuracy
    dnn_auc_kfoldcv[repeat] = (dnn_auc_AA + dnn_auc_AI + dnn_auc_AW + dnn_auc_CC + dnn_auc_QA)/5
    dnn_gmean_kfoldcv[repeat] = dnn_gmean

    for i in range(0, len(y_test)):
        lr_AA_index, lr_AI_index, lr_AW_index, lr_CC_index, lr_QA_index = 0, 0, 0, 0, 0
        if lr_pred_class_AA[i] == "Accepted/Assigned":
            if lr_pred_prob_AA[i][0] >= 0.5:
                lr_AA_index = 0
            else:
                lr_AA_index = 1
        elif lr_pred_class_AA[i] == "Others":
            if lr_pred_prob_AA[i][0] < 0.5:
                lr_AA_index = 0
            else:
                lr_AA_index = 1
        if lr_pred_class_AI[i] == "Accepted/In Progress":
            if lr_pred_prob_AI[i][0] >= 0.5:
                lr_AI_index = 0
            else:
                lr_AI_index = 1
        elif lr_pred_class_AI[i] == "Others":
            if lr_pred_prob_AI[i][0] < 0.5:
                lr_AI_index = 0
            else:
                lr_AI_index = 1
        if lr_pred_class_AW[i] == "Accepted/Wait":
            if lr_pred_prob_AW[i][0] >= 0.5:
                lr_AW_index = 0
            else:
                lr_AW_index = 1
        elif lr_pred_class_AW[i] == "Others":
            if lr_pred_prob_AW[i][0] < 0.5:
                lr_AW_index = 0
            else:
                lr_AW_index = 1
        if lr_pred_class_CC[i] == "Completed/Closed":
            if lr_pred_prob_CC[i][0] >= 0.5:
                lr_CC_index = 0
            else:
                lr_CC_index = 1
        elif lr_pred_class_CC[i] == "Others":
            if lr_pred_prob_CC[i][0] < 0.5:
                lr_CC_index = 0
            else:
                lr_CC_index = 1
        if lr_pred_class_QA[i] == "Queued/Awaiting Assignment":
            if lr_pred_prob_QA[i][0] >= 0.5:
                lr_QA_index = 0
            else:
                lr_QA_index = 1
        elif lr_pred_class_QA[i] == "Others":
            if lr_pred_prob_QA[i][0] < 0.5:
                lr_QA_index = 0
            else:
                lr_QA_index = 1
        if lr_pred_prob_AA[i][lr_AA_index] == max(lr_pred_prob_AA[i][lr_AA_index], lr_pred_prob_AI[i][lr_AI_index],
                                                  lr_pred_prob_AW[i][lr_AW_index],
                                                  lr_pred_prob_CC[i][lr_CC_index], lr_pred_prob_QA[i][lr_QA_index]):
            lr_prediction.loc[i] = "Accepted/Assigned"
        elif lr_pred_prob_AI[i][lr_AI_index] == max(lr_pred_prob_AA[i][lr_AA_index], lr_pred_prob_AI[i][lr_AI_index],
                                                    lr_pred_prob_AW[i][lr_AW_index], lr_pred_prob_CC[i][lr_CC_index],
                                                    lr_pred_prob_QA[i][lr_QA_index]):
            lr_prediction.loc[i] = "Accepted/In Progress"
        elif lr_pred_prob_AW[i][lr_AW_index] == max(lr_pred_prob_AA[i][lr_AA_index], lr_pred_prob_AI[i][lr_AI_index],
                                                    lr_pred_prob_AW[i][lr_AW_index], lr_pred_prob_CC[i][lr_CC_index],
                                                    lr_pred_prob_QA[i][lr_QA_index]):
            lr_prediction.loc[i] = "Accepted/Wait"
        elif lr_pred_prob_CC[i][lr_CC_index] == max(lr_pred_prob_AA[i][lr_AA_index], lr_pred_prob_AI[i][lr_AI_index],
                                                    lr_pred_prob_AW[i][lr_AW_index], lr_pred_prob_CC[i][lr_CC_index],
                                                    lr_pred_prob_QA[i][lr_QA_index]):
            lr_prediction.loc[i] = "Completed/Closed"
        elif lr_pred_prob_QA[i][lr_QA_index] == max(lr_pred_prob_AA[i][lr_AA_index], lr_pred_prob_AI[i][lr_AI_index],
                                                    lr_pred_prob_AW[i][lr_AW_index], lr_pred_prob_CC[i][lr_CC_index],
                                                    lr_pred_prob_QA[i][lr_QA_index]):
            lr_prediction.loc[i] = "Queued/Awaiting Assignment"

    lr_conf_matrix = confusion_matrix(y_test, lr_prediction)

    print("lr_confusion matrix:")
    print(lr_conf_matrix)
    lr_precision = get_precision(lr_conf_matrix)
    lr_recall_pen_1 = get_recall_pen_1(lr_conf_matrix)
    lr_recall_pen_5 = get_recall_pen_5(lr_conf_matrix)
    lr_f1_score_pen_1 = 2 * (lr_precision * lr_recall_pen_1) / (lr_precision + lr_recall_pen_1)
    lr_f1_score_pen_5 = 2 * (lr_precision * lr_recall_pen_5) / (lr_precision + lr_recall_pen_5)
    lr_ovr_accuracy = (lr_conf_matrix[0][0] + lr_conf_matrix[1][1] + lr_conf_matrix[2][2] + lr_conf_matrix[3][3] + lr_conf_matrix[4][4]) / (
                sum(lr_conf_matrix[0]) + sum(lr_conf_matrix[1]) + sum(lr_conf_matrix[2]) + sum(lr_conf_matrix[3]) + sum(lr_conf_matrix[4]))
    lr_auc_AA = roc_auc_score(y_true = y_test_AA, y_score = pd.DataFrame(lr_pred_prob_AA).iloc[:,0])
    lr_auc_AI = roc_auc_score(y_true = y_test_AI, y_score = pd.DataFrame(lr_pred_prob_AI).iloc[:,0])
    lr_auc_AW = roc_auc_score(y_true = y_test_AW, y_score = pd.DataFrame(lr_pred_prob_AW).iloc[:,0])
    lr_auc_CC = roc_auc_score(y_true = y_test_CC, y_score = pd.DataFrame(lr_pred_prob_CC).iloc[:,0])
    lr_auc_QA = roc_auc_score(y_true = y_test_QA, y_score = pd.DataFrame(lr_pred_prob_QA).iloc[:,0])
    lr_gmean = geometric_mean_score(y_true = y_test, y_pred = lr_prediction, average = 'macro')
    print("lr_f1 score of pen 1 is:")
    print(lr_f1_score_pen_1)
    print("lr_f1 score of pen 5 is:")
    print(lr_f1_score_pen_5)
    print("lr_overall accuracy is:")
    print(lr_ovr_accuracy)
    print("lr_auc is:")
    print((lr_auc_AA + lr_auc_AI + lr_auc_AW + lr_auc_CC + lr_auc_QA)/5)
    print("lr_gmean is:")
    print(lr_gmean)
    lr_conf_matrix = pd.DataFrame(lr_conf_matrix)
    lr_conf_matrix.to_csv('conf_matrix_'+imb_technique+'_lr_bpic2013_closed_'+ str(nsplits) +'foldcv_' + str(repeat+1)+'.csv',header=False,index=False) #First repetition
    #lr_conf_matrix.to_csv('conf_matrix_'+imb_technique+'_penalty_' + str(penalty) + '_lr_bpic2013_closed_'+ str(nsplits) +'foldcv_' + str(repeat+6)+'.csv',header=False,index=False) #Second repetition
    lr_f1_score_pen_1_kfoldcv[repeat] = lr_f1_score_pen_1
    lr_f1_score_pen_5_kfoldcv[repeat] = lr_f1_score_pen_5
    lr_ovr_accuracy_kfoldcv[repeat] = lr_ovr_accuracy
    lr_auc_kfoldcv[repeat] = (lr_auc_AA + lr_auc_AI + lr_auc_AW + lr_auc_CC + lr_auc_QA)/5
    lr_gmean_kfoldcv[repeat] = lr_gmean

    for i in range(0, len(y_test)):
        nb_AA_index, nb_AI_index, nb_AW_index, nb_CC_index, nb_QA_index = 0, 0, 0, 0, 0
        if nb_pred_class_AA[i] == "Accepted/Assigned":
            if nb_pred_prob_AA[i][0] >= 0.5:
                nb_AA_index = 0
            else:
                nb_AA_index = 1
        elif nb_pred_class_AA[i] == "Others":
            if nb_pred_prob_AA[i][0] < 0.5:
                nb_AA_index = 0
            else:
                nb_AA_index = 1
        if nb_pred_class_AI[i] == "Accepted/In Progress":
            if nb_pred_prob_AI[i][0] >= 0.5:
                nb_AI_index = 0
            else:
                nb_AI_index = 1
        elif nb_pred_class_AI[i] == "Others":
            if nb_pred_prob_AI[i][0] < 0.5:
                nb_AI_index = 0
            else:
                nb_AI_index = 1
        if nb_pred_class_AW[i] == "Accepted/Wait":
            if nb_pred_prob_AW[i][0] >= 0.5:
                nb_AW_index = 0
            else:
                nb_AW_index = 1
        elif nb_pred_class_AW[i] == "Others":
            if nb_pred_prob_AW[i][0] < 0.5:
                nb_AW_index = 0
            else:
                nb_AW_index = 1
        if nb_pred_class_CC[i] == "Completed/Closed":
            if nb_pred_prob_CC[i][0] >= 0.5:
                nb_CC_index = 0
            else:
                nb_CC_index = 1
        elif nb_pred_class_CC[i] == "Others":
            if nb_pred_prob_CC[i][0] < 0.5:
                nb_CC_index = 0
            else:
                nb_CC_index = 1
        if nb_pred_class_QA[i] == "Queued/Awaiting Assignment":
            if nb_pred_prob_QA[i][0] >= 0.5:
                nb_QA_index = 0
            else:
                nb_QA_index = 1
        elif nb_pred_class_QA[i] == "Others":
            if nb_pred_prob_QA[i][0] < 0.5:
                nb_QA_index = 0
            else:
                nb_QA_index = 1
        if nb_pred_prob_AA[i][nb_AA_index] == max(nb_pred_prob_AA[i][nb_AA_index], nb_pred_prob_AI[i][nb_AI_index], nb_pred_prob_AW[i][nb_AW_index],
                                                    nb_pred_prob_CC[i][nb_CC_index], nb_pred_prob_QA[i][nb_QA_index]):
            nb_prediction.loc[i] = "Accepted/Assigned"
        elif nb_pred_prob_AI[i][nb_AI_index] == max(nb_pred_prob_AA[i][nb_AA_index], nb_pred_prob_AI[i][nb_AI_index],
                                                      nb_pred_prob_AW[i][nb_AW_index], nb_pred_prob_CC[i][nb_CC_index],
                                                      nb_pred_prob_QA[i][nb_QA_index]):
            nb_prediction.loc[i] = "Accepted/In Progress"
        elif nb_pred_prob_AW[i][nb_AW_index] == max(nb_pred_prob_AA[i][nb_AA_index], nb_pred_prob_AI[i][nb_AI_index],
                                                      nb_pred_prob_AW[i][nb_AW_index], nb_pred_prob_CC[i][nb_CC_index],
                                                      nb_pred_prob_QA[i][nb_QA_index]):
            nb_prediction.loc[i] = "Accepted/Wait"
        elif nb_pred_prob_CC[i][nb_CC_index] == max(nb_pred_prob_AA[i][nb_AA_index], nb_pred_prob_AI[i][nb_AI_index],
                                                      nb_pred_prob_AW[i][nb_AW_index], nb_pred_prob_CC[i][nb_CC_index],
                                                      nb_pred_prob_QA[i][nb_QA_index]):
            nb_prediction.loc[i] = "Completed/Closed"
        elif nb_pred_prob_QA[i][nb_QA_index] == max(nb_pred_prob_AA[i][nb_AA_index], nb_pred_prob_AI[i][nb_AI_index],
                                                      nb_pred_prob_AW[i][nb_AW_index], nb_pred_prob_CC[i][nb_CC_index],
                                                      nb_pred_prob_QA[i][nb_QA_index]):
            nb_prediction.loc[i] = "Queued/Awaiting Assignment"

    nb_conf_matrix = confusion_matrix(y_test, nb_prediction)

    print("nb_confusion matrix:")
    print(nb_conf_matrix)
    nb_precision = get_precision(nb_conf_matrix)
    nb_recall_pen_1 = get_recall_pen_1(nb_conf_matrix)
    nb_recall_pen_5 = get_recall_pen_5(nb_conf_matrix)
    nb_f1_score_pen_1 = 2 * (nb_precision * nb_recall_pen_1) / (nb_precision + nb_recall_pen_1)
    nb_f1_score_pen_5 = 2 * (nb_precision * nb_recall_pen_5) / (nb_precision + nb_recall_pen_5)
    nb_ovr_accuracy = (nb_conf_matrix[0][0] + nb_conf_matrix[1][1] + nb_conf_matrix[2][2] + nb_conf_matrix[3][3] + nb_conf_matrix[4][4]) / (
                sum(nb_conf_matrix[0]) + sum(nb_conf_matrix[1]) + sum(nb_conf_matrix[2]) + sum(nb_conf_matrix[3]) + sum(nb_conf_matrix[4]))
    nb_auc_AA = roc_auc_score(y_true = y_test_AA, y_score = pd.DataFrame(nb_pred_prob_AA).iloc[:,0])
    nb_auc_AI = roc_auc_score(y_true = y_test_AI, y_score = pd.DataFrame(nb_pred_prob_AI).iloc[:,0])
    nb_auc_AW = roc_auc_score(y_true = y_test_AW, y_score = pd.DataFrame(nb_pred_prob_AW).iloc[:,0])
    nb_auc_CC = roc_auc_score(y_true = y_test_CC, y_score = pd.DataFrame(nb_pred_prob_CC).iloc[:,0])
    nb_auc_QA = roc_auc_score(y_true = y_test_QA, y_score = pd.DataFrame(nb_pred_prob_QA).iloc[:,0])
    nb_gmean = geometric_mean_score(y_true = y_test, y_pred = nb_prediction, average = 'macro')
    print("nb_f1 score of pen 1 is:")
    print(nb_f1_score_pen_1)
    print("nb_f1 score of pen 5 is:")
    print(nb_f1_score_pen_5)
    print("nb_overall accuracy is:")
    print(nb_ovr_accuracy)
    print("nb_auc is:")
    print((nb_auc_AA + nb_auc_AI + nb_auc_AW + nb_auc_CC + nb_auc_QA)/5)
    print("nb_gmean is:")
    print(nb_gmean)
    nb_conf_matrix = pd.DataFrame(nb_conf_matrix)
    nb_conf_matrix.to_csv('conf_matrix_'+imb_technique+'_nb_bpic2013_closed_'+ str(nsplits) +'foldcv_' + str(repeat+1)+'.csv',header=False,index=False) #First repetition
    #nb_conf_matrix.to_csv('conf_matrix_'+imb_technique+'_penalty_' + str(penalty) + '_nb_bpic2013_closed_'+ str(nsplits) +'foldcv_' + str(repeat+6)+'.csv',header=False,index=False) #Second repetition
    nb_f1_score_pen_1_kfoldcv[repeat] = nb_f1_score_pen_1
    nb_f1_score_pen_5_kfoldcv[repeat] = nb_f1_score_pen_5
    nb_ovr_accuracy_kfoldcv[repeat] = nb_ovr_accuracy
    nb_auc_kfoldcv[repeat] = (nb_auc_AA + nb_auc_AI + nb_auc_AW + nb_auc_CC + nb_auc_QA)/5
    nb_gmean_kfoldcv[repeat] = nb_gmean

    for i in range(0, len(y_test)):
        rf_AA_index, rf_AI_index, rf_AW_index, rf_CC_index, rf_QA_index = 0, 0, 0, 0, 0
        if rf_pred_class_AA[i] == "Accepted/Assigned":
            if rf_pred_prob_AA[i][0] >= 0.5:
                rf_AA_index = 0
            else:
                rf_AA_index = 1
        elif rf_pred_class_AA[i] == "Others":
            if rf_pred_prob_AA[i][0] < 0.5:
                rf_AA_index = 0
            else:
                rf_AA_index = 1
        if rf_pred_class_AI[i] == "Accepted/In Progress":
            if rf_pred_prob_AI[i][0] >= 0.5:
                rf_AI_index = 0
            else:
                rf_AI_index = 1
        elif rf_pred_class_AI[i] == "Others":
            if rf_pred_prob_AI[i][0] < 0.5:
                rf_AI_index = 0
            else:
                rf_AI_index = 1
        if rf_pred_class_AW[i] == "Accepted/Wait":
            if rf_pred_prob_AW[i][0] >= 0.5:
                rf_AW_index = 0
            else:
                rf_AW_index = 1
        elif rf_pred_class_AW[i] == "Others":
            if rf_pred_prob_AW[i][0] < 0.5:
                rf_AW_index = 0
            else:
                rf_AW_index = 1
        if rf_pred_class_CC[i] == "Completed/Closed":
            if rf_pred_prob_CC[i][0] >= 0.5:
                rf_CC_index = 0
            else:
                rf_CC_index = 1
        elif rf_pred_class_CC[i] == "Others":
            if rf_pred_prob_CC[i][0] < 0.5:
                rf_CC_index = 0
            else:
                rf_CC_index = 1
        if rf_pred_class_QA[i] == "Queued/Awaiting Assignment":
            if rf_pred_prob_QA[i][0] >= 0.5:
                rf_QA_index = 0
            else:
                rf_QA_index = 1
        elif rf_pred_class_QA[i] == "Others":
            if rf_pred_prob_QA[i][0] < 0.5:
                rf_QA_index = 0
            else:
                rf_QA_index = 1
        if rf_pred_prob_AA[i][rf_AA_index] == max(rf_pred_prob_AA[i][rf_AA_index], rf_pred_prob_AI[i][rf_AI_index], rf_pred_prob_AW[i][rf_AW_index],
                                                    rf_pred_prob_CC[i][rf_CC_index], rf_pred_prob_QA[i][rf_QA_index]):
            rf_prediction.loc[i] = "Accepted/Assigned"
        elif rf_pred_prob_AI[i][rf_AI_index] == max(rf_pred_prob_AA[i][rf_AA_index], rf_pred_prob_AI[i][rf_AI_index],
                                                      rf_pred_prob_AW[i][rf_AW_index], rf_pred_prob_CC[i][rf_CC_index],
                                                      rf_pred_prob_QA[i][rf_QA_index]):
            rf_prediction.loc[i] = "Accepted/In Progress"
        elif rf_pred_prob_AW[i][rf_AW_index] == max(rf_pred_prob_AA[i][rf_AA_index], rf_pred_prob_AI[i][rf_AI_index],
                                                      rf_pred_prob_AW[i][rf_AW_index], rf_pred_prob_CC[i][rf_CC_index],
                                                      rf_pred_prob_QA[i][rf_QA_index]):
            rf_prediction.loc[i] = "Accepted/Wait"
        elif rf_pred_prob_CC[i][rf_CC_index] == max(rf_pred_prob_AA[i][rf_AA_index], rf_pred_prob_AI[i][rf_AI_index],
                                                      rf_pred_prob_AW[i][rf_AW_index], rf_pred_prob_CC[i][rf_CC_index],
                                                      rf_pred_prob_QA[i][rf_QA_index]):
            rf_prediction.loc[i] = "Completed/Closed"
        elif rf_pred_prob_QA[i][rf_QA_index] == max(rf_pred_prob_AA[i][rf_AA_index], rf_pred_prob_AI[i][rf_AI_index],
                                                      rf_pred_prob_AW[i][rf_AW_index], rf_pred_prob_CC[i][rf_CC_index],
                                                      rf_pred_prob_QA[i][rf_QA_index]):
            rf_prediction.loc[i] = "Queued/Awaiting Assignment"

    rf_conf_matrix = confusion_matrix(y_test, rf_prediction)

    print("rf_confusion matrix:")
    print(rf_conf_matrix)
    rf_precision = get_precision(rf_conf_matrix)
    rf_recall_pen_1 = get_recall_pen_1(rf_conf_matrix)
    rf_recall_pen_5 = get_recall_pen_5(rf_conf_matrix)
    rf_f1_score_pen_1 = 2 * (rf_precision * rf_recall_pen_1) / (rf_precision + rf_recall_pen_1)
    rf_f1_score_pen_5 = 2 * (rf_precision * rf_recall_pen_5) / (rf_precision + rf_recall_pen_5)
    rf_ovr_accuracy = (rf_conf_matrix[0][0] + rf_conf_matrix[1][1] + rf_conf_matrix[2][2] + rf_conf_matrix[3][3] + rf_conf_matrix[4][4]) / (
                sum(rf_conf_matrix[0]) + sum(rf_conf_matrix[1]) + sum(rf_conf_matrix[2]) + sum(rf_conf_matrix[3]) + sum(rf_conf_matrix[4]))
    rf_auc_AA = roc_auc_score(y_true = y_test_AA, y_score = pd.DataFrame(rf_pred_prob_AA).iloc[:,0])
    rf_auc_AI = roc_auc_score(y_true = y_test_AI, y_score = pd.DataFrame(rf_pred_prob_AI).iloc[:,0])
    rf_auc_AW = roc_auc_score(y_true = y_test_AW, y_score = pd.DataFrame(rf_pred_prob_AW).iloc[:,0])
    rf_auc_CC = roc_auc_score(y_true = y_test_CC, y_score = pd.DataFrame(rf_pred_prob_CC).iloc[:,0])
    rf_auc_QA = roc_auc_score(y_true = y_test_QA, y_score = pd.DataFrame(rf_pred_prob_QA).iloc[:,0])
    rf_gmean = geometric_mean_score(y_true = y_test, y_pred = rf_prediction, average = 'macro')
    print("rf_f1 score of pen 1 is:")
    print(rf_f1_score_pen_1)
    print("rf_f1 score of pen 5 is:")
    print(rf_f1_score_pen_5)
    print("rf_overall accuracy is:")
    print(rf_ovr_accuracy)
    print("rf_auc is:")
    print((rf_auc_AA + rf_auc_AI + rf_auc_AW + rf_auc_CC + rf_auc_QA)/5)
    print("rf_gmean is:")
    print(rf_gmean)
    rf_conf_matrix = pd.DataFrame(rf_conf_matrix)
    rf_conf_matrix.to_csv('conf_matrix_'+imb_technique+'_rf_bpic2013_closed_'+ str(nsplits) +'foldcv_' + str(repeat+1)+'.csv',header=False,index=False) #First repetition
    #rf_conf_matrix.to_csv('conf_matrix_'+imb_technique+'_penalty_' + str(penalty) + '_rf_bpic2013_closed_'+ str(nsplits) +'foldcv_' + str(repeat+6)+'.csv',header=False,index=False) #Second repetition
    rf_f1_score_pen_1_kfoldcv[repeat] = rf_f1_score_pen_1
    rf_f1_score_pen_5_kfoldcv[repeat] = rf_f1_score_pen_5
    rf_ovr_accuracy_kfoldcv[repeat] = rf_ovr_accuracy
    rf_auc_kfoldcv[repeat] = (rf_auc_AA + rf_auc_AI + rf_auc_AW + rf_auc_CC + rf_auc_QA)/5
    rf_gmean_kfoldcv[repeat] = rf_gmean

    for i in range(0, len(y_test)):
        svm_AA_index, svm_AI_index, svm_AW_index, svm_CC_index, svm_QA_index = 0, 0, 0, 0, 0
        if svm_pred_class_AA[i] == "Accepted/Assigned":
            if svm_pred_prob_AA[i][0] >= 0.5:
                svm_AA_index = 0
            else:
                svm_AA_index = 1
        elif svm_pred_class_AA[i] == "Others":
            if svm_pred_prob_AA[i][0] < 0.5:
                svm_AA_index = 0
            else:
                svm_AA_index = 1
        if svm_pred_class_AI[i] == "Accepted/In Progress":
            if svm_pred_prob_AI[i][0] >= 0.5:
                svm_AI_index = 0
            else:
                svm_AI_index = 1
        elif svm_pred_class_AI[i] == "Others":
            if svm_pred_prob_AI[i][0] < 0.5:
                svm_AI_index = 0
            else:
                svm_AI_index = 1
        if svm_pred_class_AW[i] == "Accepted/Wait":
            if svm_pred_prob_AW[i][0] >= 0.5:
                svm_AW_index = 0
            else:
                svm_AW_index = 1
        elif svm_pred_class_AW[i] == "Others":
            if svm_pred_prob_AW[i][0] < 0.5:
                svm_AW_index = 0
            else:
                svm_AW_index = 1
        if svm_pred_class_CC[i] == "Completed/Closed":
            if svm_pred_prob_CC[i][0] >= 0.5:
                svm_CC_index = 0
            else:
                svm_CC_index = 1
        elif svm_pred_class_CC[i] == "Others":
            if svm_pred_prob_CC[i][0] < 0.5:
                svm_CC_index = 0
            else:
                svm_CC_index = 1
        if svm_pred_class_QA[i] == "Queued/Awaiting Assignment":
            if svm_pred_prob_QA[i][0] >= 0.5:
                svm_QA_index = 0
            else:
                svm_QA_index = 1
        elif svm_pred_class_QA[i] == "Others":
            if svm_pred_prob_QA[i][0] < 0.5:
                svm_QA_index = 0
            else:
                svm_QA_index = 1
        if svm_pred_prob_AA[i][svm_AA_index] == max(svm_pred_prob_AA[i][svm_AA_index], svm_pred_prob_AI[i][svm_AI_index], svm_pred_prob_AW[i][svm_AW_index],
                                                    svm_pred_prob_CC[i][svm_CC_index], svm_pred_prob_QA[i][svm_QA_index]):
            svm_prediction.loc[i] = "Accepted/Assigned"
        elif svm_pred_prob_AI[i][svm_AI_index] == max(svm_pred_prob_AA[i][svm_AA_index], svm_pred_prob_AI[i][svm_AI_index],
                                                      svm_pred_prob_AW[i][svm_AW_index], svm_pred_prob_CC[i][svm_CC_index],
                                                      svm_pred_prob_QA[i][svm_QA_index]):
            svm_prediction.loc[i] = "Accepted/In Progress"
        elif svm_pred_prob_AW[i][svm_AW_index] == max(svm_pred_prob_AA[i][svm_AA_index], svm_pred_prob_AI[i][svm_AI_index],
                                                      svm_pred_prob_AW[i][svm_AW_index], svm_pred_prob_CC[i][svm_CC_index],
                                                      svm_pred_prob_QA[i][svm_QA_index]):
            svm_prediction.loc[i] = "Accepted/Wait"
        elif svm_pred_prob_CC[i][svm_CC_index] == max(svm_pred_prob_AA[i][svm_AA_index], svm_pred_prob_AI[i][svm_AI_index],
                                                      svm_pred_prob_AW[i][svm_AW_index], svm_pred_prob_CC[i][svm_CC_index],
                                                      svm_pred_prob_QA[i][svm_QA_index]):
            svm_prediction.loc[i] = "Completed/Closed"
        elif svm_pred_prob_QA[i][svm_QA_index] == max(svm_pred_prob_AA[i][svm_AA_index], svm_pred_prob_AI[i][svm_AI_index],
                                                      svm_pred_prob_AW[i][svm_AW_index], svm_pred_prob_CC[i][svm_CC_index],
                                                      svm_pred_prob_QA[i][svm_QA_index]):
            svm_prediction.loc[i] = "Queued/Awaiting Assignment"

    svm_conf_matrix = confusion_matrix(y_test, svm_prediction)

    print("svm_confusion matrix:")
    print(svm_conf_matrix)
    svm_precision = get_precision(svm_conf_matrix)
    svm_recall_pen_1 = get_recall_pen_1(svm_conf_matrix)
    svm_recall_pen_5 = get_recall_pen_5(svm_conf_matrix)
    svm_f1_score_pen_1 = 2 * (svm_precision * svm_recall_pen_1) / (svm_precision + svm_recall_pen_1)
    svm_f1_score_pen_5 = 2 * (svm_precision * svm_recall_pen_5) / (svm_precision + svm_recall_pen_5)
    svm_ovr_accuracy = (svm_conf_matrix[0][0] + svm_conf_matrix[1][1] + svm_conf_matrix[2][2] + svm_conf_matrix[3][3] + svm_conf_matrix[4][4]) / (
                sum(svm_conf_matrix[0]) + sum(svm_conf_matrix[1]) + sum(svm_conf_matrix[2]) + sum(svm_conf_matrix[3]) + sum(svm_conf_matrix[4]))
    svm_auc_AA = roc_auc_score(y_true = y_test_AA, y_score = pd.DataFrame(svm_pred_prob_AA).iloc[:,0])
    svm_auc_AI = roc_auc_score(y_true = y_test_AI, y_score = pd.DataFrame(svm_pred_prob_AI).iloc[:,0])
    svm_auc_AW = roc_auc_score(y_true = y_test_AW, y_score = pd.DataFrame(svm_pred_prob_AW).iloc[:,0])
    svm_auc_CC = roc_auc_score(y_true = y_test_CC, y_score = pd.DataFrame(svm_pred_prob_CC).iloc[:,0])
    svm_auc_QA = roc_auc_score(y_true = y_test_QA, y_score = pd.DataFrame(svm_pred_prob_QA).iloc[:,0])
    svm_gmean = geometric_mean_score(y_true = y_test, y_pred = svm_prediction, average = 'macro')
    print("svm_f1 score of pen 1 is:")
    print(svm_f1_score_pen_1)
    print("svm_f1 score of pen 5 is:")
    print(svm_f1_score_pen_5)
    print("svm_overall accuracy is:")
    print(svm_ovr_accuracy)
    print("svm_auc is:")
    print((svm_auc_AA + svm_auc_AI + svm_auc_AW + svm_auc_CC + svm_auc_QA)/5)
    print("svm_gmean is:")
    print(svm_gmean)
    svm_conf_matrix = pd.DataFrame(svm_conf_matrix)
    svm_conf_matrix.to_csv('conf_matrix_'+imb_technique+'_svm_bpic2013_closed_'+ str(nsplits) +'foldcv_' + str(repeat+1)+'.csv',header=False,index=False) #First repetition
    #svm_conf_matrix.to_csv('conf_matrix_'+imb_technique+'_penalty_' + str(penalty) + '_svm_bpic2013_closed_'+ str(nsplits) +'foldcv_' + str(repeat+6)+'.csv',header=False,index=False) #Second repetition
    svm_f1_score_pen_1_kfoldcv[repeat] = svm_f1_score_pen_1
    svm_f1_score_pen_5_kfoldcv[repeat] = svm_f1_score_pen_5
    svm_ovr_accuracy_kfoldcv[repeat] = svm_ovr_accuracy
    svm_auc_kfoldcv[repeat] = (svm_auc_AA + svm_auc_AI + svm_auc_AW + svm_auc_CC + svm_auc_QA)/5
    svm_gmean_kfoldcv[repeat] = svm_gmean

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

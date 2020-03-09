import tensorflow as tf
import numpy as np
import pandas as pd
from collections import Counter
#Pandas was used to import csv file. Encoding parameter is set to "cp437" since the data contains English text
data_sample_percentage = "" #Delete after experiments. If data is not to be reduced, type in "" here. If to be reduced, type in, for example, "_20percent"
data_dir = "/home/jongchan/BPIC2013_closed/window_3_closed_problems_preprocessed" + data_sample_percentage + ".csv"
data = pd.read_csv(data_dir, encoding='cp437')
#data = data.sample(frac=1).reset_index(drop=True) #I use this for ADASYN since ADASYN returns error on specific fold
X = data[['ACT_COMB_1', 'ACT_COMB_2', 'ACT_COMB_3','duration_in_days']]
y = data[['ACT_COMB_4']]

#################################################
########## Choose resampling technique ##########
#################################################
imb_technique = input("Type in the name of the resampling technique among the followings -> Baseline / ADASYN / ALLKNN / CNN / ENN / IHT / NCR / NM / OSS / RENN / ROS / RUS / SMOTE / BSMOTE / SMOTEENN / SMOTETOMEK / TOMEK: ")
#imb_technique = "Baseline"
#imb_technique = "ADASYN"
#imb_technique = "ALLKNN"
#imb_technique = "CNN"
#imb_technique = "ENN"
#imb_technique = "GAN" #Not implemented yet
#imb_technique = "IHT"
#imb_technique = "NCR"
#imb_technique = "NM"
#imb_technique = "OSS"
#imb_technique = "RENN"
#imb_technique = "ROS"
#imb_technique = "RUS"
#imb_technique = "SMOTE"
#imb_technique = "BSMOTE"
#imb_technique = "SMOTEENN"
#imb_technique = "SMOTETOMEK"
#imb_technique = "TOMEK"
print("Data: BPIC 2013 closed")
print("Resampling technique: " + imb_technique)

# Dummification
X_dummy = pd.get_dummies(X, prefix="ACT_COMB_1", columns=['ACT_COMB_1'])
X_dummy = pd.get_dummies(X_dummy, prefix="ACT_COMB_2", columns=['ACT_COMB_2'])
X_dummy = pd.get_dummies(X_dummy, prefix="ACT_COMB_3", columns=['ACT_COMB_3'])
X_dummy.iloc[:, 0] = (X_dummy.iloc[:, 0] - X_dummy.iloc[:, 0].mean()) / X_dummy.iloc[:, 0].std()

# X and y here will be used for hyperparameter tuning using random search
X_randomsearch = X.replace(regex=True, to_replace="Accepted/Assigned", value=1)
X_randomsearch = X_randomsearch.replace(regex=True, to_replace="Accepted/In Progress", value=2)
X_randomsearch = X_randomsearch.replace(regex=True, to_replace="Accepted/Wait", value=3)
X_randomsearch = X_randomsearch.replace(regex=True, to_replace="Completed/Closed", value=4)
X_randomsearch = X_randomsearch.replace(regex=True, to_replace="Queued/Awaiting Assignment", value=5)
X_randomsearch = X_randomsearch.replace(regex=True, to_replace="Completed/Cancelled", value=6)
X_randomsearch = X_randomsearch.replace(regex=True, to_replace="Unmatched/Unmatched", value=7)

y_randomsearch = y.replace(regex=True, to_replace="Accepted/Assigned", value=1)
y_randomsearch = y_randomsearch.replace(regex=True, to_replace="Accepted/In Progress", value=2)
y_randomsearch = y_randomsearch.replace(regex=True, to_replace="Accepted/Wait", value=3)
y_randomsearch = y_randomsearch.replace(regex=True, to_replace="Completed/Closed", value=4)
y_randomsearch = y_randomsearch.replace(regex=True, to_replace="Queued/Awaiting Assignment", value=5)

from sklearn.model_selection import KFold
nsplits = 5 # Set the number of k for cross validation
kf = KFold(n_splits=nsplits)
kf.get_n_splits(X_dummy)
print(kf)

dnn_f1_score_pen_1_kfoldcv = [None] * (nsplits+2)
dnn_f1_score_pen_5_kfoldcv = [None] * (nsplits+2)
dnn_ovr_accuracy_kfoldcv = [None] * (nsplits+2)

lr_f1_score_pen_1_kfoldcv = [None] * (nsplits+2)
lr_f1_score_pen_5_kfoldcv = [None] * (nsplits+2)
lr_ovr_accuracy_kfoldcv = [None] * (nsplits+2)

nb_f1_score_pen_1_kfoldcv = [None] * (nsplits+2)
nb_f1_score_pen_5_kfoldcv = [None] * (nsplits+2)
nb_ovr_accuracy_kfoldcv = [None] * (nsplits+2)

rf_f1_score_pen_1_kfoldcv = [None] * (nsplits+2)
rf_f1_score_pen_5_kfoldcv = [None] * (nsplits+2)
rf_ovr_accuracy_kfoldcv = [None] * (nsplits+2)

svm_f1_score_pen_1_kfoldcv = [None] * (nsplits+2)
svm_f1_score_pen_5_kfoldcv = [None] * (nsplits+2)
svm_ovr_accuracy_kfoldcv = [None] * (nsplits+2)

repeat = 0
for train_index, test_index in kf.split(X_dummy):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X_dummy.iloc[train_index], X_dummy.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    train = pd.concat([X_train, y_train], axis=1)
    ACT_COMB_4_index = np.unique(data['ACT_COMB_1']).size + np.unique(data['ACT_COMB_2']).size + np.unique(data['ACT_COMB_3']).size + 1

    AA = train[train.ACT_COMB_4 == "Accepted/Assigned"]
    AA_rest = train[train.ACT_COMB_4 != "Accepted/Assigned"]
    AA_rest = AA_rest.copy()
    AA_rest.iloc[:, ACT_COMB_4_index] = "Others"
    AA = AA.reset_index()
    AA_rest = AA_rest.reset_index()
    AA = AA.iloc[:, 1:ACT_COMB_4_index+2]
    AA_rest = AA_rest.iloc[:, 1:ACT_COMB_4_index+2]
    AA_ova = pd.concat([AA, AA_rest])
    AA_ova_X_train = AA_ova.iloc[:, 0:ACT_COMB_4_index]
    AA_ova_y_train = AA_ova.iloc[:, ACT_COMB_4_index]
    AA_X_res = AA_ova_X_train
    AA_y_res = AA_ova_y_train
    Counter(AA_ova_y_train)

    AI = train[train.ACT_COMB_4 == "Accepted/In Progress"]
    AI_rest = train[train.ACT_COMB_4 != "Accepted/In Progress"]
    AI_rest = AI_rest.copy()
    AI_rest.iloc[:, ACT_COMB_4_index] = "Others"
    AI = AI.reset_index()
    AI_rest = AI_rest.reset_index()
    AI = AI.iloc[:, 1:ACT_COMB_4_index+2]
    AI_rest = AI_rest.iloc[:, 1:ACT_COMB_4_index+2]
    AI_ova = pd.concat([AI, AI_rest])
    AI_ova_X_train = AI_ova.iloc[:, 0:ACT_COMB_4_index]
    AI_ova_y_train = AI_ova.iloc[:, ACT_COMB_4_index]
    AI_X_res = AI_ova_X_train
    AI_y_res = AI_ova_y_train
    Counter(AI_ova_y_train)

    AW = train[train.ACT_COMB_4 == "Accepted/Wait"]
    AW_rest = train[train.ACT_COMB_4 != "Accepted/Wait"]
    AW_rest = AW_rest.copy()
    AW_rest.iloc[:, ACT_COMB_4_index] = "Others"
    AW = AW.reset_index()
    AW_rest = AW_rest.reset_index()
    AW = AW.iloc[:, 1:ACT_COMB_4_index+2]
    AW_rest = AW_rest.iloc[:, 1:ACT_COMB_4_index+2]
    AW_ova = pd.concat([AW, AW_rest])
    AW_ova_X_train = AW_ova.iloc[:, 0:ACT_COMB_4_index]
    AW_ova_y_train = AW_ova.iloc[:, ACT_COMB_4_index]
    AW_X_res = AW_ova_X_train
    AW_y_res = AW_ova_y_train
    Counter(AW_ova_y_train)

    CC = train[train.ACT_COMB_4 == "Completed/Closed"]
    CC_rest = train[train.ACT_COMB_4 != "Completed/Closed"]
    CC_rest = CC_rest.copy()
    CC_rest.iloc[:, ACT_COMB_4_index] = "Others"
    CC = CC.reset_index()
    CC_rest = CC_rest.reset_index()
    CC = CC.iloc[:, 1:ACT_COMB_4_index+2]
    CC_rest = CC_rest.iloc[:, 1:ACT_COMB_4_index+2]
    CC_ova = pd.concat([CC, CC_rest])
    CC_ova_X_train = CC_ova.iloc[:, 0:ACT_COMB_4_index]
    CC_ova_y_train = CC_ova.iloc[:, ACT_COMB_4_index]
    CC_X_res = CC_ova_X_train
    CC_y_res = CC_ova_y_train
    Counter(CC_ova_y_train)

    QA = train[train.ACT_COMB_4 == "Queued/Awaiting Assignment"]
    QA_rest = train[train.ACT_COMB_4 != "Queued/Awaiting Assignment"]
    QA_rest = QA_rest.copy()
    QA_rest.iloc[:, ACT_COMB_4_index] = "Others"
    QA = QA.reset_index()
    QA_rest = QA_rest.reset_index()
    QA = QA.iloc[:, 1:ACT_COMB_4_index+2]
    QA_rest = QA_rest.iloc[:, 1:ACT_COMB_4_index+2]
    QA_ova = pd.concat([QA, QA_rest])
    QA_ova_X_train = QA_ova.iloc[:, 0:ACT_COMB_4_index]
    QA_ova_y_train = QA_ova.iloc[:, ACT_COMB_4_index]
    QA_X_res = QA_ova_X_train
    QA_y_res = QA_ova_y_train
    Counter(QA_ova_y_train)

    if imb_technique == "ADASYN":
        from imblearn.over_sampling import ADASYN
        print("ADASYN")

        AA_ada = ADASYN()
        AA_X_res, AA_y_res = AA_ada.fit_resample(AA_ova_X_train, AA_ova_y_train)
        AI_ada = ADASYN()
        AI_X_res, AI_y_res = AI_ada.fit_resample(AI_ova_X_train, AI_ova_y_train)
        AW_ada = ADASYN()
        AW_X_res, AW_y_res = AW_ada.fit_resample(AW_ova_X_train, AW_ova_y_train)
        CC_ada = ADASYN()
        CC_X_res, CC_y_res = CC_ada.fit_resample(CC_ova_X_train, CC_ova_y_train)
        QA_ada = ADASYN()
        QA_X_res, QA_y_res = QA_ada.fit_resample(QA_ova_X_train, QA_ova_y_train)

    if imb_technique == "ALLKNN":
        from imblearn.under_sampling import AllKNN
        print("ALLKNN")

        AA_allknn = AllKNN()
        AA_X_res, AA_y_res = AA_allknn.fit_resample(AA_ova_X_train, AA_ova_y_train)
        AI_allknn = AllKNN()
        AI_X_res, AI_y_res = AI_allknn.fit_resample(AI_ova_X_train, AI_ova_y_train)
        AW_allknn = AllKNN()
        AW_X_res, AW_y_res = AW_allknn.fit_resample(AW_ova_X_train, AW_ova_y_train)
        CC_allknn = AllKNN()
        CC_X_res, CC_y_res = CC_allknn.fit_resample(CC_ova_X_train, CC_ova_y_train)
        QA_allknn = AllKNN()
        QA_X_res, QA_y_res = QA_allknn.fit_resample(QA_ova_X_train, QA_ova_y_train)

    if imb_technique == "CNN":
        from imblearn.under_sampling import CondensedNearestNeighbour

        print("CNN")
        AA_cnn = CondensedNearestNeighbour()
        AA_X_res, AA_y_res = AA_cnn.fit_resample(AA_ova_X_train, AA_ova_y_train)
        AI_cnn = CondensedNearestNeighbour()
        AI_X_res, AI_y_res = AI_cnn.fit_resample(AI_ova_X_train, AI_ova_y_train)
        AW_cnn = CondensedNearestNeighbour()
        AW_X_res, AW_y_res = AW_cnn.fit_resample(AW_ova_X_train, AW_ova_y_train)
        CC_cnn = CondensedNearestNeighbour()
        CC_X_res, CC_y_res = CC_cnn.fit_resample(CC_ova_X_train, CC_ova_y_train)
        QA_cnn = CondensedNearestNeighbour()
        QA_X_res, QA_y_res = QA_cnn.fit_resample(QA_ova_X_train, QA_ova_y_train)

    if imb_technique == "ENN":
        from imblearn.under_sampling import EditedNearestNeighbours
        print("ENN")

        AA_enn = EditedNearestNeighbours()
        AA_X_res, AA_y_res = AA_enn.fit_resample(AA_ova_X_train, AA_ova_y_train)
        AI_enn = EditedNearestNeighbours()
        AI_X_res, AI_y_res = AI_enn.fit_resample(AI_ova_X_train, AI_ova_y_train)
        AW_enn = EditedNearestNeighbours()
        AW_X_res, AW_y_res = AW_enn.fit_resample(AW_ova_X_train, AW_ova_y_train)
        CC_enn = EditedNearestNeighbours()
        CC_X_res, CC_y_res = CC_enn.fit_resample(CC_ova_X_train, CC_ova_y_train)
        QA_enn = EditedNearestNeighbours()
        QA_X_res, QA_y_res = QA_enn.fit_resample(QA_ova_X_train, QA_ova_y_train)

    if imb_technique == "IHT":
        from imblearn.under_sampling import InstanceHardnessThreshold
        print("IHT")
        AA_iht = InstanceHardnessThreshold()
        AA_X_res, AA_y_res = AA_iht.fit_resample(AA_ova_X_train, AA_ova_y_train)
        AI_iht = InstanceHardnessThreshold()
        AI_X_res, AI_y_res = AI_iht.fit_resample(AI_ova_X_train, AI_ova_y_train)
        AW_iht = InstanceHardnessThreshold()
        AW_X_res, AW_y_res = AW_iht.fit_resample(AW_ova_X_train, AW_ova_y_train)
        CC_iht = InstanceHardnessThreshold()
        CC_X_res, CC_y_res = CC_iht.fit_resample(CC_ova_X_train, CC_ova_y_train)
        QA_iht = InstanceHardnessThreshold()
        QA_X_res, QA_y_res = QA_iht.fit_resample(QA_ova_X_train, QA_ova_y_train)

    if imb_technique == "NCR":
        from imblearn.under_sampling import NeighbourhoodCleaningRule

        print("NCR")
        AA_ncr = NeighbourhoodCleaningRule()
        AA_ova_y_train = [0 if i == "Accepted/Assigned" else 1 for i in AA_ova_y_train]
        AA_X_res, AA_y_res = AA_ncr.fit_resample(AA_ova_X_train, AA_ova_y_train)
        AI_ncr = NeighbourhoodCleaningRule()
        AI_ova_y_train = [0 if i == "Accepted/In Progress" else 1 for i in AI_ova_y_train]
        AI_X_res, AI_y_res = AI_ncr.fit_resample(AI_ova_X_train, AI_ova_y_train)
        AW_ncr = NeighbourhoodCleaningRule()
        AW_ova_y_train = [0 if i == "Accepted/Wait" else 1 for i in AW_ova_y_train]
        AW_X_res, AW_y_res = AW_ncr.fit_resample(AW_ova_X_train, AW_ova_y_train)
        CC_ncr = NeighbourhoodCleaningRule()
        CC_ova_y_train = [0 if i == "Completed/Closed" else 1 for i in CC_ova_y_train]
        CC_X_res, CC_y_res = CC_ncr.fit_resample(CC_ova_X_train, CC_ova_y_train)
        QA_ncr = NeighbourhoodCleaningRule()
        QA_ova_y_train = [0 if i == "Queued/Awaiting Assignment" else 1 for i in QA_ova_y_train]
        QA_X_res, QA_y_res = QA_ncr.fit_resample(QA_ova_X_train, QA_ova_y_train)

    if imb_technique == "NM":
        from imblearn.under_sampling import NearMiss
        print("NM")

        AA_nm = NearMiss()
        AA_X_res, AA_y_res = AA_nm.fit_resample(AA_ova_X_train, AA_ova_y_train)
        AI_nm = NearMiss()
        AI_X_res, AI_y_res = AI_nm.fit_resample(AI_ova_X_train, AI_ova_y_train)
        AW_nm = NearMiss()
        AW_X_res, AW_y_res = AW_nm.fit_resample(AW_ova_X_train, AW_ova_y_train)
        CC_nm = NearMiss()
        CC_X_res, CC_y_res = CC_nm.fit_resample(CC_ova_X_train, CC_ova_y_train)
        QA_nm = NearMiss()
        QA_X_res, QA_y_res = QA_nm.fit_resample(QA_ova_X_train, QA_ova_y_train)

    if imb_technique == "OSS":
        from imblearn.under_sampling import OneSidedSelection
        print("OSS")

        AA_oss = OneSidedSelection()
        AA_X_res, AA_y_res = AA_oss.fit_resample(AA_ova_X_train, AA_ova_y_train)
        AI_oss = OneSidedSelection()
        AI_X_res, AI_y_res = AI_oss.fit_resample(AI_ova_X_train, AI_ova_y_train)
        AW_oss = OneSidedSelection()
        AW_X_res, AW_y_res = AW_oss.fit_resample(AW_ova_X_train, AW_ova_y_train)
        CC_oss = OneSidedSelection()
        CC_X_res, CC_y_res = CC_oss.fit_resample(CC_ova_X_train, CC_ova_y_train)
        QA_oss = OneSidedSelection()
        QA_X_res, QA_y_res = QA_oss.fit_resample(QA_ova_X_train, QA_ova_y_train)

    if imb_technique == "RENN":
        from imblearn.under_sampling import RepeatedEditedNearestNeighbours
        print("RENN")

        AA_renn = RepeatedEditedNearestNeighbours()
        AA_X_res, AA_y_res = AA_renn.fit_resample(AA_ova_X_train, AA_ova_y_train)
        AI_renn = RepeatedEditedNearestNeighbours()
        AI_X_res, AI_y_res = AI_renn.fit_resample(AI_ova_X_train, AI_ova_y_train)
        AW_renn = RepeatedEditedNearestNeighbours()
        AW_X_res, AW_y_res = AW_renn.fit_resample(AW_ova_X_train, AW_ova_y_train)
        CC_renn = RepeatedEditedNearestNeighbours()
        CC_X_res, CC_y_res = CC_renn.fit_resample(CC_ova_X_train, CC_ova_y_train)
        QA_renn = RepeatedEditedNearestNeighbours()
        QA_X_res, QA_y_res = QA_renn.fit_resample(QA_ova_X_train, QA_ova_y_train)

    if imb_technique == "SMOTE":
        from imblearn.over_sampling import SMOTE
        print("SMOTE")
        print("Original dataset shape %s" % Counter(AA_ova_y_train))
        AA_sm = SMOTE()
        AA_X_res, AA_y_res = AA_sm.fit_resample(AA_ova_X_train, AA_ova_y_train)
        print("Resampled dataset shape %s" % Counter(AA_y_res))
        print("Original dataset shape %s" % Counter(AI_ova_y_train))
        AI_sm = SMOTE()
        AI_X_res, AI_y_res = AI_sm.fit_resample(AI_ova_X_train, AI_ova_y_train)
        print("Resampled dataset shape %s" % Counter(AI_y_res))
        print("Original dataset shape %s" % Counter(AW_ova_y_train))
        AW_sm = SMOTE()
        AW_X_res, AW_y_res = AW_sm.fit_resample(AW_ova_X_train, AW_ova_y_train)
        print("Resampled dataset shape %s" % Counter(AW_y_res))
        print("Original dataset shape %s" % Counter(CC_ova_y_train))
        CC_sm = SMOTE()
        CC_X_res, CC_y_res = CC_sm.fit_resample(CC_ova_X_train, CC_ova_y_train)
        print("Resampled dataset shape %s" % Counter(CC_y_res))
        print("Original dataset shape %s" % Counter(QA_ova_y_train))
        QA_sm = SMOTE()
        QA_X_res, QA_y_res = QA_sm.fit_resample(QA_ova_X_train, QA_ova_y_train)
        print("Resampled dataset shape %s" % Counter(QA_y_res))

    if imb_technique == "BSMOTE":
        from imblearn.over_sampling import BorderlineSMOTE
        print("BorderlineSMOTE")
        print("Original dataset shape %s" % Counter(AA_ova_y_train))
        AA_bsm = BorderlineSMOTE()
        AA_X_res, AA_y_res = AA_bsm.fit_resample(AA_ova_X_train, AA_ova_y_train)
        print("Resampled dataset shape %s" % Counter(AA_y_res))
        print("Original dataset shape %s" % Counter(AI_ova_y_train))
        AI_bsm = BorderlineSMOTE()
        AI_X_res, AI_y_res = AI_bsm.fit_resample(AI_ova_X_train, AI_ova_y_train)
        print("Resampled dataset shape %s" % Counter(AI_y_res))
        print("Original dataset shape %s" % Counter(AW_ova_y_train))
        AW_bsm = BorderlineSMOTE()
        AW_X_res, AW_y_res = AW_bsm.fit_resample(AW_ova_X_train, AW_ova_y_train)
        print("Resampled dataset shape %s" % Counter(AW_y_res))
        print("Original dataset shape %s" % Counter(CC_ova_y_train))
        CC_bsm = BorderlineSMOTE()
        CC_X_res, CC_y_res = CC_bsm.fit_resample(CC_ova_X_train, CC_ova_y_train)
        print("Resampled dataset shape %s" % Counter(CC_y_res))
        print("Original dataset shape %s" % Counter(QA_ova_y_train))
        QA_bsm = BorderlineSMOTE()
        QA_X_res, QA_y_res = QA_bsm.fit_resample(QA_ova_X_train, QA_ova_y_train)
        print("Resampled dataset shape %s" % Counter(QA_y_res))

    if imb_technique == "SMOTEENN":
        from imblearn.combine import SMOTEENN

        AA_smenn = SMOTEENN()
        AA_X_res, AA_y_res = AA_smenn.fit_resample(AA_ova_X_train, AA_ova_y_train)
        AI_smenn = SMOTEENN()
        AI_X_res, AI_y_res = AI_smenn.fit_resample(AI_ova_X_train, AI_ova_y_train)
        AW_smenn = SMOTEENN()
        AW_X_res, AW_y_res = AW_smenn.fit_resample(AW_ova_X_train, AW_ova_y_train)
        CC_smenn = SMOTEENN()
        CC_X_res, CC_y_res = CC_smenn.fit_resample(CC_ova_X_train, CC_ova_y_train)
        QA_smenn = SMOTEENN()
        QA_X_res, QA_y_res = QA_smenn.fit_resample(QA_ova_X_train, QA_ova_y_train)

    if imb_technique == "SMOTETOMEK":
        from imblearn.combine import SMOTETomek

        AA_smtm = SMOTETomek()
        AA_X_res, AA_y_res = AA_smtm.fit_resample(AA_ova_X_train, AA_ova_y_train)
        AI_smtm = SMOTETomek()
        AI_X_res, AI_y_res = AI_smtm.fit_resample(AI_ova_X_train, AI_ova_y_train)
        AW_smtm = SMOTETomek()
        AW_X_res, AW_y_res = AW_smtm.fit_resample(AW_ova_X_train, AW_ova_y_train)
        CC_smtm = SMOTETomek()
        CC_X_res, CC_y_res = CC_smtm.fit_resample(CC_ova_X_train, CC_ova_y_train)
        QA_smtm = SMOTETomek()
        QA_X_res, QA_y_res = QA_smtm.fit_resample(QA_ova_X_train, QA_ova_y_train)

    if imb_technique == "TOMEK":
        from imblearn.under_sampling import TomekLinks

        AA_tm = TomekLinks()
        AA_X_res, AA_y_res = AA_tm.fit_resample(AA_ova_X_train, AA_ova_y_train)
        AI_tm = TomekLinks()
        AI_X_res, AI_y_res = AI_tm.fit_resample(AI_ova_X_train, AI_ova_y_train)
        AW_tm = TomekLinks()
        AW_X_res, AW_y_res = AW_tm.fit_resample(AW_ova_X_train, AW_ova_y_train)
        CC_tm = TomekLinks()
        CC_X_res, CC_y_res = CC_tm.fit_resample(CC_ova_X_train, CC_ova_y_train)
        QA_tm = TomekLinks()
        QA_X_res, QA_y_res = QA_tm.fit_resample(QA_ova_X_train, QA_ova_y_train)

    if imb_technique == "ROS":
        from imblearn.over_sampling import RandomOverSampler

        AA_ros = RandomOverSampler()
        AA_X_res, AA_y_res = AA_ros.fit_resample(AA_ova_X_train, AA_ova_y_train)
        AI_ros = RandomOverSampler()
        AI_X_res, AI_y_res = AI_ros.fit_resample(AI_ova_X_train, AI_ova_y_train)
        AW_ros = RandomOverSampler()
        AW_X_res, AW_y_res = AW_ros.fit_resample(AW_ova_X_train, AW_ova_y_train)
        CC_ros = RandomOverSampler()
        CC_X_res, CC_y_res = CC_ros.fit_resample(CC_ova_X_train, CC_ova_y_train)
        QA_ros = RandomOverSampler()
        QA_X_res, QA_y_res = QA_ros.fit_resample(QA_ova_X_train, QA_ova_y_train)

    if imb_technique == "RUS":
        from imblearn.under_sampling import RandomUnderSampler

        AA_rus = RandomUnderSampler()
        AA_X_res, AA_y_res = AA_rus.fit_resample(AA_ova_X_train, AA_ova_y_train)
        AI_rus = RandomUnderSampler()
        AI_X_res, AI_y_res = AI_rus.fit_resample(AI_ova_X_train, AI_ova_y_train)
        AW_rus = RandomUnderSampler()
        AW_X_res, AW_y_res = AW_rus.fit_resample(AW_ova_X_train, AW_ova_y_train)
        CC_rus = RandomUnderSampler()
        CC_X_res, CC_y_res = CC_rus.fit_resample(CC_ova_X_train, CC_ova_y_train)
        QA_rus = RandomUnderSampler()
        QA_X_res, QA_y_res = QA_rus.fit_resample(QA_ova_X_train, QA_ova_y_train)

    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import RandomizedSearchCV
    import itertools

    first_digit_parameters = [x for x in itertools.product((5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100), repeat=1)]
    all_digit_parameters = first_digit_parameters
    learning_rate_init_parameters = [0.1, 0.01, 0.001]
    parameters = {'hidden_layer_sizes': all_digit_parameters,
                  'learning_rate_init': learning_rate_init_parameters}
    dnn_AA = MLPClassifier(max_iter=10000, activation='relu')
    dnn_AA_clf = RandomizedSearchCV(dnn_AA, parameters, n_jobs=-1, cv=5)
    dnn_AA_clf.fit(AA_X_res, AA_y_res)
    print(dnn_AA_clf.best_params_)
    dnn_AI = MLPClassifier(max_iter=10000, activation='relu')
    dnn_AI_clf = RandomizedSearchCV(dnn_AI, parameters, n_jobs=-1, cv=5)
    dnn_AI_clf.fit(AI_X_res, AI_y_res)
    print(dnn_AI_clf.best_params_)
    dnn_AW = MLPClassifier(max_iter=10000, activation='relu')
    dnn_AW_clf = RandomizedSearchCV(dnn_AW, parameters, n_jobs=-1, cv=5)
    dnn_AW_clf.fit(AW_X_res, AW_y_res)
    print(dnn_AW_clf.best_params_)
    dnn_CC = MLPClassifier(max_iter=10000, activation='relu')
    dnn_CC_clf = RandomizedSearchCV(dnn_CC, parameters, n_jobs=-1, cv=5)
    dnn_CC_clf.fit(CC_X_res, CC_y_res)
    print(dnn_CC_clf.best_params_)
    dnn_QA = MLPClassifier(max_iter=10000, activation='relu')
    dnn_QA_clf = RandomizedSearchCV(dnn_QA, parameters, n_jobs=-1, cv=5)
    dnn_QA_clf.fit(QA_X_res, QA_y_res)
    print(dnn_QA_clf.best_params_)

    from sklearn.linear_model import LogisticRegression
    solver = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
    tol = [1e-2, 1e-3, 1e-4, 1e-5]
    reg_strength = [0.5, 1.0, 1.5]
    parameters = {'solver': solver,
	          'tol': tol,
	          'C': reg_strength}
    lr_AA = LogisticRegression()
    lr_AA_clf = RandomizedSearchCV(lr_AA, parameters, n_jobs = -1, cv = 5)
    lr_AA_clf.fit(AA_X_res, AA_y_res)
    print(lr_AA_clf.best_params_)
    lr_AI = LogisticRegression()
    lr_AI_clf = RandomizedSearchCV(lr_AI, parameters, n_jobs = -1, cv = 5)
    lr_AI_clf.fit(AI_X_res, AI_y_res)
    print(lr_AI_clf.best_params_)
    lr_AW = LogisticRegression()
    lr_AW_clf = RandomizedSearchCV(lr_AW, parameters, n_jobs = -1, cv = 5)
    lr_AW_clf.fit(AW_X_res, AW_y_res)
    print(lr_AW_clf.best_params_)
    lr_CC = LogisticRegression()
    lr_CC_clf = RandomizedSearchCV(lr_CC, parameters, n_jobs = -1, cv = 5)
    lr_CC_clf.fit(CC_X_res, CC_y_res)
    print(lr_CC_clf.best_params_)
    lr_QA = LogisticRegression()
    lr_QA_clf = RandomizedSearchCV(lr_QA, parameters, n_jobs = -1, cv = 5)
    lr_QA_clf.fit(QA_X_res, QA_y_res)
    print(lr_QA_clf.best_params_)

    # Below codes are for the implementation of Gaussian Naive Bayes training
    from sklearn.naive_bayes import GaussianNB
    #In Gaussian NB, 'var_smoothing' parameter optimization makes convergence errors
    nb_AA_clf = GaussianNB()
    nb_AA_clf.fit(AA_X_res, AA_y_res)
    nb_AI_clf = GaussianNB()
    nb_AI_clf.fit(AI_X_res, AI_y_res)
    nb_AW_clf = GaussianNB()
    nb_AW_clf.fit(AW_X_res, AW_y_res)
    nb_CC_clf = GaussianNB()
    nb_CC_clf.fit(CC_X_res, CC_y_res)
    nb_QA_clf = GaussianNB()
    nb_QA_clf.fit(QA_X_res, QA_y_res)

    # Below codes are for the implementation of random forest training
    from sklearn.ensemble import RandomForestClassifier
    n_tree = [50, 100, 200, 300, 400, 500, 600, 700]
    max_depth = [10, 20, 30, 40, 50, 60, 70]
    min_samples_split = [5, 10, 15, 20, 25, 30]
    parameters = {'n_estimators': n_tree,
		  'max_depth': max_depth,
		  'min_samples_split': min_samples_split}
    rf_AA = RandomForestClassifier()
    rf_AA_clf = RandomizedSearchCV(rf_AA, parameters, n_jobs = -1, cv = 5)
    rf_AA_clf.fit(AA_X_res, AA_y_res)
    print(rf_AA_clf.best_params_)
    rf_AI = RandomForestClassifier()
    rf_AI_clf = RandomizedSearchCV(rf_AI, parameters, n_jobs = -1, cv = 5)
    rf_AI_clf.fit(AI_X_res, AI_y_res)
    print(rf_AI_clf.best_params_)
    rf_AW = RandomForestClassifier()
    rf_AW_clf = RandomizedSearchCV(rf_AW, parameters, n_jobs = -1, cv=5)
    rf_AW_clf.fit(AW_X_res, AW_y_res)
    print(rf_AW_clf.best_params_)
    rf_CC = RandomForestClassifier()
    rf_CC_clf = RandomizedSearchCV(rf_CC, parameters, n_jobs = -1, cv=5)
    rf_CC_clf.fit(CC_X_res, CC_y_res)
    print(rf_CC_clf.best_params_)
    rf_QA = RandomForestClassifier()
    rf_QA_clf = RandomizedSearchCV(rf_QA, parameters, n_jobs = -1, cv=5)
    rf_QA_clf.fit(QA_X_res, QA_y_res)
    print(rf_QA_clf.best_params_)

    # Below codes are for the implementation of support vector machine training
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV
    #reg_param = [0.5, 1.0, 1.5]
    #degree = [1, 2, 3, 4, 5]
    #kernel = ['rbf', 'linear', 'poly', 'sigmoid']
    #gamma = ['scale', 'auto']
    #tol = [1e-2, 1e-3, 1e-4]
    svm_AA = LinearSVC()
    svm_AA_clf = CalibratedClassifierCV(svm_AA, cv = 5)
    #svm_AA_clf = RandomizedSearchCV(svm_AA_clf, parameters, n_jobs = -1, cv = 5)
    svm_AA_clf.fit(AA_X_res, AA_y_res)
    #print(svm_AA_clf.best_params_)
    svm_AI = LinearSVC()
    svm_AI_clf = CalibratedClassifierCV(svm_AI, cv = 5)
    #svm_AI_clf = RandomizedSearchCV(svm_AI_clf, parameters, n_jobs = -1, cv = 5)
    svm_AI_clf.fit(AI_X_res, AI_y_res)
    #print(svm_AI_clf.best_params_)
    svm_AW = LinearSVC()
    svm_AW_clf = CalibratedClassifierCV(svm_AW, cv = 5)
    #svm_AW_clf = RandomizedSearchCV(svm_AW_clf, parameters, n_jobs = -1, cv = 5)
    svm_AW_clf.fit(AW_X_res, AW_y_res)
    #print(svm_AW_clf.best_params_)
    svm_CC = LinearSVC()
    svm_CC_clf = CalibratedClassifierCV(svm_CC, cv = 5)
    #svm_CC_clf = RandomizedSearchCV(svm_CC_clf, parameters, n_jobs = -1, cv = 5)
    svm_CC_clf.fit(CC_X_res, CC_y_res)
    #print(svm_CC_clf.best_params_)
    svm_QA = LinearSVC()
    svm_QA_clf = CalibratedClassifierCV(svm_QA, cv = 5)
    #svm_QA_clf = RandomizedSearchCV(svm_QA_clf, parameters, n_jobs = -1, cv = 5)
    svm_QA_clf.fit(QA_X_res, QA_y_res)
    #print(svm_QA_clf.best_params_)

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

    dnn_prediction = pd.DataFrame(columns=['Prediction'])
    lr_prediction = pd.DataFrame(columns=['Prediction'])
    nb_prediction = pd.DataFrame(columns=['Prediction'])
    rf_prediction = pd.DataFrame(columns=['Prediction'])
    svm_prediction = pd.DataFrame(columns=['Prediction'])

    for i in range(0, len(y_test)):
        dnn_AA_index = 0
        dnn_AI_index = 0
        dnn_AW_index = 0
        dnn_CC_index = 0
        dnn_QA_index = 0
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


    def get_precision(dnn_conf_matrix):
        dnn_tp_1 = dnn_conf_matrix[0][0]
        dnn_tp_2 = dnn_conf_matrix[1][1]
        dnn_tp_3 = dnn_conf_matrix[2][2]
        dnn_tp_4 = dnn_conf_matrix[3][3]
        dnn_tp_5 = dnn_conf_matrix[4][4]
        dnn_fp_1 = dnn_conf_matrix[1][0] + dnn_conf_matrix[2][0] + dnn_conf_matrix[3][0] + dnn_conf_matrix[4][0]
        dnn_fp_2 = dnn_conf_matrix[0][1] + dnn_conf_matrix[2][1] + dnn_conf_matrix[3][1] + dnn_conf_matrix[4][1]
        dnn_fp_3 = dnn_conf_matrix[0][2] + dnn_conf_matrix[1][2] + dnn_conf_matrix[3][2] + dnn_conf_matrix[4][2]
        dnn_fp_4 = dnn_conf_matrix[0][3] + dnn_conf_matrix[1][3] + dnn_conf_matrix[2][3] + dnn_conf_matrix[4][3]
        dnn_fp_5 = dnn_conf_matrix[0][4] + dnn_conf_matrix[1][4] + dnn_conf_matrix[2][4] + dnn_conf_matrix[3][4]

        if dnn_tp_1 + dnn_fp_1 == 0:
            dnn_precision_1 = 0
        else:
            dnn_precision_1 = dnn_tp_1 / (dnn_tp_1 + dnn_fp_1)
        if dnn_tp_2 + dnn_fp_2 == 0:
            dnn_precision_2 = 0
        else:
            dnn_precision_2 = dnn_tp_2 / (dnn_tp_2 + dnn_fp_2)
        if dnn_tp_3 + dnn_fp_3 == 0:
            dnn_precision_3 = 0
        else:
            dnn_precision_3 = dnn_tp_3 / (dnn_tp_3 + dnn_fp_3)
        if dnn_tp_4 + dnn_fp_4 == 0:
            dnn_precision_4 = 0
        else:
            dnn_precision_4 = dnn_tp_4 / (dnn_tp_4 + dnn_fp_4)
        if dnn_tp_5 + dnn_fp_5 == 0:
            dnn_precision_5 = 0
        else:
            dnn_precision_5 = dnn_tp_5 / (dnn_tp_5 + dnn_fp_5)
        dnn_precision_avg = (dnn_precision_1 + dnn_precision_2 + dnn_precision_3 + dnn_precision_4 + dnn_precision_5) / 5
        print(dnn_precision_3)
        return dnn_precision_avg


    def get_recall_pen_1(dnn_conf_matrix):
        dnn_tp_1 = dnn_conf_matrix[0][0]
        dnn_tp_2 = dnn_conf_matrix[1][1]
        dnn_tp_3 = dnn_conf_matrix[2][2]
        dnn_tp_4 = dnn_conf_matrix[3][3]
        dnn_tp_5 = dnn_conf_matrix[4][4]
        dnn_fn_1 = dnn_conf_matrix[0][1] + dnn_conf_matrix[0][2] + dnn_conf_matrix[0][3] + dnn_conf_matrix[0][4]
        dnn_fn_2 = dnn_conf_matrix[1][0] + dnn_conf_matrix[1][2] + dnn_conf_matrix[1][3] + dnn_conf_matrix[1][4]
        dnn_fn_3 = dnn_conf_matrix[2][0] + dnn_conf_matrix[2][1] + dnn_conf_matrix[2][3] + dnn_conf_matrix[2][4]
        dnn_fn_4 = dnn_conf_matrix[3][0] + dnn_conf_matrix[3][1] + dnn_conf_matrix[3][2] + dnn_conf_matrix[3][4]
        dnn_fn_5 = dnn_conf_matrix[4][0] + dnn_conf_matrix[4][1] + dnn_conf_matrix[4][2] + dnn_conf_matrix[4][3]
        if dnn_tp_1 + dnn_fn_1 == 0:
            dnn_recall_1 = 0
        else:
            dnn_recall_1 = dnn_tp_1 / (dnn_tp_1 + dnn_fn_1)
        if dnn_tp_2 + dnn_fn_2 == 0:
            dnn_recall_2 = 0
        else:
            dnn_recall_2 = dnn_tp_2 / (dnn_tp_2 + dnn_fn_2)
        if dnn_tp_3 + dnn_fn_3 == 0:
            dnn_recall_3 = 0
        else:
            dnn_recall_3 = dnn_tp_3 / (dnn_tp_3 + dnn_fn_3)
        if dnn_tp_4 + dnn_fn_4 == 0:
            dnn_recall_4 = 0
        else:
            dnn_recall_4 = dnn_tp_4 / (dnn_tp_4 + dnn_fn_4)
        if dnn_tp_5 + dnn_fn_5 == 0:
            dnn_recall_5 = 0
        else:
            dnn_recall_5 = dnn_tp_5 / (dnn_tp_5 +dnn_fn_5)
        dnn_recall_avg_pen_1 = (dnn_recall_1 + dnn_recall_2 + dnn_recall_3 +dnn_recall_4 + dnn_recall_5) / (5+1-1)
        return dnn_recall_avg_pen_1

    def get_recall_pen_5(dnn_conf_matrix):
        dnn_tp_1 = dnn_conf_matrix[0][0]
        dnn_tp_2 = dnn_conf_matrix[1][1]
        dnn_tp_3 = dnn_conf_matrix[2][2]
        dnn_tp_4 = dnn_conf_matrix[3][3]
        dnn_tp_5 = dnn_conf_matrix[4][4]
        dnn_fn_1 = dnn_conf_matrix[0][1] + dnn_conf_matrix[0][2] + dnn_conf_matrix[0][3] + dnn_conf_matrix[0][4]
        dnn_fn_2 = dnn_conf_matrix[1][0] + dnn_conf_matrix[1][2] + dnn_conf_matrix[1][3] + dnn_conf_matrix[1][4]
        dnn_fn_3 = dnn_conf_matrix[2][0] + dnn_conf_matrix[2][1] + dnn_conf_matrix[2][3] + dnn_conf_matrix[2][4]
        dnn_fn_4 = dnn_conf_matrix[3][0] + dnn_conf_matrix[3][1] + dnn_conf_matrix[3][2] + dnn_conf_matrix[3][4]
        dnn_fn_5 = dnn_conf_matrix[4][0] + dnn_conf_matrix[4][1] + dnn_conf_matrix[4][2] + dnn_conf_matrix[4][3]
        if dnn_tp_1 + dnn_fn_1 == 0:
            dnn_recall_1 = 0
        else:
            dnn_recall_1 = dnn_tp_1 / (dnn_tp_1 + dnn_fn_1)
        if dnn_tp_2 + dnn_fn_2 == 0:
            dnn_recall_2 = 0
        else:
            dnn_recall_2 = dnn_tp_2 / (dnn_tp_2 + dnn_fn_2)
        if dnn_tp_3 + dnn_fn_3 == 0:
            dnn_recall_3 = 0
        else:
            dnn_recall_3 = dnn_tp_3 / (dnn_tp_3 + dnn_fn_3)
        if dnn_tp_4 + dnn_fn_4 == 0:
            dnn_recall_4 = 0
        else:
            dnn_recall_4 = dnn_tp_4 / (dnn_tp_4 + dnn_fn_4)
        if dnn_tp_5 + dnn_fn_5 == 0:
            dnn_recall_5 = 0
        else:
            dnn_recall_5 = dnn_tp_5 / (dnn_tp_5 +dnn_fn_5)
        dnn_recall_avg_pen_5 = (dnn_recall_1 + dnn_recall_2 + (5*dnn_recall_3) +dnn_recall_4 + dnn_recall_5) / (5+5-1)
        return dnn_recall_avg_pen_5


    from sklearn.metrics import classification_report, confusion_matrix

    dnn_conf_matrix = confusion_matrix(y_test, dnn_prediction)

    ### PENALTY OPERATION ###
    # Penalty for first class(Prediction of first class is important here)
    #dnn_conf_matrix[0] = penalty * dnn_conf_matrix[0]
    #dnn_conf_matrix[0][0] = dnn_conf_matrix[0][0] / penalty
    # Penalty for second class(Prediction of first class is important here)
    #dnn_conf_matrix[1] = penalty * dnn_conf_matrix[1]
    #dnn_conf_matrix[1][1] = dnn_conf_matrix[1][1] / penalty
    # Penalty for third class(Prediction of first class is important here)
    #dnn_conf_matrix[2] = penalty * dnn_conf_matrix[2]
    #dnn_conf_matrix[2][2] = dnn_conf_matrix[2][2] / penalty
    # Penalty for fourth class(Prediction of first class is important here)
    #dnn_conf_matrix[3] = penalty * dnn_conf_matrix[3]
    #dnn_conf_matrix[3][3] = dnn_conf_matrix[3][3] / penalty
    # Penalty for fifth class(Prediction of first class is important here)
    #dnn_conf_matrix[4] = penalty * dnn_conf_matrix[4]
    #dnn_conf_matrix[4][4] = dnn_conf_matrix[4][4] / penalty

    print("dnn_confusion matrix:")
    print(dnn_conf_matrix)
    dnn_precision = get_precision(dnn_conf_matrix)
    dnn_recall_pen_1 = get_recall_pen_1(dnn_conf_matrix)
    dnn_recall_pen_5 = get_recall_pen_5(dnn_conf_matrix)
    dnn_f1_score_pen_1 = 2 * (dnn_precision * dnn_recall_pen_1) / (dnn_precision + dnn_recall_pen_1)
    dnn_f1_score_pen_5 = 2 * (dnn_precision * dnn_recall_pen_5) / (dnn_precision + dnn_recall_pen_5)
    dnn_ovr_accuracy = (dnn_conf_matrix[0][0] + dnn_conf_matrix[1][1] + dnn_conf_matrix[2][2] + dnn_conf_matrix[3][3] + dnn_conf_matrix[4][4]) / (
                sum(dnn_conf_matrix[0]) + sum(dnn_conf_matrix[1]) + sum(dnn_conf_matrix[2]) + sum(dnn_conf_matrix[3]) + sum(dnn_conf_matrix[4]))
    print("dnn_f1 score of pen 1 is:")
    print(dnn_f1_score_pen_1)
    print("dnn_f1 score of pen 5 is:")
    print(dnn_f1_score_pen_5)
    print("dnn_overall accuracy is:")
    print(dnn_ovr_accuracy)
    dnn_conf_matrix = pd.DataFrame(dnn_conf_matrix)
    dnn_conf_matrix.to_csv('conf_matrix_'+imb_technique+'_dnn_bpic2013_closed_'+ str(nsplits) +'foldcv_' + str(repeat+1)+'.csv',header=False,index=False) #First repetition
    #dnn_conf_matrix.to_csv('conf_matrix_'+imb_technique+'_penalty_' + str(penalty) + '_dnn_bpic2013_closed_'+ str(nsplits) +'foldcv_' + str(repeat+6)+'.csv',header=False,index=False) #Second repetition
    dnn_f1_score_pen_1_kfoldcv[repeat] = dnn_f1_score_pen_1
    dnn_f1_score_pen_5_kfoldcv[repeat] = dnn_f1_score_pen_5
    dnn_ovr_accuracy_kfoldcv[repeat] = dnn_ovr_accuracy

    for i in range(0, len(y_test)):
        lr_AA_index = 0
        lr_AI_index = 0
        lr_AW_index = 0
        lr_CC_index = 0
        lr_QA_index = 0
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


    def get_precision(lr_conf_matrix):
        lr_tp_1 = lr_conf_matrix[0][0]
        lr_tp_2 = lr_conf_matrix[1][1]
        lr_tp_3 = lr_conf_matrix[2][2]
        lr_tp_4 = lr_conf_matrix[3][3]
        lr_tp_5 = lr_conf_matrix[4][4]
        lr_fp_1 = lr_conf_matrix[1][0] + lr_conf_matrix[2][0] + lr_conf_matrix[3][0] + lr_conf_matrix[4][0]
        lr_fp_2 = lr_conf_matrix[0][1] + lr_conf_matrix[2][1] + lr_conf_matrix[3][1] + lr_conf_matrix[4][1]
        lr_fp_3 = lr_conf_matrix[0][2] + lr_conf_matrix[1][2] + lr_conf_matrix[3][2] + lr_conf_matrix[4][2]
        lr_fp_4 = lr_conf_matrix[0][3] + lr_conf_matrix[1][3] + lr_conf_matrix[2][3] + lr_conf_matrix[4][3]
        lr_fp_5 = lr_conf_matrix[0][4] + lr_conf_matrix[1][4] + lr_conf_matrix[2][4] + lr_conf_matrix[3][4]

        if lr_tp_1 + lr_fp_1 == 0:
            lr_precision_1 = 0
        else:
            lr_precision_1 = lr_tp_1 / (lr_tp_1 + lr_fp_1)
        if lr_tp_2 + lr_fp_2 == 0:
            lr_precision_2 = 0
        else:
            lr_precision_2 = lr_tp_2 / (lr_tp_2 + lr_fp_2)
        if lr_tp_3 + lr_fp_3 == 0:
            lr_precision_3 = 0
        else:
            lr_precision_3 = lr_tp_3 / (lr_tp_3 + lr_fp_3)
        if lr_tp_4 + lr_fp_4 == 0:
            lr_precision_4 = 0
        else:
            lr_precision_4 = lr_tp_4 / (lr_tp_4 + lr_fp_4)
        if lr_tp_5 + lr_fp_5 == 0:
            lr_precision_5 = 0
        else:
            lr_precision_5 = lr_tp_5 / (lr_tp_5 + lr_fp_5)
        lr_precision_avg = (lr_precision_1 + lr_precision_2 + lr_precision_3 + lr_precision_4 + lr_precision_5) / 5
        return lr_precision_avg

    def get_recall_pen_1(lr_conf_matrix):
        lr_tp_1 = lr_conf_matrix[0][0]
        lr_tp_2 = lr_conf_matrix[1][1]
        lr_tp_3 = lr_conf_matrix[2][2]
        lr_tp_4 = lr_conf_matrix[3][3]
        lr_tp_5 = lr_conf_matrix[4][4]
        lr_fn_1 = lr_conf_matrix[0][1] + lr_conf_matrix[0][2] + lr_conf_matrix[0][3] + lr_conf_matrix[0][4]
        lr_fn_2 = lr_conf_matrix[1][0] + lr_conf_matrix[1][2] + lr_conf_matrix[1][3] + lr_conf_matrix[1][4]
        lr_fn_3 = lr_conf_matrix[2][0] + lr_conf_matrix[2][1] + lr_conf_matrix[2][3] + lr_conf_matrix[2][4]
        lr_fn_4 = lr_conf_matrix[3][0] + lr_conf_matrix[3][1] + lr_conf_matrix[3][2] + lr_conf_matrix[3][4]
        lr_fn_5 = lr_conf_matrix[4][0] + lr_conf_matrix[4][1] + lr_conf_matrix[4][2] + lr_conf_matrix[4][3]
        if lr_tp_1 + lr_fn_1 == 0:
            lr_recall_1 = 0
        else:
            lr_recall_1 = lr_tp_1 / (lr_tp_1 + lr_fn_1)
        if lr_tp_2 + lr_fn_2 == 0:
            lr_recall_2 = 0
        else:
            lr_recall_2 = lr_tp_2 / (lr_tp_2 + lr_fn_2)
        if lr_tp_3 + lr_fn_3 == 0:
            lr_recall_3 = 0
        else:
            lr_recall_3 = lr_tp_3 / (lr_tp_3 + lr_fn_3)
        if lr_tp_4 + lr_fn_4 == 0:
            lr_recall_4 = 0
        else:
            lr_recall_4 = lr_tp_4 / (lr_tp_4 + lr_fn_4)
        if lr_tp_5 + lr_fn_5 == 0:
            lr_recall_5 = 0
        else:
            lr_recall_5 = lr_tp_5 / (lr_tp_5 +lr_fn_5)
        lr_recall_avg_pen_1 = (lr_recall_1 + lr_recall_2 + lr_recall_3 +lr_recall_4 + lr_recall_5) / (5+1-1)
        return lr_recall_avg_pen_1

    def get_recall_pen_5(lr_conf_matrix):
        lr_tp_1 = lr_conf_matrix[0][0]
        lr_tp_2 = lr_conf_matrix[1][1]
        lr_tp_3 = lr_conf_matrix[2][2]
        lr_tp_4 = lr_conf_matrix[3][3]
        lr_tp_5 = lr_conf_matrix[4][4]
        lr_fn_1 = lr_conf_matrix[0][1] + lr_conf_matrix[0][2] + lr_conf_matrix[0][3] + lr_conf_matrix[0][4]
        lr_fn_2 = lr_conf_matrix[1][0] + lr_conf_matrix[1][2] + lr_conf_matrix[1][3] + lr_conf_matrix[1][4]
        lr_fn_3 = lr_conf_matrix[2][0] + lr_conf_matrix[2][1] + lr_conf_matrix[2][3] + lr_conf_matrix[2][4]
        lr_fn_4 = lr_conf_matrix[3][0] + lr_conf_matrix[3][1] + lr_conf_matrix[3][2] + lr_conf_matrix[3][4]
        lr_fn_5 = lr_conf_matrix[4][0] + lr_conf_matrix[4][1] + lr_conf_matrix[4][2] + lr_conf_matrix[4][3]
        if lr_tp_1 + lr_fn_1 == 0:
            lr_recall_1 = 0
        else:
            lr_recall_1 = lr_tp_1 / (lr_tp_1 + lr_fn_1)
        if lr_tp_2 + lr_fn_2 == 0:
            lr_recall_2 = 0
        else:
            lr_recall_2 = lr_tp_2 / (lr_tp_2 + lr_fn_2)
        if lr_tp_3 + lr_fn_3 == 0:
            lr_recall_3 = 0
        else:
            lr_recall_3 = lr_tp_3 / (lr_tp_3 + lr_fn_3)
        if lr_tp_4 + lr_fn_4 == 0:
            lr_recall_4 = 0
        else:
            lr_recall_4 = lr_tp_4 / (lr_tp_4 + lr_fn_4)
        if lr_tp_5 + lr_fn_5 == 0:
            lr_recall_5 = 0
        else:
            lr_recall_5 = lr_tp_5 / (lr_tp_5 +lr_fn_5)
        lr_recall_avg_pen_5 = (lr_recall_1 + lr_recall_2 + (5*lr_recall_3) +lr_recall_4 + lr_recall_5) / (5+5-1)
        return lr_recall_avg_pen_5


    from sklearn.metrics import classification_report, confusion_matrix

    lr_conf_matrix = confusion_matrix(y_test, lr_prediction)

    ### PENALTY OPERATION ###
    # Penalty for first class(Prediction of first class is important here)
    #lr_conf_matrix[0] = penalty * lr_conf_matrix[0]
    #lr_conf_matrix[0][0] = lr_conf_matrix[0][0] / penalty
    # Penalty for second class(Prediction of first class is important here)
    #lr_conf_matrix[1] = penalty * lr_conf_matrix[1]
    #lr_conf_matrix[1][1] = lr_conf_matrix[1][1] / penalty
    # Penalty for third class(Prediction of first class is important here)
    #lr_conf_matrix[2] = penalty * lr_conf_matrix[2]
    #lr_conf_matrix[2][2] = lr_conf_matrix[2][2] / penalty
    # Penalty for fourth class(Prediction of first class is important here)
    #lr_conf_matrix[3] = penalty * lr_conf_matrix[3]
    #lr_conf_matrix[3][3] = lr_conf_matrix[3][3] / penalty
    # Penalty for fifth class(Prediction of first class is important here)
    #lr_conf_matrix[4] = penalty * lr_conf_matrix[4]
    #lr_conf_matrix[4][4] = lr_conf_matrix[4][4] / penalty

    print("lr_confusion matrix:")
    print(lr_conf_matrix)
    lr_precision = get_precision(lr_conf_matrix)
    lr_recall_pen_1 = get_recall_pen_1(lr_conf_matrix)
    lr_recall_pen_5 = get_recall_pen_5(lr_conf_matrix)
    lr_f1_score_pen_1 = 2 * (lr_precision * lr_recall_pen_1) / (lr_precision + lr_recall_pen_1)
    lr_f1_score_pen_5 = 2 * (lr_precision * lr_recall_pen_5) / (lr_precision + lr_recall_pen_5)
    lr_ovr_accuracy = (lr_conf_matrix[0][0] + lr_conf_matrix[1][1] + lr_conf_matrix[2][2] + lr_conf_matrix[3][3] + lr_conf_matrix[4][4]) / (
                sum(lr_conf_matrix[0]) + sum(lr_conf_matrix[1]) + sum(lr_conf_matrix[2]) + sum(lr_conf_matrix[3]) + sum(lr_conf_matrix[4]))
    print("lr_f1 score of pen 1 is:")
    print(lr_f1_score_pen_1)
    print("lr_f1 score of pen 5 is:")
    print(lr_f1_score_pen_5)
    print("lr_overall accuracy is:")
    print(lr_ovr_accuracy)
    lr_conf_matrix = pd.DataFrame(lr_conf_matrix)
    lr_conf_matrix.to_csv('conf_matrix_'+imb_technique+'_lr_bpic2013_closed_'+ str(nsplits) +'foldcv_' + str(repeat+1)+'.csv',header=False,index=False) #First repetition
    #lr_conf_matrix.to_csv('conf_matrix_'+imb_technique+'_penalty_' + str(penalty) + '_lr_bpic2013_closed_'+ str(nsplits) +'foldcv_' + str(repeat+6)+'.csv',header=False,index=False) #Second repetition
    lr_f1_score_pen_1_kfoldcv[repeat] = lr_f1_score_pen_1
    lr_f1_score_pen_5_kfoldcv[repeat] = lr_f1_score_pen_5
    lr_ovr_accuracy_kfoldcv[repeat] = lr_ovr_accuracy



    for i in range(0, len(y_test)):
        nb_AA_index = 0
        nb_AI_index = 0
        nb_AW_index = 0
        nb_CC_index = 0
        nb_QA_index = 0
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


    def get_precision(nb_conf_matrix):
        nb_tp_1 = nb_conf_matrix[0][0]
        nb_tp_2 = nb_conf_matrix[1][1]
        nb_tp_3 = nb_conf_matrix[2][2]
        nb_tp_4 = nb_conf_matrix[3][3]
        nb_tp_5 = nb_conf_matrix[4][4]
        nb_fp_1 = nb_conf_matrix[1][0] + nb_conf_matrix[2][0] + nb_conf_matrix[3][0] + nb_conf_matrix[4][0]
        nb_fp_2 = nb_conf_matrix[0][1] + nb_conf_matrix[2][1] + nb_conf_matrix[3][1] + nb_conf_matrix[4][1]
        nb_fp_3 = nb_conf_matrix[0][2] + nb_conf_matrix[1][2] + nb_conf_matrix[3][2] + nb_conf_matrix[4][2]
        nb_fp_4 = nb_conf_matrix[0][3] + nb_conf_matrix[1][3] + nb_conf_matrix[2][3] + nb_conf_matrix[4][3]
        nb_fp_5 = nb_conf_matrix[0][4] + nb_conf_matrix[1][4] + nb_conf_matrix[2][4] + nb_conf_matrix[3][4]

        if nb_tp_1 + nb_fp_1 == 0:
            nb_precision_1 = 0
        else:
            nb_precision_1 = nb_tp_1 / (nb_tp_1 + nb_fp_1)
        if nb_tp_2 + nb_fp_2 == 0:
            nb_precision_2 = 0
        else:
            nb_precision_2 = nb_tp_2 / (nb_tp_2 + nb_fp_2)
        if nb_tp_3 + nb_fp_3 == 0:
            nb_precision_3 = 0
        else:
            nb_precision_3 = nb_tp_3 / (nb_tp_3 + nb_fp_3)
        if nb_tp_4 + nb_fp_4 == 0:
            nb_precision_4 = 0
        else:
            nb_precision_4 = nb_tp_4 / (nb_tp_4 + nb_fp_4)
        if nb_tp_5 + nb_fp_5 == 0:
            nb_precision_5 = 0
        else:
            nb_precision_5 = nb_tp_5 / (nb_tp_5 + nb_fp_5)
        nb_precision_avg = (nb_precision_1 + nb_precision_2 + nb_precision_3 + nb_precision_4 + nb_precision_5) / 5
        return nb_precision_avg

    def get_recall_pen_1(nb_conf_matrix):
        nb_tp_1 = nb_conf_matrix[0][0]
        nb_tp_2 = nb_conf_matrix[1][1]
        nb_tp_3 = nb_conf_matrix[2][2]
        nb_tp_4 = nb_conf_matrix[3][3]
        nb_tp_5 = nb_conf_matrix[4][4]
        nb_fn_1 = nb_conf_matrix[0][1] + nb_conf_matrix[0][2] + nb_conf_matrix[0][3] + nb_conf_matrix[0][4]
        nb_fn_2 = nb_conf_matrix[1][0] + nb_conf_matrix[1][2] + nb_conf_matrix[1][3] + nb_conf_matrix[1][4]
        nb_fn_3 = nb_conf_matrix[2][0] + nb_conf_matrix[2][1] + nb_conf_matrix[2][3] + nb_conf_matrix[2][4]
        nb_fn_4 = nb_conf_matrix[3][0] + nb_conf_matrix[3][1] + nb_conf_matrix[3][2] + nb_conf_matrix[3][4]
        nb_fn_5 = nb_conf_matrix[4][0] + nb_conf_matrix[4][1] + nb_conf_matrix[4][2] + nb_conf_matrix[4][3]
        if nb_tp_1 + nb_fn_1 == 0:
            nb_recall_1 = 0
        else:
            nb_recall_1 = nb_tp_1 / (nb_tp_1 + nb_fn_1)
        if nb_tp_2 + nb_fn_2 == 0:
            nb_recall_2 = 0
        else:
            nb_recall_2 = nb_tp_2 / (nb_tp_2 + nb_fn_2)
        if nb_tp_3 + nb_fn_3 == 0:
            nb_recall_3 = 0
        else:
            nb_recall_3 = nb_tp_3 / (nb_tp_3 + nb_fn_3)
        if nb_tp_4 + nb_fn_4 == 0:
            nb_recall_4 = 0
        else:
            nb_recall_4 = nb_tp_4 / (nb_tp_4 + nb_fn_4)
        if nb_tp_5 + nb_fn_5 == 0:
            nb_recall_5 = 0
        else:
            nb_recall_5 = nb_tp_5 / (nb_tp_5 +nb_fn_5)
        nb_recall_avg_pen_1 = (nb_recall_1 + nb_recall_2 + nb_recall_3 +nb_recall_4 + nb_recall_5) / (5+1-1)
        return nb_recall_avg_pen_1

    def get_recall_pen_5(nb_conf_matrix):
        nb_tp_1 = nb_conf_matrix[0][0]
        nb_tp_2 = nb_conf_matrix[1][1]
        nb_tp_3 = nb_conf_matrix[2][2]
        nb_tp_4 = nb_conf_matrix[3][3]
        nb_tp_5 = nb_conf_matrix[4][4]
        nb_fn_1 = nb_conf_matrix[0][1] + nb_conf_matrix[0][2] + nb_conf_matrix[0][3] + nb_conf_matrix[0][4]
        nb_fn_2 = nb_conf_matrix[1][0] + nb_conf_matrix[1][2] + nb_conf_matrix[1][3] + nb_conf_matrix[1][4]
        nb_fn_3 = nb_conf_matrix[2][0] + nb_conf_matrix[2][1] + nb_conf_matrix[2][3] + nb_conf_matrix[2][4]
        nb_fn_4 = nb_conf_matrix[3][0] + nb_conf_matrix[3][1] + nb_conf_matrix[3][2] + nb_conf_matrix[3][4]
        nb_fn_5 = nb_conf_matrix[4][0] + nb_conf_matrix[4][1] + nb_conf_matrix[4][2] + nb_conf_matrix[4][3]
        if nb_tp_1 + nb_fn_1 == 0:
            nb_recall_1 = 0
        else:
            nb_recall_1 = nb_tp_1 / (nb_tp_1 + nb_fn_1)
        if nb_tp_2 + nb_fn_2 == 0:
            nb_recall_2 = 0
        else:
            nb_recall_2 = nb_tp_2 / (nb_tp_2 + nb_fn_2)
        if nb_tp_3 + nb_fn_3 == 0:
            nb_recall_3 = 0
        else:
            nb_recall_3 = nb_tp_3 / (nb_tp_3 + nb_fn_3)
        if nb_tp_4 + nb_fn_4 == 0:
            nb_recall_4 = 0
        else:
            nb_recall_4 = nb_tp_4 / (nb_tp_4 + nb_fn_4)
        if nb_tp_5 + nb_fn_5 == 0:
            nb_recall_5 = 0
        else:
            nb_recall_5 = nb_tp_5 / (nb_tp_5 +nb_fn_5)
        nb_recall_avg_pen_5 = (nb_recall_1 + nb_recall_2 + (5*nb_recall_3) +nb_recall_4 + nb_recall_5) / (5+5-1)
        return nb_recall_avg_pen_5


    from sklearn.metrics import classification_report, confusion_matrix

    nb_conf_matrix = confusion_matrix(y_test, nb_prediction)

    ### PENALTY OPERATION ###
    # Penalty for first class(Prediction of first class is important here)
    #nb_conf_matrix[0] = penalty * nb_conf_matrix[0]
    #nb_conf_matrix[0][0] = nb_conf_matrix[0][0] / penalty
    # Penalty for second class(Prediction of first class is important here)
    #nb_conf_matrix[1] = penalty * nb_conf_matrix[1]
    #nb_conf_matrix[1][1] = nb_conf_matrix[1][1] / penalty
    # Penalty for third class(Prediction of first class is important here)
    #nb_conf_matrix[2] = penalty * nb_conf_matrix[2]
    #nb_conf_matrix[2][2] = nb_conf_matrix[2][2] / penalty
    # Penalty for fourth class(Prediction of first class is important here)
    #nb_conf_matrix[3] = penalty * nb_conf_matrix[3]
    #nb_conf_matrix[3][3] = nb_conf_matrix[3][3] / penalty
    # Penalty for fifth class(Prediction of first class is important here)
    #nb_conf_matrix[4] = penalty * nb_conf_matrix[4]
    #nb_conf_matrix[4][4] = nb_conf_matrix[4][4] / penalty

    print("nb_confusion matrix:")
    print(nb_conf_matrix)
    nb_precision = get_precision(nb_conf_matrix)
    nb_recall_pen_1 = get_recall_pen_1(nb_conf_matrix)
    nb_recall_pen_5 = get_recall_pen_5(nb_conf_matrix)
    nb_f1_score_pen_1 = 2 * (nb_precision * nb_recall_pen_1) / (nb_precision + nb_recall_pen_1)
    nb_f1_score_pen_5 = 2 * (nb_precision * nb_recall_pen_5) / (nb_precision + nb_recall_pen_5)
    nb_ovr_accuracy = (nb_conf_matrix[0][0] + nb_conf_matrix[1][1] + nb_conf_matrix[2][2] + nb_conf_matrix[3][3] + nb_conf_matrix[4][4]) / (
                sum(nb_conf_matrix[0]) + sum(nb_conf_matrix[1]) + sum(nb_conf_matrix[2]) + sum(nb_conf_matrix[3]) + sum(nb_conf_matrix[4]))
    print("nb_f1 score of pen 1 is:")
    print(nb_f1_score_pen_1)
    print("nb_f1 score of pen 5 is:")
    print(nb_f1_score_pen_5)
    print("nb_overall accuracy is:")
    print(nb_ovr_accuracy)
    nb_conf_matrix = pd.DataFrame(nb_conf_matrix)
    nb_conf_matrix.to_csv('conf_matrix_'+imb_technique+'_nb_bpic2013_closed_'+ str(nsplits) +'foldcv_' + str(repeat+1)+'.csv',header=False,index=False) #First repetition
    #nb_conf_matrix.to_csv('conf_matrix_'+imb_technique+'_penalty_' + str(penalty) + '_nb_bpic2013_closed_'+ str(nsplits) +'foldcv_' + str(repeat+6)+'.csv',header=False,index=False) #Second repetition
    nb_f1_score_pen_1_kfoldcv[repeat] = nb_f1_score_pen_1
    nb_f1_score_pen_5_kfoldcv[repeat] = nb_f1_score_pen_5
    nb_ovr_accuracy_kfoldcv[repeat] = nb_ovr_accuracy



    for i in range(0, len(y_test)):
        rf_AA_index = 0
        rf_AI_index = 0
        rf_AW_index = 0
        rf_CC_index = 0
        rf_QA_index = 0
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


    def get_precision(rf_conf_matrix):
        rf_tp_1 = rf_conf_matrix[0][0]
        rf_tp_2 = rf_conf_matrix[1][1]
        rf_tp_3 = rf_conf_matrix[2][2]
        rf_tp_4 = rf_conf_matrix[3][3]
        rf_tp_5 = rf_conf_matrix[4][4]
        rf_fp_1 = rf_conf_matrix[1][0] + rf_conf_matrix[2][0] + rf_conf_matrix[3][0] + rf_conf_matrix[4][0]
        rf_fp_2 = rf_conf_matrix[0][1] + rf_conf_matrix[2][1] + rf_conf_matrix[3][1] + rf_conf_matrix[4][1]
        rf_fp_3 = rf_conf_matrix[0][2] + rf_conf_matrix[1][2] + rf_conf_matrix[3][2] + rf_conf_matrix[4][2]
        rf_fp_4 = rf_conf_matrix[0][3] + rf_conf_matrix[1][3] + rf_conf_matrix[2][3] + rf_conf_matrix[4][3]
        rf_fp_5 = rf_conf_matrix[0][4] + rf_conf_matrix[1][4] + rf_conf_matrix[2][4] + rf_conf_matrix[3][4]

        if rf_tp_1 + rf_fp_1 == 0:
            rf_precision_1 = 0
        else:
            rf_precision_1 = rf_tp_1 / (rf_tp_1 + rf_fp_1)
        if rf_tp_2 + rf_fp_2 == 0:
            rf_precision_2 = 0
        else:
            rf_precision_2 = rf_tp_2 / (rf_tp_2 + rf_fp_2)
        if rf_tp_3 + rf_fp_3 == 0:
            rf_precision_3 = 0
        else:
            rf_precision_3 = rf_tp_3 / (rf_tp_3 + rf_fp_3)
        if rf_tp_4 + rf_fp_4 == 0:
            rf_precision_4 = 0
        else:
            rf_precision_4 = rf_tp_4 / (rf_tp_4 + rf_fp_4)
        if rf_tp_5 + rf_fp_5 == 0:
            rf_precision_5 = 0
        else:
            rf_precision_5 = rf_tp_5 / (rf_tp_5 + rf_fp_5)
        rf_precision_avg = (rf_precision_1 + rf_precision_2 + rf_precision_3 + rf_precision_4 + rf_precision_5) / 5
        return rf_precision_avg

    def get_recall_pen_1(rf_conf_matrix):
        rf_tp_1 = rf_conf_matrix[0][0]
        rf_tp_2 = rf_conf_matrix[1][1]
        rf_tp_3 = rf_conf_matrix[2][2]
        rf_tp_4 = rf_conf_matrix[3][3]
        rf_tp_5 = rf_conf_matrix[4][4]
        rf_fn_1 = rf_conf_matrix[0][1] + rf_conf_matrix[0][2] + rf_conf_matrix[0][3] + rf_conf_matrix[0][4]
        rf_fn_2 = rf_conf_matrix[1][0] + rf_conf_matrix[1][2] + rf_conf_matrix[1][3] + rf_conf_matrix[1][4]
        rf_fn_3 = rf_conf_matrix[2][0] + rf_conf_matrix[2][1] + rf_conf_matrix[2][3] + rf_conf_matrix[2][4]
        rf_fn_4 = rf_conf_matrix[3][0] + rf_conf_matrix[3][1] + rf_conf_matrix[3][2] + rf_conf_matrix[3][4]
        rf_fn_5 = rf_conf_matrix[4][0] + rf_conf_matrix[4][1] + rf_conf_matrix[4][2] + rf_conf_matrix[4][3]
        if rf_tp_1 + rf_fn_1 == 0:
            rf_recall_1 = 0
        else:
            rf_recall_1 = rf_tp_1 / (rf_tp_1 + rf_fn_1)
        if rf_tp_2 + rf_fn_2 == 0:
            rf_recall_2 = 0
        else:
            rf_recall_2 = rf_tp_2 / (rf_tp_2 + rf_fn_2)
        if rf_tp_3 + rf_fn_3 == 0:
            rf_recall_3 = 0
        else:
            rf_recall_3 = rf_tp_3 / (rf_tp_3 + rf_fn_3)
        if rf_tp_4 + rf_fn_4 == 0:
            rf_recall_4 = 0
        else:
            rf_recall_4 = rf_tp_4 / (rf_tp_4 + rf_fn_4)
        if rf_tp_5 + rf_fn_5 == 0:
            rf_recall_5 = 0
        else:
            rf_recall_5 = rf_tp_5 / (rf_tp_5 +rf_fn_5)
        rf_recall_avg_pen_1 = (rf_recall_1 + rf_recall_2 + rf_recall_3 +rf_recall_4 + rf_recall_5) / (5+1-1)
        return rf_recall_avg_pen_1

    def get_recall_pen_5(rf_conf_matrix):
        rf_tp_1 = rf_conf_matrix[0][0]
        rf_tp_2 = rf_conf_matrix[1][1]
        rf_tp_3 = rf_conf_matrix[2][2]
        rf_tp_4 = rf_conf_matrix[3][3]
        rf_tp_5 = rf_conf_matrix[4][4]
        rf_fn_1 = rf_conf_matrix[0][1] + rf_conf_matrix[0][2] + rf_conf_matrix[0][3] + rf_conf_matrix[0][4]
        rf_fn_2 = rf_conf_matrix[1][0] + rf_conf_matrix[1][2] + rf_conf_matrix[1][3] + rf_conf_matrix[1][4]
        rf_fn_3 = rf_conf_matrix[2][0] + rf_conf_matrix[2][1] + rf_conf_matrix[2][3] + rf_conf_matrix[2][4]
        rf_fn_4 = rf_conf_matrix[3][0] + rf_conf_matrix[3][1] + rf_conf_matrix[3][2] + rf_conf_matrix[3][4]
        rf_fn_5 = rf_conf_matrix[4][0] + rf_conf_matrix[4][1] + rf_conf_matrix[4][2] + rf_conf_matrix[4][3]
        if rf_tp_1 + rf_fn_1 == 0:
            rf_recall_1 = 0
        else:
            rf_recall_1 = rf_tp_1 / (rf_tp_1 + rf_fn_1)
        if rf_tp_2 + rf_fn_2 == 0:
            rf_recall_2 = 0
        else:
            rf_recall_2 = rf_tp_2 / (rf_tp_2 + rf_fn_2)
        if rf_tp_3 + rf_fn_3 == 0:
            rf_recall_3 = 0
        else:
            rf_recall_3 = rf_tp_3 / (rf_tp_3 + rf_fn_3)
        if rf_tp_4 + rf_fn_4 == 0:
            rf_recall_4 = 0
        else:
            rf_recall_4 = rf_tp_4 / (rf_tp_4 + rf_fn_4)
        if rf_tp_5 + rf_fn_5 == 0:
            rf_recall_5 = 0
        else:
            rf_recall_5 = rf_tp_5 / (rf_tp_5 +rf_fn_5)
        rf_recall_avg_pen_5 = (rf_recall_1 + rf_recall_2 + (5*rf_recall_3) +rf_recall_4 + rf_recall_5) / (5+5-1)
        return rf_recall_avg_pen_5


    from sklearn.metrics import classification_report, confusion_matrix

    rf_conf_matrix = confusion_matrix(y_test, rf_prediction)

    ### PENALTY OPERATION ###
    # Penalty for first class(Prediction of first class is important here)
    #rf_conf_matrix[0] = penalty * rf_conf_matrix[0]
    #rf_conf_matrix[0][0] = rf_conf_matrix[0][0] / penalty
    # Penalty for second class(Prediction of first class is important here)
    #rf_conf_matrix[1] = penalty * rf_conf_matrix[1]
    #rf_conf_matrix[1][1] = rf_conf_matrix[1][1] / penalty
    # Penalty for third class(Prediction of first class is important here)
    #rf_conf_matrix[2] = penalty * rf_conf_matrix[2]
    #rf_conf_matrix[2][2] = rf_conf_matrix[2][2] / penalty
    # Penalty for fourth class(Prediction of first class is important here)
    #rf_conf_matrix[3] = penalty * rf_conf_matrix[3]
    #rf_conf_matrix[3][3] = rf_conf_matrix[3][3] / penalty
    # Penalty for fifth class(Prediction of first class is important here)
    #rf_conf_matrix[4] = penalty * rf_conf_matrix[4]
    #rf_conf_matrix[4][4] = rf_conf_matrix[4][4] / penalty

    print("rf_confusion matrix:")
    print(rf_conf_matrix)
    rf_precision = get_precision(rf_conf_matrix)
    rf_recall_pen_1 = get_recall_pen_1(rf_conf_matrix)
    rf_recall_pen_5 = get_recall_pen_5(rf_conf_matrix)
    rf_f1_score_pen_1 = 2 * (rf_precision * rf_recall_pen_1) / (rf_precision + rf_recall_pen_1)
    rf_f1_score_pen_5 = 2 * (rf_precision * rf_recall_pen_5) / (rf_precision + rf_recall_pen_5)
    rf_ovr_accuracy = (rf_conf_matrix[0][0] + rf_conf_matrix[1][1] + rf_conf_matrix[2][2] + rf_conf_matrix[3][3] + rf_conf_matrix[4][4]) / (
                sum(rf_conf_matrix[0]) + sum(rf_conf_matrix[1]) + sum(rf_conf_matrix[2]) + sum(rf_conf_matrix[3]) + sum(rf_conf_matrix[4]))
    print("rf_f1 score of pen 1 is:")
    print(rf_f1_score_pen_1)
    print("rf_f1 score of pen 5 is:")
    print(rf_f1_score_pen_5)
    print("rf_overall accuracy is:")
    print(rf_ovr_accuracy)
    rf_conf_matrix = pd.DataFrame(rf_conf_matrix)
    rf_conf_matrix.to_csv('conf_matrix_'+imb_technique+'_rf_bpic2013_closed_'+ str(nsplits) +'foldcv_' + str(repeat+1)+'.csv',header=False,index=False) #First repetition
    #rf_conf_matrix.to_csv('conf_matrix_'+imb_technique+'_penalty_' + str(penalty) + '_rf_bpic2013_closed_'+ str(nsplits) +'foldcv_' + str(repeat+6)+'.csv',header=False,index=False) #Second repetition
    rf_f1_score_pen_1_kfoldcv[repeat] = rf_f1_score_pen_1
    rf_f1_score_pen_5_kfoldcv[repeat] = rf_f1_score_pen_5
    rf_ovr_accuracy_kfoldcv[repeat] = rf_ovr_accuracy



    for i in range(0, len(y_test)):
        svm_AA_index = 0
        svm_AI_index = 0
        svm_AW_index = 0
        svm_CC_index = 0
        svm_QA_index = 0
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


    def get_precision(svm_conf_matrix):
        svm_tp_1 = svm_conf_matrix[0][0]
        svm_tp_2 = svm_conf_matrix[1][1]
        svm_tp_3 = svm_conf_matrix[2][2]
        svm_tp_4 = svm_conf_matrix[3][3]
        svm_tp_5 = svm_conf_matrix[4][4]
        svm_fp_1 = svm_conf_matrix[1][0] + svm_conf_matrix[2][0] + svm_conf_matrix[3][0] + svm_conf_matrix[4][0]
        svm_fp_2 = svm_conf_matrix[0][1] + svm_conf_matrix[2][1] + svm_conf_matrix[3][1] + svm_conf_matrix[4][1]
        svm_fp_3 = svm_conf_matrix[0][2] + svm_conf_matrix[1][2] + svm_conf_matrix[3][2] + svm_conf_matrix[4][2]
        svm_fp_4 = svm_conf_matrix[0][3] + svm_conf_matrix[1][3] + svm_conf_matrix[2][3] + svm_conf_matrix[4][3]
        svm_fp_5 = svm_conf_matrix[0][4] + svm_conf_matrix[1][4] + svm_conf_matrix[2][4] + svm_conf_matrix[3][4]

        if svm_tp_1 + svm_fp_1 == 0:
            svm_precision_1 = 0
        else:
            svm_precision_1 = svm_tp_1 / (svm_tp_1 + svm_fp_1)
        if svm_tp_2 + svm_fp_2 == 0:
            svm_precision_2 = 0
        else:
            svm_precision_2 = svm_tp_2 / (svm_tp_2 + svm_fp_2)
        if svm_tp_3 + svm_fp_3 == 0:
            svm_precision_3 = 0
        else:
            svm_precision_3 = svm_tp_3 / (svm_tp_3 + svm_fp_3)
        if svm_tp_4 + svm_fp_4 == 0:
            svm_precision_4 = 0
        else:
            svm_precision_4 = svm_tp_4 / (svm_tp_4 + svm_fp_4)
        if svm_tp_5 + svm_fp_5 == 0:
            svm_precision_5 = 0
        else:
            svm_precision_5 = svm_tp_5 / (svm_tp_5 + svm_fp_5)
        svm_precision_avg = (svm_precision_1 + svm_precision_2 + svm_precision_3 + svm_precision_4 + svm_precision_5) / 5
        return svm_precision_avg

    def get_recall_pen_1(svm_conf_matrix):
        svm_tp_1 = svm_conf_matrix[0][0]
        svm_tp_2 = svm_conf_matrix[1][1]
        svm_tp_3 = svm_conf_matrix[2][2]
        svm_tp_4 = svm_conf_matrix[3][3]
        svm_tp_5 = svm_conf_matrix[4][4]
        svm_fn_1 = svm_conf_matrix[0][1] + svm_conf_matrix[0][2] + svm_conf_matrix[0][3] + svm_conf_matrix[0][4]
        svm_fn_2 = svm_conf_matrix[1][0] + svm_conf_matrix[1][2] + svm_conf_matrix[1][3] + svm_conf_matrix[1][4]
        svm_fn_3 = svm_conf_matrix[2][0] + svm_conf_matrix[2][1] + svm_conf_matrix[2][3] + svm_conf_matrix[2][4]
        svm_fn_4 = svm_conf_matrix[3][0] + svm_conf_matrix[3][1] + svm_conf_matrix[3][2] + svm_conf_matrix[3][4]
        svm_fn_5 = svm_conf_matrix[4][0] + svm_conf_matrix[4][1] + svm_conf_matrix[4][2] + svm_conf_matrix[4][3]
        if svm_tp_1 + svm_fn_1 == 0:
            svm_recall_1 = 0
        else:
            svm_recall_1 = svm_tp_1 / (svm_tp_1 + svm_fn_1)
        if svm_tp_2 + svm_fn_2 == 0:
            svm_recall_2 = 0
        else:
            svm_recall_2 = svm_tp_2 / (svm_tp_2 + svm_fn_2)
        if svm_tp_3 + svm_fn_3 == 0:
            svm_recall_3 = 0
        else:
            svm_recall_3 = svm_tp_3 / (svm_tp_3 + svm_fn_3)
        if svm_tp_4 + svm_fn_4 == 0:
            svm_recall_4 = 0
        else:
            svm_recall_4 = svm_tp_4 / (svm_tp_4 + svm_fn_4)
        if svm_tp_5 + svm_fn_5 == 0:
            svm_recall_5 = 0
        else:
            svm_recall_5 = svm_tp_5 / (svm_tp_5 +svm_fn_5)
        svm_recall_avg_pen_1 = (svm_recall_1 + svm_recall_2 + svm_recall_3 +svm_recall_4 + svm_recall_5) / (5+1-1)
        return svm_recall_avg_pen_1

    def get_recall_pen_5(svm_conf_matrix):
        svm_tp_1 = svm_conf_matrix[0][0]
        svm_tp_2 = svm_conf_matrix[1][1]
        svm_tp_3 = svm_conf_matrix[2][2]
        svm_tp_4 = svm_conf_matrix[3][3]
        svm_tp_5 = svm_conf_matrix[4][4]
        svm_fn_1 = svm_conf_matrix[0][1] + svm_conf_matrix[0][2] + svm_conf_matrix[0][3] + svm_conf_matrix[0][4]
        svm_fn_2 = svm_conf_matrix[1][0] + svm_conf_matrix[1][2] + svm_conf_matrix[1][3] + svm_conf_matrix[1][4]
        svm_fn_3 = svm_conf_matrix[2][0] + svm_conf_matrix[2][1] + svm_conf_matrix[2][3] + svm_conf_matrix[2][4]
        svm_fn_4 = svm_conf_matrix[3][0] + svm_conf_matrix[3][1] + svm_conf_matrix[3][2] + svm_conf_matrix[3][4]
        svm_fn_5 = svm_conf_matrix[4][0] + svm_conf_matrix[4][1] + svm_conf_matrix[4][2] + svm_conf_matrix[4][3]
        if svm_tp_1 + svm_fn_1 == 0:
            svm_recall_1 = 0
        else:
            svm_recall_1 = svm_tp_1 / (svm_tp_1 + svm_fn_1)
        if svm_tp_2 + svm_fn_2 == 0:
            svm_recall_2 = 0
        else:
            svm_recall_2 = svm_tp_2 / (svm_tp_2 + svm_fn_2)
        if svm_tp_3 + svm_fn_3 == 0:
            svm_recall_3 = 0
        else:
            svm_recall_3 = svm_tp_3 / (svm_tp_3 + svm_fn_3)
        if svm_tp_4 + svm_fn_4 == 0:
            svm_recall_4 = 0
        else:
            svm_recall_4 = svm_tp_4 / (svm_tp_4 + svm_fn_4)
        if svm_tp_5 + svm_fn_5 == 0:
            svm_recall_5 = 0
        else:
            svm_recall_5 = svm_tp_5 / (svm_tp_5 +svm_fn_5)
        svm_recall_avg_pen_5 = (svm_recall_1 + svm_recall_2 + (5*svm_recall_3) +svm_recall_4 + svm_recall_5) / (5+5-1)
        return svm_recall_avg_pen_5


    from sklearn.metrics import classification_report, confusion_matrix

    svm_conf_matrix = confusion_matrix(y_test, svm_prediction)

    ### PENALTY OPERATION ###
    # Penalty for first class(Prediction of first class is important here)
    #svm_conf_matrix[0] = penalty * svm_conf_matrix[0]
    #svm_conf_matrix[0][0] = svm_conf_matrix[0][0] / penalty
    # Penalty for second class(Prediction of first class is important here)
    #svm_conf_matrix[1] = penalty * svm_conf_matrix[1]
    #svm_conf_matrix[1][1] = svm_conf_matrix[1][1] / penalty
    # Penalty for third class(Prediction of first class is important here)
    #svm_conf_matrix[2] = penalty * svm_conf_matrix[2]
    #svm_conf_matrix[2][2] = svm_conf_matrix[2][2] / penalty
    # Penalty for fourth class(Prediction of first class is important here)
    #svm_conf_matrix[3] = penalty * svm_conf_matrix[3]
    #svm_conf_matrix[3][3] = svm_conf_matrix[3][3] / penalty
    # Penalty for fifth class(Prediction of first class is important here)
    #svm_conf_matrix[4] = penalty * svm_conf_matrix[4]
    #svm_conf_matrix[4][4] = svm_conf_matrix[4][4] / penalty

    print("svm_confusion matrix:")
    print(svm_conf_matrix)
    svm_precision = get_precision(svm_conf_matrix)
    svm_recall_pen_1 = get_recall_pen_1(svm_conf_matrix)
    svm_recall_pen_5 = get_recall_pen_5(svm_conf_matrix)
    svm_f1_score_pen_1 = 2 * (svm_precision * svm_recall_pen_1) / (svm_precision + svm_recall_pen_1)
    svm_f1_score_pen_5 = 2 * (svm_precision * svm_recall_pen_5) / (svm_precision + svm_recall_pen_5)
    svm_ovr_accuracy = (svm_conf_matrix[0][0] + svm_conf_matrix[1][1] + svm_conf_matrix[2][2] + svm_conf_matrix[3][3] + svm_conf_matrix[4][4]) / (
                sum(svm_conf_matrix[0]) + sum(svm_conf_matrix[1]) + sum(svm_conf_matrix[2]) + sum(svm_conf_matrix[3]) + sum(svm_conf_matrix[4]))
    print("svm_f1 score of pen 1 is:")
    print(svm_f1_score_pen_1)
    print("svm_f1 score of pen 5 is:")
    print(svm_f1_score_pen_5)
    print("svm_overall accuracy is:")
    print(svm_ovr_accuracy)
    svm_conf_matrix = pd.DataFrame(svm_conf_matrix)
    svm_conf_matrix.to_csv('conf_matrix_'+imb_technique+'_svm_bpic2013_closed_'+ str(nsplits) +'foldcv_' + str(repeat+1)+'.csv',header=False,index=False) #First repetition
    #svm_conf_matrix.to_csv('conf_matrix_'+imb_technique+'_penalty_' + str(penalty) + '_svm_bpic2013_closed_'+ str(nsplits) +'foldcv_' + str(repeat+6)+'.csv',header=False,index=False) #Second repetition
    svm_f1_score_pen_1_kfoldcv[repeat] = svm_f1_score_pen_1
    svm_f1_score_pen_5_kfoldcv[repeat] = svm_f1_score_pen_5
    svm_ovr_accuracy_kfoldcv[repeat] = svm_ovr_accuracy
    repeat = repeat + 1
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
#dnn_ovr_accuracy_kfoldcv.to_csv('ovr_accuracy_'+imb_technique+'_penalty_' + str(penalty) + '_dnn_bpic2013_closed_' + str(nsplits) + 'foldcv_1~5.csv',header=False,index=False) #First repetition
#dnn_ovr_accuracy_kfoldcv.to_csv('ovr_accuracy_'+imb_technique+'_penalty_' + str(penalty) + '_dnn_bpic2013_closed_' + str(nsplits) + 'foldcv_6~10.csv',header=False,index=False) #Second repetition

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
#lr_ovr_accuracy_kfoldcv.to_csv('ovr_accuracy_'+imb_technique+'_penalty_' + str(penalty) + '_lr_bpic2013_closed_' + str(nsplits) + 'foldcv_1~5.csv',header=False,index=False) #First repetition
#lr_ovr_accuracy_kfoldcv.to_csv('ovr_accuracy_'+imb_technique+'_penalty_' + str(penalty) + '_lr_bpic2013_closed_' + str(nsplits) + 'foldcv_6~10.csv',header=False,index=False) #Second repetition

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
#nb_ovr_accuracy_kfoldcv.to_csv('ovr_accuracy_'+imb_technique+'_penalty_' + str(penalty) + '_nb_bpic2013_closed_' + str(nsplits) + 'foldcv_1~5.csv',header=False,index=False) #First repetition
#nb_ovr_accuracy_kfoldcv.to_csv('ovr_accuracy_'+imb_technique+'_penalty_' + str(penalty) + '_nb_bpic2013_closed_' + str(nsplits) + 'foldcv_6~10.csv',header=False,index=False) #Second repetition

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
#rf_ovr_accuracy_kfoldcv.to_csv('ovr_accuracy_'+imb_technique+'_penalty_' + str(penalty) + '_rf_bpic2013_closed_' + str(nsplits) + 'foldcv_1~5.csv',header=False,index=False) #First repetition
#rf_ovr_accuracy_kfoldcv.to_csv('ovr_accuracy_'+imb_technique+'_penalty_' + str(penalty) + '_rf_bpic2013_closed_' + str(nsplits) + 'foldcv_6~10.csv',header=False,index=False) #Second repetition

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
#svm_ovr_accuracy_kfoldcv.to_csv('ovr_accuracy_'+imb_technique+'_penalty_' + str(penalty) + '_svm_bpic2013_closed_' + str(nsplits) + 'foldcv_1~5.csv',header=False,index=False) #First repetition
#svm_ovr_accuracy_kfoldcv.to_csv('ovr_accuracy_'+imb_technique+'_penalty_' + str(penalty) + '_svm_bpic2013_closed_' + str(nsplits) + 'foldcv_6~10.csv',header=False,index=False) #Second repetition

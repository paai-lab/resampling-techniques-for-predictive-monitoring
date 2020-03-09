import tensorflow as tf
import numpy as np
import pandas as pd
from collections import Counter
#Pandas was used to import csv file. Encoding parameter is set to "cp437" since the data contains English text
data_sample_percentage = "" #Delete after experiments. If data is not to be reduced, type in "" here. If to be reduced, type in, for example, "_20percent"
data_dir = "/home/jongchan/Road_traffic/window_3_road_traffic_reduced_preprocessed" + data_sample_percentage + ".csv"
data = pd.read_csv(data_dir, encoding='cp437')
#data = data.sample(frac=1).reset_index(drop=True) #I use this for ADASYN since ADASYN returns error on specific fold
X = data[['ACT_1', 'ACT_2', 'ACT_3','duration_in_days']]
y = data[['ACT_4']]

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
print("Data: Road traffic")
print("Resampling technique: " + imb_technique)

penalty = 5

# Dummification
X_dummy = pd.get_dummies(X, prefix="ACT_1", columns=['ACT_1'])
X_dummy = pd.get_dummies(X_dummy, prefix="ACT_2", columns=['ACT_2'])
X_dummy = pd.get_dummies(X_dummy, prefix="ACT_3", columns=['ACT_3'])
X_dummy.iloc[:, 0] = (X_dummy.iloc[:, 0] - X_dummy.iloc[:, 0].mean()) / X_dummy.iloc[:, 0].std()

# X and y here will be used for hyperparameter tuning using random search
X_randomsearch = X.replace(regex=True, to_replace="Add penalty", value=1)
X_randomsearch = X_randomsearch.replace(regex=True, to_replace="Payment", value=2)
X_randomsearch = X_randomsearch.replace(regex=True, to_replace="Send for Credit Collection", value=3)
X_randomsearch = X_randomsearch.replace(regex=True, to_replace="Insert Fine Notification", value=4)
X_randomsearch = X_randomsearch.replace(regex=True, to_replace="Send Fine", value=5)
X_randomsearch = X_randomsearch.replace(regex=True, to_replace="Create Fine", value=6)

y_randomsearch = y.replace(regex=True, to_replace="Add penalty", value=1)
y_randomsearch = y_randomsearch.replace(regex=True, to_replace="Payment", value=2)
y_randomsearch = y_randomsearch.replace(regex=True, to_replace="Send for Credit Collection", value=3)
y_randomsearch = y_randomsearch.replace(regex=True, to_replace="Insert Fine Notification", value=4)
y_randomsearch = y_randomsearch.replace(regex=True, to_replace="Send Fine", value=5)
y_randomsearch = y_randomsearch.replace(regex=True, to_replace="Create Fine", value=6)



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
#X_dummy = X_dummy.sample(frac=1).reset_index(drop=True)

for train_index, test_index in kf.split(X_dummy):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X_dummy.iloc[train_index], X_dummy.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    train = pd.concat([X_train, y_train], axis=1)
    ACT_4_index = np.unique(data['ACT_1']).size + np.unique(data['ACT_2']).size + np.unique(data['ACT_3']).size + 1

    #Below codes are used to select features that we are going to use in the training procedure
    AP = train[train.ACT_4 == "Add penalty"]
    AP_rest = train[train.ACT_4 != "Add penalty"]
    AP_rest = AP_rest.copy()
    AP_rest.iloc[:, ACT_4_index] = "Others"
    AP = AP.reset_index()
    AP_rest = AP_rest.reset_index()
    AP = AP.iloc[:, 1:ACT_4_index+2]
    AP_rest = AP_rest.iloc[:, 1:ACT_4_index+2]
    AP_ova = pd.concat([AP, AP_rest])
    AP_ova_X_train = AP_ova.iloc[:, 0:ACT_4_index]
    AP_ova_y_train = AP_ova.iloc[:, ACT_4_index]
    AP_X_res = AP_ova_X_train
    AP_y_res = AP_ova_y_train
    Counter(AP_ova_y_train)

    # CF = train[train.ACT_4 == "Create Fine"]
    # CF_rest = train[train.ACT_4 != "Create Fine"]
    # CF_rest = CF_rest.copy()
    # CF_rest.iloc[:,ACT_4_index] = "Others"
    # CF = CF.reset_index()
    # CF_rest = CF_rest.reset_index()
    # CF = CF.iloc[:,1:ACT_4_index+2]
    # CF_rest = CF_rest.iloc[:,1:ACT_4_index+2]
    # CF_ova = pd.concat([CF, CF_rest])
    # CF_ova_X_train = CF_ova.iloc[:,0:ACT_4_index]
    # CF_ova_y_train = CF_ova.iloc[:,ACT_4_index]
    # CF_X_res = CF_ova_X_train
    # CF_y_res = CF_ova_y_train
    # Counter(CF_ova_y_train)


    # IFN = train[train.ACT_4 == "Insert Fine Notification"]
    # IFN_rest = train[train.ACT_4 != "Insert Fine Notification"]
    # IFN_rest = IFN_rest.copy()
    # IFN_rest.iloc[:,ACT_4_index] = "Others"
    # IFN = IFN.reset_index()
    # IFN_rest = IFN_rest.reset_index()
    # IFN = IFN.iloc[:,1:ACT_4_index+2]
    # IFN_rest = IFN_rest.iloc[:,1:ACT_4_index+2]
    # IFN_ova = pd.concat([IFN, IFN_rest])
    # IFN_ova_X_train = IFN_ova.iloc[:,0:ACT_4_index]
    # IFN_ova_y_train = IFN_ova.iloc[:,ACT_4_index]
    # IFN_X_res = IFN_ova_X_train
    # IFN_y_res = IFN_ova_y_train
    # Counter(IFN_ova_y_train)


    PM = train[train.ACT_4 == "Payment"]
    PM_rest = train[train.ACT_4 != "Payment"]
    PM_rest = PM_rest.copy()
    PM_rest.iloc[:, ACT_4_index] = "Others"
    PM = PM.reset_index()
    PM_rest = PM_rest.reset_index()
    PM = PM.iloc[:, 1:ACT_4_index+2]
    PM_rest = PM_rest.iloc[:, 1:ACT_4_index+2]
    PM_ova = pd.concat([PM, PM_rest])
    PM_ova_X_train = PM_ova.iloc[:, 0:ACT_4_index]
    PM_ova_y_train = PM_ova.iloc[:, ACT_4_index]
    PM_X_res = PM_ova_X_train
    PM_y_res = PM_ova_y_train
    Counter(PM_ova_y_train)

    # SF = train[train.ACT_4 == "Send Fine"]
    # SF_rest = train[train.ACT_4 != "Send Fine"]
    # SF_rest = SF_rest.copy()
    # SF_rest.iloc[:,ACT_4_index] = "Others"
    # SF = SF.reset_index()
    # SF_rest = SF_rest.reset_index()
    # SF = SF.iloc[:,1:ACT_4_index+2]
    # SF_rest = SF_rest.iloc[:,1:ACT_4_index+2]
    # SF_ova = pd.concat([SF, SF_rest])
    # SF_ova_X_train = SF_ova.iloc[:,0:ACT_4_index]
    # SF_ova_y_train = SF_ova.iloc[:,ACT_4_index]
    # SF_X_res = SF_ova_X_train
    # SF_y_res = SF_ova_y_train
    # Counter(SF_ova_y_train)


    SC = train[train.ACT_4 == "Send for Credit Collection"]
    SC_rest = train[train.ACT_4 != "Send for Credit Collection"]
    SC_rest = SC_rest.copy()
    SC_rest.iloc[:, ACT_4_index] = "Others"
    SC = SC.reset_index()
    SC_rest = SC_rest.reset_index()
    SC = SC.iloc[:, 1:ACT_4_index+2]
    SC_rest = SC_rest.iloc[:, 1:ACT_4_index+2]
    SC_ova = pd.concat([SC, SC_rest])
    SC_ova_X_train = SC_ova.iloc[:, 0:ACT_4_index]
    SC_ova_y_train = SC_ova.iloc[:, ACT_4_index]
    SC_X_res = SC_ova_X_train
    SC_y_res = SC_ova_y_train
    Counter(SC_ova_y_train)

    print("Resampling started")

    #Below codes are used to resample data
    if imb_technique == "ADASYN":
        from imblearn.over_sampling import ADASYN
        print("Original dataset shape %s" % Counter(AP_ova_y_train))
        AP_ada = ADASYN()
        AP_X_res, AP_y_res = AP_ada.fit_resample(AP_ova_X_train, AP_ova_y_train)
        print("Resampled dataset shape %s" % Counter(AP_y_res))
        # CF_ada = ADASYN()
        # CF_X_res, CF_y_res = CF_ada.fit_resample(CF_ova_X_train, CF_ova_y_train)
        # IFN_ada = ADASYN()
        # IFN_X_res, IFN_y_res = IFN_ada.fit_resample(IFN_ova_X_train, IFN_ova_y_train)
        print("Original dataset shape %s" % Counter(PM_ova_y_train))
        PM_ada = ADASYN()
        PM_X_res, PM_y_res = PM_ada.fit_resample(PM_ova_X_train, PM_ova_y_train)
        print("Resampled dataset shape %s" % Counter(PM_y_res))
        # SF_ada = ADASYN()
        # SF_X_res, SF_y_res = SF_ada.fit_resample(SF_ova_X_train, SF_ova_y_train)
        print("Original dataset shape %s" % Counter(SC_ova_y_train))
        SC_ada = ADASYN()
        SC_X_res, SC_y_res = SC_ada.fit_resample(SC_ova_X_train, SC_ova_y_train)
        print("Resampled dataset shape %s" % Counter(SC_y_res))

    if imb_technique == "ALLKNN":
        from imblearn.under_sampling import AllKNN
        print("Original dataset shape %s" % Counter(AP_ova_y_train))
        AP_allknn = AllKNN()
        AP_X_res, AP_y_res = AP_allknn.fit_resample(AP_ova_X_train, AP_ova_y_train)
        print("Resampled dataset shape %s" % Counter(AP_y_res))
        # CF_allknn = AllKNN()
        # CF_X_res, CF_y_res = CF_allknn.fit_resample(CF_ova_X_train, CF_ova_y_train)
        # IFN_allknn = AllKNN()
        # IFN_X_res, IFN_y_res = IFN_allknn.fit_resample(IFN_ova_X_train, IFN_ova_y_train)
        print("Original dataset shape %s" % Counter(PM_ova_y_train))
        PM_allknn = AllKNN()
        PM_X_res, PM_y_res = PM_allknn.fit_resample(PM_ova_X_train, PM_ova_y_train)
        print("Resampled dataset shape %s" % Counter(PM_y_res))
        # SF_allknn = AllKNN()
        # SF_X_res, SF_y_res = SF_allknn.fit_resample(SF_ova_X_train, SF_ova_y_train)
        print("Original dataset shape %s" % Counter(SC_ova_y_train))
        SC_allknn = AllKNN()
        SC_X_res, SC_y_res = SC_allknn.fit_resample(SC_ova_X_train, SC_ova_y_train)
        print("Resampled dataset shape %s" % Counter(SC_y_res))

    if imb_technique == "CNN":
        from imblearn.under_sampling import CondensedNearestNeighbour
        print("Original dataset shape %s" % Counter(AP_ova_y_train))
        AP_cnn = CondensedNearestNeighbour(sampling_strategy='auto')
        #AP_X_res, AP_y_res = AP_cnn.fit_resample(AP_ova_X_train, AP_ova_y_train)
        AP_X_res = AP_ova_X_train
        AP_y_res = AP_ova_y_train
        print("Resampled dataset shape %s" % Counter(AP_y_res))
        # CF_cnn = CondensedNearestNeighbour(sampling_strategy='auto')
        # CF_X_res, CF_y_res = CF_cnn.fit_resample(CF_ova_X_train, CF_ova_y_train)
        # IFN_cnn = CondensedNearestNeighbour(sampling_strategy='auto')
        # IFN_X_res, IFN_y_res = IFN_cnn.fit_resample(IFN_ova_X_train, IFN_ova_y_train)
        print("Original dataset shape %s" % Counter(PM_ova_y_train))
        PM_cnn = CondensedNearestNeighbour(sampling_strategy='auto')
        PM_X_res, PM_y_res = PM_cnn.fit_resample(PM_ova_X_train, PM_ova_y_train)
        print("Resampled dataset shape %s" % Counter(PM_y_res))
        # SF_cnn = CondensedNearestNeighbour(sampling_strategy='auto')
        # SF_X_res, SF_y_res = SF_cnn.fit_resample(SF_ova_X_train, SF_ova_y_train)
        print("Original dataset shape %s" % Counter(SC_ova_y_train))
        SC_cnn = CondensedNearestNeighbour(sampling_strategy='auto')
        SC_X_res, SC_y_res = SC_cnn.fit_resample(SC_ova_X_train, SC_ova_y_train)
        print("Resampled dataset shape %s" % Counter(SC_y_res))

    if imb_technique == "ENN":
        from imblearn.under_sampling import EditedNearestNeighbours
        print("Original dataset shape %s" % Counter(AP_ova_y_train))
        AP_enn = EditedNearestNeighbours()
        AP_X_res, AP_y_res = AP_enn.fit_resample(AP_ova_X_train, AP_ova_y_train)
        print("Resampled dataset shape %s" % Counter(AP_y_res))
        # CF_enn = EditedNearestNeighbours()
        # CF_X_res, CF_y_res = CF_enn.fit_resample(CF_ova_X_train, CF_ova_y_train)
        # IFN_enn = EditedNearestNeighbours()
        # IFN_X_res, IFN_y_res = IFN_enn.fit_resample(IFN_ova_X_train, IFN_ova_y_train)
        print("Original dataset shape %s" % Counter(PM_ova_y_train))
        PM_enn = EditedNearestNeighbours()
        PM_X_res, PM_y_res = PM_enn.fit_resample(PM_ova_X_train, PM_ova_y_train)
        print("Resampled dataset shape %s" % Counter(PM_y_res))
        # SF_enn = EditedNearestNeighbours()
        # SF_X_res, SF_y_res = SF_enn.fit_resample(SF_ova_X_train, SF_ova_y_train)
        print("Original dataset shape %s" % Counter(SC_ova_y_train))
        SC_enn = EditedNearestNeighbours()
        SC_X_res, SC_y_res = SC_enn.fit_resample(SC_ova_X_train, SC_ova_y_train)
        print("Resampled dataset shape %s" % Counter(SC_y_res))

    if imb_technique == "IHT":
        from imblearn.under_sampling import InstanceHardnessThreshold
        print("Original dataset shape %s" % Counter(AP_ova_y_train))
        AP_iht = InstanceHardnessThreshold()
        AP_X_res, AP_y_res = AP_iht.fit_resample(AP_ova_X_train, AP_ova_y_train)
        print("Resampled dataset shape %s" % Counter(AP_y_res))
        # CF_iht = InstanceHardnessThreshold()
        # CF_X_res, CF_y_res = CF_iht.fit_resample(CF_ova_X_train, CF_ova_y_train)
        # IFN_iht = InstanceHardnessThreshold()
        # IFN_X_res, IFN_y_res = IFN_iht.fit_resample(IFN_ova_X_train, IFN_ova_y_train)
        print("Original dataset shape %s" % Counter(PM_ova_y_train))
        PM_iht = InstanceHardnessThreshold()
        PM_X_res, PM_y_res = PM_iht.fit_resample(PM_ova_X_train, PM_ova_y_train)
        print("Resampled dataset shape %s" % Counter(PM_y_res))
        # SF_iht = InstanceHardnessThreshold()
        # SF_X_res, SF_y_res = SF_iht.fit_resample(SF_ova_X_train, SF_ova_y_train)
        print("Original dataset shape %s" % Counter(SC_ova_y_train))
        SC_iht = InstanceHardnessThreshold()
        SC_X_res, SC_y_res = SC_iht.fit_resample(SC_ova_X_train, SC_ova_y_train)
        print("Resampled dataset shape %s" % Counter(SC_y_res))

    if imb_technique == "NCR":
        from imblearn.under_sampling import NeighbourhoodCleaningRule
        print("Original dataset shape %s" % Counter(AP_ova_y_train))
        AP_ncr = NeighbourhoodCleaningRule()
        AP_ova_y_train = [0 if i == "Add penalty" else 1 for i in AP_ova_y_train]
        AP_X_res, AP_y_res = AP_ncr.fit_resample(AP_ova_X_train, AP_ova_y_train)
        print("Resampled dataset shape %s" % Counter(AP_y_res))
        # CF_ncr = NeighbourhoodCleaningRule()
        # CF_ova_y_train = [0 if i == "Create Fine" else 1 for i in CF_ova_y_train]
        # CF_X_res, CF_y_res = CF_ncr.fit_resample(CF_ova_X_train, CF_ova_y_train)
        # IFN_ncr = NeighbourhoodCleaningRule()
        # IFN_ova_y_train = [0 if i == "Insert Fine Notification" else 1 for i in IFN_ova_y_train]
        # IFN_X_res, IFN_y_res = IFN_ncr.fit_resample(IFN_ova_X_train, IFN_ova_y_train)
        print("Original dataset shape %s" % Counter(PM_ova_y_train))
        PM_ncr = NeighbourhoodCleaningRule()
        PM_ova_y_train = [0 if i == "Payment" else 1 for i in PM_ova_y_train]
        PM_X_res, PM_y_res = PM_ncr.fit_resample(PM_ova_X_train, PM_ova_y_train)
        print("Resampled dataset shape %s" % Counter(PM_y_res))
        # SF_ncr = NeighbourhoodCleaningRule()
        # SF_ova_y_train = [0 if i == "Send Fine" else 1 for i in SF_ova_y_train]
        # SF_X_res, SF_y_res = SF_ncr.fit_resample(SF_ova_X_train, SF_ova_y_train)
        print("Original dataset shape %s" % Counter(SC_ova_y_train))
        SC_ncr = NeighbourhoodCleaningRule()
        SC_ova_y_train = [0 if i == "Send for Credit Collection" else 1 for i in SC_ova_y_train]
        SC_X_res, SC_y_res = SC_ncr.fit_resample(SC_ova_X_train, SC_ova_y_train)
        print("Resampled dataset shape %s" % Counter(SC_y_res))

    if imb_technique == "NM":
        from imblearn.under_sampling import NearMiss
        print("Original dataset shape %s" % Counter(AP_ova_y_train))
        AP_nm = NearMiss()
        AP_X_res, AP_y_res = AP_nm.fit_resample(AP_ova_X_train, AP_ova_y_train)
        print("Resampled dataset shape %s" % Counter(AP_y_res))
        # CF_nm = NearMiss()
        # CF_X_res, CF_y_res = CF_nm.fit_resample(CF_ova_X_train, CF_ova_y_train)
        # IFN_nm = NearMiss()
        # IFN_X_res, IFN_y_res = IFN_nm.fit_resample(IFN_ova_X_train, IFN_ova_y_train)
        print("Original dataset shape %s" % Counter(PM_ova_y_train))
        PM_nm = NearMiss()
        PM_X_res, PM_y_res = PM_nm.fit_resample(PM_ova_X_train, PM_ova_y_train)
        print("Resampled dataset shape %s" % Counter(PM_y_res))
        # SF_nm = NearMiss()
        # SF_X_res, SF_y_res = SF_nm.fit_resample(SF_ova_X_train, SF_ova_y_train)
        print("Original dataset shape %s" % Counter(SC_ova_y_train))
        SC_nm = NearMiss()
        SC_X_res, SC_y_res = SC_nm.fit_resample(SC_ova_X_train, SC_ova_y_train)
        print("Resampled dataset shape %s" % Counter(SC_y_res))

    if imb_technique == "OSS":
        from imblearn.under_sampling import OneSidedSelection
        print("Original dataset shape %s" % Counter(AP_ova_y_train))
        AP_oss = OneSidedSelection()
        AP_X_res = AP_ova_X_train
        AP_y_res = AP_ova_y_train
        #AP_X_res, AP_y_res = AP_oss.fit_resample(AP_ova_X_train, AP_ova_y_train)
        print("Resampled dataset shape %s" % Counter(AP_y_res))
        # CF_oss = OneSidedSelection()
        # CF_X_res, CF_y_res = CF_oss.fit_resample(CF_ova_X_train, CF_ova_y_train)
        # IFN_oss = OneSidedSelection()
        # IFN_X_res, IFN_y_res = IFN_oss.fit_resample(IFN_ova_X_train, IFN_ova_y_train)
        print("Original dataset shape %s" % Counter(PM_ova_y_train))
        PM_oss = OneSidedSelection()
        PM_X_res, PM_y_res = PM_oss.fit_resample(PM_ova_X_train, PM_ova_y_train)
        print("Resampled dataset shape %s" % Counter(PM_y_res))
        # SF_oss = OneSidedSelection()
        # SF_X_res, SF_y_res = SF_oss.fit_resample(SF_ova_X_train, SF_ova_y_train)
        print("Original dataset shape %s" % Counter(SC_ova_y_train))
        SC_oss = OneSidedSelection()
        SC_X_res, SC_y_res = SC_oss.fit_resample(SC_ova_X_train, SC_ova_y_train)
        print("Resampled dataset shape %s" % Counter(SC_y_res))

    if imb_technique == "RENN":
        from imblearn.under_sampling import RepeatedEditedNearestNeighbours
        print("Original dataset shape %s" % Counter(AP_ova_y_train))
        AP_renn = RepeatedEditedNearestNeighbours()
        AP_X_res, AP_y_res = AP_renn.fit_resample(AP_ova_X_train, AP_ova_y_train)
        print("Resampled dataset shape %s" % Counter(AP_y_res))
        # CF_renn = RepeatedEditedNearestNeighbours()
        # CF_X_res, CF_y_res = CF_renn.fit_resample(CF_ova_X_train, CF_ova_y_train)
        # IFN_renn = RepeatedEditedNearestNeighbours()
        # IFN_X_res, IFN_y_res = IFN_renn.fit_resample(IFN_ova_X_train, IFN_ova_y_train)
        print("Original dataset shape %s" % Counter(PM_ova_y_train))
        PM_renn = RepeatedEditedNearestNeighbours()
        PM_X_res, PM_y_res = PM_renn.fit_resample(PM_ova_X_train, PM_ova_y_train)
        print("Resampled dataset shape %s" % Counter(PM_y_res))
        # SF_renn = RepeatedEditedNearestNeighbours()
        # SF_X_res, SF_y_res = SF_renn.fit_resample(SF_ova_X_train, SF_ova_y_train)
        print("Original dataset shape %s" % Counter(SC_ova_y_train))
        SC_renn = RepeatedEditedNearestNeighbours()
        SC_X_res, SC_y_res = SC_renn.fit_resample(SC_ova_X_train, SC_ova_y_train)
        print("Resampled dataset shape %s" % Counter(SC_y_res))

    if imb_technique == "BSMOTE":
        from imblearn.over_sampling import BorderlineSMOTE
        print("Original dataset shape %s" % Counter(AP_ova_y_train))
        AP_bsm = BorderlineSMOTE()
        AP_X_res, AP_y_res = AP_bsm.fit_resample(AP_ova_X_train, AP_ova_y_train)
        print("Resampled dataset shape %s" % Counter(AP_y_res))
        # CF_bsm = BorderlineSMOTE()
        # CF_X_res, CF_y_res = CF_bsm.fit_resample(CF_ova_X_train, CF_ova_y_train)
        # IFN_bsm = BorderlineSMOTE()
        # IFN_X_res, IFN_y_res = IFN_bsm.fit_resample(IFN_ova_X_train, IFN_ova_y_train)
        print("Original dataset shape %s" % Counter(PM_ova_y_train))
        PM_bsm = BorderlineSMOTE()
        PM_X_res, PM_y_res = PM_bsm.fit_resample(PM_ova_X_train, PM_ova_y_train)
        print("Resampled dataset shape %s" % Counter(PM_y_res))
        # SF_bsm = BorderlineSMOTE()
        # SF_X_res, SF_y_res = SF_bsm.fit_resample(SF_ova_X_train, SF_ova_y_train)
        print("Original dataset shape %s" % Counter(SC_ova_y_train))
        SC_bsm = BorderlineSMOTE()
        SC_X_res, SC_y_res = SC_bsm.fit_resample(SC_ova_X_train, SC_ova_y_train)
        print("Resampled dataset shape %s" % Counter(SC_y_res))

    if imb_technique == "SMOTE":
        from imblearn.over_sampling import SMOTE
        print("Original dataset shape %s" % Counter(AP_ova_y_train))
        AP_sm = SMOTE()
        AP_X_res, AP_y_res = AP_sm.fit_resample(AP_ova_X_train, AP_ova_y_train)
        print("Resampled dataset shape %s" % Counter(AP_y_res))
        # CF_sm = SMOTE()
        # CF_X_res, CF_y_res = CF_sm.fit_resample(CF_ova_X_train, CF_ova_y_train)
        # IFN_sm = SMOTE()
        # IFN_X_res, IFN_y_res = IFN_sm.fit_resample(IFN_ova_X_train, IFN_ova_y_train)
        print("Original dataset shape %s" % Counter(PM_ova_y_train))
        PM_sm = SMOTE()
        PM_X_res, PM_y_res = PM_sm.fit_resample(PM_ova_X_train, PM_ova_y_train)
        print("Resampled dataset shape %s" % Counter(PM_y_res))
        # SF_sm = SMOTE()
        # SF_X_res, SF_y_res = SF_sm.fit_resample(SF_ova_X_train, SF_ova_y_train)
        print("Original dataset shape %s" % Counter(SC_ova_y_train))
        SC_sm = SMOTE()
        SC_X_res, SC_y_res = SC_sm.fit_resample(SC_ova_X_train, SC_ova_y_train)
        print("Resampled dataset shape %s" % Counter(SC_y_res))

    if imb_technique == "SMOTEENN":
        from imblearn.combine import SMOTEENN
        print("Original dataset shape %s" % Counter(AP_ova_y_train))
        AP_smenn = SMOTEENN()
        AP_X_res, AP_y_res = AP_smenn.fit_resample(AP_ova_X_train, AP_ova_y_train)
        print("Resampled dataset shape %s" % Counter(AP_y_res))
        # CF_smenn = SMOTEENN()
        # CF_X_res, CF_y_res = CF_smenn.fit_resample(CF_ova_X_train, CF_ova_y_train)
        # IFN_smenn = SMOTEENN()
        # IFN_X_res, IFN_y_res = IFN_smenn.fit_resample(IFN_ova_X_train, IFN_ova_y_train)
        print("Original dataset shape %s" % Counter(PM_ova_y_train))
        PM_smenn = SMOTEENN()
        PM_X_res, PM_y_res = PM_smenn.fit_resample(PM_ova_X_train, PM_ova_y_train)
        print("Resampled dataset shape %s" % Counter(PM_y_res))
        # SF_smenn = SMOTEENN()
        # SF_X_res, SF_y_res = SF_smenn.fit_resample(SF_ova_X_train, SF_ova_y_train)
        print("Original dataset shape %s" % Counter(SC_ova_y_train))
        SC_smenn = SMOTEENN()
        SC_X_res, SC_y_res = SC_smenn.fit_resample(SC_ova_X_train, SC_ova_y_train)
        print("Resampled dataset shape %s" % Counter(SC_y_res))

    if imb_technique == "SMOTETOMEK":
        from imblearn.combine import SMOTETomek
        print("Original dataset shape %s" % Counter(AP_ova_y_train))
        AP_smtm = SMOTETomek()
        AP_X_res, AP_y_res = AP_smtm.fit_resample(AP_ova_X_train, AP_ova_y_train)
        print("Resampled dataset shape %s" % Counter(AP_y_res))
        # CF_smtm = SMOTETomek()
        # CF_X_res, CF_y_res = CF_smtm.fit_resample(CF_ova_X_train, CF_ova_y_train)
        # IFN_smtm = SMOTETomek()
        # IFN_X_res, IFN_y_res = IFN_smtm.fit_resample(IFN_ova_X_train, IFN_ova_y_train)
        print("Original dataset shape %s" % Counter(PM_ova_y_train))
        PM_smtm = SMOTETomek()
        PM_X_res, PM_y_res = PM_smtm.fit_resample(PM_ova_X_train, PM_ova_y_train)
        print("Resampled dataset shape %s" % Counter(PM_y_res))
        # SF_smtm = SMOTETomek()
        # SF_X_res, SF_y_res = SF_smtm.fit_resample(SF_ova_X_train, SF_ova_y_train)
        print("Original dataset shape %s" % Counter(SC_ova_y_train))
        SC_smtm = SMOTETomek()
        SC_X_res, SC_y_res = SC_smtm.fit_resample(SC_ova_X_train, SC_ova_y_train)
        print("Resampled dataset shape %s" % Counter(SC_y_res))

    if imb_technique == "TOMEK":
        from imblearn.under_sampling import TomekLinks
        print("Original dataset shape %s" % Counter(AP_ova_y_train))
        AP_tm = TomekLinks()
        AP_X_res, AP_y_res = AP_tm.fit_resample(AP_ova_X_train, AP_ova_y_train)
        print("Resampled dataset shape %s" % Counter(AP_y_res))
        # CF_tm = TomekLinks()
        # CF_X_res, CF_y_res = CF_tm.fit_resample(CF_ova_X_train, CF_ova_y_train)
        # IFN_tm = TomekLinks()
        # IFN_X_res, IFN_y_res = IFN_tm.fit_resample(IFN_ova_X_train, IFN_ova_y_train)
        print("Original dataset shape %s" % Counter(PM_ova_y_train))
        PM_tm = TomekLinks()
        PM_X_res, PM_y_res = PM_tm.fit_resample(PM_ova_X_train, PM_ova_y_train)
        print("Resampled dataset shape %s" % Counter(PM_y_res))
        # SF_tm = TomekLinks()
        # SF_X_res, SF_y_res = SF_tm.fit_resample(SF_ova_X_train, SF_ova_y_train)
        print("Original dataset shape %s" % Counter(SC_ova_y_train))
        SC_tm = TomekLinks()
        SC_X_res, SC_y_res = SC_tm.fit_resample(SC_ova_X_train, SC_ova_y_train)
        print("Resampled dataset shape %s" % Counter(SC_y_res))

    if imb_technique == "ROS":
        from imblearn.over_sampling import RandomOverSampler
        print("Original dataset shape %s" % Counter(AP_ova_y_train))
        AP_ros = RandomOverSampler()
        AP_X_res, AP_y_res = AP_ros.fit_resample(AP_ova_X_train, AP_ova_y_train)
        print("Resampled dataset shape %s" % Counter(AP_y_res))
        # CF_ros = RandomOverSampler()
        # CF_X_res, CF_y_res = CF_ros.fit_resample(CF_ova_X_train, CF_ova_y_train)
        # IFN_ros = RandomOverSampler()
        # IFN_X_res, IFN_y_res = IFN_ros.fit_resample(IFN_ova_X_train, IFN_ova_y_train)
        print("Original dataset shape %s" % Counter(PM_ova_y_train))
        PM_ros = RandomOverSampler()
        PM_X_res, PM_y_res = PM_ros.fit_resample(PM_ova_X_train, PM_ova_y_train)
        print("Resampled dataset shape %s" % Counter(PM_y_res))
        # SF_ros = RandomOverSampler()
        # SF_X_res, SF_y_res = SF_ros.fit_resample(SF_ova_X_train, SF_ova_y_train)
        print("Original dataset shape %s" % Counter(SC_ova_y_train))
        SC_ros = RandomOverSampler()
        SC_X_res, SC_y_res = SC_ros.fit_resample(SC_ova_X_train, SC_ova_y_train)
        print("Resampled dataset shape %s" % Counter(SC_y_res))

    if imb_technique == "RUS":
        from imblearn.under_sampling import RandomUnderSampler
        print("Original dataset shape %s" % Counter(AP_ova_y_train))
        AP_rus = RandomUnderSampler()
        AP_X_res, AP_y_res = AP_rus.fit_resample(AP_ova_X_train, AP_ova_y_train)
        print("Resampled dataset shape %s" % Counter(AP_y_res))
        # CF_rus = RandomUnderSampler()
        # CF_X_res, CF_y_res = CF_rus.fit_resample(CF_ova_X_train, CF_ova_y_train)
        # IFN_rus = RandomUnderSampler()
        # IFN_X_res, IFN_y_res = IFN_rus.fit_resample(IFN_ova_X_train, IFN_ova_y_train)
        print("Original dataset shape %s" % Counter(PM_ova_y_train))
        PM_rus = RandomUnderSampler()
        PM_X_res, PM_y_res = PM_rus.fit_resample(PM_ova_X_train, PM_ova_y_train)
        print("Resampled dataset shape %s" % Counter(PM_y_res))
        # SF_rus = RandomUnderSampler()
        # SF_X_res, SF_y_res = SF_rus.fit_resample(SF_ova_X_train, SF_ova_y_train)
        print("Original dataset shape %s" % Counter(SC_ova_y_train))
        SC_rus = RandomUnderSampler()
        SC_X_res, SC_y_res = SC_rus.fit_resample(SC_ova_X_train, SC_ova_y_train)
        print("Resampled dataset shape %s" % Counter(SC_y_res))

    #Below codes are for the implementation of deep neural network training
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import RandomizedSearchCV
    import itertools

    first_digit_parameters = [x for x in itertools.product((5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100), repeat=1)]
    all_digit_parameters = first_digit_parameters
    learning_rate_init_parameters = [0.1, 0.01, 0.001]
    parameters = {'hidden_layer_sizes': all_digit_parameters,
                  'learning_rate_init': learning_rate_init_parameters}
    print(AP_y_res)
    print(PM_y_res)
    print(SC_y_res)
    np.savetxt("AP_y_res.csv", AP_y_res, fmt='%s')
    np.savetxt("PM_y_res.csv", PM_y_res, fmt='%s')
    np.savetxt("SC_y_res.csv", SC_y_res, fmt='%s')


    dnn_AP = MLPClassifier(max_iter=10000, activation='relu')
    dnn_AP_clf = RandomizedSearchCV(dnn_AP, parameters, n_jobs=-1, cv=5)
    dnn_AP_clf.fit(AP_X_res, AP_y_res)
    print(dnn_AP_clf.best_params_)
    dnn_PM = MLPClassifier(max_iter=10000, activation='relu')
    dnn_PM_clf = RandomizedSearchCV(dnn_PM, parameters, n_jobs=-1, cv=5)
    dnn_PM_clf.fit(PM_X_res, PM_y_res)
    print(dnn_PM_clf.best_params_)
    dnn_SC = MLPClassifier(max_iter=10000, activation='relu')
    dnn_SC_clf = RandomizedSearchCV(dnn_SC, parameters, n_jobs=-1, cv=5)
    dnn_SC_clf.fit(SC_X_res, SC_y_res)
    print(dnn_SC_clf.best_params_)

    # Below codes are for the implementation of logistic regression training
    from sklearn.linear_model import LogisticRegression
    solver = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
    tol = [1e-2, 1e-3, 1e-4, 1e-5]
    reg_strength = [0.5, 1.0, 1.5]
    parameters = {'solver': solver,
	          'tol': tol,
	          'C': reg_strength}
    lr_AP = LogisticRegression()
    lr_AP_clf = RandomizedSearchCV(lr_AP, parameters, n_jobs = -1, cv = 5)
    lr_AP_clf.fit(AP_X_res, AP_y_res)
    print(lr_AP_clf.best_params_)
    lr_PM = LogisticRegression()
    lr_PM_clf = RandomizedSearchCV(lr_PM, parameters, n_jobs = -1, cv = 5)
    lr_PM_clf.fit(PM_X_res, PM_y_res)
    print(lr_PM_clf.best_params_)
    lr_SC = LogisticRegression()
    lr_SC_clf = RandomizedSearchCV(lr_SC, parameters, n_jobs = -1, cv = 5)
    lr_SC_clf.fit(SC_X_res, SC_y_res)
    print(lr_SC_clf.best_params_)

    # Below codes are for the implementation of Gaussian Naive Bayes training
    from sklearn.naive_bayes import GaussianNB
    #In Gaussian NB, 'var_smoothing' parameter optimization makes convergence errors
    nb_AP_clf = GaussianNB()
    nb_AP_clf.fit(AP_X_res, AP_y_res)
    nb_PM_clf = GaussianNB()
    nb_PM_clf.fit(PM_X_res, PM_y_res)
    nb_SC_clf = GaussianNB()
    nb_SC_clf.fit(SC_X_res, SC_y_res)

    # Below codes are for the implementation of random forest training
    from sklearn.ensemble import RandomForestClassifier
    n_tree = [50, 100, 200, 300, 400, 500, 600, 700]
    max_depth = [10, 20, 30, 40, 50, 60, 70]
    min_samples_split = [5, 10, 15, 20, 25, 30]
    parameters = {'n_estimators': n_tree,
		  'max_depth': max_depth,
		  'min_samples_split': min_samples_split}
    rf_AP = RandomForestClassifier()
    rf_AP_clf = RandomizedSearchCV(rf_AP, parameters, n_jobs = -1, cv=5)
    rf_AP_clf.fit(AP_X_res, AP_y_res)
    print(rf_AP_clf.best_params_)
    rf_PM = RandomForestClassifier()
    rf_PM_clf = RandomizedSearchCV(rf_PM, parameters, n_jobs = -1, cv=5)
    rf_PM_clf.fit(PM_X_res, PM_y_res)
    print(rf_PM_clf.best_params_)
    rf_SC = RandomForestClassifier()
    rf_SC_clf = RandomizedSearchCV(rf_SC, parameters, n_jobs = -1, cv=5)
    rf_SC_clf.fit(SC_X_res, SC_y_res)
    print(rf_SC_clf.best_params_)

    # Below codes are for the implementation of support vector machine training
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV
    #reg_param = [0.5, 1.0, 1.5]
    #degree = [1, 2, 3, 4, 5]
    #degree = [1, 2]
    #kernel = ['rbf', 'linear', 'sigmoid']
    #gamma = ['scale', 'auto']
    #tol = [1e-2, 1e-3, 1e-4]
    #tol = [1e-2, 1e-3]
    
    svm_AP = LinearSVC()
    svm_AP_clf = CalibratedClassifierCV(svm_AP, cv = 5)
    #svm_AP_clf = RandomizedSearchCV(svm_AP_clf, parameters, n_jobs = -1, cv = 5)
    svm_AP_clf.fit(AP_X_res, AP_y_res)
    print("svm_AP fitted")
    #print(svm_AP_clf.best_params_)
    svm_PM = LinearSVC()
    svm_PM_clf = CalibratedClassifierCV(svm_PM, cv = 5)
    #svm_PM_clf = RandomizedSearchCV(svm_PM_clf, parameters, n_jobs = -1, cv = 5)
    svm_PM_clf.fit(PM_X_res, PM_y_res)
    print("svm_PM fitted")
    #print(svm_PM_clf.best_params_)
    svm_SC = LinearSVC()
    svm_SC_clf = CalibratedClassifierCV(svm_SC, cv = 5)
    #svm_SC_clf = RandomizedSearchCV(svm_SC_clf, parameters, n_jobs = -1, cv = 5)
    svm_SC_clf.fit(SC_X_res, SC_y_res)
    print("svm_SC fitted")
    #print(svm_SC_clf.best_params_)

    dnn_pred_class_AP = dnn_AP_clf.predict(X_test)
    dnn_pred_prob_AP = dnn_AP_clf.predict_proba(X_test)
    dnn_pred_class_PM = dnn_PM_clf.predict(X_test)
    dnn_pred_prob_PM = dnn_PM_clf.predict_proba(X_test)
    dnn_pred_class_SC = dnn_SC_clf.predict(X_test)
    dnn_pred_prob_SC = dnn_SC_clf.predict_proba(X_test)

    lr_pred_class_AP = lr_AP_clf.predict(X_test)
    lr_pred_prob_AP = lr_AP_clf.predict_proba(X_test)
    lr_pred_class_PM = lr_PM_clf.predict(X_test)
    lr_pred_prob_PM = lr_PM_clf.predict_proba(X_test)
    lr_pred_class_SC = lr_SC_clf.predict(X_test)
    lr_pred_prob_SC = lr_SC_clf.predict_proba(X_test)

    nb_pred_class_AP = nb_AP_clf.predict(X_test)
    nb_pred_prob_AP= nb_AP_clf.predict_proba(X_test)
    nb_pred_class_PM = nb_PM_clf.predict(X_test)
    nb_pred_prob_PM = nb_PM_clf.predict_proba(X_test)
    nb_pred_class_SC = nb_SC_clf.predict(X_test)
    nb_pred_prob_SC = nb_SC_clf.predict_proba(X_test)

    rf_pred_class_AP = rf_AP_clf.predict(X_test)
    rf_pred_prob_AP = rf_AP_clf.predict_proba(X_test)
    rf_pred_class_PM = rf_PM_clf.predict(X_test)
    rf_pred_prob_PM = rf_PM_clf.predict_proba(X_test)
    rf_pred_class_SC = rf_SC_clf.predict(X_test)
    rf_pred_prob_SC = rf_SC_clf.predict_proba(X_test)

    svm_pred_class_AP = svm_AP_clf.predict(X_test)
    svm_pred_prob_AP = svm_AP_clf.predict_proba(X_test)
    svm_pred_class_PM = svm_PM_clf.predict(X_test)
    svm_pred_prob_PM = svm_PM_clf.predict_proba(X_test)
    svm_pred_class_SC = svm_SC_clf.predict(X_test)
    svm_pred_prob_SC = svm_SC_clf.predict_proba(X_test)

    #Below dataframes are generated to store classification result based on predict probabilities
    #This procedure is necessary since this analysis uses one-versus-all classification method
    dnn_prediction = pd.DataFrame(columns=['Prediction'])
    lr_prediction = pd.DataFrame(columns=['Prediction'])
    nb_prediction = pd.DataFrame(columns=['Prediction'])
    rf_prediction = pd.DataFrame(columns=['Prediction'])
    svm_prediction = pd.DataFrame(columns=['Prediction'])

    #Below codes are for aggregating test results from one-versus-all deep neural network training
    for i in range(0, len(y_test)):
        dnn_AP_index = 0
        # dnn_CF_index = 0
        # dnn_IFN_index = 0
        dnn_PM_index = 0
        # dnn_SF_index = 0
        dnn_SC_index = 0
        if dnn_pred_class_AP[i] == "Add penalty":
            if dnn_pred_prob_AP[i][0] >= 0.5:
                dnn_AP_index = 0
            else:
                dnn_AP_index = 1
        elif dnn_pred_class_AP[i] == "Others":
            if dnn_pred_prob_AP[i][0] < 0.5:
                dnn_AP_index = 0
            else:
                dnn_AP_index = 1
        # if dnn_pred_class_CF[i] == "Create Fine":
        #    if dnn_pred_prob_CF[i][0] >= 0.5:
        #        dnn_CF_index = 0
        #    else:
        #        dnn_CF_index = 1
        # elif dnn_pred_class_CF[i] == "Others":
        #    if dnn_pred_prob_CF[i][0] < 0.5:
        #        dnn_CF_index = 0
        #    else:
        #        dnn_CF_index = 1
        # if dnn_pred_class_IFN[i] == "Insert Fine Notification":
        #    if dnn_pred_prob_IFN[i][0] >= 0.5:
        #        dnn_IFN_index = 0
        #    else:
        #        dnn_IFN_index = 1
        # elif dnn_pred_class_IFN[i] == "Others":
        #    if dnn_pred_prob_IFN[i][0] < 0.5:
        #        dnn_IFN_index = 0
        #    else:
        #        dnn_IFN_index = 1
        if dnn_pred_class_PM[i] == "Payment":
            if dnn_pred_prob_PM[i][0] >= 0.5:
                dnn_PM_index = 0
            else:
                dnn_PM_index = 1
        elif dnn_pred_class_PM[i] == "Others":
            if dnn_pred_prob_PM[i][0] < 0.5:
                dnn_PM_index = 0
            else:
                dnn_PM_index = 1
        # if dnn_pred_class_SF[i] == "Send for Credit Collection":
        #    if dnn_pred_prob_SF[i][0] >= 0.5:
        #        dnn_SF_index = 0
        #    else:
        #        dnn_SF_index = 1
        # elif dnn_pred_class_SF[i] == "Others":
        #    if dnn_pred_prob_SF[i][0] < 0.5:
        #        dnn_SF_index = 0
        #    else:
        #        dnn_SF_index = 1
        if dnn_pred_class_SC[i] == "Send for Credit Collection":
            if dnn_pred_prob_SC[i][0] >= 0.5:
                dnn_SC_index = 0
            else:
                dnn_SC_index = 1
        elif dnn_pred_class_SC[i] == "Others":
            if dnn_pred_prob_SC[i][0] < 0.5:
                dnn_SC_index = 0
            else:
                dnn_SC_index = 1
        if dnn_pred_prob_AP[i][dnn_AP_index] == max(dnn_pred_prob_AP[i][dnn_AP_index], dnn_pred_prob_PM[i][dnn_PM_index], dnn_pred_prob_SC[i][dnn_SC_index]):
            dnn_prediction.loc[i] = "Add penalty"
        # elif dnn_pred_prob_CF[i][dnn_CF_index] == max(dnn_pred_prob_AP[i][dnn_AP_index],dnn_pred_prob_PM[i][dnn_PM_index], dnn_pred_prob_SC[i][dnn_SC_index]):
        #    dnn_prediction.loc[i] = "Create Fine"
        # elif dnn_pred_prob_IFN[i][dnn_IFN_index] == max(dnn_pred_prob_AP[i][dnn_AP_index],dnn_pred_prob_PM[i][dnn_PM_index],dnn_pred_prob_SC[i][dnn_SC_index]):
        #    dnn_prediction.loc[i] = "Insert Fine Notification"
        elif dnn_pred_prob_PM[i][dnn_PM_index] == max(dnn_pred_prob_AP[i][dnn_AP_index], dnn_pred_prob_PM[i][dnn_PM_index], dnn_pred_prob_SC[i][dnn_SC_index]):
            dnn_prediction.loc[i] = "Payment"
        # elif dnn_pred_prob_SF[i][dnn_SF_index] == max(dnn_pred_prob_AP[i][dnn_AP_index],dnn_pred_prob_PM[i][dnn_PM_index],dnn_pred_prob_SC[i][dnn_SC_index]):
        #    dnn_prediction.loc[i] = "Send Fine"
        elif dnn_pred_prob_SC[i][dnn_SC_index] == max(dnn_pred_prob_AP[i][dnn_AP_index], dnn_pred_prob_PM[i][dnn_PM_index], dnn_pred_prob_SC[i][dnn_SC_index]):
            dnn_prediction.loc[i] = "Send for Credit Collection"


    def get_precision(dnn_conf_matrix):
        dnn_tp_1 = dnn_conf_matrix[0][0]
        dnn_tp_2 = dnn_conf_matrix[1][1]
        dnn_tp_3 = dnn_conf_matrix[2][2]
        dnn_fp_1 = dnn_conf_matrix[1][0] + dnn_conf_matrix[2][0]
        dnn_fp_2 = dnn_conf_matrix[0][1] + dnn_conf_matrix[2][1]
        dnn_fp_3 = dnn_conf_matrix[0][2] + dnn_conf_matrix[1][2]

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

        dnn_precision_avg = (dnn_precision_1 + dnn_precision_2 + dnn_precision_3) / 3
        return dnn_precision_avg


    def get_recall_pen_1(dnn_conf_matrix):
        dnn_tp_1 = dnn_conf_matrix[0][0]
        dnn_tp_2 = dnn_conf_matrix[1][1]
        dnn_tp_3 = dnn_conf_matrix[2][2]
        dnn_fn_1 = dnn_conf_matrix[0][1] + dnn_conf_matrix[0][2]
        dnn_fn_2 = dnn_conf_matrix[1][0] + dnn_conf_matrix[1][2]
        dnn_fn_3 = dnn_conf_matrix[2][0] + dnn_conf_matrix[2][1]
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
        dnn_recall_avg_pen_1 = (dnn_recall_1 + dnn_recall_2 + dnn_recall_3) / (3+1-1)
        return dnn_recall_avg_pen_1

    def get_recall_pen_5(dnn_conf_matrix):
        dnn_tp_1 = dnn_conf_matrix[0][0]
        dnn_tp_2 = dnn_conf_matrix[1][1]
        dnn_tp_3 = dnn_conf_matrix[2][2]
        dnn_fn_1 = dnn_conf_matrix[0][1] + dnn_conf_matrix[0][2]
        dnn_fn_2 = dnn_conf_matrix[1][0] + dnn_conf_matrix[1][2]
        dnn_fn_3 = dnn_conf_matrix[2][0] + dnn_conf_matrix[2][1]
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
        dnn_recall_avg_pen_5 = (dnn_recall_1 + (5*dnn_recall_2) + dnn_recall_3) / (3+5-1)
        return dnn_recall_avg_pen_5


    from sklearn.metrics import classification_report, confusion_matrix

    dnn_conf_matrix = confusion_matrix(y_test, dnn_prediction, labels = np.unique(data['ACT_4']))
    print("dnn_confusion matrix:")
    print(dnn_conf_matrix)
    dnn_precision = get_precision(dnn_conf_matrix)
    dnn_recall_pen_1 = get_recall_pen_1(dnn_conf_matrix)
    dnn_recall_pen_5 = get_recall_pen_5(dnn_conf_matrix)
    dnn_f1_score_pen_1 = 2 * (dnn_precision * dnn_recall_pen_1) / (dnn_precision + dnn_recall_pen_1)
    dnn_f1_score_pen_5 = 2 * (dnn_precision * dnn_recall_pen_5) / (dnn_precision + dnn_recall_pen_5)
    dnn_ovr_accuracy = (dnn_conf_matrix[0][0] + dnn_conf_matrix[1][1] + dnn_conf_matrix[2][2]) / (
                sum(dnn_conf_matrix[0]) + sum(dnn_conf_matrix[1]) + sum(dnn_conf_matrix[2]))
    print("dnn_f1 score of pen 1 is:")
    print(dnn_f1_score_pen_1)
    print("dnn_f1 score of pen 5 is:")
    print(dnn_f1_score_pen_5)
    print("dnn_overall accuracy is:")
    print(dnn_ovr_accuracy)
    dnn_conf_matrix = pd.DataFrame(dnn_conf_matrix)
    dnn_conf_matrix.to_csv('conf_matrix_' + imb_technique + '_dnn_road_traffic_' + str(nsplits) + 'foldcv_' + str(repeat + 1) + '.csv', header=False, index=False)  # First repetition
    #dnn_conf_matrix.to_csv('conf_matrix_' + imb_technique + '_dnn_road_traffic_' + str(nsplits) + 'foldcv_' + str(repeat + 5) + '.csv', header=False, index=False)  # Second repetition
    dnn_f1_score_pen_1_kfoldcv[repeat] = dnn_f1_score_pen_1
    dnn_f1_score_pen_5_kfoldcv[repeat] = dnn_f1_score_pen_5
    dnn_ovr_accuracy_kfoldcv[repeat] = dnn_ovr_accuracy

    #Below codes are for aggregating test results from one-versus-all logistic regression training
    for i in range(0, len(y_test)):
        lr_AP_index = 0
        # lr_CF_index = 0
        # lr_IFN_index = 0
        lr_PM_index = 0
        # lr_SF_index = 0
        lr_SC_index = 0
        if lr_pred_class_AP[i] == "Add penalty":
            if lr_pred_prob_AP[i][0] >= 0.5:
                lr_AP_index = 0
            else:
                lr_AP_index = 1
        elif lr_pred_class_AP[i] == "Others":
            if lr_pred_prob_AP[i][0] < 0.5:
                lr_AP_index = 0
            else:
                lr_AP_index = 1
        # if lr_pred_class_CF[i] == "Create Fine":
        #    if lr_pred_prob_CF[i][0] >= 0.5:
        #        lr_CF_index = 0
        #    else:
        #        lr_CF_index = 1
        # elif lr_pred_class_CF[i] == "Others":
        #    if lr_pred_prob_CF[i][0] < 0.5:
        #        lr_CF_index = 0
        #    else:
        #        lr_CF_index = 1
        # if lr_pred_class_IFN[i] == "Insert Fine Notification":
        #    if lr_pred_prob_IFN[i][0] >= 0.5:
        #        lr_IFN_index = 0
        #    else:
        #        lr_IFN_index = 1
        # elif lr_pred_class_IFN[i] == "Others":
        #    if lr_pred_prob_IFN[i][0] < 0.5:
        #        lr_IFN_index = 0
        #    else:
        #        lr_IFN_index = 1
        if lr_pred_class_PM[i] == "Payment":
            if lr_pred_prob_PM[i][0] >= 0.5:
                lr_PM_index = 0
            else:
                lr_PM_index = 1
        elif lr_pred_class_PM[i] == "Others":
            if lr_pred_prob_PM[i][0] < 0.5:
                lr_PM_index = 0
            else:
                lr_PM_index = 1
        # if lr_pred_class_SF[i] == "Send for Credit Collection":
        #    if lr_pred_prob_SF[i][0] >= 0.5:
        #        lr_SF_index = 0
        #    else:
        #        lr_SF_index = 1
        # elif lr_pred_class_SF[i] == "Others":
        #    if lr_pred_prob_SF[i][0] < 0.5:
        #        lr_SF_index = 0
        #    else:
        #        lr_SF_index = 1
        if lr_pred_class_SC[i] == "Send for Credit Collection":
            if lr_pred_prob_SC[i][0] >= 0.5:
                lr_SC_index = 0
            else:
                lr_SC_index = 1
        elif lr_pred_class_SC[i] == "Others":
            if lr_pred_prob_SC[i][0] < 0.5:
                lr_SC_index = 0
            else:
                lr_SC_index = 1
        if lr_pred_prob_AP[i][lr_AP_index] == max(lr_pred_prob_AP[i][lr_AP_index], lr_pred_prob_PM[i][lr_PM_index],
                                                    lr_pred_prob_SC[i][lr_SC_index]):
            lr_prediction.loc[i] = "Add penalty"
        # elif lr_pred_prob_CF[i][lr_CF_index] == max(lr_pred_prob_AP[i][lr_AP_index],lr_pred_prob_PM[i][lr_PM_index],lr_pred_prob_SC[i][lr_SC_index]):
        #    lr_prediction.loc[i] = "Create Fine"
        # elif lr_pred_prob_IFN[i][lr_IFN_index] == max(lr_pred_prob_AP[i][lr_AP_index],lr_pred_prob_PM[i][lr_PM_index],lr_pred_prob_SC[i][lr_SC_index]):
        #    lr_prediction.loc[i] = "Insert Fine Notification"
        elif lr_pred_prob_PM[i][lr_PM_index] == max(lr_pred_prob_AP[i][lr_AP_index], lr_pred_prob_PM[i][lr_PM_index],
                                                      lr_pred_prob_SC[i][lr_SC_index]):
            lr_prediction.loc[i] = "Payment"
        # elif lr_pred_prob_SF[i][lr_SF_index] == max(lr_pred_prob_AP[i][lr_AP_index],lr_pred_prob_PM[i][lr_PM_index],lr_pred_prob_SC[i][lr_SC_index]):
        #    lr_prediction.loc[i] = "Send Fine"
        elif lr_pred_prob_SC[i][lr_SC_index] == max(lr_pred_prob_AP[i][lr_AP_index], lr_pred_prob_PM[i][lr_PM_index],
                                                      lr_pred_prob_SC[i][lr_SC_index]):
            lr_prediction.loc[i] = "Send for Credit Collection"


    def get_precision(lr_conf_matrix):
        lr_tp_1 = lr_conf_matrix[0][0]
        lr_tp_2 = lr_conf_matrix[1][1]
        lr_tp_3 = lr_conf_matrix[2][2]
        lr_fp_1 = lr_conf_matrix[1][0] + lr_conf_matrix[2][0]
        lr_fp_2 = lr_conf_matrix[0][1] + lr_conf_matrix[2][1]
        lr_fp_3 = lr_conf_matrix[0][2] + lr_conf_matrix[1][2]

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

        lr_precision_avg = (lr_precision_1 + lr_precision_2 + lr_precision_3) / 3
        return lr_precision_avg


    def get_recall_pen_1(lr_conf_matrix):
        lr_tp_1 = lr_conf_matrix[0][0]
        lr_tp_2 = lr_conf_matrix[1][1]
        lr_tp_3 = lr_conf_matrix[2][2]
        lr_fn_1 = lr_conf_matrix[0][1] + lr_conf_matrix[0][2]
        lr_fn_2 = lr_conf_matrix[1][0] + lr_conf_matrix[1][2]
        lr_fn_3 = lr_conf_matrix[2][0] + lr_conf_matrix[2][1]
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
        lr_recall_avg_pen_1 = (lr_recall_1 + lr_recall_2 + lr_recall_3) / (3+1-1)
        return lr_recall_avg_pen_1

    def get_recall_pen_5(lr_conf_matrix):
        lr_tp_1 = lr_conf_matrix[0][0]
        lr_tp_2 = lr_conf_matrix[1][1]
        lr_tp_3 = lr_conf_matrix[2][2]
        lr_fn_1 = lr_conf_matrix[0][1] + lr_conf_matrix[0][2]
        lr_fn_2 = lr_conf_matrix[1][0] + lr_conf_matrix[1][2]
        lr_fn_3 = lr_conf_matrix[2][0] + lr_conf_matrix[2][1]
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
        lr_recall_avg_pen_5 = (lr_recall_1 + (5*lr_recall_2) + lr_recall_3) / (3+5-1)
        return lr_recall_avg_pen_5



    from sklearn.metrics import classification_report, confusion_matrix

    lr_conf_matrix = confusion_matrix(y_test, lr_prediction, labels = np.unique(data['ACT_4']))
    print("lr_confusion matrix:")
    print(lr_conf_matrix)
    lr_precision = get_precision(lr_conf_matrix)
    lr_recall_pen_1 = get_recall_pen_1(lr_conf_matrix)
    lr_recall_pen_5 = get_recall_pen_5(lr_conf_matrix)
    lr_f1_score_pen_1 = 2 * (lr_precision * lr_recall_pen_1) / (lr_precision + lr_recall_pen_1)
    lr_f1_score_pen_5 = 2 * (lr_precision * lr_recall_pen_5) / (lr_precision + lr_recall_pen_5)
    lr_ovr_accuracy = (lr_conf_matrix[0][0] + lr_conf_matrix[1][1] + lr_conf_matrix[2][2]) / (
                sum(lr_conf_matrix[0]) + sum(lr_conf_matrix[1]) + sum(lr_conf_matrix[2]))
    print("lr_f1 score of pen 1 is:")
    print(lr_f1_score_pen_1)
    print("lr_f1 score of pen 5 is:")
    print(lr_f1_score_pen_5)
    print("lr_overall accuracy is:")
    print(lr_ovr_accuracy)
    lr_conf_matrix = pd.DataFrame(lr_conf_matrix)
    lr_conf_matrix.to_csv('conf_matrix_' + imb_technique + '_lr_road_traffic_' + str(nsplits) + 'foldcv_' + str(repeat + 1) + '.csv', header=False, index=False)  # First repetition
    #lr_conf_matrix.to_csv('conf_matrix_'+imb_technique+'_lr_road_traffic_'+ str(nsplits) +'foldcv_' + str(repeat+6)+'.csv',header=False,index=False) #Second repetition
    lr_f1_score_pen_1_kfoldcv[repeat] = lr_f1_score_pen_1
    lr_f1_score_pen_5_kfoldcv[repeat] = lr_f1_score_pen_5
    lr_ovr_accuracy_kfoldcv[repeat] = lr_ovr_accuracy

    #Below codes are for aggregating test results from one-versus-all Naive Bayes training
    for i in range(0, len(y_test)):
        nb_AP_index = 0
        # nb_CF_index = 0
        # nb_IFN_index = 0
        nb_PM_index = 0
        # nb_SF_index = 0
        nb_SC_index = 0
        if nb_pred_class_AP[i] == "Add penalty":
            if nb_pred_prob_AP[i][0] >= 0.5:
                nb_AP_index = 0
            else:
                nb_AP_index = 1
        elif nb_pred_class_AP[i] == "Others":
            if nb_pred_prob_AP[i][0] < 0.5:
                nb_AP_index = 0
            else:
                nb_AP_index = 1
        # if nb_pred_class_CF[i] == "Create Fine":
        #    if nb_pred_prob_CF[i][0] >= 0.5:
        #        nb_CF_index = 0
        #    else:
        #        nb_CF_index = 1
        # elif nb_pred_class_CF[i] == "Others":
        #    if nb_pred_prob_CF[i][0] < 0.5:
        #        nb_CF_index = 0
        #    else:
        #        nb_CF_index = 1
        # if nb_pred_class_IFN[i] == "Insert Fine Notification":
        #    if nb_pred_prob_IFN[i][0] >= 0.5:
        #        nb_IFN_index = 0
        #    else:
        #        nb_IFN_index = 1
        # elif nb_pred_class_IFN[i] == "Others":
        #    if nb_pred_prob_IFN[i][0] < 0.5:
        #        nb_IFN_index = 0
        #    else:
        #        nb_IFN_index = 1
        if nb_pred_class_PM[i] == "Payment":
            if nb_pred_prob_PM[i][0] >= 0.5:
                nb_PM_index = 0
            else:
                nb_PM_index = 1
        elif nb_pred_class_PM[i] == "Others":
            if nb_pred_prob_PM[i][0] < 0.5:
                nb_PM_index = 0
            else:
                nb_PM_index = 1
        # if nb_pred_class_SF[i] == "Send for Credit Collection":
        #    if nb_pred_prob_SF[i][0] >= 0.5:
        #        nb_SF_index = 0
        #    else:
        #        nb_SF_index = 1
        # elif nb_pred_class_SF[i] == "Others":
        #    if nb_pred_prob_SF[i][0] < 0.5:
        #        nb_SF_index = 0
        #    else:
        #        nb_SF_index = 1
        if nb_pred_class_SC[i] == "Send for Credit Collection":
            if nb_pred_prob_SC[i][0] >= 0.5:
                nb_SC_index = 0
            else:
                nb_SC_index = 1
        elif nb_pred_class_SC[i] == "Others":
            if nb_pred_prob_SC[i][0] < 0.5:
                nb_SC_index = 0
            else:
                nb_SC_index = 1
        if nb_pred_prob_AP[i][nb_AP_index] == max(nb_pred_prob_AP[i][nb_AP_index], nb_pred_prob_PM[i][nb_PM_index],
                                                    nb_pred_prob_SC[i][nb_SC_index]):
            nb_prediction.loc[i] = "Add penalty"
        # elif nb_pred_prob_CF[i][nb_CF_index] == max(nb_pred_prob_AP[i][nb_AP_index],nb_pred_prob_PM[i][nb_PM_index],nb_pred_prob_SC[i][nb_SC_index]):
        #    nb_prediction.loc[i] = "Create Fine"
        # elif nb_pred_prob_IFN[i][nb_IFN_index] == max(nb_pred_prob_AP[i][nb_AP_index],nb_pred_prob_PM[i][nb_PM_index],nb_pred_prob_SC[i][nb_SC_index]):
        #    nb_prediction.loc[i] = "Insert Fine Notification"
        elif nb_pred_prob_PM[i][nb_PM_index] == max(nb_pred_prob_AP[i][nb_AP_index], nb_pred_prob_PM[i][nb_PM_index],
                                                      nb_pred_prob_SC[i][nb_SC_index]):
            nb_prediction.loc[i] = "Payment"
        # elif nb_pred_prob_SF[i][nb_SF_index] == max(nb_pred_prob_AP[i][nb_AP_index],nb_pred_prob_PM[i][nb_PM_index],nb_pred_prob_SC[i][nb_SC_index]):
        #    nb_prediction.loc[i] = "Send Fine"
        elif nb_pred_prob_SC[i][nb_SC_index] == max(nb_pred_prob_AP[i][nb_AP_index], nb_pred_prob_PM[i][nb_PM_index],
                                                      nb_pred_prob_SC[i][nb_SC_index]):
            nb_prediction.loc[i] = "Send for Credit Collection"


    def get_precision(nb_conf_matrix):
        nb_tp_1 = nb_conf_matrix[0][0]
        nb_tp_2 = nb_conf_matrix[1][1]
        nb_tp_3 = nb_conf_matrix[2][2]
        nb_fp_1 = nb_conf_matrix[1][0] + nb_conf_matrix[2][0]
        nb_fp_2 = nb_conf_matrix[0][1] + nb_conf_matrix[2][1]
        nb_fp_3 = nb_conf_matrix[0][2] + nb_conf_matrix[1][2]

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

        nb_precision_avg = (nb_precision_1 + nb_precision_2 + nb_precision_3) / 3
        return nb_precision_avg


    def get_recall_pen_1(nb_conf_matrix):
        nb_tp_1 = nb_conf_matrix[0][0]
        nb_tp_2 = nb_conf_matrix[1][1]
        nb_tp_3 = nb_conf_matrix[2][2]
        nb_fn_1 = nb_conf_matrix[0][1] + nb_conf_matrix[0][2]
        nb_fn_2 = nb_conf_matrix[1][0] + nb_conf_matrix[1][2]
        nb_fn_3 = nb_conf_matrix[2][0] + nb_conf_matrix[2][1]
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
        nb_recall_avg_pen_1 = (nb_recall_1 + nb_recall_2 + nb_recall_3) / (3+1-1)
        return nb_recall_avg_pen_1

    def get_recall_pen_5(nb_conf_matrix):
        nb_tp_1 = nb_conf_matrix[0][0]
        nb_tp_2 = nb_conf_matrix[1][1]
        nb_tp_3 = nb_conf_matrix[2][2]
        nb_fn_1 = nb_conf_matrix[0][1] + nb_conf_matrix[0][2]
        nb_fn_2 = nb_conf_matrix[1][0] + nb_conf_matrix[1][2]
        nb_fn_3 = nb_conf_matrix[2][0] + nb_conf_matrix[2][1]
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
        nb_recall_avg_pen_5 = (nb_recall_1 + nb_recall_2 + nb_recall_3) / (3+5-1)
        return nb_recall_avg_pen_5



    from sklearn.metrics import classification_report, confusion_matrix

    nb_conf_matrix = confusion_matrix(y_test, nb_prediction, labels = np.unique(data['ACT_4']))
    print("nb_confusion matrix:")
    print(nb_conf_matrix)
    nb_precision = get_precision(nb_conf_matrix)
    nb_recall_pen_1 = get_recall_pen_1(nb_conf_matrix)
    nb_recall_pen_5 = get_recall_pen_5(nb_conf_matrix)
    nb_f1_score_pen_1 = 2 * (nb_precision * nb_recall_pen_1) / (nb_precision + nb_recall_pen_1)
    nb_f1_score_pen_5 = 2 * (nb_precision * nb_recall_pen_5) / (nb_precision + nb_recall_pen_5)
    nb_ovr_accuracy = (nb_conf_matrix[0][0] + nb_conf_matrix[1][1] + nb_conf_matrix[2][2]) / (
                sum(nb_conf_matrix[0]) + sum(nb_conf_matrix[1]) + sum(nb_conf_matrix[2]))
    nb_f1_score_pen_5 = 2 * (nb_precision * nb_recall_pen_5) / (nb_precision + nb_recall_pen_5)
    nb_ovr_accuracy = (nb_conf_matrix[0][0] + nb_conf_matrix[1][1] + nb_conf_matrix[2][2]) / (
                sum(nb_conf_matrix[0]) + sum(nb_conf_matrix[1]) + sum(nb_conf_matrix[2]))
    print("nb_f1 score of pen 1 is:")
    print(nb_f1_score_pen_1)
    print("nb_f1 score of pen 5 is:")
    print(nb_f1_score_pen_5)
    print("nb_overall accuracy is:")
    print(nb_ovr_accuracy)
    nb_conf_matrix = pd.DataFrame(nb_conf_matrix)
    nb_conf_matrix.to_csv('conf_matrix_' + imb_technique + '_nb_road_traffic_' + str(nsplits) + 'foldcv_' + str(repeat + 1) + '.csv', header=False, index=False)  # First repetition
    #nb_conf_matrix.to_csv('conf_matrix_'+imb_technique+'_nb_road_traffic_'+ str(nsplits) +'foldcv_' + str(repeat+6)+'.csv',header=False,index=False) #Second repetition
    nb_f1_score_pen_1_kfoldcv[repeat] = nb_f1_score_pen_1
    nb_f1_score_pen_5_kfoldcv[repeat] = nb_f1_score_pen_5
    nb_ovr_accuracy_kfoldcv[repeat] = nb_ovr_accuracy


    for i in range(0, len(y_test)):
        rf_AP_index = 0
        # rf_CF_index = 0
        # rf_IFN_index = 0
        rf_PM_index = 0
        # rf_SF_index = 0
        rf_SC_index = 0
        if rf_pred_class_AP[i] == "Add penalty":
            if rf_pred_prob_AP[i][0] >= 0.5:
                rf_AP_index = 0
            else:
                rf_AP_index = 1
        elif rf_pred_class_AP[i] == "Others":
            if rf_pred_prob_AP[i][0] < 0.5:
                rf_AP_index = 0
            else:
                rf_AP_index = 1
        # if rf_pred_class_CF[i] == "Create Fine":
        #    if rf_pred_prob_CF[i][0] >= 0.5:
        #        rf_CF_index = 0
        #    else:
        #        rf_CF_index = 1
        # elif rf_pred_class_CF[i] == "Others":
        #    if rf_pred_prob_CF[i][0] < 0.5:
        #        rf_CF_index = 0
        #    else:
        #        rf_CF_index = 1
        # if rf_pred_class_IFN[i] == "Insert Fine Notification":
        #    if rf_pred_prob_IFN[i][0] >= 0.5:
        #        rf_IFN_index = 0
        #    else:
        #        rf_IFN_index = 1
        # elif rf_pred_class_IFN[i] == "Others":
        #    if rf_pred_prob_IFN[i][0] < 0.5:
        #        rf_IFN_index = 0
        #    else:
        #        rf_IFN_index = 1
        if rf_pred_class_PM[i] == "Payment":
            if rf_pred_prob_PM[i][0] >= 0.5:
                rf_PM_index = 0
            else:
                rf_PM_index = 1
        elif rf_pred_class_PM[i] == "Others":
            if rf_pred_prob_PM[i][0] < 0.5:
                rf_PM_index = 0
            else:
                rf_PM_index = 1
        # if rf_pred_class_SF[i] == "Send for Credit Collection":
        #    if rf_pred_prob_SF[i][0] >= 0.5:
        #        rf_SF_index = 0
        #    else:
        #        rf_SF_index = 1
        # elif rf_pred_class_SF[i] == "Others":
        #    if rf_pred_prob_SF[i][0] < 0.5:
        #        rf_SF_index = 0
        #    else:
        #        rf_SF_index = 1
        if rf_pred_class_SC[i] == "Send for Credit Collection":
            if rf_pred_prob_SC[i][0] >= 0.5:
                rf_SC_index = 0
            else:
                rf_SC_index = 1
        elif rf_pred_class_SC[i] == "Others":
            if rf_pred_prob_SC[i][0] < 0.5:
                rf_SC_index = 0
            else:
                rf_SC_index = 1
        if rf_pred_prob_AP[i][rf_AP_index] == max(rf_pred_prob_AP[i][rf_AP_index], rf_pred_prob_PM[i][rf_PM_index],
                                                    rf_pred_prob_SC[i][rf_SC_index]):
            rf_prediction.loc[i] = "Add penalty"
        # elif rf_pred_prob_CF[i][rf_CF_index] == max(rf_pred_prob_AP[i][rf_AP_index],rf_pred_prob_PM[i][rf_PM_index],rf_pred_prob_SC[i][rf_SC_index]):
        #    rf_prediction.loc[i] = "Create Fine"
        # elif rf_pred_prob_IFN[i][rf_IFN_index] == max(rf_pred_prob_AP[i][rf_AP_index],rf_pred_prob_PM[i][rf_PM_index],rf_pred_prob_SC[i][rf_SC_index]):
        #    rf_prediction.loc[i] = "Insert Fine Notification"
        elif rf_pred_prob_PM[i][rf_PM_index] == max(rf_pred_prob_AP[i][rf_AP_index], rf_pred_prob_PM[i][rf_PM_index],
                                                      rf_pred_prob_SC[i][rf_SC_index]):
            rf_prediction.loc[i] = "Payment"
        # elif rf_pred_prob_SF[i][rf_SF_index] == max(rf_pred_prob_AP[i][rf_AP_index],rf_pred_prob_PM[i][rf_PM_index],rf_pred_prob_SC[i][rf_SC_index]):
        #    rf_prediction.loc[i] = "Send Fine"
        elif rf_pred_prob_SC[i][rf_SC_index] == max(rf_pred_prob_AP[i][rf_AP_index], rf_pred_prob_PM[i][rf_PM_index],
                                                      rf_pred_prob_SC[i][rf_SC_index]):
            rf_prediction.loc[i] = "Send for Credit Collection"


    def get_precision(rf_conf_matrix):
        rf_tp_1 = rf_conf_matrix[0][0]
        rf_tp_2 = rf_conf_matrix[1][1]
        rf_tp_3 = rf_conf_matrix[2][2]
        rf_fp_1 = rf_conf_matrix[1][0] + rf_conf_matrix[2][0]
        rf_fp_2 = rf_conf_matrix[0][1] + rf_conf_matrix[2][1]
        rf_fp_3 = rf_conf_matrix[0][2] + rf_conf_matrix[1][2]

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

        rf_precision_avg = (rf_precision_1 + rf_precision_2 + rf_precision_3) / 3
        return rf_precision_avg


    def get_recall_pen_1(rf_conf_matrix):
        rf_tp_1 = rf_conf_matrix[0][0]
        rf_tp_2 = rf_conf_matrix[1][1]
        rf_tp_3 = rf_conf_matrix[2][2]
        rf_fn_1 = rf_conf_matrix[0][1] + rf_conf_matrix[0][2]
        rf_fn_2 = rf_conf_matrix[1][0] + rf_conf_matrix[1][2]
        rf_fn_3 = rf_conf_matrix[2][0] + rf_conf_matrix[2][1]
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
        rf_recall_avg_pen_1 = (rf_recall_1 + rf_recall_2 + rf_recall_3) / (3+1-1)
        return rf_recall_avg_pen_1

    def get_recall_pen_5(rf_conf_matrix):
        rf_tp_1 = rf_conf_matrix[0][0]
        rf_tp_2 = rf_conf_matrix[1][1]
        rf_tp_3 = rf_conf_matrix[2][2]
        rf_fn_1 = rf_conf_matrix[0][1] + rf_conf_matrix[0][2]
        rf_fn_2 = rf_conf_matrix[1][0] + rf_conf_matrix[1][2]
        rf_fn_3 = rf_conf_matrix[2][0] + rf_conf_matrix[2][1]
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
        rf_recall_avg_pen_5 = (rf_recall_1 + (5*rf_recall_2) + rf_recall_3) / (3+5-1)
        return rf_recall_avg_pen_5



    from sklearn.metrics import classification_report, confusion_matrix

    rf_conf_matrix = confusion_matrix(y_test, rf_prediction, labels = np.unique(data['ACT_4']))
    print("rf_confusion matrix:")
    print(rf_conf_matrix)
    rf_precision = get_precision(rf_conf_matrix)
    rf_recall_pen_1 = get_recall_pen_1(rf_conf_matrix)
    rf_recall_pen_5 = get_recall_pen_5(rf_conf_matrix)
    rf_f1_score_pen_1 = 2 * (rf_precision * rf_recall_pen_1) / (rf_precision + rf_recall_pen_1)
    rf_ovr_accuracy = (rf_conf_matrix[0][0] + rf_conf_matrix[1][1] + rf_conf_matrix[2][2]) / (
                sum(rf_conf_matrix[0]) + sum(rf_conf_matrix[1]) + sum(rf_conf_matrix[2]))
    rf_f1_score_pen_5 = 2 * (rf_precision * rf_recall_pen_5) / (rf_precision + rf_recall_pen_5)
    rf_ovr_accuracy = (rf_conf_matrix[0][0] + rf_conf_matrix[1][1] + rf_conf_matrix[2][2]) / (
                sum(rf_conf_matrix[0]) + sum(rf_conf_matrix[1]) + sum(rf_conf_matrix[2]))
    print("rf_f1 score of pen 1 is:")
    print(rf_f1_score_pen_1)
    print("rf_f1 score of pen 5 is:")
    print(rf_f1_score_pen_5)
    print("rf_overall accuracy is:")
    print(rf_ovr_accuracy)
    rf_conf_matrix = pd.DataFrame(rf_conf_matrix)
    rf_conf_matrix.to_csv('conf_matrix_' + imb_technique + '_rf_road_traffic_' + str(nsplits) + 'foldcv_' + str(repeat + 1) + '.csv', header=False, index=False)  # First repetition
    #rf_conf_matrix.to_csv('conf_matrix_'+imb_technique+'_rf_road_traffic_'+ str(nsplits) +'foldcv_' + str(repeat+6)+'.csv',header=False,index=False) #Second repetition
    rf_f1_score_pen_1_kfoldcv[repeat] = rf_f1_score_pen_1
    rf_f1_score_pen_5_kfoldcv[repeat] = rf_f1_score_pen_5
    rf_ovr_accuracy_kfoldcv[repeat] = rf_ovr_accuracy


    for i in range(0, len(y_test)):
        svm_AP_index = 0
        # svm_CF_index = 0
        # svm_IFN_index = 0
        svm_PM_index = 0
        # svm_SF_index = 0
        svm_SC_index = 0
        if svm_pred_class_AP[i] == "Add penalty":
            if svm_pred_prob_AP[i][0] >= 0.5:
                svm_AP_index = 0
            else:
                svm_AP_index = 1
        elif svm_pred_class_AP[i] == "Others":
            if svm_pred_prob_AP[i][0] < 0.5:
                svm_AP_index = 0
            else:
                svm_AP_index = 1
        # if svm_pred_class_CF[i] == "Create Fine":
        #    if svm_pred_prob_CF[i][0] >= 0.5:
        #        svm_CF_index = 0
        #    else:
        #        svm_CF_index = 1
        # elif svm_pred_class_CF[i] == "Others":
        #    if svm_pred_prob_CF[i][0] < 0.5:
        #        svm_CF_index = 0
        #    else:
        #        svm_CF_index = 1
        # if svm_pred_class_IFN[i] == "Insert Fine Notification":
        #    if svm_pred_prob_IFN[i][0] >= 0.5:
        #        svm_IFN_index = 0
        #    else:
        #        svm_IFN_index = 1
        # elif svm_pred_class_IFN[i] == "Others":
        #    if svm_pred_prob_IFN[i][0] < 0.5:
        #        svm_IFN_index = 0
        #    else:
        #        svm_IFN_index = 1
        if svm_pred_class_PM[i] == "Payment":
            if svm_pred_prob_PM[i][0] >= 0.5:
                svm_PM_index = 0
            else:
                svm_PM_index = 1
        elif svm_pred_class_PM[i] == "Others":
            if svm_pred_prob_PM[i][0] < 0.5:
                svm_PM_index = 0
            else:
                svm_PM_index = 1
        # if svm_pred_class_SF[i] == "Send for Credit Collection":
        #    if svm_pred_prob_SF[i][0] >= 0.5:
        #        svm_SF_index = 0
        #    else:
        #        svm_SF_index = 1
        # elif svm_pred_class_SF[i] == "Others":
        #    if svm_pred_prob_SF[i][0] < 0.5:
        #        svm_SF_index = 0
        #    else:
        #        svm_SF_index = 1
        if svm_pred_class_SC[i] == "Send for Credit Collection":
            if svm_pred_prob_SC[i][0] >= 0.5:
                svm_SC_index = 0
            else:
                svm_SC_index = 1
        elif svm_pred_class_SC[i] == "Others":
            if svm_pred_prob_SC[i][0] < 0.5:
                svm_SC_index = 0
            else:
                svm_SC_index = 1
        if svm_pred_prob_AP[i][svm_AP_index] == max(svm_pred_prob_AP[i][svm_AP_index], svm_pred_prob_PM[i][svm_PM_index],
                                                    svm_pred_prob_SC[i][svm_SC_index]):
            svm_prediction.loc[i] = "Add penalty"
        # elif svm_pred_prob_CF[i][svm_CF_index] == max(svm_pred_prob_AP[i][svm_AP_index],svm_pred_prob_PM[i][svm_PM_index],svm_pred_prob_SC[i][svm_SC_index]):
        #    svm_prediction.loc[i] = "Create Fine"
        # elif svm_pred_prob_IFN[i][svm_IFN_index] == max(svm_pred_prob_AP[i][svm_AP_index],svm_pred_prob_PM[i][svm_PM_index],svm_pred_prob_SC[i][svm_SC_index]):
        #    svm_prediction.loc[i] = "Insert Fine Notification"
        elif svm_pred_prob_PM[i][svm_PM_index] == max(svm_pred_prob_AP[i][svm_AP_index], svm_pred_prob_PM[i][svm_PM_index],
                                                      svm_pred_prob_SC[i][svm_SC_index]):
            svm_prediction.loc[i] = "Payment"
        # elif svm_pred_prob_SF[i][svm_SF_index] == max(svm_pred_prob_AP[i][svm_AP_index],svm_pred_prob_PM[i][svm_PM_index],svm_pred_prob_SC[i][svm_SC_index]):
        #    svm_prediction.loc[i] = "Send Fine"
        elif svm_pred_prob_SC[i][svm_SC_index] == max(svm_pred_prob_AP[i][svm_AP_index], svm_pred_prob_PM[i][svm_PM_index],
                                                      svm_pred_prob_SC[i][svm_SC_index]):
            svm_prediction.loc[i] = "Send for Credit Collection"


    def get_precision(svm_conf_matrix):
        svm_tp_1 = svm_conf_matrix[0][0]
        svm_tp_2 = svm_conf_matrix[1][1]
        svm_tp_3 = svm_conf_matrix[2][2]
        svm_fp_1 = svm_conf_matrix[1][0] + svm_conf_matrix[2][0]
        svm_fp_2 = svm_conf_matrix[0][1] + svm_conf_matrix[2][1]
        svm_fp_3 = svm_conf_matrix[0][2] + svm_conf_matrix[1][2]

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

        svm_precision_avg = (svm_precision_1 + svm_precision_2 + svm_precision_3) / 3
        return svm_precision_avg


    def get_recall_pen_1(svm_conf_matrix):
        svm_tp_1 = svm_conf_matrix[0][0]
        svm_tp_2 = svm_conf_matrix[1][1]
        svm_tp_3 = svm_conf_matrix[2][2]
        svm_fn_1 = svm_conf_matrix[0][1] + svm_conf_matrix[0][2]
        svm_fn_2 = svm_conf_matrix[1][0] + svm_conf_matrix[1][2]
        svm_fn_3 = svm_conf_matrix[2][0] + svm_conf_matrix[2][1]
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
        svm_recall_avg_pen_1 = (svm_recall_1 + svm_recall_2 + svm_recall_3) / (3+1-1)
        return svm_recall_avg_pen_1

    def get_recall_pen_5(svm_conf_matrix):
        svm_tp_1 = svm_conf_matrix[0][0]
        svm_tp_2 = svm_conf_matrix[1][1]
        svm_tp_3 = svm_conf_matrix[2][2]
        svm_fn_1 = svm_conf_matrix[0][1] + svm_conf_matrix[0][2]
        svm_fn_2 = svm_conf_matrix[1][0] + svm_conf_matrix[1][2]
        svm_fn_3 = svm_conf_matrix[2][0] + svm_conf_matrix[2][1]
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
        svm_recall_avg_pen_5 = (svm_recall_1 + (5*svm_recall_2) + svm_recall_3) / (3+5-1)
        return svm_recall_avg_pen_5


    from sklearn.metrics import classification_report, confusion_matrix

    svm_conf_matrix = confusion_matrix(y_test, svm_prediction, labels = np.unique(data['ACT_4']))
    print("svm_confusion matrix:")
    print(svm_conf_matrix)
    svm_precision = get_precision(svm_conf_matrix)
    svm_recall_pen_1 = get_recall_pen_1(svm_conf_matrix)
    svm_recall_pen_5 = get_recall_pen_5(svm_conf_matrix)
    svm_f1_score_pen_1 = 2 * (svm_precision * svm_recall_pen_1) / (svm_precision + svm_recall_pen_1)
    svm_f1_score_pen_5 = 2 * (svm_precision * svm_recall_pen_5) / (svm_precision + svm_recall_pen_5)
    svm_ovr_accuracy = (svm_conf_matrix[0][0] + svm_conf_matrix[1][1] + svm_conf_matrix[2][2]) / (
                sum(svm_conf_matrix[0]) + sum(svm_conf_matrix[1]) + sum(svm_conf_matrix[2]))
    print("svm_f1 score of pen 1 is:")
    print(svm_f1_score_pen_1)
    print("svm_f1 score of pen 5 is:")
    print(svm_f1_score_pen_5)
    print("svm_overall accuracy is:")
    print(svm_ovr_accuracy)
    svm_conf_matrix = pd.DataFrame(svm_conf_matrix)
    svm_conf_matrix.to_csv('conf_matrix_' + imb_technique + '_svm_road_traffic_' + str(nsplits) + 'foldcv_' + str(repeat + 1) + '.csv', header=False, index=False)  # First repetition
    #svm_conf_matrix.to_csv('conf_matrix_' + imb_technique + '_svm_road_traffic_' + str(nsplits) + 'foldcv_' + str(repeat + 6) + '.csv', header=False, index=False)  # Second repetition

    svm_f1_score_pen_1_kfoldcv[repeat] = svm_f1_score_pen_1
    svm_f1_score_pen_5_kfoldcv[repeat] = svm_f1_score_pen_5
    svm_ovr_accuracy_kfoldcv[repeat] = svm_ovr_accuracy

    repeat = repeat + 1

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
dnn_ovr_accuracy_kfoldcv.to_csv('ovr_accuracy_'+imb_technique+'_penalty_' + str(penalty) + '_dnn_road_traffic_' + str(nsplits) + 'foldcv_1~5.csv',header=False,index=False) #First repetition
#dnn_ovr_accuracy_kfoldcv.to_csv('ovr_accuracy_'+imb_technique+'_penalty_' + str(penalty) + '_dnn_road_traffic_' + str(nsplits) + 'foldcv_6~10.csv',header=False,index=False) #Second repetition

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
lr_ovr_accuracy_kfoldcv.to_csv('ovr_accuracy_'+imb_technique+'_penalty_' + str(penalty) + '_lr_road_traffic_' + str(nsplits) + 'foldcv_1~5.csv',header=False,index=False) #First repetition
#lr_ovr_accuracy_kfoldcv.to_csv('ovr_accuracy_'+imb_technique+'_penalty_' + str(penalty) + '_lr_road_traffic_' + str(nsplits) + 'foldcv_6~10.csv',header=False,index=False) #Second repetition

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
nb_ovr_accuracy_kfoldcv.to_csv('ovr_accuracy_'+imb_technique+'_penalty_' + str(penalty) + '_nb_road_traffic_' + str(nsplits) + 'foldcv_1~5.csv',header=False,index=False) #First repetition
#nb_ovr_accuracy_kfoldcv.to_csv('ovr_accuracy_'+imb_technique+'_penalty_' + str(penalty) + '_nb_road_traffic_' + str(nsplits) + 'foldcv_6~10.csv',header=False,index=False) #Second repetition

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
rf_ovr_accuracy_kfoldcv.to_csv('ovr_accuracy_'+imb_technique+'_penalty_' + str(penalty) + '_rf_road_traffic_' + str(nsplits) + 'foldcv_1~5.csv',header=False,index=False) #First repetition
#rf_ovr_accuracy_kfoldcv.to_csv('ovr_accuracy_'+imb_technique+'_penalty_' + str(penalty) + '_rf_road_traffic_' + str(nsplits) + 'foldcv_6~10.csv',header=False,index=False) #Second repetition

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
svm_ovr_accuracy_kfoldcv.to_csv('ovr_accuracy_'+imb_technique+'_penalty_' + str(penalty) + '_svm_road_traffic_' + str(nsplits) + 'foldcv_1~5.csv',header=False,index=False) #First repetition
#svm_ovr_accuracy_kfoldcv.to_csv('ovr_accuracy_'+imb_technique+'_penalty_' + str(penalty) + '_svm_road_traffic_' + str(nsplits) + 'foldcv_6~10.csv',header=False,index=False) #Second repetition

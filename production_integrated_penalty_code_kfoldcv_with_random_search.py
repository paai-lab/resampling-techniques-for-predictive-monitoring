import tensorflow as tf
import numpy as np
import pandas as pd
from collections import Counter
#Pandas was used to import csv file. Encoding parameter is set to "cp437" since the data contains English text
data_sample_percentage = "" #Delete after experiments. If data is not to be reduced, type in "" here. If to be reduced, type in, for example, "_20percent"
data_dir = "/home/jongchan/Production/window_3_production_preprocessed" + data_sample_percentage + ".csv"
data = pd.read_csv(data_dir, encoding='cp437')
#data = data.sample(frac=1).reset_index(drop=True) #I use this for ADASYN since ADASYN returns error on specific fold
X = data[['ACT_1', 'ACT_2', 'ACT_3','Span_in_minutes']]
y = data[['ACT_4']]

#################################################
########## Choose resampling technique ##########
#################################################
imb_technique = input("Type in the namne of the resampling technique among the followings -> Baseline / ADASYN / ALLKNN / CNN / ENN / IHT / NCR / NM / OSS / RENN / ROS / RUS / SMOTE / BSMOTE / SMOTEENN / SMOTETOMEK / BC / EE / TOMEK: ")
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
#imb_technique = "BC"
#imb_technique = "EE"
print("Data: Production")
print("Resampling technique: " + imb_technique)
round = "1st"
#round = "2nd"

# Dummification
X_dummy = pd.get_dummies(X, prefix="ACT_1", columns=['ACT_1'])
X_dummy = pd.get_dummies(X_dummy, prefix="ACT_2", columns=['ACT_2'])
X_dummy = pd.get_dummies(X_dummy, prefix="ACT_3", columns=['ACT_3'])
X_dummy.iloc[:, 0] = (X_dummy.iloc[:, 0] - X_dummy.iloc[:, 0].mean()) / X_dummy.iloc[:, 0].std()

# X and y here will be used for hyperparameter tuning using random search
X_randomsearch = X.replace(regex=True, to_replace="Final Inspection Q.C.", value=1)
X_randomsearch = X_randomsearch.replace(regex=True, to_replace="Flat Grinding - Machine 11", value=2)
X_randomsearch = X_randomsearch.replace(regex=True, to_replace="Grinding Rework - Machine 27", value=3)
X_randomsearch = X_randomsearch.replace(regex=True, to_replace="Lapping - Machine 1", value=4)
X_randomsearch = X_randomsearch.replace(regex=True, to_replace="Laser Marking - Machine 7", value=5)
X_randomsearch = X_randomsearch.replace(regex=True, to_replace="Packing", value=6)
X_randomsearch = X_randomsearch.replace(regex=True, to_replace="Round Grinding - Machine 12", value=7)
X_randomsearch = X_randomsearch.replace(regex=True, to_replace="Round Grinding - Machine 2", value=8)
X_randomsearch = X_randomsearch.replace(regex=True, to_replace="Round Grinding - Machine 3", value=9)
X_randomsearch = X_randomsearch.replace(regex=True, to_replace="Round Grinding - Manual", value=10)
X_randomsearch = X_randomsearch.replace(regex=True, to_replace="Round Grinding - Q.C.", value=11)
X_randomsearch = X_randomsearch.replace(regex=True, to_replace="Turning - Machine 8", value=12)
X_randomsearch = X_randomsearch.replace(regex=True, to_replace="Turning & Milling - Machine 10", value=13)
X_randomsearch = X_randomsearch.replace(regex=True, to_replace="Turning & Milling - Machine 4", value=14)
X_randomsearch = X_randomsearch.replace(regex=True, to_replace="Turning & Milling - Machine 5", value=15)
X_randomsearch = X_randomsearch.replace(regex=True, to_replace="Turning & Milling - Machine 6", value=16)
X_randomsearch = X_randomsearch.replace(regex=True, to_replace="Turning & Milling - Machine 8", value=17)
X_randomsearch = X_randomsearch.replace(regex=True, to_replace="Turning & Milling - Machine 9", value=18)
X_randomsearch = X_randomsearch.replace(regex=True, to_replace="Turning & Milling Q.C.", value=19)
X_randomsearch = X_randomsearch.replace(regex=True, to_replace="Turning Q.C.", value=20)
X_randomsearch = X_randomsearch.replace(regex=True, to_replace="Wire Cut - Machine 13", value=21)
X_randomsearch = X_randomsearch.replace(regex=True, to_replace="Deburring - Manual", value=22)
X_randomsearch = X_randomsearch.replace(regex=True, to_replace="Stress Relief", value=23)
X_randomsearch = X_randomsearch.replace(regex=True, to_replace="Turning - Machine 9", value=24)
X_randomsearch = X_randomsearch.replace(regex=True, to_replace="Grinding Rework", value=25)
X_randomsearch = X_randomsearch.replace(regex=True, to_replace="Fix EDM", value=26)
X_randomsearch = X_randomsearch.replace(regex=True, to_replace="Wire Cut - Machine 18", value=27)
X_randomsearch = X_randomsearch.replace(regex=True, to_replace="Turn & Mill. & Screw Assem - Machine 10", value=28)
X_randomsearch = X_randomsearch.replace(regex=True, to_replace="Fix - Machine 15M", value=29)
X_randomsearch = X_randomsearch.replace(regex=True, to_replace="Nitration Q.C.", value=30)
X_randomsearch = X_randomsearch.replace(regex=True, to_replace="Fix - Machine 15", value=31)
X_randomsearch = X_randomsearch.replace(regex=True, to_replace="Milling - Machine 16", value=32)
X_randomsearch = X_randomsearch.replace(regex=True, to_replace="Turning Rework - Machine 21", value=33)
X_randomsearch = X_randomsearch.replace(regex=True, to_replace="Turn & Mill. & Screw Assem - Machine 9", value=34)
X_randomsearch = X_randomsearch.replace(regex=True, to_replace="Fix - Machine 3", value=35)
X_randomsearch = X_randomsearch.replace(regex=True, to_replace="Round Grinding - Machine 19", value=36)
X_randomsearch = X_randomsearch.replace(regex=True, to_replace="Change Version - Machine 22", value=37)
X_randomsearch = X_randomsearch.replace(regex=True, to_replace="Milling - Machine 14", value=38)
X_randomsearch = X_randomsearch.replace(regex=True, to_replace="Setup - Machine 4", value=39)
X_randomsearch = X_randomsearch.replace(regex=True, to_replace="Turning - Machine 21", value=40)
X_randomsearch = X_randomsearch.replace(regex=True, to_replace="Turning - Machine 5", value=41)
X_randomsearch = X_randomsearch.replace(regex=True, to_replace="Milling - Machine 10", value=42)
X_randomsearch = X_randomsearch.replace(regex=True, to_replace="Round  Q.C.", value=43)
X_randomsearch = X_randomsearch.replace(regex=True, to_replace="Rework Milling - Machine 28", value=44)
X_randomsearch = X_randomsearch.replace(regex=True, to_replace="Milling Q.C.", value=45)
X_randomsearch = X_randomsearch.replace(regex=True, to_replace="Turning - Machine 4", value=46)
X_randomsearch = X_randomsearch.replace(regex=True, to_replace="Flat Grinding - Machine 26", value=47)
X_randomsearch = X_randomsearch.replace(regex=True, to_replace="Fix - Machine 19", value=48)
X_randomsearch = X_randomsearch.replace(regex=True, to_replace="Setup - Machine 8", value=49)



y_randomsearch = y.replace(regex=True, to_replace="Final Inspection Q.C.", value=1)
y_randomsearch = y_randomsearch.replace(regex=True, to_replace="Flat Grinding - Machine 11", value=2)
y_randomsearch = y_randomsearch.replace(regex=True, to_replace="Grinding Rework - Machine 27", value=3)
y_randomsearch = y_randomsearch.replace(regex=True, to_replace="Lapping - Machine 1", value=4)
y_randomsearch = y_randomsearch.replace(regex=True, to_replace="Laser Marking - Machine 7", value=5)
y_randomsearch = y_randomsearch.replace(regex=True, to_replace="Packing", value=6)
y_randomsearch = y_randomsearch.replace(regex=True, to_replace="Round Grinding - Machine 12", value=7)
y_randomsearch = y_randomsearch.replace(regex=True, to_replace="Round Grinding - Machine 2", value=8)
y_randomsearch = y_randomsearch.replace(regex=True, to_replace="Round Grinding - Machine 3", value=9)
y_randomsearch = y_randomsearch.replace(regex=True, to_replace="Round Grinding - Manual", value=10)
y_randomsearch = y_randomsearch.replace(regex=True, to_replace="Round Grinding - Q.C.", value=11)
y_randomsearch = y_randomsearch.replace(regex=True, to_replace="Turning - Machine 8", value=12)
y_randomsearch = y_randomsearch.replace(regex=True, to_replace="Turning & Milling - Machine 10", value=13)
y_randomsearch = y_randomsearch.replace(regex=True, to_replace="Turning & Milling - Machine 4", value=14)
y_randomsearch = y_randomsearch.replace(regex=True, to_replace="Turning & Milling - Machine 5", value=15)
y_randomsearch = y_randomsearch.replace(regex=True, to_replace="Turning & Milling - Machine 6", value=16)
y_randomsearch = y_randomsearch.replace(regex=True, to_replace="Turning & Milling - Machine 8", value=17)
y_randomsearch = y_randomsearch.replace(regex=True, to_replace="Turning & Milling - Machine 9", value=18)
y_randomsearch = y_randomsearch.replace(regex=True, to_replace="Turning & Milling Q.C.", value=19)
y_randomsearch = y_randomsearch.replace(regex=True, to_replace="Turning Q.C.", value=20)
y_randomsearch = y_randomsearch.replace(regex=True, to_replace="Wire Cut - Machine 13", value=21)
y_randomsearch = y_randomsearch.replace(regex=True, to_replace="Deburring - Manual", value=22)
y_randomsearch = y_randomsearch.replace(regex=True, to_replace="Stress Relief", value=23)
y_randomsearch = y_randomsearch.replace(regex=True, to_replace="Turning - Machine 9", value=24)
y_randomsearch = y_randomsearch.replace(regex=True, to_replace="Grinding Rework", value=25)
y_randomsearch = y_randomsearch.replace(regex=True, to_replace="Fix EDM", value=26)
y_randomsearch = y_randomsearch.replace(regex=True, to_replace="Wire Cut - Machine 18", value=27)
y_randomsearch = y_randomsearch.replace(regex=True, to_replace="Turn & Mill. & Screw Assem - Machine 10", value=28)
y_randomsearch = y_randomsearch.replace(regex=True, to_replace="Fix - Machine 15M", value=29)
y_randomsearch = y_randomsearch.replace(regex=True, to_replace="Nitration Q.C.", value=30)
y_randomsearch = y_randomsearch.replace(regex=True, to_replace="Fix - Machine 15", value=31)
y_randomsearch = y_randomsearch.replace(regex=True, to_replace="Milling - Machine 16", value=32)
y_randomsearch = y_randomsearch.replace(regex=True, to_replace="Turning Rework - Machine 21", value=33)
y_randomsearch = y_randomsearch.replace(regex=True, to_replace="Turn & Mill. & Screw Assem - Machine 9", value=34)
y_randomsearch = y_randomsearch.replace(regex=True, to_replace="Fix - Machine 3", value=35)
y_randomsearch = y_randomsearch.replace(regex=True, to_replace="Round Grinding - Machine 19", value=36)
y_randomsearch = y_randomsearch.replace(regex=True, to_replace="Change Version - Machine 22", value=37)
y_randomsearch = y_randomsearch.replace(regex=True, to_replace="Milling - Machine 14", value=38)
y_randomsearch = y_randomsearch.replace(regex=True, to_replace="Setup - Machine 4", value=39)
y_randomsearch = y_randomsearch.replace(regex=True, to_replace="Turning - Machine 21", value=40)
y_randomsearch = y_randomsearch.replace(regex=True, to_replace="Turning - Machine 5", value=41)
y_randomsearch = y_randomsearch.replace(regex=True, to_replace="Milling - Machine 10", value=42)
y_randomsearch = y_randomsearch.replace(regex=True, to_replace="Round  Q.C.", value=43)
y_randomsearch = y_randomsearch.replace(regex=True, to_replace="Rework Milling - Machine 28", value=44)
y_randomsearch = y_randomsearch.replace(regex=True, to_replace="Milling Q.C.", value=45)
y_randomsearch = y_randomsearch.replace(regex=True, to_replace="Turning - Machine 4", value=46)
y_randomsearch = y_randomsearch.replace(regex=True, to_replace="Flat Grinding - Machine 26", value=47)
y_randomsearch = y_randomsearch.replace(regex=True, to_replace="Fix - Machine 19", value=48)
y_randomsearch = y_randomsearch.replace(regex=True, to_replace="Setup - Machine 8", value=49)



from sklearn.model_selection import KFold
nsplits = 5 # Set the number of k for cross validation
kf = KFold(n_splits=nsplits)
kf.get_n_splits(X_dummy)
print(kf)

#Lists below are used to store the f1 score and overall accuracy values generated from each fold
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


#Training & Testing
for train_index, test_index in kf.split(X_dummy):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X_dummy.iloc[train_index], X_dummy.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    train = pd.concat([X_train, y_train], axis=1)
    ACT_4_index = np.unique(data['ACT_1']).size + np.unique(data['ACT_2']).size + np.unique(data['ACT_3']).size + 1

    #Below codes are used to select features that we are going to use in the training procedure

    '''
    DM = train[train.ACT_4 == "Deburring - Manual"]
    DM_rest = train[train.ACT_4 != "Deburring - Manual"]
    DM_rest = DM_rest.copy()
    DM_rest.iloc[:, ACT_4_index] = "Others"
    DM = DM.reset_index()
    DM_rest = DM_rest.reset_index()
    DM = DM.iloc[:, 1:ACT_4_index+2]
    DM_rest = DM_rest.iloc[:, 1:ACT_4_index+2]
    DM_ova = pd.concat([DM, DM_rest])
    DM_ova_X_train = DM_ova.iloc[:, 0:ACT_4_index]
    DM_ova_y_train = DM_ova.iloc[:, ACT_4_index]
    DM_X_res = DM_ova_X_train
    DM_y_res = DM_ova_y_train
    Counter(DM_ova_y_train)
    '''
    FI = train[train.ACT_4 == "Final Inspection Q.C."]
    FI_rest = train[train.ACT_4 != "Final Inspection Q.C."]
    FI_rest = FI_rest.copy()
    ACT_4_index = np.unique(data['ACT_1']).size + np.unique(data['ACT_2']).size + np.unique(data['ACT_3']).size + 1
    FI_rest.iloc[:, ACT_4_index] = "Others"
    FI = FI.reset_index()
    FI_rest = FI_rest.reset_index()
    FI = FI.iloc[:, 1:ACT_4_index+2]
    FI_rest = FI_rest.iloc[:, 1:ACT_4_index+2]
    FI_ova = pd.concat([FI, FI_rest])
    FI_ova_X_train = FI_ova.iloc[:, 0:ACT_4_index]
    FI_ova_y_train = FI_ova.iloc[:, ACT_4_index]
    FI_X_res = FI_ova_X_train
    FI_y_res = FI_ova_y_train
    Counter(FI_ova_y_train)

    FG = train[train.ACT_4 == "Flat Grinding - Machine 11"]
    FG_rest = train[train.ACT_4 != "Flat Grinding - Machine 11"]
    FG_rest = FG_rest.copy()
    FG_rest.iloc[:, ACT_4_index] = "Others"
    FG = FG.reset_index()
    FG_rest = FG_rest.reset_index()
    FG = FG.iloc[:, 1:ACT_4_index+2]
    FG_rest = FG_rest.iloc[:, 1:ACT_4_index+2]
    FG_ova = pd.concat([FG, FG_rest])
    FG_ova_X_train = FG_ova.iloc[:, 0:ACT_4_index]
    FG_ova_y_train = FG_ova.iloc[:, ACT_4_index]
    FG_X_res = FG_ova_X_train
    FG_y_res = FG_ova_y_train
    Counter(FG_ova_y_train)
    """
    GR = train[train.ACT_4 == "Grinding Rework"]
    GR_rest = train[train.ACT_4 != "Grinding Rework"]
    GR_rest = GR_rest.copy()
    GR_rest.iloc[:, ACT_4_index] = "Others"
    GR = GR.reset_index()
    GR_rest = GR_rest.reset_index()
    GR = GR.iloc[:, 1:ACT_4_index+2]
    GR_rest = GR_rest.iloc[:, 1:ACT_4_index+2]
    GR_ova = pd.concat([GR, GR_rest])
    GR_ova_X_train = GR_ova.iloc[:, 0:ACT_4_index+2]
    GR_ova_y_train = GR_ova.iloc[:, ACT_4_index]
    GR_X_res = GR_ova_X_train
    GR_y_res = GR_ova_y_train
    Counter(GR_ova_y_train)
    """
    """
    GR12 = train[train.ACT_4 == "Grinding Rework - Machine 12"]
    GR12_rest = train[train.ACT_4 != "Grinding Rework - Machine 12"]
    GR12_rest = GR12_rest.copy()
    GR12_rest.iloc[:, ACT_4_index] = "Others"
    GR12 = GR12.reset_index()
    GR12_rest = GR12_rest.reset_index()
    GR12 = GR12.iloc[:, 1:ACT_4_index+2]
    GR12_rest = GR12_rest.iloc[:, 1:ACT_4_index+2]
    GR12_ova = pd.concat([GR12, GR12_rest])
    GR12_ova_X_train = GR12_ova.iloc[:, 0:ACT_4_index]
    GR12_ova_y_train = GR12_ova.iloc[:, ACT_4_index]
    GR12_X_res = GR12_ova_X_train
    GR12_y_res = GR12_ova_y_train
    Counter(GR12_ova_y_train)
    """
    GR27 = train[train.ACT_4 == "Grinding Rework - Machine 27"]
    GR27_rest = train[train.ACT_4 != "Grinding Rework - Machine 27"]
    GR27_rest = GR27_rest.copy()
    GR27_rest.iloc[:, ACT_4_index] = "Others"
    GR27 = GR27.reset_index()
    GR27_rest = GR27_rest.reset_index()
    GR27 = GR27.iloc[:, 1:ACT_4_index+2]
    GR27_rest = GR27_rest.iloc[:, 1:ACT_4_index+2]
    GR27_ova = pd.concat([GR27, GR27_rest])
    GR27_ova_X_train = GR27_ova.iloc[:, 0:ACT_4_index]
    GR27_ova_y_train = GR27_ova.iloc[:, ACT_4_index]
    GR27_X_res = GR27_ova_X_train
    GR27_y_res = GR27_ova_y_train
    Counter(GR27_ova_y_train)

    LM = train[train.ACT_4 == "Lapping - Machine 1"]
    LM_rest = train[train.ACT_4 != "Lapping - Machine 1"]
    LM_rest = LM_rest.copy()
    LM_rest.iloc[:, ACT_4_index] = "Others"
    LM = LM.reset_index()
    LM_rest = LM_rest.reset_index()
    LM = LM.iloc[:, 1:ACT_4_index+2]
    LM_rest = LM_rest.iloc[:, 1:ACT_4_index+2]
    LM_ova = pd.concat([LM, LM_rest])
    LM_ova_X_train = LM_ova.iloc[:, 0:ACT_4_index]
    LM_ova_y_train = LM_ova.iloc[:, ACT_4_index]
    LM_X_res = LM_ova_X_train
    LM_y_res = LM_ova_y_train
    Counter(LM_ova_y_train)

    LMM = train[train.ACT_4 == "Laser Marking - Machine 7"]
    LMM_rest = train[train.ACT_4 != "Laser Marking - Machine 7"]
    LMM_rest = LMM_rest.copy()
    LMM_rest.iloc[:, ACT_4_index] = "Others"
    LMM = LMM.reset_index()
    LMM_rest = LMM_rest.reset_index()
    LMM = LMM.iloc[:, 1:ACT_4_index+2]
    LMM_rest = LMM_rest.iloc[:, 1:ACT_4_index+2]
    LMM_ova = pd.concat([LMM, LMM_rest])
    LMM_ova_X_train = LMM_ova.iloc[:, 0:ACT_4_index]
    LMM_ova_y_train = LMM_ova.iloc[:, ACT_4_index]
    LMM_X_res = LMM_ova_X_train
    LMM_y_res = LMM_ova_y_train
    Counter(LMM_ova_y_train)
    '''
    MM14 = train[train.ACT_4 == "Milling - Machine 14"]
    MM14_rest = train[train.ACT_4 != "Milling - Machine 14"]
    MM14_rest = MM14_rest.copy()
    MM14_rest.iloc[:, ACT_4_index] = "Others"
    MM14 = MM14.reset_index()
    MM14_rest = MM14_rest.reset_index()
    MM14 = MM14.iloc[:, 1:ACT_4_index+2]
    MM14_rest = MM14_rest.iloc[:, 1:ACT_4_index+2]
    MM14_ova = pd.concat([MM14, MM14_rest])
    MM14_ova_X_train = MM14_ova.iloc[:, 0:ACT_4_index]
    MM14_ova_y_train = MM14_ova.iloc[:, ACT_4_index]
    MM14_X_res = MM14_ova_X_train
    MM14_y_res = MM14_ova_y_train
    Counter(MM14_ova_y_train)
    '''
    '''
    MM16 = train[train.ACT_4 == "Milling - Machine 16"]
    MM16_rest = train[train.ACT_4 != "Milling - Machine 16"]
    MM16_rest = MM16_rest.copy()
    MM16_rest.iloc[:, ACT_4_index] = "Others"
    MM16 = MM16.reset_index()
    MM16_rest = MM16_rest.reset_index()
    MM16 = MM16.iloc[:, 1:ACT_4_index+2]
    MM16_rest = MM16_rest.iloc[:, 1:ACT_4_index+2]
    MM16_ova = pd.concat([MM16, MM16_rest])
    MM16_ova_X_train = MM16_ova.iloc[:, 0:ACT_4_index]
    MM16_ova_y_train = MM16_ova.iloc[:, ACT_4_index]
    MM16_X_res = MM16_ova_X_train
    MM16_y_res = MM16_ova_y_train
    Counter(MM16_ova_y_train)
    '''
    PC = train[train.ACT_4 == "Packing"]
    PC_rest = train[train.ACT_4 != "Packing"]
    PC_rest = PC_rest.copy()
    PC_rest.iloc[:, ACT_4_index] = "Others"
    PC = PC.reset_index()
    PC_rest = PC_rest.reset_index()
    PC = PC.iloc[:, 1:ACT_4_index+2]
    PC_rest = PC_rest.iloc[:, 1:ACT_4_index+2]
    PC_ova = pd.concat([PC, PC_rest])
    PC_ova_X_train = PC_ova.iloc[:, 0:ACT_4_index]
    PC_ova_y_train = PC_ova.iloc[:, ACT_4_index]
    PC_X_res = PC_ova_X_train
    PC_y_res = PC_ova_y_train
    Counter(PC_ova_y_train)

    RG12 = train[train.ACT_4 == "Round Grinding - Machine 12"]
    RG12_rest = train[train.ACT_4 != "Round Grinding - Machine 12"]
    RG12_rest = RG12_rest.copy()
    RG12_rest.iloc[:, ACT_4_index] = "Others"
    RG12 = RG12.reset_index()
    RG12_rest = RG12_rest.reset_index()
    RG12 = RG12.iloc[:, 1:ACT_4_index+2]
    RG12_rest = RG12_rest.iloc[:, 1:ACT_4_index+2]
    RG12_ova = pd.concat([RG12, RG12_rest])
    RG12_ova_X_train = RG12_ova.iloc[:, 0:ACT_4_index]
    RG12_ova_y_train = RG12_ova.iloc[:, ACT_4_index]
    RG12_X_res = RG12_ova_X_train
    RG12_y_res = RG12_ova_y_train
    Counter(RG12_ova_y_train)
    '''
    RG19 = train[train.ACT_4 == "Round Grinding - Machine 19"]
    RG19_rest = train[train.ACT_4 != "Round Grinding - Machine 19"]
    RG19_rest = RG19_rest.copy()
    RG19_rest.iloc[:, ACT_4_index] = "Others"
    RG19 = RG19.reset_index()
    RG19_rest = RG19_rest.reset_index()
    RG19 = RG19.iloc[:, 1:ACT_4_index+2]
    RG19_rest = RG19_rest.iloc[:, 1:ACT_4_index+2]
    RG19_ova = pd.concat([RG19, RG19_rest])
    RG19_ova_X_train = RG19_ova.iloc[:, 0:ACT_4_index]
    RG19_ova_y_train = RG19_ova.iloc[:, ACT_4_index]
    RG19_X_res = RG19_ova_X_train
    RG19_y_res = RG19_ova_y_train
    Counter(RG19_ova_y_train)
    '''
    RG2 = train[train.ACT_4 == "Round Grinding - Machine 2"]
    RG2_rest = train[train.ACT_4 != "Round Grinding - Machine 2"]
    RG2rest = RG2_rest.copy()
    RG2_rest.iloc[:, ACT_4_index] = "Others"
    RG2 = RG2.reset_index()
    RG2_rest = RG2_rest.reset_index()
    RG2 = RG2.iloc[:, 1:ACT_4_index+2]
    RG2_rest = RG2_rest.iloc[:, 1:ACT_4_index+2]
    RG2_ova = pd.concat([RG2, RG2_rest])
    RG2_ova_X_train = RG2_ova.iloc[:, 0:ACT_4_index]
    RG2_ova_y_train = RG2_ova.iloc[:, ACT_4_index]
    RG2_X_res = RG2_ova_X_train
    RG2_y_res = RG2_ova_y_train
    Counter(RG2_ova_y_train)

    RG3 = train[train.ACT_4 == "Round Grinding - Machine 3"]
    RG3_rest = train[train.ACT_4 != "Round Grinding - Machine 3"]
    RG3_rest = RG3_rest.copy()
    RG3_rest.iloc[:, ACT_4_index] = "Others"
    RG3 = RG3.reset_index()
    RG3_rest = RG3_rest.reset_index()
    RG3 = RG3.iloc[:, 1:ACT_4_index+2]
    RG3_rest = RG3_rest.iloc[:, 1:ACT_4_index+2]
    RG3_ova = pd.concat([RG3, RG3_rest])
    RG3_ova_X_train = RG3_ova.iloc[:, 0:ACT_4_index]
    RG3_ova_y_train = RG3_ova.iloc[:, ACT_4_index]
    RG3_X_res = RG3_ova_X_train
    RG3_y_res = RG3_ova_y_train
    Counter(RG3_ova_y_train)

    RGM = train[train.ACT_4 == "Round Grinding - Manual"]
    RGM_rest = train[train.ACT_4 != "Round Grinding - Manual"]
    RGM_rest = RGM_rest.copy()
    RGM_rest.iloc[:, ACT_4_index] = "Others"
    RGM = RGM.reset_index()
    RGM_rest = RGM_rest.reset_index()
    RGM = RGM.iloc[:, 1:ACT_4_index+2]
    RGM_rest = RGM_rest.iloc[:, 1:ACT_4_index+2]
    RGM_ova = pd.concat([RGM, RGM_rest])
    RGM_ova_X_train = RGM_ova.iloc[:, 0:ACT_4_index]
    RGM_ova_y_train = RGM_ova.iloc[:, ACT_4_index]
    RGM_X_res = RGM_ova_X_train
    RGM_y_res = RGM_ova_y_train
    Counter(RGM_ova_y_train)

    RGQC = train[train.ACT_4 == "Round Grinding - Q.C."]
    RGQC_rest = train[train.ACT_4 != "Round Grinding - Q.C."]
    RGQC_rest = RGQC_rest.copy()
    RGQC_rest.iloc[:, ACT_4_index] = "Others"
    RGQC = RGQC.reset_index()
    RGQC_rest = RGQC_rest.reset_index()
    RGQC = RGQC.iloc[:, 1:ACT_4_index+2]
    RGQC_rest = RGQC_rest.iloc[:, 1:ACT_4_index+2]
    RGQC_ova = pd.concat([RGQC, RGQC_rest])
    RGQC_ova_X_train = RGQC_ova.iloc[:, 0:ACT_4_index]
    RGQC_ova_y_train = RGQC_ova.iloc[:, ACT_4_index]
    RGQC_X_res = RGQC_ova_X_train
    RGQC_y_res = RGQC_ova_y_train
    Counter(RGQC_ova_y_train)
    """
    TMSA10 = train[train.ACT_4 == "Turn & Mill. & Screw Assem - Machine 10"]
    TMSA10_rest = train[train.ACT_4 != "Turn & Mill. & Screw Assem - Machine 10"]
    TMSA10_rest = TMSA10_rest.copy()
    TMSA10_rest.iloc[:, ACT_4_index] = "Others"
    TMSA10 = TMSA10.reset_index()
    TMSA10_rest = TMSA10_rest.reset_index()
    TMSA10 = TMSA10.iloc[:, 1:ACT_4_index+2]
    TMSA10_rest = TMSA10_rest.iloc[:, 1:ACT_4_index+2]
    TMSA10_ova = pd.concat([TMSA10, TMSA10_rest])
    TMSA10_ova_X_train = TMSA10_ova.iloc[:, 0:ACT_4_index]
    TMSA10_ova_y_train = TMSA10_ova.iloc[:, ACT_4_index]
    TMSA10_X_res = TMSA10_ova_X_train
    TMSA10_y_res = TMSA10_ova_y_train
    Counter(TMSA10_ova_y_train)
    """
    T8 = train[train.ACT_4 == "Turning - Machine 8"]
    T8_rest = train[train.ACT_4 != "Turning - Machine 8"]
    T8_rest = T8_rest.copy()
    T8_rest.iloc[:, ACT_4_index] = "Others"
    T8 = T8.reset_index()
    T8_rest = T8_rest.reset_index()
    T8 = T8.iloc[:, 1:ACT_4_index+2]
    T8_rest = T8_rest.iloc[:, 1:ACT_4_index+2]
    T8_ova = pd.concat([T8, T8_rest])
    T8_ova_X_train = T8_ova.iloc[:, 0:ACT_4_index]
    T8_ova_y_train = T8_ova.iloc[:, ACT_4_index]
    T8_X_res = T8_ova_X_train
    T8_y_res = T8_ova_y_train
    Counter(T8_ova_y_train)
    """
    T9 = train[train.ACT_4 == "Turning - Machine 9"]
    T9_rest = train[train.ACT_4 != "Turning - Machine 9"]
    T9_rest = T9_rest.copy()
    T9_rest.iloc[:, ACT_4_index] = "Others"
    T9 = T9.reset_index()
    T9_rest = T9_rest.reset_index()
    T9 = T9.iloc[:, 1:ACT_4_index+2]
    T9_rest = T9_rest.iloc[:, 1:ACT_4_index+2]
    T9_ova = pd.concat([T9, T9_rest])
    T9_ova_X_train = T9_ova.iloc[:, 0:ACT_4_index]
    T9_ova_y_train = T9_ova.iloc[:, ACT_4_index]
    T9_X_res = T9_ova_X_train
    T9_y_res = T9_ova_y_train
    Counter(T9_ova_y_train)
    """
    TM10 = train[train.ACT_4 == "Turning & Milling - Machine 10"]
    TM10_rest = train[train.ACT_4 != "Turning & Milling - Machine 10"]
    TM10_rest = TM10_rest.copy()
    TM10_rest.iloc[:, ACT_4_index] = "Others"
    TM10 = TM10.reset_index()
    TM10_rest = TM10_rest.reset_index()
    TM10 = TM10.iloc[:, 1:ACT_4_index+2]
    TM10_rest = TM10_rest.iloc[:, 1:ACT_4_index+2]
    TM10_ova = pd.concat([TM10, TM10_rest])
    TM10_ova_X_train = TM10_ova.iloc[:, 0:ACT_4_index]
    TM10_ova_y_train = TM10_ova.iloc[:, ACT_4_index]
    TM10_X_res = TM10_ova_X_train
    TM10_y_res = TM10_ova_y_train
    Counter(TM10_ova_y_train)

    TM4 = train[train.ACT_4 == "Turning & Milling - Machine 4"]
    TM4_rest = train[train.ACT_4 != "Turning & Milling - Machine 4"]
    TM4_rest = TM4_rest.copy()
    TM4_rest.iloc[:, ACT_4_index] = "Others"
    TM4 = TM4.reset_index()
    TM4_rest = TM4_rest.reset_index()
    TM4 = TM4.iloc[:, 1:ACT_4_index+2]
    TM4_rest = TM4_rest.iloc[:, 1:ACT_4_index+2]
    TM4_ova = pd.concat([TM4, TM4_rest])
    TM4_ova_X_train = TM4_ova.iloc[:, 0:ACT_4_index]
    TM4_ova_y_train = TM4_ova.iloc[:, ACT_4_index]
    TM4_X_res = TM4_ova_X_train
    TM4_y_res = TM4_ova_y_train
    Counter(TM4_ova_y_train)

    TM5 = train[train.ACT_4 == "Turning & Milling - Machine 5"]
    TM5_rest = train[train.ACT_4 != "Turning & Milling - Machine 5"]
    TM5_rest = TM5_rest.copy()
    TM5_rest.iloc[:, ACT_4_index] = "Others"
    TM5 = TM5.reset_index()
    TM5_rest = TM5_rest.reset_index()
    TM5 = TM5.iloc[:, 1:ACT_4_index+2]
    TM5_rest = TM5_rest.iloc[:, 1:ACT_4_index+2]
    TM5_ova = pd.concat([TM5, TM5_rest])
    TM5_ova_X_train = TM5_ova.iloc[:, 0:ACT_4_index]
    TM5_ova_y_train = TM5_ova.iloc[:, ACT_4_index]
    TM5_X_res = TM5_ova_X_train
    TM5_y_res = TM5_ova_y_train
    Counter(TM5_ova_y_train)

    TM6 = train[train.ACT_4 == "Turning & Milling - Machine 6"]
    TM6_rest = train[train.ACT_4 != "Turning & Milling - Machine 6"]
    TM6_rest = TM6_rest.copy()
    TM6_rest.iloc[:, ACT_4_index] = "Others"
    TM6 = TM6.reset_index()
    TM6_rest = TM6_rest.reset_index()
    TM6 = TM6.iloc[:, 1:ACT_4_index+2]
    TM6_rest = TM6_rest.iloc[:, 1:ACT_4_index+2]
    TM6_ova = pd.concat([TM6, TM6_rest])
    TM6_ova_X_train = TM6_ova.iloc[:, 0:ACT_4_index]
    TM6_ova_y_train = TM6_ova.iloc[:, ACT_4_index]
    TM6_X_res = TM6_ova_X_train
    TM6_y_res = TM6_ova_y_train
    Counter(TM6_ova_y_train)

    TM8 = train[train.ACT_4 == "Turning & Milling - Machine 8"]
    TM8_rest = train[train.ACT_4 != "Turning & Milling - Machine 8"]
    TM8_rest = TM8_rest.copy()
    TM8_rest.iloc[:, ACT_4_index] = "Others"
    TM8 = TM8.reset_index()
    TM8_rest = TM8_rest.reset_index()
    TM8 = TM8.iloc[:, 1:ACT_4_index+2]
    TM8_rest = TM8_rest.iloc[:, 1:ACT_4_index+2]
    TM8_ova = pd.concat([TM8, TM8_rest])
    TM8_ova_X_train = TM8_ova.iloc[:, 0:ACT_4_index]
    TM8_ova_y_train = TM8_ova.iloc[:, ACT_4_index]
    TM8_X_res = TM8_ova_X_train
    TM8_y_res = TM8_ova_y_train
    Counter(TM8_ova_y_train)

    TM9 = train[train.ACT_4 == "Turning & Milling - Machine 9"]
    TM9_rest = train[train.ACT_4 != "Turning & Milling - Machine 9"]
    TM9_rest = TM9_rest.copy()
    TM9_rest.iloc[:, ACT_4_index] = "Others"
    TM9 = TM9.reset_index()
    TM9_rest = TM9_rest.reset_index()
    TM9 = TM9.iloc[:, 1:ACT_4_index+2]
    TM9_rest = TM9_rest.iloc[:, 1:ACT_4_index+2]
    TM9_ova = pd.concat([TM9, TM9_rest])
    TM9_ova_X_train = TM9_ova.iloc[:, 0:ACT_4_index]
    TM9_ova_y_train = TM9_ova.iloc[:, ACT_4_index]
    TM9_X_res = TM9_ova_X_train
    TM9_y_res = TM9_ova_y_train
    Counter(TM9_ova_y_train)

    TMQC = train[train.ACT_4 == "Turning & Milling Q.C."]
    TMQC_rest = train[train.ACT_4 != "Turning & Milling Q.C."]
    TMQC_rest = TMQC_rest.copy()
    TMQC_rest.iloc[:, ACT_4_index] = "Others"
    TMQC = TMQC.reset_index()
    TMQC_rest = TMQC_rest.reset_index()
    TMQC = TMQC.iloc[:, 1:ACT_4_index+2]
    TMQC_rest = TMQC_rest.iloc[:, 1:ACT_4_index+2]
    TMQC_ova = pd.concat([TMQC, TMQC_rest])
    TMQC_ova_X_train = TMQC_ova.iloc[:, 0:ACT_4_index]
    TMQC_ova_y_train = TMQC_ova.iloc[:, ACT_4_index]
    TMQC_X_res = TMQC_ova_X_train
    TMQC_y_res = TMQC_ova_y_train
    Counter(TMQC_ova_y_train)

    TQC = train[train.ACT_4 == "Turning Q.C."]
    TQC_rest = train[train.ACT_4 != "Turning Q.C."]
    TQC_rest = TQC_rest.copy()
    TQC_rest.iloc[:, ACT_4_index] = "Others"
    TQC = TQC.reset_index()
    TQC_rest = TQC_rest.reset_index()
    TQC = TQC.iloc[:, 1:ACT_4_index+2]
    TQC_rest = TQC_rest.iloc[:, 1:ACT_4_index+2]
    TQC_ova = pd.concat([TQC, TQC_rest])
    TQC_ova_X_train = TQC_ova.iloc[:, 0:ACT_4_index]
    TQC_ova_y_train = TQC_ova.iloc[:, ACT_4_index]
    TQC_X_res = TQC_ova_X_train
    TQC_y_res = TQC_ova_y_train
    Counter(TQC_ova_y_train)
    """
    WC13 = train[train.ACT_4 == "Wire Cut - Machine 13"]
    WC13_rest = train[train.ACT_4 != "Wire Cut - Machine 13"]
    WC13_rest = WC13_rest.copy()
    WC13_rest.iloc[:, ACT_4_index] = "Others"
    WC13 = WC13.reset_index()
    WC13_rest = WC13_rest.reset_index()
    WC13 = WC13.iloc[:, 1:ACT_4_index+2]
    WC13_rest = WC13_rest.iloc[:, 1:ACT_4_index+2]
    WC13_ova = pd.concat([WC13, WC13_rest])
    WC13_ova_X_train = WC13_ova.iloc[:, 0:ACT_4_index]
    WC13_ova_y_train = WC13_ova.iloc[:, ACT_4_index]
    WC13_X_res = WC13_ova_X_train
    WC13_y_res = WC13_ova_y_train
    Counter(WC13_ova_y_train)
    """

    #Below codes are used to resample data
    if imb_technique == "ADASYN":
        from imblearn.over_sampling import ADASYN

        #DM_ada = ADASYN()
        #DM_X_res, DM_y_res = DM_ada.fit_resample(DM_ova_X_train, DM_ova_y_train)
        FI_ada = ADASYN()
        FI_X_res, FI_y_res = FI_ada.fit_resample(FI_ova_X_train, FI_ova_y_train)
        FG_ada = ADASYN()
        FG_X_res, FG_y_res = FG_ada.fit_resample(FG_ova_X_train, FG_ova_y_train)
        #GR_ada = ADASYN()
        #GR_X_res, GR_y_res = GR_ada.fit_resample(GR_ova_X_train, GR_ova_y_train)
        #GR12_ada = ADASYN()
        #GR12_X_res, GR12_y_res = GR12_ada.fit_resample(GR12_ova_X_train, GR12_ova_y_train)
        GR27_ada = ADASYN()
        GR27_X_res, GR27_y_res = GR27_ada.fit_resample(GR27_ova_X_train, GR27_ova_y_train)
        LM_ada = ADASYN()
        LM_X_res, LM_y_res = LM_ada.fit_resample(LM_ova_X_train, LM_ova_y_train)
        LMM_ada = ADASYN()
        LMM_X_res, LMM_y_res = LMM_ada.fit_resample(LMM_ova_X_train, LMM_ova_y_train)
        #MM14_ada = ADASYN()
        #MM14_X_res, MM14_y_res = MM14_ada.fit_resample(MM14_ova_X_train, MM14_ova_y_train)
        #MM16_ada = ADASYN()
        #MM16_X_res, MM16_y_res = MM16_ada.fit_resample(MM16_ova_X_train, MM16_ova_y_train)
        PC_ada = ADASYN()
        PC_X_res, PC_y_res = PC_ada.fit_resample(PC_ova_X_train, PC_ova_y_train)
        RG12_ada = ADASYN()
        RG12_X_res, RG12_y_res = RG12_ada.fit_resample(RG12_ova_X_train, RG12_ova_y_train)
        #RG19_ada = ADASYN()
        #RG19_X_res, RG19_y_res = RG19_ada.fit_resample(RG19_ova_X_train, RG19_ova_y_train)
        RG2_ada = ADASYN()
        RG2_X_res, RG2_y_res = RG2_ada.fit_resample(RG2_ova_X_train, RG2_ova_y_train)
        RG3_ada = ADASYN()
        RG3_X_res, RG3_y_res = RG3_ada.fit_resample(RG3_ova_X_train, RG3_ova_y_train)
        RGM_ada = ADASYN()
        RGM_X_res, RGM_y_res = RGM_ada.fit_resample(RGM_ova_X_train, RGM_ova_y_train)
        RGQC_ada = ADASYN()
        RGQC_X_res, RGQC_y_res = RGQC_ada.fit_resample(RGQC_ova_X_train, RGQC_ova_y_train)
        #TMSA10_ada = ADASYN()
        #TMSA10_X_res, TMSA10_y_res = TMSA10_ada.fit_resample(TMSA10_ova_X_train, TMSA10_ova_y_train)
        T8_ada = ADASYN()
        T8_X_res, T8_y_res = T8_ada.fit_resample(T8_ova_X_train, T8_ova_y_train)
        #T9_ada = ADASYN()
        #T9_X_res, T9_y_res = T9_ada.fit_resample(T9_ova_X_train, T9_ova_y_train)
        TM10_ada = ADASYN()
        TM10_X_res, TM10_y_res = TM10_ada.fit_resample(TM10_ova_X_train, TM10_ova_y_train)
        TM4_ada = ADASYN()
        TM4_X_res, TM4_y_res = TM4_ada.fit_resample(TM4_ova_X_train, TM4_ova_y_train)
        TM5_ada = ADASYN()
        TM5_X_res, TM5_y_res = TM5_ada.fit_resample(TM5_ova_X_train, TM5_ova_y_train)
        TM6_ada = ADASYN()
        TM6_X_res, TM6_y_res = TM6_ada.fit_resample(TM6_ova_X_train, TM6_ova_y_train)
        TM8_ada = ADASYN()
        TM8_X_res, TM8_y_res = TM8_ada.fit_resample(TM8_ova_X_train, TM8_ova_y_train)
        TM9_ada = ADASYN()
        TM9_X_res, TM9_y_res = TM9_ada.fit_resample(TM9_ova_X_train, TM9_ova_y_train)
        TMQC_ada = ADASYN()
        TMQC_X_res, TMQC_y_res = TMQC_ada.fit_resample(TMQC_ova_X_train, TMQC_ova_y_train)
        TQC_ada = ADASYN()
        TQC_X_res, TQC_y_res = TQC_ada.fit_resample(TQC_ova_X_train, TQC_ova_y_train)
        #WC13_ada = ADASYN()
        #WC13_X_res, WC13_y_res = WC13_ada.fit_resample(WC13_ova_X_train, WC13_ova_y_train)

    if imb_technique == "ALLKNN":
        from imblearn.under_sampling import AllKNN

        #DM_allknn = AllKNN()
        #DM_X_res, DM_y_res = DM_allknn.fit_resample(DM_ova_X_train, DM_ova_y_train)
        FI_allknn = AllKNN()
        FI_X_res, FI_y_res = FI_allknn.fit_resample(FI_ova_X_train, FI_ova_y_train)
        FG_allknn = AllKNN()
        FG_X_res, FG_y_res = FG_allknn.fit_resample(FG_ova_X_train, FG_ova_y_train)
        #GR_allknn = AllKNN()
        #GR_X_res, GR_y_res = GR_allknn.fit_resample(GR_ova_X_train, GR_ova_y_train)
        #GR12_allknn = AllKNN()
        #GR12_X_res, GR12_y_res = GR12_allknn.fit_resample(GR12_ova_X_train, GR12_ova_y_train)
        GR27_allknn = AllKNN()
        GR27_X_res, GR27_y_res = GR27_allknn.fit_resample(GR27_ova_X_train, GR27_ova_y_train)
        LM_allknn = AllKNN()
        LM_X_res, LM_y_res = LM_allknn.fit_resample(LM_ova_X_train, LM_ova_y_train)
        LMM_allknn = AllKNN()
        LMM_X_res, LMM_y_res = LMM_allknn.fit_resample(LMM_ova_X_train, LMM_ova_y_train)
        #MM14_allknn = AllKNN()
        #MM14_X_res, MM14_y_res = MM14_allknn.fit_resample(MM14_ova_X_train, MM14_ova_y_train)
        #MM16_allknn = AllKNN()
        #MM16_X_res, MM16_y_res = MM16_allknn.fit_resample(MM16_ova_X_train, MM16_ova_y_train)
        PC_allknn = AllKNN()
        PC_X_res, PC_y_res = PC_allknn.fit_resample(PC_ova_X_train, PC_ova_y_train)
        RG12_allknn = AllKNN()
        RG12_X_res, RG12_y_res = RG12_allknn.fit_resample(RG12_ova_X_train, RG12_ova_y_train)
        #RG19_allknn = AllKNN()
        #RG19_X_res, RG19_y_res = RG19_allknn.fit_resample(RG19_ova_X_train, RG19_ova_y_train)
        RG2_allknn = AllKNN()
        RG2_X_res, RG2_y_res = RG2_allknn.fit_resample(RG2_ova_X_train, RG2_ova_y_train)
        RG3_allknn = AllKNN()
        RG3_X_res, RG3_y_res = RG3_allknn.fit_resample(RG3_ova_X_train, RG3_ova_y_train)
        RGM_allknn = AllKNN()
        RGM_X_res, RGM_y_res = RGM_allknn.fit_resample(RGM_ova_X_train, RGM_ova_y_train)
        RGQC_allknn = AllKNN()
        RGQC_X_res, RGQC_y_res = RGQC_allknn.fit_resample(RGQC_ova_X_train, RGQC_ova_y_train)
        #TMSA10_allknn = AllKNN()
        #TMSA10_X_res, TMSA10_y_res = TMSA10_allknn.fit_resample(TMSA10_ova_X_train, TMSA10_ova_y_train)
        T8_allknn = AllKNN()
        T8_X_res, T8_y_res = T8_allknn.fit_resample(T8_ova_X_train, T8_ova_y_train)
        #T9_allknn = AllKNN()
        #T9_X_res, T9_y_res = T9_allknn.fit_resample(T9_ova_X_train, T9_ova_y_train)
        TM10_allknn = AllKNN()
        TM10_X_res, TM10_y_res = TM10_allknn.fit_resample(TM10_ova_X_train, TM10_ova_y_train)
        TM4_allknn = AllKNN()
        TM4_X_res, TM4_y_res = TM4_allknn.fit_resample(TM4_ova_X_train, TM4_ova_y_train)
        TM5_allknn = AllKNN()
        TM5_X_res, TM5_y_res = TM5_allknn.fit_resample(TM5_ova_X_train, TM5_ova_y_train)
        TM6_allknn = AllKNN()
        TM6_X_res, TM6_y_res = TM6_allknn.fit_resample(TM6_ova_X_train, TM6_ova_y_train)
        TM8_allknn = AllKNN()
        TM8_X_res, TM8_y_res = TM8_allknn.fit_resample(TM8_ova_X_train, TM8_ova_y_train)
        TM9_allknn = AllKNN()
        TM9_X_res, TM9_y_res = TM9_allknn.fit_resample(TM9_ova_X_train, TM9_ova_y_train)
        TMQC_allknn = AllKNN()
        TMQC_X_res, TMQC_y_res = TMQC_allknn.fit_resample(TMQC_ova_X_train, TMQC_ova_y_train)
        TQC_allknn = AllKNN()
        TQC_X_res, TQC_y_res = TQC_allknn.fit_resample(TQC_ova_X_train, TQC_ova_y_train)
        #WC13_allknn = AllKNN()
        #WC13_X_res, WC13_y_res = WC13_allknn.fit_resample(WC13_ova_X_train, WC13_ova_y_train)

    if imb_technique == "CNN":
        from imblearn.under_sampling import CondensedNearestNeighbour

        #DM_cnn = CondensedNearestNeighbour()
        #DM_X_res, DM_y_res = DM_cnn.fit_resample(DM_ova_X_train, DM_ova_y_train)
        FI_cnn = CondensedNearestNeighbour()
        FI_X_res, FI_y_res = FI_cnn.fit_resample(FI_ova_X_train, FI_ova_y_train)
        FG_cnn = CondensedNearestNeighbour()
        FG_X_res, FG_y_res = FG_cnn.fit_resample(FG_ova_X_train, FG_ova_y_train)
        #GR_cnn = CondensedNearestNeighbour()
        #GR_X_res, GR_y_res = GR_cnn.fit_resample(GR_ova_X_train, GR_ova_y_train)
        #GR12_cnn = CondensedNearestNeighbour()
        #GR12_X_res, GR12_y_res = GR12_cnn.fit_resample(GR12_ova_X_train, GR12_ova_y_train)
        GR27_cnn = CondensedNearestNeighbour()
        GR27_X_res, GR27_y_res = GR27_cnn.fit_resample(GR27_ova_X_train, GR27_ova_y_train)
        LM_cnn = CondensedNearestNeighbour()
        LM_X_res, LM_y_res = LM_cnn.fit_resample(LM_ova_X_train, LM_ova_y_train)
        LMM_cnn = CondensedNearestNeighbour()
        LMM_X_res, LMM_y_res = LMM_cnn.fit_resample(LMM_ova_X_train, LMM_ova_y_train)
        #MM14_cnn = CondensedNearestNeighbour()
        #MM14_X_res, MM14_y_res = MM14_cnn.fit_resample(MM14_ova_X_train, MM14_ova_y_train)
        #MM16_cnn = CondensedNearestNeighbour()
        #MM16_X_res, MM16_y_res = MM16_cnn.fit_resample(MM16_ova_X_train, MM16_ova_y_train)
        PC_cnn = CondensedNearestNeighbour()
        PC_X_res, PC_y_res = PC_cnn.fit_resample(PC_ova_X_train, PC_ova_y_train)
        RG12_cnn = CondensedNearestNeighbour()
        RG12_X_res, RG12_y_res = RG12_cnn.fit_resample(RG12_ova_X_train, RG12_ova_y_train)
        #RG19_cnn = CondensedNearestNeighbour()
        #RG19_X_res, RG19_y_res = RG19_cnn.fit_resample(RG19_ova_X_train, RG19_ova_y_train)
        RG2_cnn = CondensedNearestNeighbour()
        RG2_X_res, RG2_y_res = RG2_cnn.fit_resample(RG2_ova_X_train, RG2_ova_y_train)
        RG3_cnn = CondensedNearestNeighbour()
        RG3_X_res, RG3_y_res = RG3_cnn.fit_resample(RG3_ova_X_train, RG3_ova_y_train)
        RGM_cnn = CondensedNearestNeighbour()
        RGM_X_res, RGM_y_res = RGM_cnn.fit_resample(RGM_ova_X_train, RGM_ova_y_train)
        RGQC_cnn = CondensedNearestNeighbour()
        RGQC_X_res, RGQC_y_res = RGQC_cnn.fit_resample(RGQC_ova_X_train, RGQC_ova_y_train)
        #TMSA10_cnn = CondensedNearestNeighbour()
        #TMSA10_X_res, TMSA10_y_res = TMSA10_cnn.fit_resample(TMSA10_ova_X_train, TMSA10_ova_y_train)
        T8_cnn = CondensedNearestNeighbour()
        T8_X_res, T8_y_res = T8_cnn.fit_resample(T8_ova_X_train, T8_ova_y_train)
        #T9_cnn = CondensedNearestNeighbour()
        #T9_X_res, T9_y_res = T9_cnn.fit_resample(T9_ova_X_train, T9_ova_y_train)
        TM10_cnn = CondensedNearestNeighbour()
        TM10_X_res, TM10_y_res = TM10_cnn.fit_resample(TM10_ova_X_train, TM10_ova_y_train)
        TM4_cnn = CondensedNearestNeighbour()
        TM4_X_res, TM4_y_res = TM4_cnn.fit_resample(TM4_ova_X_train, TM4_ova_y_train)
        TM5_cnn = CondensedNearestNeighbour()
        TM5_X_res, TM5_y_res = TM5_cnn.fit_resample(TM5_ova_X_train, TM5_ova_y_train)
        TM6_cnn = CondensedNearestNeighbour()
        TM6_X_res, TM6_y_res = TM6_cnn.fit_resample(TM6_ova_X_train, TM6_ova_y_train)
        TM8_cnn = CondensedNearestNeighbour()
        TM8_X_res, TM8_y_res = TM8_cnn.fit_resample(TM8_ova_X_train, TM8_ova_y_train)
        TM9_cnn = CondensedNearestNeighbour()
        TM9_X_res, TM9_y_res = TM9_cnn.fit_resample(TM9_ova_X_train, TM9_ova_y_train)
        TMQC_cnn = CondensedNearestNeighbour()
        TMQC_X_res, TMQC_y_res = TMQC_cnn.fit_resample(TMQC_ova_X_train, TMQC_ova_y_train)
        TQC_cnn = CondensedNearestNeighbour()
        TQC_X_res, TQC_y_res = TQC_cnn.fit_resample(TQC_ova_X_train, TQC_ova_y_train)
        #WC13_cnn = CondensedNearestNeighbour()
        #WC13_X_res, WC13_y_res = WC13_cnn.fit_resample(WC13_ova_X_train, WC13_ova_y_train)

    if imb_technique == "ENN":
        from imblearn.under_sampling import EditedNearestNeighbours

        #DM_enn = EditedNearestNeighbours()
        #DM_X_res, DM_y_res = DM_enn.fit_resample(DM_ova_X_train, DM_ova_y_train)
        FI_enn = EditedNearestNeighbours()
        FI_X_res, FI_y_res = FI_enn.fit_resample(FI_ova_X_train, FI_ova_y_train)
        FG_enn = EditedNearestNeighbours()
        FG_X_res, FG_y_res = FG_enn.fit_resample(FG_ova_X_train, FG_ova_y_train)
        #GR_enn = EditedNearestNeighbours()
        #GR_X_res, GR_y_res = GR_enn.fit_resample(GR_ova_X_train, GR_ova_y_train)
        #GR12_enn = EditedNearestNeighbours()
        #GR12_X_res, GR12_y_res = GR12_enn.fit_resample(GR12_ova_X_train, GR12_ova_y_train)
        GR27_enn = EditedNearestNeighbours()
        GR27_X_res, GR27_y_res = GR27_enn.fit_resample(GR27_ova_X_train, GR27_ova_y_train)
        LM_enn = EditedNearestNeighbours()
        LM_X_res, LM_y_res = LM_enn.fit_resample(LM_ova_X_train, LM_ova_y_train)
        LMM_enn = EditedNearestNeighbours()
        LMM_X_res, LMM_y_res = LMM_enn.fit_resample(LMM_ova_X_train, LMM_ova_y_train)
        #MM14_enn = EditedNearestNeighbours()
        #MM14_X_res, MM14_y_res = MM14_enn.fit_resample(MM14_ova_X_train, MM14_ova_y_train)
        #MM16_enn = EditedNearestNeighbours()
        #MM16_X_res, MM16_y_res = MM16_enn.fit_resample(MM16_ova_X_train, MM16_ova_y_train)
        PC_enn = EditedNearestNeighbours()
        PC_X_res, PC_y_res = PC_enn.fit_resample(PC_ova_X_train, PC_ova_y_train)
        RG12_enn = EditedNearestNeighbours()
        RG12_X_res, RG12_y_res = RG12_enn.fit_resample(RG12_ova_X_train, RG12_ova_y_train)
        #RG19_enn = EditedNearestNeighbours()
        #RG19_X_res, RG19_y_res = RG19_enn.fit_resample(RG19_ova_X_train, RG19_ova_y_train)
        RG2_enn = EditedNearestNeighbours()
        RG2_X_res, RG2_y_res = RG2_enn.fit_resample(RG2_ova_X_train, RG2_ova_y_train)
        RG3_enn = EditedNearestNeighbours()
        RG3_X_res, RG3_y_res = RG3_enn.fit_resample(RG3_ova_X_train, RG3_ova_y_train)
        RGM_enn = EditedNearestNeighbours()
        RGM_X_res, RGM_y_res = RGM_enn.fit_resample(RGM_ova_X_train, RGM_ova_y_train)
        RGQC_enn = EditedNearestNeighbours()
        RGQC_X_res, RGQC_y_res = RGQC_enn.fit_resample(RGQC_ova_X_train, RGQC_ova_y_train)
        #TMSA10_enn = EditedNearestNeighbours()
        #TMSA10_X_res, TMSA10_y_res = TMSA10_enn.fit_resample(TMSA10_ova_X_train, TMSA10_ova_y_train)
        T8_enn = EditedNearestNeighbours()
        T8_X_res, T8_y_res = T8_enn.fit_resample(T8_ova_X_train, T8_ova_y_train)
        #T9_enn = EditedNearestNeighbours()
        #T9_X_res, T9_y_res = T9_enn.fit_resample(T9_ova_X_train, T9_ova_y_train)
        TM10_enn = EditedNearestNeighbours()
        TM10_X_res, TM10_y_res = TM10_enn.fit_resample(TM10_ova_X_train, TM10_ova_y_train)
        TM4_enn = EditedNearestNeighbours()
        TM4_X_res, TM4_y_res = TM4_enn.fit_resample(TM4_ova_X_train, TM4_ova_y_train)
        TM5_enn = EditedNearestNeighbours()
        TM5_X_res, TM5_y_res = TM5_enn.fit_resample(TM5_ova_X_train, TM5_ova_y_train)
        TM6_enn = EditedNearestNeighbours()
        TM6_X_res, TM6_y_res = TM6_enn.fit_resample(TM6_ova_X_train, TM6_ova_y_train)
        TM8_enn = EditedNearestNeighbours()
        TM8_X_res, TM8_y_res = TM8_enn.fit_resample(TM8_ova_X_train, TM8_ova_y_train)
        TM9_enn = EditedNearestNeighbours()
        TM9_X_res, TM9_y_res = TM9_enn.fit_resample(TM9_ova_X_train, TM9_ova_y_train)
        TMQC_enn = EditedNearestNeighbours()
        TMQC_X_res, TMQC_y_res = TMQC_enn.fit_resample(TMQC_ova_X_train, TMQC_ova_y_train)
        TQC_enn = EditedNearestNeighbours()
        TQC_X_res, TQC_y_res = TQC_enn.fit_resample(TQC_ova_X_train, TQC_ova_y_train)
        #WC13_enn = EditedNearestNeighbours()
        #WC13_X_res, WC13_y_res = WC13_enn.fit_resample(WC13_ova_X_train, WC13_ova_y_train)

    if imb_technique == "IHT":
        from imblearn.under_sampling import InstanceHardnessThreshold

        #DM_iht = InstanceHardnessThreshold()
        #DM_X_res, DM_y_res = DM_iht.fit_resample(DM_ova_X_train, DM_ova_y_train)
        FI_iht = InstanceHardnessThreshold()
        FI_X_res, FI_y_res = FI_iht.fit_resample(FI_ova_X_train, FI_ova_y_train)
        FG_iht = InstanceHardnessThreshold()
        FG_X_res, FG_y_res = FG_iht.fit_resample(FG_ova_X_train, FG_ova_y_train)
        #GR_iht = InstanceHardnessThreshold()
        #GR_X_res, GR_y_res = GR_iht.fit_resample(GR_ova_X_train, GR_ova_y_train)
        #GR12_iht = InstanceHardnessThreshold()
        #GR12_X_res, GR12_y_res = GR12_iht.fit_resample(GR12_ova_X_train, GR12_ova_y_train)
        GR27_iht = InstanceHardnessThreshold()
        GR27_X_res, GR27_y_res = GR27_iht.fit_resample(GR27_ova_X_train, GR27_ova_y_train)
        LM_iht = InstanceHardnessThreshold()
        LM_X_res, LM_y_res = LM_iht.fit_resample(LM_ova_X_train, LM_ova_y_train)
        LMM_iht = InstanceHardnessThreshold()
        LMM_X_res, LMM_y_res = LMM_iht.fit_resample(LMM_ova_X_train, LMM_ova_y_train)
        #MM14_iht = InstanceHardnessThreshold()
        #MM14_X_res, MM14_y_res = MM14_iht.fit_resample(MM14_ova_X_train, MM14_ova_y_train)
        #MM16_iht = InstanceHardnessThreshold()
        #MM16_X_res, MM16_y_res = MM16_iht.fit_resample(MM16_ova_X_train, MM16_ova_y_train)
        PC_iht = InstanceHardnessThreshold()
        PC_X_res, PC_y_res = PC_iht.fit_resample(PC_ova_X_train, PC_ova_y_train)
        RG12_iht = InstanceHardnessThreshold()
        RG12_X_res, RG12_y_res = RG12_iht.fit_resample(RG12_ova_X_train, RG12_ova_y_train)
        #RG19_iht = InstanceHardnessThreshold()
        #RG19_X_res, RG19_y_res = RG19_iht.fit_resample(RG19_ova_X_train, RG19_ova_y_train)
        RG2_iht = InstanceHardnessThreshold()
        RG2_X_res, RG2_y_res = RG2_iht.fit_resample(RG2_ova_X_train, RG2_ova_y_train)
        RG3_iht = InstanceHardnessThreshold()
        RG3_X_res, RG3_y_res = RG3_iht.fit_resample(RG3_ova_X_train, RG3_ova_y_train)
        RGM_iht = InstanceHardnessThreshold()
        RGM_X_res, RGM_y_res = RGM_iht.fit_resample(RGM_ova_X_train, RGM_ova_y_train)
        RGQC_iht = InstanceHardnessThreshold()
        RGQC_X_res, RGQC_y_res = RGQC_iht.fit_resample(RGQC_ova_X_train, RGQC_ova_y_train)
        #TMSA10_iht = InstanceHardnessThreshold()
        #TMSA10_X_res, TMSA10_y_res = TMSA10_iht.fit_resample(TMSA10_ova_X_train, TMSA10_ova_y_train)
        T8_iht = InstanceHardnessThreshold()
        T8_X_res, T8_y_res = T8_iht.fit_resample(T8_ova_X_train, T8_ova_y_train)
        #T9_iht = InstanceHardnessThreshold()
        #T9_X_res, T9_y_res = T9_iht.fit_resample(T9_ova_X_train, T9_ova_y_train)
        TM10_iht = InstanceHardnessThreshold()
        TM10_X_res, TM10_y_res = TM10_iht.fit_resample(TM10_ova_X_train, TM10_ova_y_train)
        TM4_iht = InstanceHardnessThreshold()
        TM4_X_res, TM4_y_res = TM4_iht.fit_resample(TM4_ova_X_train, TM4_ova_y_train)
        TM5_iht = InstanceHardnessThreshold()
        TM5_X_res, TM5_y_res = TM5_iht.fit_resample(TM5_ova_X_train, TM5_ova_y_train)
        TM6_iht = InstanceHardnessThreshold()
        TM6_X_res, TM6_y_res = TM6_iht.fit_resample(TM6_ova_X_train, TM6_ova_y_train)
        TM8_iht = InstanceHardnessThreshold()
        TM8_X_res, TM8_y_res = TM8_iht.fit_resample(TM8_ova_X_train, TM8_ova_y_train)
        TM9_iht = InstanceHardnessThreshold()
        TM9_X_res, TM9_y_res = TM9_iht.fit_resample(TM9_ova_X_train, TM9_ova_y_train)
        TMQC_iht = InstanceHardnessThreshold()
        TMQC_X_res, TMQC_y_res = TMQC_iht.fit_resample(TMQC_ova_X_train, TMQC_ova_y_train)
        TQC_iht = InstanceHardnessThreshold()
        TQC_X_res, TQC_y_res = TQC_iht.fit_resample(TQC_ova_X_train, TQC_ova_y_train)
        #WC13_iht = InstanceHardnessThreshold()
        #WC13_X_res, WC13_y_res = WC13_iht.fit_resample(WC13_ova_X_train, WC13_ova_y_train)

    if imb_technique == "NCR":
        from imblearn.under_sampling import NeighbourhoodCleaningRule

        #DM_ncr = NeighbourhoodCleaningRule()
        #DM_ova_y_train = [0 if i == "Deburring - Manual" else 1 for i in DM_ova_y_train]
        #DM_X_res, DM_y_res = DM_ncr.fit_resample(DM_ova_X_train, DM_ova_y_train)
        FI_ncr = NeighbourhoodCleaningRule()
        FI_ova_y_train = [0 if i == "Final Inspection Q.C." else 1 for i in FI_ova_y_train]
        FI_X_res, FI_y_res = FI_ncr.fit_resample(FI_ova_X_train, FI_ova_y_train)
        FG_ncr = NeighbourhoodCleaningRule()
        FG_ova_y_train = [0 if i == "Flat Grinding - Machine 11" else 1 for i in FG_ova_y_train]
        FG_X_res, FG_y_res = FG_ncr.fit_resample(FG_ova_X_train, FG_ova_y_train)
        #GR_ncr = NeighbourhoodCleaningRule()
        #GR_ova_y_train = [0 if i == "Grinding Rework" else 1 for i in GR_ova_y_train]
        #GR_X_res, GR_y_res = GR_ncr.fit_resample(GR_ova_X_train, GR_ova_y_train)
        #GR12_ncr = NeighbourhoodCleaningRule()
        #GR12_ova_y_train = [0 if i == "Grinding Rework - Machine 12" else 1 for i in GR12_ova_y_train]
        #GR12_X_res, GR12_y_res = GR12_ncr.fit_resample(GR12_ova_X_train, GR12_ova_y_train)
        GR27_ncr = NeighbourhoodCleaningRule()
        GR27_ova_y_train = [0 if i == "Grinding Rework - Machine 27" else 1 for i in GR27_ova_y_train]
        GR27_X_res, GR27_y_res = GR27_ncr.fit_resample(GR27_ova_X_train, GR27_ova_y_train)
        LM_ncr = NeighbourhoodCleaningRule()
        LM_ova_y_train = [0 if i == "Lapping - Machine 1" else 1 for i in LM_ova_y_train]
        LM_X_res, LM_y_res = LM_ncr.fit_resample(LM_ova_X_train, LM_ova_y_train)
        LMM_ncr = NeighbourhoodCleaningRule()
        LMM_ova_y_train = [0 if i == "Laser Marking - Machine 7" else 1 for i in LMM_ova_y_train]
        LMM_X_res, LMM_y_res = LMM_ncr.fit_resample(LMM_ova_X_train, LMM_ova_y_train)
        #MM14_ncr = NeighbourhoodCleaningRule()
        #MM14_ova_y_train = [0 if i == "Milling - Machine 14" else 1 for i in MM14_ova_y_train]
        #MM14_X_res, MM14_y_res = MM14_ncr.fit_resample(MM14_ova_X_train, MM14_ova_y_train)
        #MM16_ncr = NeighbourhoodCleaningRule()
        #MM16_ova_y_train = [0 if i == "Milling - Machine 16" else 1 for i in MM16_ova_y_train]
        #MM16_X_res, MM16_y_res = MM16_ncr.fit_resample(MM16_ova_X_train, MM16_ova_y_train)
        PC_ncr = NeighbourhoodCleaningRule()
        PC_ova_y_train = [0 if i == "Packing" else 1 for i in PC_ova_y_train]
        PC_X_res, PC_y_res = PC_ncr.fit_resample(PC_ova_X_train, PC_ova_y_train)
        RG12_ncr = NeighbourhoodCleaningRule()
        RG12_ova_y_train = [0 if i == "Round Grinding - Machine 12" else 1 for i in RG12_ova_y_train]
        RG12_X_res, RG12_y_res = RG12_ncr.fit_resample(RG12_ova_X_train, RG12_ova_y_train)
        #RG19_ncr = NeighbourhoodCleaningRule()
        #RG19_ova_y_train = [0 if i == "Round Grinding - Machine 19" else 1 for i in RG19_ova_y_train]
        #RG19_X_res, RG19_y_res = RG19_ncr.fit_resample(RG19_ova_X_train, RG19_ova_y_train)
        RG2_ncr = NeighbourhoodCleaningRule()
        RG2_ova_y_train = [0 if i == "Round Grinding - Machine 2" else 1 for i in RG2_ova_y_train]
        RG2_X_res, RG2_y_res = RG2_ncr.fit_resample(RG2_ova_X_train, RG2_ova_y_train)
        RG3_ncr = NeighbourhoodCleaningRule()
        RG3_ova_y_train = [0 if i == "Round Grinding - Machine 3" else 1 for i in RG3_ova_y_train]
        RG3_X_res, RG3_y_res = RG3_ncr.fit_resample(RG3_ova_X_train, RG3_ova_y_train)
        RGM_ncr = NeighbourhoodCleaningRule()
        RGM_ova_y_train = [0 if i == "Round Grinding - Manual" else 1 for i in RGM_ova_y_train]
        RGM_X_res, RGM_y_res = RGM_ncr.fit_resample(RGM_ova_X_train, RGM_ova_y_train)
        RGQC_ncr = NeighbourhoodCleaningRule()
        RGQC_ova_y_train = [0 if i == "Round Grinding - Q.C." else 1 for i in RGQC_ova_y_train]
        RGQC_X_res, RGQC_y_res = RGQC_ncr.fit_resample(RGQC_ova_X_train, RGQC_ova_y_train)
        #TMSA10_ncr = NeighbourhoodCleaningRule()
        #TMSA10_ova_y_train = [0 if i == "Turn & Mill. & Screw Assem - Machine 10" else 1 for i in TMSA10_ova_y_train]
        #TMSA10_X_res, TMSA10_y_res = TMSA10_ncr.fit_resample(TMSA10_ova_X_train, TMSA10_ova_y_train)
        T8_ncr = NeighbourhoodCleaningRule()
        T8_ova_y_train = [0 if i == "Turning - Machine 8" else 1 for i in T8_ova_y_train]
        T8_X_res, T8_y_res = T8_ncr.fit_resample(T8_ova_X_train, T8_ova_y_train)
        #T9_ncr = NeighbourhoodCleaningRule()
        #T9_ova_y_train = [0 if i == "Turning - Machine 9" else 1 for i in T9_ova_y_train]
        #T9_X_res, T9_y_res = T9_ncr.fit_resample(T9_ova_X_train, T9_ova_y_train)
        TM10_ncr = NeighbourhoodCleaningRule()
        TM10_ova_y_train = [0 if i == "Turning & Milling - Machine 10" else 1 for i in TM10_ova_y_train]
        TM10_X_res, TM10_y_res = TM10_ncr.fit_resample(TM10_ova_X_train, TM10_ova_y_train)
        TM4_ncr = NeighbourhoodCleaningRule()
        TM4_ova_y_train = [0 if i == "Turning & Milling - Machine 4" else 1 for i in TM4_ova_y_train]
        TM4_X_res, TM4_y_res = TM4_ncr.fit_resample(TM4_ova_X_train, TM4_ova_y_train)
        TM5_ncr = NeighbourhoodCleaningRule()
        TM5_ova_y_train = [0 if i == "Turning & Milling - Machine 5" else 1 for i in TM5_ova_y_train]
        TM5_X_res, TM5_y_res = TM5_ncr.fit_resample(TM5_ova_X_train, TM5_ova_y_train)
        TM6_ncr = NeighbourhoodCleaningRule()
        TM6_ova_y_train = [0 if i == "Turning & Milling - Machine 6" else 1 for i in TM6_ova_y_train]
        TM6_X_res, TM6_y_res = TM6_ncr.fit_resample(TM6_ova_X_train, TM6_ova_y_train)
        TM8_ncr = NeighbourhoodCleaningRule()
        TM8_ova_y_train = [0 if i == "Turning & Milling - Machine 8" else 1 for i in TM8_ova_y_train]
        TM8_X_res, TM8_y_res = TM8_ncr.fit_resample(TM8_ova_X_train, TM8_ova_y_train)
        TM9_ncr = NeighbourhoodCleaningRule()
        TM9_ova_y_train = [0 if i == "Turning & Milling - Machine 9" else 1 for i in TM9_ova_y_train]
        TM9_X_res, TM9_y_res = TM9_ncr.fit_resample(TM9_ova_X_train, TM9_ova_y_train)
        TMQC_ncr = NeighbourhoodCleaningRule()
        TMQC_ova_y_train = [0 if i == "Turning & Milling Q.C." else 1 for i in TMQC_ova_y_train]
        TMQC_X_res, TMQC_y_res = TMQC_ncr.fit_resample(TMQC_ova_X_train, TMQC_ova_y_train)
        TQC_ncr = NeighbourhoodCleaningRule()
        TQC_ova_y_train = [0 if i == "Turning Q.C." else 1 for i in TQC_ova_y_train]
        TQC_X_res, TQC_y_res = TQC_ncr.fit_resample(TQC_ova_X_train, TQC_ova_y_train)
        #WC13_ncr = NeighbourhoodCleaningRule()
        #WC13_ova_y_train = [0 if i == "Wire Cut - Machine 13" else 1 for i in WC13_ova_y_train]
        #WC13_X_res, WC13_y_res = WC13_ncr.fit_resample(WC13_ova_X_train, WC13_ova_y_train)

    if imb_technique == "NM":
        from imblearn.under_sampling import NearMiss

        #DM_nm = NearMiss()
        #DM_X_res, DM_y_res = DM_nm.fit_resample(DM_ova_X_train, DM_ova_y_train)
        FI_nm = NearMiss()
        FI_X_res, FI_y_res = FI_nm.fit_resample(FI_ova_X_train, FI_ova_y_train)
        FG_nm = NearMiss()
        FG_X_res, FG_y_res = FG_nm.fit_resample(FG_ova_X_train, FG_ova_y_train)
        #GR_nm = NearMiss()
        #GR_X_res, GR_y_res = GR_nm.fit_resample(GR_ova_X_train, GR_ova_y_train)
        #GR12_nm = NearMiss()
        #GR12_X_res, GR12_y_res = GR12_nm.fit_resample(GR12_ova_X_train, GR12_ova_y_train)
        GR27_nm = NearMiss()
        GR27_X_res, GR27_y_res = GR27_nm.fit_resample(GR27_ova_X_train, GR27_ova_y_train)
        LM_nm = NearMiss()
        LM_X_res, LM_y_res = LM_nm.fit_resample(LM_ova_X_train, LM_ova_y_train)
        LMM_nm = NearMiss()
        LMM_X_res, LMM_y_res = LMM_nm.fit_resample(LMM_ova_X_train, LMM_ova_y_train)
        #MM14_nm = NearMiss()
        #MM14_X_res, MM14_y_res = MM14_nm.fit_resample(MM14_ova_X_train, MM14_ova_y_train)
        #MM16_nm = NearMiss()
        #MM16_X_res, MM16_y_res = MM16_nm.fit_resample(MM16_ova_X_train, MM16_ova_y_train)
        PC_nm = NearMiss()
        PC_X_res, PC_y_res = PC_nm.fit_resample(PC_ova_X_train, PC_ova_y_train)
        RG12_nm = NearMiss()
        RG12_X_res, RG12_y_res = RG12_nm.fit_resample(RG12_ova_X_train, RG12_ova_y_train)
        #RG19_nm = NearMiss()
        #RG19_X_res, RG19_y_res = RG19_nm.fit_resample(RG19_ova_X_train, RG19_ova_y_train)
        RG2_nm = NearMiss()
        RG2_X_res, RG2_y_res = RG2_nm.fit_resample(RG2_ova_X_train, RG2_ova_y_train)
        RG3_nm = NearMiss()
        RG3_X_res, RG3_y_res = RG3_nm.fit_resample(RG3_ova_X_train, RG3_ova_y_train)
        RGM_nm = NearMiss()
        RGM_X_res, RGM_y_res = RGM_nm.fit_resample(RGM_ova_X_train, RGM_ova_y_train)
        RGQC_nm = NearMiss()
        RGQC_X_res, RGQC_y_res = RGQC_nm.fit_resample(RGQC_ova_X_train, RGQC_ova_y_train)
        #TMSA10_nm = NearMiss()
        #TMSA10_X_res, TMSA10_y_res = TMSA10_nm.fit_resample(TMSA10_ova_X_train, TMSA10_ova_y_train)
        T8_nm = NearMiss()
        T8_X_res, T8_y_res = T8_nm.fit_resample(T8_ova_X_train, T8_ova_y_train)
        #T9_nm = NearMiss()
        #T9_X_res, T9_y_res = T9_nm.fit_resample(T9_ova_X_train, T9_ova_y_train)
        TM10_nm = NearMiss()
        TM10_X_res, TM10_y_res = TM10_nm.fit_resample(TM10_ova_X_train, TM10_ova_y_train)
        TM4_nm = NearMiss()
        TM4_X_res, TM4_y_res = TM4_nm.fit_resample(TM4_ova_X_train, TM4_ova_y_train)
        TM5_nm = NearMiss()
        TM5_X_res, TM5_y_res = TM5_nm.fit_resample(TM5_ova_X_train, TM5_ova_y_train)
        TM6_nm = NearMiss()
        TM6_X_res, TM6_y_res = TM6_nm.fit_resample(TM6_ova_X_train, TM6_ova_y_train)
        TM8_nm = NearMiss()
        TM8_X_res, TM8_y_res = TM8_nm.fit_resample(TM8_ova_X_train, TM8_ova_y_train)
        TM9_nm = NearMiss()
        TM9_X_res, TM9_y_res = TM9_nm.fit_resample(TM9_ova_X_train, TM9_ova_y_train)
        TMQC_nm = NearMiss()
        TMQC_X_res, TMQC_y_res = TMQC_nm.fit_resample(TMQC_ova_X_train, TMQC_ova_y_train)
        TQC_nm = NearMiss()
        TQC_X_res, TQC_y_res = TQC_nm.fit_resample(TQC_ova_X_train, TQC_ova_y_train)
        #WC13_nm = NearMiss()
        #WC13_X_res, WC13_y_res = WC13_nm.fit_resample(WC13_ova_X_train, WC13_ova_y_train)

    if imb_technique == "OSS":
        from imblearn.under_sampling import OneSidedSelection

        #DM_oss = OneSidedSelection()
        #DM_X_res, DM_y_res = DM_oss.fit_resample(DM_ova_X_train, DM_ova_y_train)
        FI_oss = OneSidedSelection()
        FI_X_res, FI_y_res = FI_oss.fit_resample(FI_ova_X_train, FI_ova_y_train)
        FG_oss = OneSidedSelection()
        FG_X_res, FG_y_res = FG_oss.fit_resample(FG_ova_X_train, FG_ova_y_train)
        #GR_oss = OneSidedSelection()
        #GR_X_res, GR_y_res = GR_oss.fit_resample(GR_ova_X_train, GR_ova_y_train)
        #GR12_oss = OneSidedSelection()
        #GR12_X_res, GR12_y_res = GR12_oss.fit_resample(GR12_ova_X_train, GR12_ova_y_train)
        GR27_oss = OneSidedSelection()
        GR27_X_res, GR27_y_res = GR27_oss.fit_resample(GR27_ova_X_train, GR27_ova_y_train)
        LM_oss = OneSidedSelection()
        LM_X_res, LM_y_res = LM_oss.fit_resample(LM_ova_X_train, LM_ova_y_train)
        LMM_oss = OneSidedSelection()
        LMM_X_res, LMM_y_res = LMM_oss.fit_resample(LMM_ova_X_train, LMM_ova_y_train)
        #MM14_oss = OneSidedSelection()
        #MM14_X_res, MM14_y_res = MM14_oss.fit_resample(MM14_ova_X_train, MM14_ova_y_train)
        #MM16_oss = OneSidedSelection()
        #MM16_X_res, MM16_y_res = MM16_oss.fit_resample(MM16_ova_X_train, MM16_ova_y_train)
        PC_oss = OneSidedSelection()
        PC_X_res, PC_y_res = PC_oss.fit_resample(PC_ova_X_train, PC_ova_y_train)
        RG12_oss = OneSidedSelection()
        RG12_X_res, RG12_y_res = RG12_oss.fit_resample(RG12_ova_X_train, RG12_ova_y_train)
        #RG19_oss = OneSidedSelection()
        #RG19_X_res, RG19_y_res = RG19_oss.fit_resample(RG19_ova_X_train, RG19_ova_y_train)
        RG2_oss = OneSidedSelection()
        RG2_X_res, RG2_y_res = RG2_oss.fit_resample(RG2_ova_X_train, RG2_ova_y_train)
        RG3_oss = OneSidedSelection()
        RG3_X_res, RG3_y_res = RG3_oss.fit_resample(RG3_ova_X_train, RG3_ova_y_train)
        RGM_oss = OneSidedSelection()
        RGM_X_res, RGM_y_res = RGM_oss.fit_resample(RGM_ova_X_train, RGM_ova_y_train)
        RGQC_oss = OneSidedSelection()
        RGQC_X_res, RGQC_y_res = RGQC_oss.fit_resample(RGQC_ova_X_train, RGQC_ova_y_train)
        #TMSA10_oss = OneSidedSelection()
        #TMSA10_X_res, TMSA10_y_res = TMSA10_oss.fit_resample(TMSA10_ova_X_train, TMSA10_ova_y_train)
        T8_oss = OneSidedSelection()
        T8_X_res, T8_y_res = T8_oss.fit_resample(T8_ova_X_train, T8_ova_y_train)
        #T9_oss = OneSidedSelection()
        #T9_X_res, T9_y_res = T9_oss.fit_resample(T9_ova_X_train, T9_ova_y_train)
        TM10_oss = OneSidedSelection()
        TM10_X_res, TM10_y_res = TM10_oss.fit_resample(TM10_ova_X_train, TM10_ova_y_train)
        TM4_oss = OneSidedSelection()
        TM4_X_res, TM4_y_res = TM4_oss.fit_resample(TM4_ova_X_train, TM4_ova_y_train)
        TM5_oss = OneSidedSelection()
        TM5_X_res, TM5_y_res = TM5_oss.fit_resample(TM5_ova_X_train, TM5_ova_y_train)
        TM6_oss = OneSidedSelection()
        TM6_X_res, TM6_y_res = TM6_oss.fit_resample(TM6_ova_X_train, TM6_ova_y_train)
        TM8_oss = OneSidedSelection()
        TM8_X_res, TM8_y_res = TM8_oss.fit_resample(TM8_ova_X_train, TM8_ova_y_train)
        TM9_oss = OneSidedSelection()
        TM9_X_res, TM9_y_res = TM9_oss.fit_resample(TM9_ova_X_train, TM9_ova_y_train)
        TMQC_oss = OneSidedSelection()
        TMQC_X_res, TMQC_y_res = TMQC_oss.fit_resample(TMQC_ova_X_train, TMQC_ova_y_train)
        TQC_oss = OneSidedSelection()
        TQC_X_res, TQC_y_res = TQC_oss.fit_resample(TQC_ova_X_train, TQC_ova_y_train)
        #WC13_oss = OneSidedSelection()
        #WC13_X_res, WC13_y_res = WC13_oss.fit_resample(WC13_ova_X_train, WC13_ova_y_train)

    if imb_technique == "RENN":
        from imblearn.under_sampling import RepeatedEditedNearestNeighbours

        #DM_renn = RepeatedEditedNearestNeighbours()
        #DM_X_res, DM_y_res = DM_renn.fit_resample(DM_ova_X_train, DM_ova_y_train)
        FI_renn = RepeatedEditedNearestNeighbours()
        FI_X_res, FI_y_res = FI_renn.fit_resample(FI_ova_X_train, FI_ova_y_train)
        FG_renn = RepeatedEditedNearestNeighbours()
        FG_X_res, FG_y_res = FG_renn.fit_resample(FG_ova_X_train, FG_ova_y_train)
        #GR_renn = RepeatedEditedNearestNeighbours()
        #GR_X_res, GR_y_res = GR_renn.fit_resample(GR_ova_X_train, GR_ova_y_train)
        #GR12_renn = RepeatedEditedNearestNeighbours()
        #GR12_X_res, GR12_y_res = GR12_renn.fit_resample(GR12_ova_X_train, GR12_ova_y_train)
        GR27_renn = RepeatedEditedNearestNeighbours()
        GR27_X_res, GR27_y_res = GR27_renn.fit_resample(GR27_ova_X_train, GR27_ova_y_train)
        LM_renn = RepeatedEditedNearestNeighbours()
        LM_X_res, LM_y_res = LM_renn.fit_resample(LM_ova_X_train, LM_ova_y_train)
        LMM_renn = RepeatedEditedNearestNeighbours()
        LMM_X_res, LMM_y_res = LMM_renn.fit_resample(LMM_ova_X_train, LMM_ova_y_train)
        #MM14_renn = RepeatedEditedNearestNeighbours()
        #MM14_X_res, MM14_y_res = MM14_renn.fit_resample(MM14_ova_X_train, MM14_ova_y_train)
        #MM16_renn = RepeatedEditedNearestNeighbours()
        #MM16_X_res, MM16_y_res = MM16_renn.fit_resample(MM16_ova_X_train, MM16_ova_y_train)
        PC_renn = RepeatedEditedNearestNeighbours()
        PC_X_res, PC_y_res = PC_renn.fit_resample(PC_ova_X_train, PC_ova_y_train)
        RG12_renn = RepeatedEditedNearestNeighbours()
        RG12_X_res, RG12_y_res = RG12_renn.fit_resample(RG12_ova_X_train, RG12_ova_y_train)
        #RG19_renn = RepeatedEditedNearestNeighbours()
        #RG19_X_res, RG19_y_res = RG19_renn.fit_resample(RG19_ova_X_train, RG19_ova_y_train)
        RG2_renn = RepeatedEditedNearestNeighbours()
        RG2_X_res, RG2_y_res = RG2_renn.fit_resample(RG2_ova_X_train, RG2_ova_y_train)
        RG3_renn = RepeatedEditedNearestNeighbours()
        RG3_X_res, RG3_y_res = RG3_renn.fit_resample(RG3_ova_X_train, RG3_ova_y_train)
        RGM_renn = RepeatedEditedNearestNeighbours()
        RGM_X_res, RGM_y_res = RGM_renn.fit_resample(RGM_ova_X_train, RGM_ova_y_train)
        RGQC_renn = RepeatedEditedNearestNeighbours()
        RGQC_X_res, RGQC_y_res = RGQC_renn.fit_resample(RGQC_ova_X_train, RGQC_ova_y_train)
        #TMSA10_renn = RepeatedEditedNearestNeighbours()
        #TMSA10_X_res, TMSA10_y_res = TMSA10_renn.fit_resample(TMSA10_ova_X_train, TMSA10_ova_y_train)
        T8_renn = RepeatedEditedNearestNeighbours()
        T8_X_res, T8_y_res = T8_renn.fit_resample(T8_ova_X_train, T8_ova_y_train)
        #T9_renn = RepeatedEditedNearestNeighbours()
        #T9_X_res, T9_y_res = T9_renn.fit_resample(T9_ova_X_train, T9_ova_y_train)
        TM10_renn = RepeatedEditedNearestNeighbours()
        TM10_X_res, TM10_y_res = TM10_renn.fit_resample(TM10_ova_X_train, TM10_ova_y_train)
        TM4_renn = RepeatedEditedNearestNeighbours()
        TM4_X_res, TM4_y_res = TM4_renn.fit_resample(TM4_ova_X_train, TM4_ova_y_train)
        TM5_renn = RepeatedEditedNearestNeighbours()
        TM5_X_res, TM5_y_res = TM5_renn.fit_resample(TM5_ova_X_train, TM5_ova_y_train)
        TM6_renn = RepeatedEditedNearestNeighbours()
        TM6_X_res, TM6_y_res = TM6_renn.fit_resample(TM6_ova_X_train, TM6_ova_y_train)
        TM8_renn = RepeatedEditedNearestNeighbours()
        TM8_X_res, TM8_y_res = TM8_renn.fit_resample(TM8_ova_X_train, TM8_ova_y_train)
        TM9_renn = RepeatedEditedNearestNeighbours()
        TM9_X_res, TM9_y_res = TM9_renn.fit_resample(TM9_ova_X_train, TM9_ova_y_train)
        TMQC_renn = RepeatedEditedNearestNeighbours()
        TMQC_X_res, TMQC_y_res = TMQC_renn.fit_resample(TMQC_ova_X_train, TMQC_ova_y_train)
        TQC_renn = RepeatedEditedNearestNeighbours()
        TQC_X_res, TQC_y_res = TQC_renn.fit_resample(TQC_ova_X_train, TQC_ova_y_train)
        #WC13_renn = RepeatedEditedNearestNeighbours()
        #WC13_X_res, WC13_y_res = WC13_renn.fit_resample(WC13_ova_X_train, WC13_ova_y_train)

    if imb_technique == "BSMOTE":
        from imblearn.over_sampling import BorderlineSMOTE

        #DM_bsm = BorderlineSMOTE()
        #DM_X_res, DM_y_res = DM_bsm.fit_resample(DM_ova_X_train, DM_ova_y_train)
        FI_bsm = BorderlineSMOTE()
        FI_X_res, FI_y_res = FI_bsm.fit_resample(FI_ova_X_train, FI_ova_y_train)
        FG_bsm = BorderlineSMOTE()
        FG_X_res, FG_y_res = FG_bsm.fit_resample(FG_ova_X_train, FG_ova_y_train)
        #GR_bsm = BorderlineSMOTE()
        #GR_X_res, GR_y_res = GR_bsm.fit_resample(GR_ova_X_train, GR_ova_y_train)
        #GR12_bsm = BorderlineSMOTE()
        #GR12_X_res, GR12_y_res = GR12_bsm.fit_resample(GR12_ova_X_train, GR12_ova_y_train)
        GR27_bsm = BorderlineSMOTE()
        GR27_X_res, GR27_y_res = GR27_bsm.fit_resample(GR27_ova_X_train, GR27_ova_y_train)
        LM_bsm = BorderlineSMOTE()
        LM_X_res, LM_y_res = LM_bsm.fit_resample(LM_ova_X_train, LM_ova_y_train)
        LMM_bsm = BorderlineSMOTE()
        LMM_X_res, LMM_y_res = LMM_bsm.fit_resample(LMM_ova_X_train, LMM_ova_y_train)
        #MM14_bsm = BorderlineSMOTE()
        #MM14_X_res, MM14_y_res = MM14_bsm.fit_resample(MM14_ova_X_train, MM14_ova_y_train)
        #MM16_bsm = BorderlineSMOTE()
        #MM16_X_res, MM16_y_res = MM16_bsm.fit_resample(MM16_ova_X_train, MM16_ova_y_train)
        PC_bsm = BorderlineSMOTE()
        PC_X_res, PC_y_res = PC_bsm.fit_resample(PC_ova_X_train, PC_ova_y_train)
        RG12_bsm = BorderlineSMOTE()
        RG12_X_res, RG12_y_res = RG12_bsm.fit_resample(RG12_ova_X_train, RG12_ova_y_train)
        #RG19_bsm = BorderlineSMOTE()
        #RG19_X_res, RG19_y_res = RG19_bsm.fit_resample(RG19_ova_X_train, RG19_ova_y_train)
        RG2_bsm = BorderlineSMOTE()
        RG2_X_res, RG2_y_res = RG2_bsm.fit_resample(RG2_ova_X_train, RG2_ova_y_train)
        RG3_bsm = BorderlineSMOTE()
        RG3_X_res, RG3_y_res = RG3_bsm.fit_resample(RG3_ova_X_train, RG3_ova_y_train)
        RGM_bsm = BorderlineSMOTE()
        RGM_X_res, RGM_y_res = RGM_bsm.fit_resample(RGM_ova_X_train, RGM_ova_y_train)
        RGQC_bsm = BorderlineSMOTE()
        RGQC_X_res, RGQC_y_res = RGQC_bsm.fit_resample(RGQC_ova_X_train, RGQC_ova_y_train)
        #TMSA10_bsm = BorderlineSMOTE()
        #TMSA10_X_res, TMSA10_y_res = TMSA10_bsm.fit_resample(TMSA10_ova_X_train, TMSA10_ova_y_train)
        T8_bsm = BorderlineSMOTE()
        T8_X_res, T8_y_res = T8_bsm.fit_resample(T8_ova_X_train, T8_ova_y_train)
        #T9_bsm = BorderlineSMOTE()
        #T9_X_res, T9_y_res = T9_bsm.fit_resample(T9_ova_X_train, T9_ova_y_train)
        TM10_bsm = BorderlineSMOTE()
        TM10_X_res, TM10_y_res = TM10_bsm.fit_resample(TM10_ova_X_train, TM10_ova_y_train)
        TM4_bsm = BorderlineSMOTE()
        TM4_X_res, TM4_y_res = TM4_bsm.fit_resample(TM4_ova_X_train, TM4_ova_y_train)
        TM5_bsm = BorderlineSMOTE()
        TM5_X_res, TM5_y_res = TM5_bsm.fit_resample(TM5_ova_X_train, TM5_ova_y_train)
        TM6_bsm = BorderlineSMOTE()
        TM6_X_res, TM6_y_res = TM6_bsm.fit_resample(TM6_ova_X_train, TM6_ova_y_train)
        TM8_bsm = BorderlineSMOTE()
        TM8_X_res, TM8_y_res = TM8_bsm.fit_resample(TM8_ova_X_train, TM8_ova_y_train)
        TM9_bsm = BorderlineSMOTE()
        TM9_X_res, TM9_y_res = TM9_bsm.fit_resample(TM9_ova_X_train, TM9_ova_y_train)
        TMQC_bsm = BorderlineSMOTE()
        TMQC_X_res, TMQC_y_res = TMQC_bsm.fit_resample(TMQC_ova_X_train, TMQC_ova_y_train)
        TQC_bsm = BorderlineSMOTE()
        TQC_X_res, TQC_y_res = TQC_bsm.fit_resample(TQC_ova_X_train, TQC_ova_y_train)
        #WC13_bsm = BorderlineSMOTE()
        #WC13_X_res, WC13_y_res = WC13_bsm.fit_resample(WC13_ova_X_train, WC13_ova_y_train)

    if imb_technique == "SMOTE":
        from imblearn.over_sampling import SMOTE

        #DM_sm = SMOTE()
        #DM_X_res, DM_y_res = DM_sm.fit_resample(DM_ova_X_train, DM_ova_y_train)
        FI_sm = SMOTE()
        FI_X_res, FI_y_res = FI_sm.fit_resample(FI_ova_X_train, FI_ova_y_train)
        FG_sm = SMOTE()
        FG_X_res, FG_y_res = FG_sm.fit_resample(FG_ova_X_train, FG_ova_y_train)
        #GR_sm = SMOTE()
        #GR_X_res, GR_y_res = GR_sm.fit_resample(GR_ova_X_train, GR_ova_y_train)
        #GR12_sm = SMOTE()
        #GR12_X_res, GR12_y_res = GR12_sm.fit_resample(GR12_ova_X_train, GR12_ova_y_train)
        GR27_sm = SMOTE()
        GR27_X_res, GR27_y_res = GR27_sm.fit_resample(GR27_ova_X_train, GR27_ova_y_train)
        LM_sm = SMOTE()
        LM_X_res, LM_y_res = LM_sm.fit_resample(LM_ova_X_train, LM_ova_y_train)
        LMM_sm = SMOTE()
        LMM_X_res, LMM_y_res = LMM_sm.fit_resample(LMM_ova_X_train, LMM_ova_y_train)
        #MM14_sm = SMOTE()
        #MM14_X_res, MM14_y_res = MM14_sm.fit_resample(MM14_ova_X_train, MM14_ova_y_train)
        #MM16_sm = SMOTE()
        #MM16_X_res, MM16_y_res = MM16_sm.fit_resample(MM16_ova_X_train, MM16_ova_y_train)
        PC_sm = SMOTE()
        PC_X_res, PC_y_res = PC_sm.fit_resample(PC_ova_X_train, PC_ova_y_train)
        RG12_sm = SMOTE()
        RG12_X_res, RG12_y_res = RG12_sm.fit_resample(RG12_ova_X_train, RG12_ova_y_train)
        #RG19_sm = SMOTE()
        #RG19_X_res, RG19_y_res = RG19_sm.fit_resample(RG19_ova_X_train, RG19_ova_y_train)
        RG2_sm = SMOTE()
        RG2_X_res, RG2_y_res = RG2_sm.fit_resample(RG2_ova_X_train, RG2_ova_y_train)
        RG3_sm = SMOTE()
        RG3_X_res, RG3_y_res = RG3_sm.fit_resample(RG3_ova_X_train, RG3_ova_y_train)
        RGM_sm = SMOTE()
        RGM_X_res, RGM_y_res = RGM_sm.fit_resample(RGM_ova_X_train, RGM_ova_y_train)
        RGQC_sm = SMOTE()
        RGQC_X_res, RGQC_y_res = RGQC_sm.fit_resample(RGQC_ova_X_train, RGQC_ova_y_train)
        #TMSA10_sm = SMOTE()
        #TMSA10_X_res, TMSA10_y_res = TMSA10_sm.fit_resample(TMSA10_ova_X_train, TMSA10_ova_y_train)
        T8_sm = SMOTE()
        T8_X_res, T8_y_res = T8_sm.fit_resample(T8_ova_X_train, T8_ova_y_train)
        #T9_sm = SMOTE()
        #T9_X_res, T9_y_res = T9_sm.fit_resample(T9_ova_X_train, T9_ova_y_train)
        TM10_sm = SMOTE()
        TM10_X_res, TM10_y_res = TM10_sm.fit_resample(TM10_ova_X_train, TM10_ova_y_train)
        TM4_sm = SMOTE()
        TM4_X_res, TM4_y_res = TM4_sm.fit_resample(TM4_ova_X_train, TM4_ova_y_train)
        TM5_sm = SMOTE()
        TM5_X_res, TM5_y_res = TM5_sm.fit_resample(TM5_ova_X_train, TM5_ova_y_train)
        TM6_sm = SMOTE()
        TM6_X_res, TM6_y_res = TM6_sm.fit_resample(TM6_ova_X_train, TM6_ova_y_train)
        TM8_sm = SMOTE()
        TM8_X_res, TM8_y_res = TM8_sm.fit_resample(TM8_ova_X_train, TM8_ova_y_train)
        TM9_sm = SMOTE()
        TM9_X_res, TM9_y_res = TM9_sm.fit_resample(TM9_ova_X_train, TM9_ova_y_train)
        TMQC_sm = SMOTE()
        TMQC_X_res, TMQC_y_res = TMQC_sm.fit_resample(TMQC_ova_X_train, TMQC_ova_y_train)
        TQC_sm = SMOTE()
        TQC_X_res, TQC_y_res = TQC_sm.fit_resample(TQC_ova_X_train, TQC_ova_y_train)
        #WC13_sm = SMOTE()
        #WC13_X_res, WC13_y_res = WC13_sm.fit_resample(WC13_ova_X_train, WC13_ova_y_train)

    if imb_technique == "SMOTEENN":
        from imblearn.combine import SMOTEENN

        #DM_smenn = SMOTEENN()
        #DM_X_res, DM_y_res = DM_smenn.fit_resample(DM_ova_X_train, DM_ova_y_train)
        FI_smenn = SMOTEENN()
        FI_X_res, FI_y_res = FI_smenn.fit_resample(FI_ova_X_train, FI_ova_y_train)
        FG_smenn = SMOTEENN()
        FG_X_res, FG_y_res = FG_smenn.fit_resample(FG_ova_X_train, FG_ova_y_train)
        #GR_smenn = SMOTEENN()
        #GR_X_res, GR_y_res = GR_smenn.fit_resample(GR_ova_X_train, GR_ova_y_train)
        #GR12_smenn = SMOTEENN()
        #GR12_X_res, GR12_y_res = GR12_smenn.fit_resample(GR12_ova_X_train, GR12_ova_y_train)
        GR27_smenn = SMOTEENN()
        GR27_X_res, GR27_y_res = GR27_smenn.fit_resample(GR27_ova_X_train, GR27_ova_y_train)
        LM_smenn = SMOTEENN()
        LM_X_res, LM_y_res = LM_smenn.fit_resample(LM_ova_X_train, LM_ova_y_train)
        LMM_smenn = SMOTEENN()
        LMM_X_res, LMM_y_res = LMM_smenn.fit_resample(LMM_ova_X_train, LMM_ova_y_train)
        #MM14_smenn = SMOTEENN()
        #MM14_X_res, MM14_y_res = MM14_smenn.fit_resample(MM14_ova_X_train, MM14_ova_y_train)
        #MM16_smenn = SMOTEENN()
        #MM16_X_res, MM16_y_res = MM16_smenn.fit_resample(MM16_ova_X_train, MM16_ova_y_train)
        PC_smenn = SMOTEENN()
        PC_X_res, PC_y_res = PC_smenn.fit_resample(PC_ova_X_train, PC_ova_y_train)
        RG12_smenn = SMOTEENN()
        RG12_X_res, RG12_y_res = RG12_smenn.fit_resample(RG12_ova_X_train, RG12_ova_y_train)
        #RG19_smenn = SMOTEENN()
        #RG19_X_res, RG19_y_res = RG19_smenn.fit_resample(RG19_ova_X_train, RG19_ova_y_train)
        RG2_smenn = SMOTEENN()
        RG2_X_res, RG2_y_res = RG2_smenn.fit_resample(RG2_ova_X_train, RG2_ova_y_train)
        RG3_smenn = SMOTEENN()
        RG3_X_res, RG3_y_res = RG3_smenn.fit_resample(RG3_ova_X_train, RG3_ova_y_train)
        RGM_smenn = SMOTEENN()
        RGM_X_res, RGM_y_res = RGM_smenn.fit_resample(RGM_ova_X_train, RGM_ova_y_train)
        RGQC_smenn = SMOTEENN()
        RGQC_X_res, RGQC_y_res = RGQC_smenn.fit_resample(RGQC_ova_X_train, RGQC_ova_y_train)
        #TMSA10_smenn = SMOTEENN()
        #TMSA10_X_res, TMSA10_y_res = TMSA10_smenn.fit_resample(TMSA10_ova_X_train, TMSA10_ova_y_train)
        T8_smenn = SMOTEENN()
        T8_X_res, T8_y_res = T8_smenn.fit_resample(T8_ova_X_train, T8_ova_y_train)
        #T9_smenn = SMOTEENN()
        #T9_X_res, T9_y_res = T9_smenn.fit_resample(T9_ova_X_train, T9_ova_y_train)
        TM10_smenn = SMOTEENN()
        TM10_X_res, TM10_y_res = TM10_smenn.fit_resample(TM10_ova_X_train, TM10_ova_y_train)
        TM4_smenn = SMOTEENN()
        TM4_X_res, TM4_y_res = TM4_smenn.fit_resample(TM4_ova_X_train, TM4_ova_y_train)
        TM5_smenn = SMOTEENN()
        TM5_X_res, TM5_y_res = TM5_smenn.fit_resample(TM5_ova_X_train, TM5_ova_y_train)
        TM6_smenn = SMOTEENN()
        TM6_X_res, TM6_y_res = TM6_smenn.fit_resample(TM6_ova_X_train, TM6_ova_y_train)
        TM8_smenn = SMOTEENN()
        TM8_X_res, TM8_y_res = TM8_smenn.fit_resample(TM8_ova_X_train, TM8_ova_y_train)
        TM9_smenn = SMOTEENN()
        TM9_X_res, TM9_y_res = TM9_smenn.fit_resample(TM9_ova_X_train, TM9_ova_y_train)
        TMQC_smenn = SMOTEENN()
        TMQC_X_res, TMQC_y_res = TMQC_smenn.fit_resample(TMQC_ova_X_train, TMQC_ova_y_train)
        TQC_smenn = SMOTEENN()
        TQC_X_res, TQC_y_res = TQC_smenn.fit_resample(TQC_ova_X_train, TQC_ova_y_train)
        #WC13_smenn = SMOTEENN()
        #WC13_X_res, WC13_y_res = WC13_smenn.fit_resample(WC13_ova_X_train, WC13_ova_y_train)

    if imb_technique == "SMOTETOMEK":
        from imblearn.combine import SMOTETomek

        #DM_smtm = SMOTETomek()
        #DM_X_res, DM_y_res = DM_smtm.fit_resample(DM_ova_X_train, DM_ova_y_train)
        FI_smtm = SMOTETomek()
        FI_X_res, FI_y_res = FI_smtm.fit_resample(FI_ova_X_train, FI_ova_y_train)
        FG_smtm = SMOTETomek()
        FG_X_res, FG_y_res = FG_smtm.fit_resample(FG_ova_X_train, FG_ova_y_train)
        #GR_smtm = SMOTETomek()
        #GR_X_res, GR_y_res = GR_smtm.fit_resample(GR_ova_X_train, GR_ova_y_train)
        #GR12_smtm = SMOTETomek()
        #GR12_X_res, GR12_y_res = GR12_smtm.fit_resample(GR12_ova_X_train, GR12_ova_y_train)
        GR27_smtm = SMOTETomek()
        GR27_X_res, GR27_y_res = GR27_smtm.fit_resample(GR27_ova_X_train, GR27_ova_y_train)
        LM_smtm = SMOTETomek()
        LM_X_res, LM_y_res = LM_smtm.fit_resample(LM_ova_X_train, LM_ova_y_train)
        LMM_smtm = SMOTETomek()
        LMM_X_res, LMM_y_res = LMM_smtm.fit_resample(LMM_ova_X_train, LMM_ova_y_train)
        #MM14_smtm = SMOTETomek()
        #MM14_X_res, MM14_y_res = MM14_smtm.fit_resample(MM14_ova_X_train, MM14_ova_y_train)
        #MM16_smtm = SMOTETomek()
        #MM16_X_res, MM16_y_res = MM16_smtm.fit_resample(MM16_ova_X_train, MM16_ova_y_train)
        PC_smtm = SMOTETomek()
        PC_X_res, PC_y_res = PC_smtm.fit_resample(PC_ova_X_train, PC_ova_y_train)
        RG12_smtm = SMOTETomek()
        RG12_X_res, RG12_y_res = RG12_smtm.fit_resample(RG12_ova_X_train, RG12_ova_y_train)
        #RG19_smtm = SMOTETomek()
        #RG19_X_res, RG19_y_res = RG19_smtm.fit_resample(RG19_ova_X_train, RG19_ova_y_train)
        RG2_smtm = SMOTETomek()
        RG2_X_res, RG2_y_res = RG2_smtm.fit_resample(RG2_ova_X_train, RG2_ova_y_train)
        RG3_smtm = SMOTETomek()
        RG3_X_res, RG3_y_res = RG3_smtm.fit_resample(RG3_ova_X_train, RG3_ova_y_train)
        RGM_smtm = SMOTETomek()
        RGM_X_res, RGM_y_res = RGM_smtm.fit_resample(RGM_ova_X_train, RGM_ova_y_train)
        RGQC_smtm = SMOTETomek()
        RGQC_X_res, RGQC_y_res = RGQC_smtm.fit_resample(RGQC_ova_X_train, RGQC_ova_y_train)
        #TMSA10_smtm = SMOTETomek()
        #TMSA10_X_res, TMSA10_y_res = TMSA10_smtm.fit_resample(TMSA10_ova_X_train, TMSA10_ova_y_train)
        T8_smtm = SMOTETomek()
        T8_X_res, T8_y_res = T8_smtm.fit_resample(T8_ova_X_train, T8_ova_y_train)
        #T9_smtm = SMOTETomek()
        #T9_X_res, T9_y_res = T9_smtm.fit_resample(T9_ova_X_train, T9_ova_y_train)
        TM10_smtm = SMOTETomek()
        TM10_X_res, TM10_y_res = TM10_smtm.fit_resample(TM10_ova_X_train, TM10_ova_y_train)
        TM4_smtm = SMOTETomek()
        TM4_X_res, TM4_y_res = TM4_smtm.fit_resample(TM4_ova_X_train, TM4_ova_y_train)
        TM5_smtm = SMOTETomek()
        TM5_X_res, TM5_y_res = TM5_smtm.fit_resample(TM5_ova_X_train, TM5_ova_y_train)
        TM6_smtm = SMOTETomek()
        TM6_X_res, TM6_y_res = TM6_smtm.fit_resample(TM6_ova_X_train, TM6_ova_y_train)
        TM8_smtm = SMOTETomek()
        TM8_X_res, TM8_y_res = TM8_smtm.fit_resample(TM8_ova_X_train, TM8_ova_y_train)
        TM9_smtm = SMOTETomek()
        TM9_X_res, TM9_y_res = TM9_smtm.fit_resample(TM9_ova_X_train, TM9_ova_y_train)
        TMQC_smtm = SMOTETomek()
        TMQC_X_res, TMQC_y_res = TMQC_smtm.fit_resample(TMQC_ova_X_train, TMQC_ova_y_train)
        TQC_smtm = SMOTETomek()
        TQC_X_res, TQC_y_res = TQC_smtm.fit_resample(TQC_ova_X_train, TQC_ova_y_train)
        #WC13_smtm = SMOTETomek()
        #WC13_X_res, WC13_y_res = WC13_smtm.fit_resample(WC13_ova_X_train, WC13_ova_y_train)

    if imb_technique == "BC":
        from imblearn.ensemble import BalanceCascade

        #DM_bc = BalanceCascade()
        #DM_X_res, DM_y_res = DM_bc.fit_resample(DM_ova_X_train, DM_ova_y_train)
        #DM_X_res = DM_X_res[0]
        #DM_y_res = DM_y_res[0]
        FI_bc = BalanceCascade()
        FI_X_res, FI_y_res = FI_bc.fit_resample(FI_ova_X_train, FI_ova_y_train)
        FI_X_res = FI_X_res[0]
        FI_y_res = FI_y_res[0]
        FG_bc = BalanceCascade()
        FG_X_res, FG_y_res = FG_bc.fit_resample(FG_ova_X_train, FG_ova_y_train)
        FG_X_res = FG_X_res[0]
        FG_y_res = FG_y_res[0]
        #GR_bc = BalanceCascade()
        #GR_X_res, GR_y_res = GR_bc.fit_resample(GR_ova_X_train, GR_ova_y_train)
        #GR_X_res = GR_X_res[0]
        #GR_y_res = GR_y_res[0]
        #GR12_bc = BalanceCascade()
        #GR12_X_res, GR12_y_res = GR12_bc.fit_resample(GR12_ova_X_train, GR12_ova_y_train)
        #GR12_X_res = GR12_X_res[0]
        #GR12_y_res = GR12_y_res[0]
        GR27_bc = BalanceCascade()
        GR27_X_res, GR27_y_res = GR27_bc.fit_resample(GR27_ova_X_train, GR27_ova_y_train)
        GR27_X_res = GR27_X_res[0]
        GR27_y_res = GR27_y_res[0]
        LM_bc = BalanceCascade()
        LM_X_res, LM_y_res = LM_bc.fit_resample(LM_ova_X_train, LM_ova_y_train)
        LM_X_res = LM_X_res[0]
        LM_y_res = LM_y_res[0]
        LMM_bc = BalanceCascade()
        LMM_X_res, LMM_y_res = LMM_bc.fit_resample(LMM_ova_X_train, LMM_ova_y_train)
        LMM_X_res = LMM_X_res[0]
        LMM_y_res = LMM_y_res[0]
        #MM14_bc = BalanceCascade()
        #MM14_X_res, MM14_y_res = MM14_bc.fit_resample(MM14_ova_X_train, MM14_ova_y_train)
        #MM14_X_res = MM14_X_res[0]
        #MM14_y_res = MM14_y_res[0]
        #MM16_bc = BalanceCascade()
        #MM16_X_res, MM16_y_res = MM16_bc.fit_resample(MM16_ova_X_train, MM16_ova_y_train)
        #MM16_X_res = MM16_X_res[0]
        #MM16_y_res = MM16_y_res[0]
        PC_bc = BalanceCascade()
        PC_X_res, PC_y_res = PC_bc.fit_resample(PC_ova_X_train, PC_ova_y_train)
        PC_X_res = PC_X_res[0]
        PC_y_res = PC_y_res[0]
        RG12_bc = BalanceCascade()
        RG12_X_res, RG12_y_res = RG12_bc.fit_resample(RG12_ova_X_train, RG12_ova_y_train)
        RG12_X_res = RG12_X_res[0]
        RG12_y_res = RG12_y_res[0]
        #RG19_bc = BalanceCascade()
        #RG19_X_res, RG19_y_res = RG19_bc.fit_resample(RG19_ova_X_train, RG19_ova_y_train)
        #RG19_X_res = RG19_X_res[0]
        #RG19_y_res = RG19_y_res[0]
        RG2_bc = BalanceCascade()
        RG2_X_res, RG2_y_res = RG2_bc.fit_resample(RG2_ova_X_train, RG2_ova_y_train)
        RG2_X_res = RG2_X_res[0]
        RG2_y_res = RG2_y_res[0]
        RG3_bc = BalanceCascade()
        RG3_X_res, RG3_y_res = RG3_bc.fit_resample(RG3_ova_X_train, RG3_ova_y_train)
        RG3_X_res = RG3_X_res[0]
        RG3_y_res = RG3_y_res[0]
        RGM_bc = BalanceCascade()
        RGM_X_res, RGM_y_res = RGM_bc.fit_resample(RGM_ova_X_train, RGM_ova_y_train)
        RGM_X_res = RGM_X_res[0]
        RGM_y_res = RGM_y_res[0]
        RGQC_bc = BalanceCascade()
        RGQC_X_res, RGQC_y_res = RGQC_bc.fit_resample(RGQC_ova_X_train, RGQC_ova_y_train)
        RGQC_X_res = RGQC_X_res[0]
        RGQC_y_res = RGQC_y_res[0]
        #TMSA10_bc = BalanceCascade()
        #TMSA10_X_res, TMSA10_y_res = TMSA10_bc.fit_resample(TMSA10_ova_X_train, TMSA10_ova_y_train)
        #TMSA10_X_res = TMSA10_X_res[0]
        #TMSA10_y_res = TMSA10_y_res[0]
        T8_bc = BalanceCascade()
        T8_X_res, T8_y_res = T8_bc.fit_resample(T8_ova_X_train, T8_ova_y_train)
        T8_X_res = T8_X_res[0]
        T8_y_res = T8_y_res[0]
        #T9_bc = BalanceCascade()
        #T9_X_res, T9_y_res = T9_bc.fit_resample(T9_ova_X_train, T9_ova_y_train)
        #T9_X_res = T9_X_res[0]
        #T9_y_res = T9_y_res[0]
        TM10_bc = BalanceCascade()
        TM10_X_res, TM10_y_res = TM10_bc.fit_resample(TM10_ova_X_train, TM10_ova_y_train)
        TM10_X_res = TM10_X_res[0]
        TM10_y_res = TM10_y_res[0]
        TM4_bc = BalanceCascade()
        TM4_X_res, TM4_y_res = TM4_bc.fit_resample(TM4_ova_X_train, TM4_ova_y_train)
        TM4_X_res = TM4_X_res[0]
        TM4_y_res = TM4_y_res[0]
        TM5_bc = BalanceCascade()
        TM5_X_res, TM5_y_res = TM5_bc.fit_resample(TM5_ova_X_train, TM5_ova_y_train)
        TM5_X_res = TM5_X_res[0]
        TM5_y_res = TM5_y_res[0]
        TM6_bc = BalanceCascade()
        TM6_X_res, TM6_y_res = TM6_bc.fit_resample(TM6_ova_X_train, TM6_ova_y_train)
        TM6_X_res = TM6_X_res[0]
        TM6_y_res = TM6_y_res[0]
        TM8_bc = BalanceCascade()
        TM8_X_res, TM8_y_res = TM8_bc.fit_resample(TM8_ova_X_train, TM8_ova_y_train)
        TM8_X_res = TM8_X_res[0]
        TM8_y_res = TM8_y_res[0]
        TM9_bc = BalanceCascade()
        TM9_X_res, TM9_y_res = TM9_bc.fit_resample(TM9_ova_X_train, TM9_ova_y_train)
        TM9_X_res = TM9_X_res[0]
        TM9_y_res = TM9_y_res[0]
        TMQC_bc = BalanceCascade()
        TMQC_X_res, TMQC_y_res = TMQC_bc.fit_resample(TMQC_ova_X_train, TMQC_ova_y_train)
        TMQC_X_res = TMQC_X_res[0]
        TMQC_y_res = TMQC_y_res[0]
        TQC_bc = BalanceCascade()
        TQC_X_res, TQC_y_res = TQC_bc.fit_resample(TQC_ova_X_train, TQC_ova_y_train)
        TQC_X_res = TQC_X_res[0]
        TQC_y_res = TQC_y_res[0]
        #WC13_bc = BalanceCascade()
        #WC13_X_res, WC13_y_res = WC13_bc.fit_resample(WC13_ova_X_train, WC13_ova_y_train)
        #WC13_X_res = WC13_X_res[0]
        #WC13_y_res = WC13_y_res[0]

    if imb_technique == "EE":
        from imblearn.ensemble import EasyEnsemble

        #DM_ee = EasyEnsemble()
        #DM_X_res, DM_y_res = DM_ee.fit_resample(DM_ova_X_train, DM_ova_y_train)
        #DM_X_res = DM_X_res[0]
        #DM_y_res = DM_y_res[0]
        FI_ee = EasyEnsemble()
        FI_X_res, FI_y_res = FI_ee.fit_resample(FI_ova_X_train, FI_ova_y_train)
        FI_X_res = FI_X_res[0]
        FI_y_res = FI_y_res[0]
        FG_ee = EasyEnsemble()
        FG_X_res, FG_y_res = FG_ee.fit_resample(FG_ova_X_train, FG_ova_y_train)
        FG_X_res = FG_X_res[0]
        FG_y_res = FG_y_res[0]
        #GR_ee = EasyEnsemble()
        #GR_X_res, GR_y_res = GR_ee.fit_resample(GR_ova_X_train, GR_ova_y_train)
        #GR_X_res = GR_X_res[0]
        #GR_y_res = GR_y_res[0]
        #GR12_ee = EasyEnsemble()
        #GR12_X_res, GR12_y_res = GR12_ee.fit_resample(GR12_ova_X_train, GR12_ova_y_train)
        #GR12_X_res = GR12_X_res[0]
        #GR12_y_res = GR12_y_res[0]
        GR27_ee = EasyEnsemble()
        GR27_X_res, GR27_y_res = GR27_ee.fit_resample(GR27_ova_X_train, GR27_ova_y_train)
        GR27_X_res = GR27_X_res[0]
        GR27_y_res = GR27_y_res[0]
        LM_ee = EasyEnsemble()
        LM_X_res, LM_y_res = LM_ee.fit_resample(LM_ova_X_train, LM_ova_y_train)
        LM_X_res = LM_X_res[0]
        LM_y_res = LM_y_res[0]
        LMM_ee = EasyEnsemble()
        LMM_X_res, LMM_y_res = LMM_ee.fit_resample(LMM_ova_X_train, LMM_ova_y_train)
        LMM_X_res = LMM_X_res[0]
        LMM_y_res = LMM_y_res[0]
        #MM14_ee = EasyEnsemble()
        #MM14_X_res, MM14_y_res = MM14_ee.fit_resample(MM14_ova_X_train, MM14_ova_y_train)
        #MM14_X_res = MM14_X_res[0]
        #MM14_y_res = MM14_y_res[0]
        #MM16_ee = EasyEnsemble()
        #MM16_X_res, MM16_y_res = MM16_ee.fit_resample(MM16_ova_X_train, MM16_ova_y_train)
        #MM16_X_res = MM16_X_res[0]
        #MM16_y_res = MM16_y_res[0]
        PC_ee = EasyEnsemble()
        PC_X_res, PC_y_res = PC_ee.fit_resample(PC_ova_X_train, PC_ova_y_train)
        PC_X_res = PC_X_res[0]
        PC_y_res = PC_y_res[0]
        RG12_ee = EasyEnsemble()
        RG12_X_res, RG12_y_res = RG12_ee.fit_resample(RG12_ova_X_train, RG12_ova_y_train)
        RG12_X_res = RG12_X_res[0]
        RG12_y_res = RG12_y_res[0]
        #RG19_ee = EasyEnsemble()
        #RG19_X_res, RG19_y_res = RG19_ee.fit_resample(RG19_ova_X_train, RG19_ova_y_train)
        #RG19_X_res = RG19_X_res[0]
        #RG19_y_res = RG19_y_res[0]
        RG2_ee = EasyEnsemble()
        RG2_X_res, RG2_y_res = RG2_ee.fit_resample(RG2_ova_X_train, RG2_ova_y_train)
        RG2_X_res = RG2_X_res[0]
        RG2_y_res = RG2_y_res[0]
        RG3_ee = EasyEnsemble()
        RG3_X_res, RG3_y_res = RG3_ee.fit_resample(RG3_ova_X_train, RG3_ova_y_train)
        RG3_X_res = RG3_X_res[0]
        RG3_y_res = RG3_y_res[0]
        RGM_ee = EasyEnsemble()
        RGM_X_res, RGM_y_res = RGM_ee.fit_resample(RGM_ova_X_train, RGM_ova_y_train)
        RGM_X_res = RGM_X_res[0]
        RGM_y_res = RGM_y_res[0]
        RGQC_ee = EasyEnsemble()
        RGQC_X_res, RGQC_y_res = RGQC_ee.fit_resample(RGQC_ova_X_train, RGQC_ova_y_train)
        RGQC_X_res = RGQC_X_res[0]
        RGQC_y_res = RGQC_y_res[0]
        #TMSA10_ee = EasyEnsemble()
        #TMSA10_X_res, TMSA10_y_res = TMSA10_ee.fit_resample(TMSA10_ova_X_train, TMSA10_ova_y_train)
        #TMSA10_X_res = TMSA10_X_res[0]
        #TMSA10_y_res = TMSA10_y_res[0]
        T8_ee = EasyEnsemble()
        T8_X_res, T8_y_res = T8_ee.fit_resample(T8_ova_X_train, T8_ova_y_train)
        T8_X_res = T8_X_res[0]
        T8_y_res = T8_y_res[0]
        #T9_ee = EasyEnsemble()
        #T9_X_res, T9_y_res = T9_ee.fit_resample(T9_ova_X_train, T9_ova_y_train)
        #T9_X_res = T9_X_res[0]
        #T9_y_res = T9_y_res[0]
        TM10_ee = EasyEnsemble()
        TM10_X_res, TM10_y_res = TM10_ee.fit_resample(TM10_ova_X_train, TM10_ova_y_train)
        TM10_X_res = TM10_X_res[0]
        TM10_y_res = TM10_y_res[0]
        TM4_ee = EasyEnsemble()
        TM4_X_res, TM4_y_res = TM4_ee.fit_resample(TM4_ova_X_train, TM4_ova_y_train)
        TM4_X_res = TM4_X_res[0]
        TM4_y_res = TM4_y_res[0]
        TM5_ee = EasyEnsemble()
        TM5_X_res, TM5_y_res = TM5_ee.fit_resample(TM5_ova_X_train, TM5_ova_y_train)
        TM5_X_res = TM5_X_res[0]
        TM5_y_res = TM5_y_res[0]
        TM6_ee = EasyEnsemble()
        TM6_X_res, TM6_y_res = TM6_ee.fit_resample(TM6_ova_X_train, TM6_ova_y_train)
        TM6_X_res = TM6_X_res[0]
        TM6_y_res = TM6_y_res[0]
        TM8_ee = EasyEnsemble()
        TM8_X_res, TM8_y_res = TM8_ee.fit_resample(TM8_ova_X_train, TM8_ova_y_train)
        TM8_X_res = TM8_X_res[0]
        TM8_y_res = TM8_y_res[0]
        TM9_ee = EasyEnsemble()
        TM9_X_res, TM9_y_res = TM9_ee.fit_resample(TM9_ova_X_train, TM9_ova_y_train)
        TM9_X_res = TM9_X_res[0]
        TM9_y_res = TM9_y_res[0]
        TMQC_ee = EasyEnsemble()
        TMQC_X_res, TMQC_y_res = TMQC_ee.fit_resample(TMQC_ova_X_train, TMQC_ova_y_train)
        TMQC_X_res = TMQC_X_res[0]
        TMQC_y_res = TMQC_y_res[0]
        TQC_ee = EasyEnsemble()
        TQC_X_res, TQC_y_res = TQC_ee.fit_resample(TQC_ova_X_train, TQC_ova_y_train)
        TQC_X_res = TQC_X_res[0]
        TQC_y_res = TQC_y_res[0]
        #WC13_ee = EasyEnsemble()
        #WC13_X_res, WC13_y_res = WC13_ee.fit_resample(WC13_ova_X_train, WC13_ova_y_train)
        #WC13_X_res = WC13_X_res[0]
        #WC13_y_res = WC13_y_res[0]

    if imb_technique == "TOMEK":
        from imblearn.under_sampling import TomekLinks

        #DM_tm = TomekLinks()
        #DM_X_res, DM_y_res = DM_tm.fit_resample(DM_ova_X_train, DM_ova_y_train)
        FI_tm = TomekLinks()
        FI_X_res, FI_y_res = FI_tm.fit_resample(FI_ova_X_train, FI_ova_y_train)
        FG_tm = TomekLinks()
        FG_X_res, FG_y_res = FG_tm.fit_resample(FG_ova_X_train, FG_ova_y_train)
        #GR_tm = TomekLinks()
        #GR_X_res, GR_y_res = GR_tm.fit_resample(GR_ova_X_train, GR_ova_y_train)
        #GR12_tm = TomekLinks()
        #GR12_X_res, GR12_y_res = GR12_tm.fit_resample(GR12_ova_X_train, GR12_ova_y_train)
        GR27_tm = TomekLinks()
        GR27_X_res, GR27_y_res = GR27_tm.fit_resample(GR27_ova_X_train, GR27_ova_y_train)
        LM_tm = TomekLinks()
        LM_X_res, LM_y_res = LM_tm.fit_resample(LM_ova_X_train, LM_ova_y_train)
        LMM_tm = TomekLinks()
        LMM_X_res, LMM_y_res = LMM_tm.fit_resample(LMM_ova_X_train, LMM_ova_y_train)
        #MM14_tm = TomekLinks()
        #MM14_X_res, MM14_y_res = MM14_tm.fit_resample(MM14_ova_X_train, MM14_ova_y_train)
        #MM16_tm = TomekLinks()
        #MM16_X_res, MM16_y_res = MM16_tm.fit_resample(MM16_ova_X_train, MM16_ova_y_train)
        PC_tm = TomekLinks()
        PC_X_res, PC_y_res = PC_tm.fit_resample(PC_ova_X_train, PC_ova_y_train)
        RG12_tm = TomekLinks()
        RG12_X_res, RG12_y_res = RG12_tm.fit_resample(RG12_ova_X_train, RG12_ova_y_train)
        #RG19_tm = TomekLinks()
        #RG19_X_res, RG19_y_res = RG19_tm.fit_resample(RG19_ova_X_train, RG19_ova_y_train)
        RG2_tm = TomekLinks()
        RG2_X_res, RG2_y_res = RG2_tm.fit_resample(RG2_ova_X_train, RG2_ova_y_train)
        RG3_tm = TomekLinks()
        RG3_X_res, RG3_y_res = RG3_tm.fit_resample(RG3_ova_X_train, RG3_ova_y_train)
        RGM_tm = TomekLinks()
        RGM_X_res, RGM_y_res = RGM_tm.fit_resample(RGM_ova_X_train, RGM_ova_y_train)
        RGQC_tm = TomekLinks()
        RGQC_X_res, RGQC_y_res = RGQC_tm.fit_resample(RGQC_ova_X_train, RGQC_ova_y_train)
        #TMSA10_tm = TomekLinks()
        #TMSA10_X_res, TMSA10_y_res = TMSA10_tm.fit_resample(TMSA10_ova_X_train, TMSA10_ova_y_train)
        T8_tm = TomekLinks()
        T8_X_res, T8_y_res = T8_tm.fit_resample(T8_ova_X_train, T8_ova_y_train)
        #T9_tm = TomekLinks()
        #T9_X_res, T9_y_res = T9_tm.fit_resample(T9_ova_X_train, T9_ova_y_train)
        TM10_tm = TomekLinks()
        TM10_X_res, TM10_y_res = TM10_tm.fit_resample(TM10_ova_X_train, TM10_ova_y_train)
        TM4_tm = TomekLinks()
        TM4_X_res, TM4_y_res = TM4_tm.fit_resample(TM4_ova_X_train, TM4_ova_y_train)
        TM5_tm = TomekLinks()
        TM5_X_res, TM5_y_res = TM5_tm.fit_resample(TM5_ova_X_train, TM5_ova_y_train)
        TM6_tm = TomekLinks()
        TM6_X_res, TM6_y_res = TM6_tm.fit_resample(TM6_ova_X_train, TM6_ova_y_train)
        TM8_tm = TomekLinks()
        TM8_X_res, TM8_y_res = TM8_tm.fit_resample(TM8_ova_X_train, TM8_ova_y_train)
        TM9_tm = TomekLinks()
        TM9_X_res, TM9_y_res = TM9_tm.fit_resample(TM9_ova_X_train, TM9_ova_y_train)
        TMQC_tm = TomekLinks()
        TMQC_X_res, TMQC_y_res = TMQC_tm.fit_resample(TMQC_ova_X_train, TMQC_ova_y_train)
        TQC_tm = TomekLinks()
        TQC_X_res, TQC_y_res = TQC_tm.fit_resample(TQC_ova_X_train, TQC_ova_y_train)
        #WC13_tm = TomekLinks()
        #WC13_X_res, WC13_y_res = WC13_tm.fit_resample(WC13_ova_X_train, WC13_ova_y_train)


    if imb_technique == "ROS":
        from imblearn.over_sampling import RandomOverSampler

        #DM_ros = RandomOverSampler()
        #DM_X_res, DM_y_res = DM_ros.fit_resample(DM_ova_X_train, DM_ova_y_train)
        FI_ros = RandomOverSampler()
        FI_X_res, FI_y_res = FI_ros.fit_resample(FI_ova_X_train, FI_ova_y_train)
        FG_ros = RandomOverSampler()
        FG_X_res, FG_y_res = FG_ros.fit_resample(FG_ova_X_train, FG_ova_y_train)
        #GR_ros = RandomOverSampler()
        #GR_X_res, GR_y_res = GR_ros.fit_resample(GR_ova_X_train, GR_ova_y_train)
        #GR12_ros = RandomOverSampler()
        #GR12_X_res, GR12_y_res = GR12_ros.fit_resample(GR12_ova_X_train, GR12_ova_y_train)
        GR27_ros = RandomOverSampler()
        GR27_X_res, GR27_y_res = GR27_ros.fit_resample(GR27_ova_X_train, GR27_ova_y_train)
        LM_ros = RandomOverSampler()
        LM_X_res, LM_y_res = LM_ros.fit_resample(LM_ova_X_train, LM_ova_y_train)
        LMM_ros = RandomOverSampler()
        LMM_X_res, LMM_y_res = LMM_ros.fit_resample(LMM_ova_X_train, LMM_ova_y_train)
        #MM14_ros = RandomOverSampler()
        #MM14_X_res, MM14_y_res = MM14_ros.fit_resample(MM14_ova_X_train, MM14_ova_y_train)
        #MM16_ros = RandomOverSampler()
        #MM16_X_res, MM16_y_res = MM16_ros.fit_resample(MM16_ova_X_train, MM16_ova_y_train)
        PC_ros = RandomOverSampler()
        PC_X_res, PC_y_res = PC_ros.fit_resample(PC_ova_X_train, PC_ova_y_train)
        RG12_ros = RandomOverSampler()
        RG12_X_res, RG12_y_res = RG12_ros.fit_resample(RG12_ova_X_train, RG12_ova_y_train)
        #RG19_ros = RandomOverSampler()
        #RG19_X_res, RG19_y_res = RG19_ros.fit_resample(RG19_ova_X_train, RG19_ova_y_train)
        RG2_ros = RandomOverSampler()
        RG2_X_res, RG2_y_res = RG2_ros.fit_resample(RG2_ova_X_train, RG2_ova_y_train)
        RG3_ros = RandomOverSampler()
        RG3_X_res, RG3_y_res = RG3_ros.fit_resample(RG3_ova_X_train, RG3_ova_y_train)
        RGM_ros = RandomOverSampler()
        RGM_X_res, RGM_y_res = RGM_ros.fit_resample(RGM_ova_X_train, RGM_ova_y_train)
        RGQC_ros = RandomOverSampler()
        RGQC_X_res, RGQC_y_res = RGQC_ros.fit_resample(RGQC_ova_X_train, RGQC_ova_y_train)
        #TMSA10_ros = RandomOverSampler()
        #TMSA10_X_res, TMSA10_y_res = TMSA10_ros.fit_resample(TMSA10_ova_X_train, TMSA10_ova_y_train)
        T8_ros = RandomOverSampler()
        T8_X_res, T8_y_res = T8_ros.fit_resample(T8_ova_X_train, T8_ova_y_train)
        #T9_ros = RandomOverSampler()
        #T9_X_res, T9_y_res = T9_ros.fit_resample(T9_ova_X_train, T9_ova_y_train)
        TM10_ros = RandomOverSampler()
        TM10_X_res, TM10_y_res = TM10_ros.fit_resample(TM10_ova_X_train, TM10_ova_y_train)
        TM4_ros = RandomOverSampler()
        TM4_X_res, TM4_y_res = TM4_ros.fit_resample(TM4_ova_X_train, TM4_ova_y_train)
        TM5_ros = RandomOverSampler()
        TM5_X_res, TM5_y_res = TM5_ros.fit_resample(TM5_ova_X_train, TM5_ova_y_train)
        TM6_ros = RandomOverSampler()
        TM6_X_res, TM6_y_res = TM6_ros.fit_resample(TM6_ova_X_train, TM6_ova_y_train)
        TM8_ros = RandomOverSampler()
        TM8_X_res, TM8_y_res = TM8_ros.fit_resample(TM8_ova_X_train, TM8_ova_y_train)
        TM9_ros = RandomOverSampler()
        TM9_X_res, TM9_y_res = TM9_ros.fit_resample(TM9_ova_X_train, TM9_ova_y_train)
        TMQC_ros = RandomOverSampler()
        TMQC_X_res, TMQC_y_res = TMQC_ros.fit_resample(TMQC_ova_X_train, TMQC_ova_y_train)
        TQC_ros = RandomOverSampler()
        TQC_X_res, TQC_y_res = TQC_ros.fit_resample(TQC_ova_X_train, TQC_ova_y_train)
        #WC13_ros = RandomOverSampler()
        #WC13_X_res, WC13_y_res = WC13_ros.fit_resample(WC13_ova_X_train, WC13_ova_y_train)


    if imb_technique == "RUS":
        from imblearn.under_sampling import RandomUnderSampler

        #DM_rus = RandomUnderSampler()
        #DM_X_res, DM_y_res = DM_rus.fit_resample(DM_ova_X_train, DM_ova_y_train)
        FI_rus = RandomUnderSampler()
        FI_X_res, FI_y_res = FI_rus.fit_resample(FI_ova_X_train, FI_ova_y_train)
        FG_rus = RandomUnderSampler()
        FG_X_res, FG_y_res = FG_rus.fit_resample(FG_ova_X_train, FG_ova_y_train)
        #GR_rus = RandomUnderSampler()
        #GR_X_res, GR_y_res = GR_rus.fit_resample(GR_ova_X_train, GR_ova_y_train)
        #GR12_rus = RandomUnderSampler()
        #GR12_X_res, GR12_y_res = GR12_rus.fit_resample(GR12_ova_X_train, GR12_ova_y_train)
        GR27_rus = RandomUnderSampler()
        GR27_X_res, GR27_y_res = GR27_rus.fit_resample(GR27_ova_X_train, GR27_ova_y_train)
        LM_rus = RandomUnderSampler()
        LM_X_res, LM_y_res = LM_rus.fit_resample(LM_ova_X_train, LM_ova_y_train)
        LMM_rus = RandomUnderSampler()
        LMM_X_res, LMM_y_res = LMM_rus.fit_resample(LMM_ova_X_train, LMM_ova_y_train)
        #MM14_rus = RandomUnderSampler()
        #MM14_X_res, MM14_y_res = MM14_rus.fit_resample(MM14_ova_X_train, MM14_ova_y_train)
        #MM16_rus = RandomUnderSampler()
        #MM16_X_res, MM16_y_res = MM16_rus.fit_resample(MM16_ova_X_train, MM16_ova_y_train)
        PC_rus = RandomUnderSampler()
        PC_X_res, PC_y_res = PC_rus.fit_resample(PC_ova_X_train, PC_ova_y_train)
        RG12_rus = RandomUnderSampler()
        RG12_X_res, RG12_y_res = RG12_rus.fit_resample(RG12_ova_X_train, RG12_ova_y_train)
        #RG19_rus = RandomUnderSampler()
        #RG19_X_res, RG19_y_res = RG19_rus.fit_resample(RG19_ova_X_train, RG19_ova_y_train)
        RG2_rus = RandomUnderSampler()
        RG2_X_res, RG2_y_res = RG2_rus.fit_resample(RG2_ova_X_train, RG2_ova_y_train)
        RG3_rus = RandomUnderSampler()
        RG3_X_res, RG3_y_res = RG3_rus.fit_resample(RG3_ova_X_train, RG3_ova_y_train)
        RGM_rus = RandomUnderSampler()
        RGM_X_res, RGM_y_res = RGM_rus.fit_resample(RGM_ova_X_train, RGM_ova_y_train)
        RGQC_rus = RandomUnderSampler()
        RGQC_X_res, RGQC_y_res = RGQC_rus.fit_resample(RGQC_ova_X_train, RGQC_ova_y_train)
        #TMSA10_rus = RandomUnderSampler()
        #TMSA10_X_res, TMSA10_y_res = TMSA10_rus.fit_resample(TMSA10_ova_X_train, TMSA10_ova_y_train)
        T8_rus = RandomUnderSampler()
        T8_X_res, T8_y_res = T8_rus.fit_resample(T8_ova_X_train, T8_ova_y_train)
        #T9_rus = RandomUnderSampler()
        #T9_X_res, T9_y_res = T9_rus.fit_resample(T9_ova_X_train, T9_ova_y_train)
        TM10_rus = RandomUnderSampler()
        TM10_X_res, TM10_y_res = TM10_rus.fit_resample(TM10_ova_X_train, TM10_ova_y_train)
        TM4_rus = RandomUnderSampler()
        TM4_X_res, TM4_y_res = TM4_rus.fit_resample(TM4_ova_X_train, TM4_ova_y_train)
        TM5_rus = RandomUnderSampler()
        TM5_X_res, TM5_y_res = TM5_rus.fit_resample(TM5_ova_X_train, TM5_ova_y_train)
        TM6_rus = RandomUnderSampler()
        TM6_X_res, TM6_y_res = TM6_rus.fit_resample(TM6_ova_X_train, TM6_ova_y_train)
        TM8_rus = RandomUnderSampler()
        TM8_X_res, TM8_y_res = TM8_rus.fit_resample(TM8_ova_X_train, TM8_ova_y_train)
        TM9_rus = RandomUnderSampler()
        TM9_X_res, TM9_y_res = TM9_rus.fit_resample(TM9_ova_X_train, TM9_ova_y_train)
        TMQC_rus = RandomUnderSampler()
        TMQC_X_res, TMQC_y_res = TMQC_rus.fit_resample(TMQC_ova_X_train, TMQC_ova_y_train)
        TQC_rus = RandomUnderSampler()
        TQC_X_res, TQC_y_res = TQC_rus.fit_resample(TQC_ova_X_train, TQC_ova_y_train)
        #WC13_rus = RandomUnderSampler()
        #WC13_X_res, WC13_y_res = WC13_rus.fit_resample(WC13_ova_X_train, WC13_ova_y_train)

    #Below codes are for the implementation of deep neural network training
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import RandomizedSearchCV
    import itertools

    first_digit_parameters = [x for x in itertools.product((5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100), repeat=1)]
    all_digit_parameters = first_digit_parameters
    learning_rate_init_parameters = [0.1, 0.01, 0.001]
    parameters = {'hidden_layer_sizes': all_digit_parameters,
                  'learning_rate_init': learning_rate_init_parameters}
    #dnn_DM = MLPClassifier(max_iter=10000, activation='relu')
    #dnn_DM_clf = RandomizedSearchCV(dnn_DM, parameters, n_jobs=-1, cv=5)
    #dnn_DM_clf.fit(DM_X_res, DM_y_res)
    dnn_FI = MLPClassifier(max_iter=10000, activation='relu')
    dnn_FI_clf = RandomizedSearchCV(dnn_FI, parameters, n_jobs=-1, cv=5)
    dnn_FI_clf.fit(FI_X_res, FI_y_res)
    dnn_FG = MLPClassifier(max_iter=10000, activation='relu')
    dnn_FG_clf = RandomizedSearchCV(dnn_FG, parameters, n_jobs=-1, cv=5)
    dnn_FG_clf.fit(FG_X_res, FG_y_res)
    #dnn_GR = MLPClassifier(max_iter=10000, activation='relu')
    #dnn_GR_clf = RandomizedSearchCV(dnn_GR, parameters, n_jobs=-1, cv=5)
    #dnn_GR_clf.fit(GR_X_res, GR_y_res)
    #dnn_GR12 = MLPClassifier(max_iter=10000, activation='relu')
    #dnn_GR12_clf = RandomizedSearchCV(dnn_GR12, parameters, n_jobs=-1, cv=5)
    #dnn_GR12_clf.fit(GR12_X_res, GR12_y_res)
    dnn_GR27 = MLPClassifier(max_iter=10000, activation='relu')
    dnn_GR27_clf = RandomizedSearchCV(dnn_GR27, parameters, n_jobs=-1, cv=5)
    dnn_GR27_clf.fit(GR27_X_res, GR27_y_res)
    dnn_LM = MLPClassifier(max_iter=10000, activation='relu')
    dnn_LM_clf = RandomizedSearchCV(dnn_LM, parameters, n_jobs=-1, cv=5)
    dnn_LM_clf.fit(LM_X_res, LM_y_res)
    dnn_LMM = MLPClassifier(max_iter=10000, activation='relu')
    dnn_LMM_clf = RandomizedSearchCV(dnn_LMM, parameters, n_jobs=-1, cv=5)
    dnn_LMM_clf.fit(LMM_X_res, LMM_y_res)
    #dnn_MM14 = MLPClassifier(max_iter=10000, activation='relu')
    #dnn_MM14_clf = RandomizedSearchCV(dnn_MM14, parameters, n_jobs=-1, cv=5)
    #dnn_MM14_clf.fit(MM14_X_res, MM14_y_res)
    #dnn_MM16 = MLPClassifier(max_iter=10000, activation='relu')
    #dnn_MM16_clf = RandomizedSearchCV(dnn_MM16, parameters, n_jobs=-1, cv=5)
    #dnn_MM16_clf.fit(MM16_X_res, MM16_y_res)
    dnn_PC = MLPClassifier(max_iter=10000, activation='relu')
    dnn_PC_clf = RandomizedSearchCV(dnn_PC, parameters, n_jobs=-1, cv=5)
    dnn_PC_clf.fit(PC_X_res, PC_y_res)
    dnn_RG12 = MLPClassifier(max_iter=10000, activation='relu')
    dnn_RG12_clf = RandomizedSearchCV(dnn_RG12, parameters, n_jobs=-1, cv=5)
    dnn_RG12_clf.fit(RG12_X_res, RG12_y_res)
    #dnn_RG19 = MLPClassifier(max_iter=10000, activation='relu')
    #dnn_RG19_clf = RandomizedSearchCV(dnn_RG19, parameters, n_jobs=-1, cv=5)
    #dnn_RG19_clf.fit(RG19_X_res, RG19_y_res)
    dnn_RG2 = MLPClassifier(max_iter=10000, activation='relu')
    dnn_RG2_clf = RandomizedSearchCV(dnn_RG2, parameters, n_jobs=-1, cv=5)
    dnn_RG2_clf.fit(RG2_X_res, RG2_y_res)
    dnn_RG3 = MLPClassifier(max_iter=10000, activation='relu')
    dnn_RG3_clf = RandomizedSearchCV(dnn_RG3, parameters, n_jobs=-1, cv=5)
    dnn_RG3_clf.fit(RG3_X_res, RG3_y_res)
    dnn_RGM = MLPClassifier(max_iter=10000, activation='relu')
    dnn_RGM_clf = RandomizedSearchCV(dnn_RGM, parameters, n_jobs=-1, cv=5)
    dnn_RGM_clf.fit(RGM_X_res, RGM_y_res)
    dnn_RGQC = MLPClassifier(max_iter=10000, activation='relu')
    dnn_RGQC_clf = RandomizedSearchCV(dnn_RGQC, parameters, n_jobs=-1, cv=5)
    dnn_RGQC_clf.fit(RGQC_X_res, RGQC_y_res)
    #dnn_TMSA10 = MLPClassifier(max_iter=10000, activation='relu')
    #dnn_TMSA10_clf = RandomizedSearchCV(dnn_TMSA10, parameters, n_jobs=-1, cv=5)
    #dnn_TMSA10_clf.fit(TMSA10_X_res, TMSA10_y_res)
    dnn_T8 = MLPClassifier(max_iter=10000, activation='relu')
    dnn_T8_clf = RandomizedSearchCV(dnn_T8, parameters, n_jobs=-1, cv=5)
    dnn_T8_clf.fit(T8_X_res, T8_y_res)
    #dnn_T9 = MLPClassifier(max_iter=10000, activation='relu')
    #dnn_T9_clf = RandomizedSearchCV(dnn_T9, parameters, n_jobs=-1, cv=5)
    #dnn_T9_clf.fit(T9_X_res, T9_y_res)
    dnn_TM10 = MLPClassifier(max_iter=10000, activation='relu')
    dnn_TM10_clf = RandomizedSearchCV(dnn_TM10, parameters, n_jobs=-1, cv=5)
    dnn_TM10_clf.fit(TM10_X_res, TM10_y_res)
    dnn_TM4 = MLPClassifier(max_iter=10000, activation='relu')
    dnn_TM4_clf = RandomizedSearchCV(dnn_TM4, parameters, n_jobs=-1, cv=5)
    dnn_TM4_clf.fit(TM4_X_res, TM4_y_res)
    dnn_TM5 = MLPClassifier(max_iter=10000, activation='relu')
    dnn_TM5_clf = RandomizedSearchCV(dnn_TM5, parameters, n_jobs=-1, cv=5)
    dnn_TM5_clf.fit(TM5_X_res, TM5_y_res)
    dnn_TM6 = MLPClassifier(max_iter=10000, activation='relu')
    dnn_TM6_clf = RandomizedSearchCV(dnn_TM6, parameters, n_jobs=-1, cv=5)
    dnn_TM6_clf.fit(TM6_X_res, TM6_y_res)
    dnn_TM8 = MLPClassifier(max_iter=10000, activation='relu')
    dnn_TM8_clf = RandomizedSearchCV(dnn_TM8, parameters, n_jobs=-1, cv=5)
    dnn_TM8_clf.fit(TM8_X_res, TM8_y_res)
    dnn_TM9 = MLPClassifier(max_iter=10000, activation='relu')
    dnn_TM9_clf = RandomizedSearchCV(dnn_TM9, parameters, n_jobs=-1, cv=5)
    dnn_TM9_clf.fit(TM9_X_res, TM9_y_res)
    dnn_TMQC = MLPClassifier(max_iter=10000, activation='relu')
    dnn_TMQC_clf = RandomizedSearchCV(dnn_TMQC, parameters, n_jobs=-1, cv=5)
    dnn_TMQC_clf.fit(TMQC_X_res, TMQC_y_res)
    dnn_TQC = MLPClassifier(max_iter=10000, activation='relu')
    dnn_TQC_clf = RandomizedSearchCV(dnn_TQC, parameters, n_jobs=-1, cv=5)
    dnn_TQC_clf.fit(TQC_X_res, TQC_y_res)
    #dnn_WC13 = MLPClassifier(max_iter=10000, activation='relu')
    #dnn_WC13_clf = RandomizedSearchCV(dnn_WC13, parameters, n_jobs=-1, cv=5)
    #dnn_WC13_clf.fit(WC13_X_res, WC13_y_res)

    # Below codes are for the implementation of logistic regression training
    from sklearn.linear_model import LogisticRegression
    solver = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
    tol = [1e-2, 1e-3, 1e-4, 1e-5]
    reg_strength = [0.5, 1.0, 1.5]
    parameters = {'solver': solver,
	          'tol': tol,
	          'C': reg_strength}
    #lr_DM = LogisticRegression()
    #lr_DM_clf = RandomizedSearchCV(lr_DM, parameters, n_jobs=-1, cv=5)
    #lr_DM_clf.fit(DM_X_res, DM_y_res)
    lr_FI = LogisticRegression()
    lr_FI_clf = RandomizedSearchCV(lr_FI, parameters, n_jobs=-1, cv=5)
    lr_FI_clf.fit(FI_X_res, FI_y_res)
    lr_FG = LogisticRegression()
    lr_FG_clf = RandomizedSearchCV(lr_FG, parameters, n_jobs=-1, cv=5)
    lr_FG_clf.fit(FG_X_res, FG_y_res)
    #lr_GR = LogisticRegression()
    #lr_GR_clf = RandomizedSearchCV(lr_GR, parameters, n_jobs=-1, cv=5)
    #lr_GR_clf.fit(GR_X_res, GR_y_res)
    #lr_GR12 = LogisticRegression()
    #lr_GR12_clf = RandomizedSearchCV(lr_GR12, parameters, n_jobs=-1, cv=5)
    #lr_GR12_clf.fit(GR12_X_res, GR12_y_res)
    lr_GR27 = LogisticRegression()
    lr_GR27_clf = RandomizedSearchCV(lr_GR27, parameters, n_jobs=-1, cv=5)
    lr_GR27_clf.fit(GR27_X_res, GR27_y_res)
    lr_LM = LogisticRegression()
    lr_LM_clf = RandomizedSearchCV(lr_LM, parameters, n_jobs=-1, cv=5)
    lr_LM_clf.fit(LM_X_res, LM_y_res)
    lr_LMM = LogisticRegression()
    lr_LMM_clf = RandomizedSearchCV(lr_LMM, parameters, n_jobs=-1, cv=5)
    lr_LMM_clf.fit(LMM_X_res, LMM_y_res)
    #lr_MM14 = LogisticRegression()
    #lr_MM14_clf = RandomizedSearchCV(lr_MM14, parameters, n_jobs=-1, cv=5)
    #lr_MM14_clf.fit(MM14_X_res, MM14_y_res)
    #lr_MM16 = LogisticRegression()
    #lr_MM16_clf = RandomizedSearchCV(lr_MM16, parameters, n_jobs=-1, cv=5)
    #lr_MM16_clf.fit(MM16_X_res, MM16_y_res)
    lr_PC = LogisticRegression()
    lr_PC_clf = RandomizedSearchCV(lr_PC, parameters, n_jobs=-1, cv=5)
    lr_PC_clf.fit(PC_X_res, PC_y_res)
    lr_RG12 = LogisticRegression()
    lr_RG12_clf = RandomizedSearchCV(lr_RG12, parameters, n_jobs=-1, cv=5)
    lr_RG12_clf.fit(RG12_X_res, RG12_y_res)
    #lr_RG19 = LogisticRegression()
    #lr_RG19_clf = RandomizedSearchCV(lr_RG19, parameters, n_jobs=-1, cv=5)
    #lr_RG19_clf.fit(RG19_X_res, RG19_y_res)
    lr_RG2 = LogisticRegression()
    lr_RG2_clf = RandomizedSearchCV(lr_RG2, parameters, n_jobs=-1, cv=5)
    lr_RG2_clf.fit(RG2_X_res, RG2_y_res)
    lr_RG3 = LogisticRegression()
    lr_RG3_clf = RandomizedSearchCV(lr_RG3, parameters, n_jobs=-1, cv=5)
    lr_RG3_clf.fit(RG3_X_res, RG3_y_res)
    lr_RGM = LogisticRegression()
    lr_RGM_clf = RandomizedSearchCV(lr_RGM, parameters, n_jobs=-1, cv=5)
    lr_RGM_clf.fit(RGM_X_res, RGM_y_res)
    lr_RGQC = LogisticRegression()
    lr_RGQC_clf = RandomizedSearchCV(lr_RGQC, parameters, n_jobs=-1, cv=5)
    lr_RGQC_clf.fit(RGQC_X_res, RGQC_y_res)
    #lr_TMSA10 = LogisticRegression()
    #lr_TMSA10_clf = RandomizedSearchCV(lr_TMSA10, parameters, n_jobs=-1, cv=5)
    #lr_TMSA10_clf.fit(TMSA10_X_res, TMSA10_y_res)
    lr_T8 = LogisticRegression()
    lr_T8_clf = RandomizedSearchCV(lr_T8, parameters, n_jobs=-1, cv=5)
    lr_T8_clf.fit(T8_X_res, T8_y_res)
    #lr_T9 = LogisticRegression()
    #lr_T9_clf = RandomizedSearchCV(lr_T9, parameters, n_jobs=-1, cv=5)
    #lr_T9_clf.fit(T9_X_res, T9_y_res)
    lr_TM10 = LogisticRegression()
    lr_TM10_clf = RandomizedSearchCV(lr_TM10, parameters, n_jobs=-1, cv=5)
    lr_TM10_clf.fit(TM10_X_res, TM10_y_res)
    lr_TM4 = LogisticRegression()
    lr_TM4_clf = RandomizedSearchCV(lr_TM4, parameters, n_jobs=-1, cv=5)
    lr_TM4_clf.fit(TM4_X_res, TM4_y_res)
    lr_TM5 = LogisticRegression()
    lr_TM5_clf = RandomizedSearchCV(lr_TM5, parameters, n_jobs=-1, cv=5)
    lr_TM5_clf.fit(TM5_X_res, TM5_y_res)
    lr_TM6 = LogisticRegression()
    lr_TM6_clf = RandomizedSearchCV(lr_TM6, parameters, n_jobs=-1, cv=5)
    lr_TM6_clf.fit(TM6_X_res, TM6_y_res)
    lr_TM8 = LogisticRegression()
    lr_TM8_clf = RandomizedSearchCV(lr_TM8, parameters, n_jobs=-1, cv=5)
    lr_TM8_clf.fit(TM8_X_res, TM8_y_res)
    lr_TM9 = LogisticRegression()
    lr_TM9_clf = RandomizedSearchCV(lr_TM9, parameters, n_jobs=-1, cv=5)
    lr_TM9_clf.fit(TM9_X_res, TM9_y_res)
    lr_TMQC = LogisticRegression()
    lr_TMQC_clf = RandomizedSearchCV(lr_TMQC, parameters, n_jobs=-1, cv=5)
    lr_TMQC_clf.fit(TMQC_X_res, TMQC_y_res)
    lr_TQC = LogisticRegression()
    lr_TQC_clf = RandomizedSearchCV(lr_TQC, parameters, n_jobs=-1, cv=5)
    lr_TQC_clf.fit(TQC_X_res, TQC_y_res)
    #lr_WC13 = LogisticRegression()
    #lr_WC13_clf = RandomizedSearchCV(lr_WC13, parameters, n_jobs=-1, cv=5)
    #lr_WC13_clf.fit(WC13_X_res, WC13_y_res)

    # Below codes are for the implementation of Gaussian Naive Bayes training
    from sklearn.naive_bayes import GaussianNB
    #nb_DM_clf = GaussianNB()
    #nb_DM_clf.fit(DM_X_res, DM_y_res)
    nb_FI_clf = GaussianNB()
    nb_FI_clf.fit(FI_X_res, FI_y_res)
    nb_FG_clf = GaussianNB()
    nb_FG_clf.fit(FG_X_res, FG_y_res)
    #nb_GR_clf = GaussianNB()
    #nb_GR_clf.fit(GR_X_res, GR_y_res)
    #nb_GR12_clf = GaussianNB()
    #nb_GR12_clf.fit(GR12_X_res, GR12_y_res)
    nb_GR27_clf = GaussianNB()
    nb_GR27_clf.fit(GR27_X_res, GR27_y_res)
    nb_LM_clf = GaussianNB()
    nb_LM_clf.fit(LM_X_res, LM_y_res)
    nb_LMM_clf = GaussianNB()
    nb_LMM_clf.fit(LMM_X_res, LMM_y_res)
    #nb_MM14_clf = GaussianNB()
    #nb_MM14_clf.fit(MM14_X_res, MM14_y_res)
    #nb_MM16_clf = GaussianNB()
    #nb_MM16_clf.fit(MM16_X_res, MM16_y_res)
    nb_PC_clf = GaussianNB()
    nb_PC_clf.fit(PC_X_res, PC_y_res)
    nb_RG12_clf = GaussianNB()
    nb_RG12_clf.fit(RG12_X_res, RG12_y_res)
    #nb_RG19_clf = GaussianNB()
    #nb_RG19_clf.fit(RG19_X_res, RG19_y_res)
    nb_RG2_clf = GaussianNB()
    nb_RG2_clf.fit(RG2_X_res, RG2_y_res)
    nb_RG3_clf = GaussianNB()
    nb_RG3_clf.fit(RG3_X_res, RG3_y_res)
    nb_RGM_clf = GaussianNB()
    nb_RGM_clf.fit(RGM_X_res, RGM_y_res)
    nb_RGQC_clf = GaussianNB()
    nb_RGQC_clf.fit(RGQC_X_res, RGQC_y_res)
    #nb_TMSA10_clf = GaussianNB()
    #nb_TMSA10_clf.fit(TMSA10_X_res, TMSA10_y_res)
    nb_T8_clf = GaussianNB()
    nb_T8_clf.fit(T8_X_res, T8_y_res)
    #nb_T9_clf = GaussianNB()
    #nb_T9_clf.fit(T9_X_res, T9_y_res)
    nb_TM10_clf = GaussianNB()
    nb_TM10_clf.fit(TM10_X_res, TM10_y_res)
    nb_TM4_clf = GaussianNB()
    nb_TM4_clf.fit(TM4_X_res, TM4_y_res)
    nb_TM5_clf = GaussianNB()
    nb_TM5_clf.fit(TM5_X_res, TM5_y_res)
    nb_TM6_clf = GaussianNB()
    nb_TM6_clf.fit(TM6_X_res, TM6_y_res)
    nb_TM8_clf = GaussianNB()
    nb_TM8_clf.fit(TM8_X_res, TM8_y_res)
    nb_TM9_clf = GaussianNB()
    nb_TM9_clf.fit(TM9_X_res, TM9_y_res)
    nb_TMQC_clf = GaussianNB()
    nb_TMQC_clf.fit(TMQC_X_res, TMQC_y_res)
    nb_TQC_clf = GaussianNB()
    nb_TQC_clf.fit(TQC_X_res, TQC_y_res)
    #nb_WC13_clf = GaussianNB()
    #nb_WC13_clf.fit(WC13_X_res, WC13_y_res)



    # Below codes are for the implementation of random forest training
    from sklearn.ensemble import RandomForestClassifier
    n_tree = [50, 100, 200, 300, 400, 500, 600, 700]
    max_depth = [10, 20, 30, 40, 50, 60, 70]
    min_samples_split = [5, 10, 15, 20, 25, 30]
    parameters = {'n_estimators': n_tree,
		  'max_depth': max_depth,
		  'min_samples_split': min_samples_split}
    #rf_DM = RandomForestClassifier()
    #rf_DM_clf = RandomizedSearchCV(rf_DM, parameters, n_jobs=-1, cv=5)
    #rf_DM_clf.fit(DM_X_res, DM_y_res)
    rf_FI = RandomForestClassifier()
    rf_FI_clf = RandomizedSearchCV(rf_FI, parameters, n_jobs=-1, cv=5)
    rf_FI_clf.fit(FI_X_res, FI_y_res)
    rf_FG = RandomForestClassifier()
    rf_FG_clf = RandomizedSearchCV(rf_FG, parameters, n_jobs=-1, cv=5)
    rf_FG_clf.fit(FG_X_res, FG_y_res)
    #rf_GR = RandomForestClassifier()
    #rf_GR_clf = RandomizedSearchCV(rf_GR, parameters, n_jobs=-1, cv=5)
    #rf_GR_clf.fit(GR_X_res, GR_y_res)
    #rf_GR12 = RandomForestClassifier()
    #rf_GR12_clf = RandomizedSearchCV(rf_GR12, parameters, n_jobs=-1, cv=5)
    #rf_GR12_clf.fit(GR12_X_res, GR12_y_res)
    rf_GR27 = RandomForestClassifier()
    rf_GR27_clf = RandomizedSearchCV(rf_GR27, parameters, n_jobs=-1, cv=5)
    rf_GR27_clf.fit(GR27_X_res, GR27_y_res)
    rf_LM = RandomForestClassifier()
    rf_LM_clf = RandomizedSearchCV(rf_LM, parameters, n_jobs=-1, cv=5)
    rf_LM_clf.fit(LM_X_res, LM_y_res)
    rf_LMM = RandomForestClassifier()
    rf_LMM_clf = RandomizedSearchCV(rf_LMM, parameters, n_jobs=-1, cv=5)
    rf_LMM_clf.fit(LMM_X_res, LMM_y_res)
    #rf_MM14 = RandomForestClassifier()
    #rf_MM14_clf = RandomizedSearchCV(rf_MM14, parameters, n_jobs=-1, cv=5)
    #rf_MM14_clf.fit(MM14_X_res, MM14_y_res)
    #rf_MM16 = RandomForestClassifier()
    #rf_MM16_clf = RandomizedSearchCV(rf_MM16, parameters, n_jobs=-1, cv=5)
    #rf_MM16_clf.fit(MM16_X_res, MM16_y_res)
    rf_PC = RandomForestClassifier()
    rf_PC_clf = RandomizedSearchCV(rf_PC, parameters, n_jobs=-1, cv=5)
    rf_PC_clf.fit(PC_X_res, PC_y_res)
    rf_RG12 = RandomForestClassifier()
    rf_RG12_clf = RandomizedSearchCV(rf_RG12, parameters, n_jobs=-1, cv=5)
    rf_RG12_clf.fit(RG12_X_res, RG12_y_res)
    #rf_RG19 = RandomForestClassifier()
    #rf_RG19_clf = RandomizedSearchCV(rf_RG19, parameters, n_jobs=-1, cv=5)
    #rf_RG19_clf.fit(RG19_X_res, RG19_y_res)
    rf_RG2 = RandomForestClassifier()
    rf_RG2_clf = RandomizedSearchCV(rf_RG2, parameters, n_jobs=-1, cv=5)
    rf_RG2_clf.fit(RG2_X_res, RG2_y_res)
    rf_RG3 = RandomForestClassifier()
    rf_RG3_clf = RandomizedSearchCV(rf_RG3, parameters, n_jobs=-1, cv=5)
    rf_RG3_clf.fit(RG3_X_res, RG3_y_res)
    rf_RGM = RandomForestClassifier()
    rf_RGM_clf = RandomizedSearchCV(rf_RGM, parameters, n_jobs=-1, cv=5)
    rf_RGM_clf.fit(RGM_X_res, RGM_y_res)
    rf_RGQC = RandomForestClassifier()
    rf_RGQC_clf = RandomizedSearchCV(rf_RGQC, parameters, n_jobs=-1, cv=5)
    rf_RGQC_clf.fit(RGQC_X_res, RGQC_y_res)
    #rf_TMSA10 = RandomForestClassifier()
    #rf_TMSA10_clf = RandomizedSearchCV(rf_TMSA10, parameters, n_jobs=-1, cv=5)
    #rf_TMSA10_clf.fit(TMSA10_X_res, TMSA10_y_res)
    rf_T8 = RandomForestClassifier()
    rf_T8_clf = RandomizedSearchCV(rf_T8, parameters, n_jobs=-1, cv=5)
    rf_T8_clf.fit(T8_X_res, T8_y_res)
    #rf_T9 = RandomForestClassifier()
    #rf_T9_clf = RandomizedSearchCV(rf_T9, parameters, n_jobs=-1, cv=5)
    #rf_T9_clf.fit(T9_X_res, T9_y_res)
    rf_TM10 = RandomForestClassifier()
    rf_TM10_clf = RandomizedSearchCV(rf_TM10, parameters, n_jobs=-1, cv=5)
    rf_TM10_clf.fit(TM10_X_res, TM10_y_res)
    rf_TM4 = RandomForestClassifier()
    rf_TM4_clf = RandomizedSearchCV(rf_TM4, parameters, n_jobs=-1, cv=5)
    rf_TM4_clf.fit(TM4_X_res, TM4_y_res)
    rf_TM5 = RandomForestClassifier()
    rf_TM5_clf = RandomizedSearchCV(rf_TM5, parameters, n_jobs=-1, cv=5)
    rf_TM5_clf.fit(TM5_X_res, TM5_y_res)
    rf_TM6 = RandomForestClassifier()
    rf_TM6_clf = RandomizedSearchCV(rf_TM6, parameters, n_jobs=-1, cv=5)
    rf_TM6_clf.fit(TM6_X_res, TM6_y_res)
    rf_TM8 = RandomForestClassifier()
    rf_TM8_clf = RandomizedSearchCV(rf_TM8, parameters, n_jobs=-1, cv=5)
    rf_TM8_clf.fit(TM8_X_res, TM8_y_res)
    rf_TM9 = RandomForestClassifier()
    rf_TM9_clf = RandomizedSearchCV(rf_TM9, parameters, n_jobs=-1, cv=5)
    rf_TM9_clf.fit(TM9_X_res, TM9_y_res)
    rf_TMQC = RandomForestClassifier()
    rf_TMQC_clf = RandomizedSearchCV(rf_TMQC, parameters, n_jobs=-1, cv=5)
    rf_TMQC_clf.fit(TMQC_X_res, TMQC_y_res)
    rf_TQC = RandomForestClassifier()
    rf_TQC_clf = RandomizedSearchCV(rf_TQC, parameters, n_jobs=-1, cv=5)
    rf_TQC_clf.fit(TQC_X_res, TQC_y_res)
    #rf_WC13 = RandomForestClassifier()
    #rf_WC13_clf = RandomizedSearchCV(rf_WC13, parameters, n_jobs=-1, cv=5)
    #rf_WC13_clf.fit(WC13_X_res, WC13_y_res)



    # Below codes are for the implementation of support vector machine training
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV
    #reg_param = [0.5, 1.0, 1.5]
    #degree = [1, 2, 3, 4, 5]
    #kernel = ['rbf', 'linear', 'poly', 'sigmoid']
    #gamma = ['scale', 'auto']
    #tol = [1e-2, 1e-3, 1e-4]
    #svm_DM = SVC(probability = True)
    #svm_DM_clf = RandomizedSearchCV(svm_DM, parameters, n_jobs=-1, cv=5)
    #svm_DM_clf.fit(DM_X_res, DM_y_res)
    svm_FI = LinearSVC()
    svm_FI_clf = CalibratedClassifierCV(svm_FI, cv=5)
    svm_FI_clf.fit(FI_X_res, FI_y_res)
    svm_FG = LinearSVC()
    svm_FG_clf = CalibratedClassifierCV(svm_FG, cv=5)
    svm_FG_clf.fit(FG_X_res, FG_y_res)
    #svm_GR = SVC(probability = True)
    #svm_GR_clf = RandomizedSearchCV(svm_GR, parameters, n_jobs=-1, cv=5)
    #svm_GR_clf.fit(GR_X_res, GR_y_res)
    #svm_GR12 = SVC(probability = True)
    #svm_GR12_clf = RandomizedSearchCV(svm_GR12, parameters, n_jobs=-1, cv=5)
    #svm_GR12_clf.fit(GR12_X_res, GR12_y_res)
    svm_GR27 = LinearSVC()
    svm_GR27_clf = CalibratedClassifierCV(svm_GR27, cv=5)
    svm_GR27_clf.fit(GR27_X_res, GR27_y_res)
    svm_LM = LinearSVC()
    svm_LM_clf = CalibratedClassifierCV(svm_LM, cv=5)
    svm_LM_clf.fit(LM_X_res, LM_y_res)
    svm_LMM = LinearSVC()
    svm_LMM_clf = CalibratedClassifierCV(svm_LMM, cv=5)
    svm_LMM_clf.fit(LMM_X_res, LMM_y_res)
    #svm_MM14 = SVC(probability = True)
    #svm_MM14_clf = RandomizedSearchCV(svm_MM14, parameters, n_jobs=-1, cv=5)
    #svm_MM14_clf.fit(MM14_X_res, MM14_y_res)
    #svm_MM16 = SVC(probability = True)
    #svm_MM16_clf = RandomizedSearchCV(svm_MM16, parameters, n_jobs=-1, cv=5)
    #svm_MM16_clf.fit(MM16_X_res, MM16_y_res)
    svm_PC = LinearSVC()
    svm_PC_clf = CalibratedClassifierCV(svm_PC, cv=5)
    svm_PC_clf.fit(PC_X_res, PC_y_res)
    svm_RG12 = LinearSVC()
    svm_RG12_clf = CalibratedClassifierCV(svm_RG12, cv=5)
    svm_RG12_clf.fit(RG12_X_res, RG12_y_res)
    #svm_RG19 = SVC(probability = True)
    #svm_RG19_clf = RandomizedSearchCV(svm_RG19, parameters, n_jobs=-1, cv=5)
    #svm_RG19_clf.fit(RG19_X_res, RG19_y_res)
    svm_RG2 = LinearSVC()
    svm_RG2_clf = CalibratedClassifierCV(svm_RG2, cv=5)
    svm_RG2_clf.fit(RG2_X_res, RG2_y_res)
    svm_RG3 = LinearSVC()
    svm_RG3_clf = CalibratedClassifierCV(svm_RG3, cv=5)
    svm_RG3_clf.fit(RG3_X_res, RG3_y_res)
    svm_RGM = LinearSVC()
    svm_RGM_clf = CalibratedClassifierCV(svm_RGM, cv=5)
    svm_RGM_clf.fit(RGM_X_res, RGM_y_res)
    svm_RGQC = LinearSVC()
    svm_RGQC_clf = CalibratedClassifierCV(svm_RGQC, cv=5)
    svm_RGQC_clf.fit(RGQC_X_res, RGQC_y_res)
    #svm_TMSA10 = SVC(probability = True)
    #svm_TMSA10_clf = RandomizedSearchCV(svm_TMSA10, parameters, n_jobs=-1, cv=5)
    #svm_TMSA10_clf.fit(TMSA10_X_res, TMSA10_y_res)
    svm_T8 = LinearSVC()
    svm_T8_clf = CalibratedClassifierCV(svm_T8, cv=5)
    svm_T8_clf.fit(T8_X_res, T8_y_res)
    #svm_T9 = SVC(probability = True)
    #svm_T9_clf = RandomizedSearchCV(svm_T9, parameters, n_jobs=-1, cv=5)
    #svm_T9_clf.fit(T9_X_res, T9_y_res)
    svm_TM10 = LinearSVC()
    svm_TM10_clf = CalibratedClassifierCV(svm_TM10, cv=5)
    svm_TM10_clf.fit(TM10_X_res, TM10_y_res)
    svm_TM4 = LinearSVC()
    svm_TM4_clf = CalibratedClassifierCV(svm_TM4, cv=5)
    svm_TM4_clf.fit(TM4_X_res, TM4_y_res)
    svm_TM5 = LinearSVC()
    svm_TM5_clf = CalibratedClassifierCV(svm_TM5, cv=5)
    svm_TM5_clf.fit(TM5_X_res, TM5_y_res)
    svm_TM6 = LinearSVC()
    svm_TM6_clf = CalibratedClassifierCV(svm_TM6, cv=5)
    svm_TM6_clf.fit(TM6_X_res, TM6_y_res)
    svm_TM8 = LinearSVC()
    svm_TM8_clf = CalibratedClassifierCV(svm_TM8, cv=5)
    svm_TM8_clf.fit(TM8_X_res, TM8_y_res)
    svm_TM9 = LinearSVC()
    svm_TM9_clf = CalibratedClassifierCV(svm_TM9, cv=5)
    svm_TM9_clf.fit(TM9_X_res, TM9_y_res)
    svm_TMQC = LinearSVC()
    svm_TMQC_clf = CalibratedClassifierCV(svm_TMQC, cv=5)
    svm_TMQC_clf.fit(TMQC_X_res, TMQC_y_res)
    svm_TQC = LinearSVC()
    svm_TQC_clf = CalibratedClassifierCV(svm_TQC, cv=5)
    svm_TQC_clf.fit(TQC_X_res, TQC_y_res)
    #svm_WC13 = SVC(probability = True)
    #svm_WC13_clf = RandomizedSearchCV(svm_WC13, parameters, n_jobs=-1, cv=5)
    #svm_WC13_clf.fit(WC13_X_res, WC13_y_res)

    #dnn_pred_class_DM = dnn_DM_clf.predict(X_test)
    #dnn_pred_prob_DM = dnn_DM_clf.predict_proba(X_test)
    dnn_pred_class_FI = dnn_FI_clf.predict(X_test)
    dnn_pred_prob_FI = dnn_FI_clf.predict_proba(X_test)
    dnn_pred_class_FG = dnn_FG_clf.predict(X_test)
    dnn_pred_prob_FG = dnn_FG_clf.predict_proba(X_test)
    #dnn_pred_class_GR = dnn_GR_clf.predict(X_test)
    #dnn_pred_prob_GR = dnn_GR_clf.predict_proba(X_test)
    #dnn_pred_class_GR12 = dnn_GR12_clf.predict(X_test)
    #dnn_pred_prob_GR12 = dnn_GR12_clf.predict_proba(X_test)
    dnn_pred_class_GR27 = dnn_GR27_clf.predict(X_test)
    dnn_pred_prob_GR27 = dnn_GR27_clf.predict_proba(X_test)
    dnn_pred_class_LM = dnn_LM_clf.predict(X_test)
    dnn_pred_prob_LM = dnn_LM_clf.predict_proba(X_test)
    dnn_pred_class_LMM = dnn_LMM_clf.predict(X_test)
    dnn_pred_prob_LMM = dnn_LMM_clf.predict_proba(X_test)
    #dnn_pred_class_MM14 = dnn_MM14_clf.predict(X_test)
    #dnn_pred_prob_MM14 = dnn_MM14_clf.predict_proba(X_test)
    #dnn_pred_class_MM16 = dnn_MM16_clf.predict(X_test)
    #dnn_pred_prob_MM16 = dnn_MM16_clf.predict_proba(X_test)
    dnn_pred_class_PC = dnn_PC_clf.predict(X_test)
    dnn_pred_prob_PC = dnn_PC_clf.predict_proba(X_test)
    dnn_pred_class_RG12 = dnn_RG12_clf.predict(X_test)
    dnn_pred_prob_RG12 = dnn_RG12_clf.predict_proba(X_test)
    #dnn_pred_class_RG19 = dnn_RG19_clf.predict(X_test)
    #dnn_pred_prob_RG19 = dnn_RG19_clf.predict_proba(X_test)
    dnn_pred_class_RG2 = dnn_RG2_clf.predict(X_test)
    dnn_pred_prob_RG2 = dnn_RG2_clf.predict_proba(X_test)
    dnn_pred_class_RG3 = dnn_RG3_clf.predict(X_test)
    dnn_pred_prob_RG3 = dnn_RG3_clf.predict_proba(X_test)
    dnn_pred_class_RGM = dnn_RGM_clf.predict(X_test)
    dnn_pred_prob_RGM = dnn_RGM_clf.predict_proba(X_test)
    dnn_pred_class_RGQC = dnn_RGQC_clf.predict(X_test)
    dnn_pred_prob_RGQC = dnn_RGQC_clf.predict_proba(X_test)
    #dnn_pred_class_TMSA10 = dnn_TMSA10_clf.predict(X_test)
    #dnn_pred_prob_TMSA10 = dnn_TMSA10_clf.predict_proba(X_test)
    dnn_pred_class_T8 = dnn_T8_clf.predict(X_test)
    dnn_pred_prob_T8 = dnn_T8_clf.predict_proba(X_test)
    #dnn_pred_class_T9 = dnn_T9_clf.predict(X_test)
    #dnn_pred_prob_T9 = dnn_T9_clf.predict_proba(X_test)
    dnn_pred_class_TM10 = dnn_TM10_clf.predict(X_test)
    dnn_pred_prob_TM10 = dnn_TM10_clf.predict_proba(X_test)
    dnn_pred_class_TM4 = dnn_TM4_clf.predict(X_test)
    dnn_pred_prob_TM4 = dnn_TM4_clf.predict_proba(X_test)
    dnn_pred_class_TM5 = dnn_TM5_clf.predict(X_test)
    dnn_pred_prob_TM5 = dnn_TM5_clf.predict_proba(X_test)
    dnn_pred_class_TM6 = dnn_TM6_clf.predict(X_test)
    dnn_pred_prob_TM6 = dnn_TM6_clf.predict_proba(X_test)
    dnn_pred_class_TM8 = dnn_TM8_clf.predict(X_test)
    dnn_pred_prob_TM8 = dnn_TM8_clf.predict_proba(X_test)
    dnn_pred_class_TM9 = dnn_TM9_clf.predict(X_test)
    dnn_pred_prob_TM9 = dnn_TM9_clf.predict_proba(X_test)
    dnn_pred_class_TMQC = dnn_TMQC_clf.predict(X_test)
    dnn_pred_prob_TMQC = dnn_TMQC_clf.predict_proba(X_test)
    dnn_pred_class_TQC = dnn_TQC_clf.predict(X_test)
    dnn_pred_prob_TQC = dnn_TQC_clf.predict_proba(X_test)
    #dnn_pred_class_WC13 = dnn_WC13_clf.predict(X_test)
    #dnn_pred_prob_WC13 = dnn_WC13_clf.predict_proba(X_test)

    #lr_pred_class_DM = lr_DM_clf.predict(X_test)
    #lr_pred_prob_DM = lr_DM_clf.predict_proba(X_test)
    lr_pred_class_FI = lr_FI_clf.predict(X_test)
    lr_pred_prob_FI = lr_FI_clf.predict_proba(X_test)
    lr_pred_class_FG = lr_FG_clf.predict(X_test)
    lr_pred_prob_FG = lr_FG_clf.predict_proba(X_test)
    #lr_pred_class_GR = lr_GR_clf.predict(X_test)
    #lr_pred_prob_GR = lr_GR_clf.predict_proba(X_test)
    #lr_pred_class_GR12 = lr_GR12_clf.predict(X_test)
    #lr_pred_prob_GR12 = lr_GR12_clf.predict_proba(X_test)
    lr_pred_class_GR27 = lr_GR27_clf.predict(X_test)
    lr_pred_prob_GR27 = lr_GR27_clf.predict_proba(X_test)
    lr_pred_class_LM = lr_LM_clf.predict(X_test)
    lr_pred_prob_LM = lr_LM_clf.predict_proba(X_test)
    lr_pred_class_LMM = lr_LMM_clf.predict(X_test)
    lr_pred_prob_LMM = lr_LMM_clf.predict_proba(X_test)
    #lr_pred_class_MM14 = lr_MM14_clf.predict(X_test)
    #lr_pred_prob_MM14 = lr_MM14_clf.predict_proba(X_test)
    #lr_pred_class_MM16 = lr_MM16_clf.predict(X_test)
    #lr_pred_prob_MM16 = lr_MM16_clf.predict_proba(X_test)
    lr_pred_class_PC = lr_PC_clf.predict(X_test)
    lr_pred_prob_PC = lr_PC_clf.predict_proba(X_test)
    lr_pred_class_RG12 = lr_RG12_clf.predict(X_test)
    lr_pred_prob_RG12 = lr_RG12_clf.predict_proba(X_test)
    #lr_pred_class_RG19 = lr_RG19_clf.predict(X_test)
    #lr_pred_prob_RG19 = lr_RG19_clf.predict_proba(X_test)
    lr_pred_class_RG2 = lr_RG2_clf.predict(X_test)
    lr_pred_prob_RG2 = lr_RG2_clf.predict_proba(X_test)
    lr_pred_class_RG3 = lr_RG3_clf.predict(X_test)
    lr_pred_prob_RG3 = lr_RG3_clf.predict_proba(X_test)
    lr_pred_class_RGM = lr_RGM_clf.predict(X_test)
    lr_pred_prob_RGM = lr_RGM_clf.predict_proba(X_test)
    lr_pred_class_RGQC = lr_RGQC_clf.predict(X_test)
    lr_pred_prob_RGQC = lr_RGQC_clf.predict_proba(X_test)
    #lr_pred_class_TMSA10 = lr_TMSA10_clf.predict(X_test)
    #lr_pred_prob_TMSA10 = lr_TMSA10_clf.predict_proba(X_test)
    lr_pred_class_T8 = lr_T8_clf.predict(X_test)
    lr_pred_prob_T8 = lr_T8_clf.predict_proba(X_test)
    #lr_pred_class_T9 = lr_T9_clf.predict(X_test)
    #lr_pred_prob_T9 = lr_T9_clf.predict_proba(X_test)
    lr_pred_class_TM10 = lr_TM10_clf.predict(X_test)
    lr_pred_prob_TM10 = lr_TM10_clf.predict_proba(X_test)
    lr_pred_class_TM4 = lr_TM4_clf.predict(X_test)
    lr_pred_prob_TM4 = lr_TM4_clf.predict_proba(X_test)
    lr_pred_class_TM5 = lr_TM5_clf.predict(X_test)
    lr_pred_prob_TM5 = lr_TM5_clf.predict_proba(X_test)
    lr_pred_class_TM6 = lr_TM6_clf.predict(X_test)
    lr_pred_prob_TM6 = lr_TM6_clf.predict_proba(X_test)
    lr_pred_class_TM8 = lr_TM8_clf.predict(X_test)
    lr_pred_prob_TM8 = lr_TM8_clf.predict_proba(X_test)
    lr_pred_class_TM9 = lr_TM9_clf.predict(X_test)
    lr_pred_prob_TM9 = lr_TM9_clf.predict_proba(X_test)
    lr_pred_class_TMQC = lr_TMQC_clf.predict(X_test)
    lr_pred_prob_TMQC = lr_TMQC_clf.predict_proba(X_test)
    lr_pred_class_TQC = lr_TQC_clf.predict(X_test)
    lr_pred_prob_TQC = lr_TQC_clf.predict_proba(X_test)
    #lr_pred_class_WC13 = lr_WC13_clf.predict(X_test)
    #lr_pred_prob_WC13 = lr_WC13_clf.predict_proba(X_test)

    #nb_pred_class_DM = nb_DM_clf.predict(X_test)
    #nb_pred_prob_DM = nb_DM_clf.predict_proba(X_test)
    nb_pred_class_FI = nb_FI_clf.predict(X_test)
    nb_pred_prob_FI = nb_FI_clf.predict_proba(X_test)
    nb_pred_class_FG = nb_FG_clf.predict(X_test)
    nb_pred_prob_FG = nb_FG_clf.predict_proba(X_test)
    #nb_pred_class_GR = nb_GR_clf.predict(X_test)
    #nb_pred_prob_GR = nb_GR_clf.predict_proba(X_test)
    #nb_pred_class_GR12 = nb_GR12_clf.predict(X_test)
    #nb_pred_prob_GR12 = nb_GR12_clf.predict_proba(X_test)
    nb_pred_class_GR27 = nb_GR27_clf.predict(X_test)
    nb_pred_prob_GR27 = nb_GR27_clf.predict_proba(X_test)
    nb_pred_class_LM = nb_LM_clf.predict(X_test)
    nb_pred_prob_LM = nb_LM_clf.predict_proba(X_test)
    nb_pred_class_LMM = nb_LMM_clf.predict(X_test)
    nb_pred_prob_LMM = nb_LMM_clf.predict_proba(X_test)
    #nb_pred_class_MM14 = nb_MM14_clf.predict(X_test)
    #nb_pred_prob_MM14 = nb_MM14_clf.predict_proba(X_test)
    #nb_pred_class_MM16 = nb_MM16_clf.predict(X_test)
    #nb_pred_prob_MM16 = nb_MM16_clf.predict_proba(X_test)
    nb_pred_class_PC = nb_PC_clf.predict(X_test)
    nb_pred_prob_PC = nb_PC_clf.predict_proba(X_test)
    nb_pred_class_RG12 = nb_RG12_clf.predict(X_test)
    nb_pred_prob_RG12 = nb_RG12_clf.predict_proba(X_test)
    #nb_pred_class_RG19 = nb_RG19_clf.predict(X_test)
    #nb_pred_prob_RG19 = nb_RG19_clf.predict_proba(X_test)
    nb_pred_class_RG2 = nb_RG2_clf.predict(X_test)
    nb_pred_prob_RG2 = nb_RG2_clf.predict_proba(X_test)
    nb_pred_class_RG3 = nb_RG3_clf.predict(X_test)
    nb_pred_prob_RG3 = nb_RG3_clf.predict_proba(X_test)
    nb_pred_class_RGM = nb_RGM_clf.predict(X_test)
    nb_pred_prob_RGM = nb_RGM_clf.predict_proba(X_test)
    nb_pred_class_RGQC = nb_RGQC_clf.predict(X_test)
    nb_pred_prob_RGQC = nb_RGQC_clf.predict_proba(X_test)
    #nb_pred_class_TMSA10 = nb_TMSA10_clf.predict(X_test)
    #nb_pred_prob_TMSA10 = nb_TMSA10_clf.predict_proba(X_test)
    nb_pred_class_T8 = nb_T8_clf.predict(X_test)
    nb_pred_prob_T8 = nb_T8_clf.predict_proba(X_test)
    #nb_pred_class_T9 = nb_T9_clf.predict(X_test)
    #nb_pred_prob_T9 = nb_T9_clf.predict_proba(X_test)
    nb_pred_class_TM10 = nb_TM10_clf.predict(X_test)
    nb_pred_prob_TM10 = nb_TM10_clf.predict_proba(X_test)
    nb_pred_class_TM4 = nb_TM4_clf.predict(X_test)
    nb_pred_prob_TM4 = nb_TM4_clf.predict_proba(X_test)
    nb_pred_class_TM5 = nb_TM5_clf.predict(X_test)
    nb_pred_prob_TM5 = nb_TM5_clf.predict_proba(X_test)
    nb_pred_class_TM6 = nb_TM6_clf.predict(X_test)
    nb_pred_prob_TM6 = nb_TM6_clf.predict_proba(X_test)
    nb_pred_class_TM8 = nb_TM8_clf.predict(X_test)
    nb_pred_prob_TM8 = nb_TM8_clf.predict_proba(X_test)
    nb_pred_class_TM9 = nb_TM9_clf.predict(X_test)
    nb_pred_prob_TM9 = nb_TM9_clf.predict_proba(X_test)
    nb_pred_class_TMQC = nb_TMQC_clf.predict(X_test)
    nb_pred_prob_TMQC = nb_TMQC_clf.predict_proba(X_test)
    nb_pred_class_TQC = nb_TQC_clf.predict(X_test)
    nb_pred_prob_TQC = nb_TQC_clf.predict_proba(X_test)
    #nb_pred_class_WC13 = nb_WC13_clf.predict(X_test)
    #nb_pred_prob_WC13 = nb_WC13_clf.predict_proba(X_test)

    #rf_pred_class_DM = rf_DM_clf.predict(X_test)
    #rf_pred_prob_DM = rf_DM_clf.predict_proba(X_test)
    rf_pred_class_FI = rf_FI_clf.predict(X_test)
    rf_pred_prob_FI = rf_FI_clf.predict_proba(X_test)
    rf_pred_class_FG = rf_FG_clf.predict(X_test)
    rf_pred_prob_FG = rf_FG_clf.predict_proba(X_test)
    #rf_pred_class_GR = rf_GR_clf.predict(X_test)
    #rf_pred_prob_GR = rf_GR_clf.predict_proba(X_test)
    #rf_pred_class_GR12 = rf_GR12_clf.predict(X_test)
    #rf_pred_prob_GR12 = rf_GR12_clf.predict_proba(X_test)
    rf_pred_class_GR27 = rf_GR27_clf.predict(X_test)
    rf_pred_prob_GR27 = rf_GR27_clf.predict_proba(X_test)
    rf_pred_class_LM = rf_LM_clf.predict(X_test)
    rf_pred_prob_LM = rf_LM_clf.predict_proba(X_test)
    rf_pred_class_LMM = rf_LMM_clf.predict(X_test)
    rf_pred_prob_LMM = rf_LMM_clf.predict_proba(X_test)
    #rf_pred_class_MM14 = rf_MM14_clf.predict(X_test)
    #rf_pred_prob_MM14 = rf_MM14_clf.predict_proba(X_test)
    #rf_pred_class_MM16 = rf_MM16_clf.predict(X_test)
    #rf_pred_prob_MM16 = rf_MM16_clf.predict_proba(X_test)
    rf_pred_class_PC = rf_PC_clf.predict(X_test)
    rf_pred_prob_PC = rf_PC_clf.predict_proba(X_test)
    rf_pred_class_RG12 = rf_RG12_clf.predict(X_test)
    rf_pred_prob_RG12 = rf_RG12_clf.predict_proba(X_test)
    #rf_pred_class_RG19 = rf_RG19_clf.predict(X_test)
    #rf_pred_prob_RG19 = rf_RG19_clf.predict_proba(X_test)
    rf_pred_class_RG2 = rf_RG2_clf.predict(X_test)
    rf_pred_prob_RG2 = rf_RG2_clf.predict_proba(X_test)
    rf_pred_class_RG3 = rf_RG3_clf.predict(X_test)
    rf_pred_prob_RG3 = rf_RG3_clf.predict_proba(X_test)
    rf_pred_class_RGM = rf_RGM_clf.predict(X_test)
    rf_pred_prob_RGM = rf_RGM_clf.predict_proba(X_test)
    rf_pred_class_RGQC = rf_RGQC_clf.predict(X_test)
    rf_pred_prob_RGQC = rf_RGQC_clf.predict_proba(X_test)
    #rf_pred_class_TMSA10 = rf_TMSA10_clf.predict(X_test)
    #rf_pred_prob_TMSA10 = rf_TMSA10_clf.predict_proba(X_test)
    rf_pred_class_T8 = rf_T8_clf.predict(X_test)
    rf_pred_prob_T8 = rf_T8_clf.predict_proba(X_test)
    #rf_pred_class_T9 = rf_T9_clf.predict(X_test)
    #rf_pred_prob_T9 = rf_T9_clf.predict_proba(X_test)
    rf_pred_class_TM10 = rf_TM10_clf.predict(X_test)
    rf_pred_prob_TM10 = rf_TM10_clf.predict_proba(X_test)
    rf_pred_class_TM4 = rf_TM4_clf.predict(X_test)
    rf_pred_prob_TM4 = rf_TM4_clf.predict_proba(X_test)
    rf_pred_class_TM5 = rf_TM5_clf.predict(X_test)
    rf_pred_prob_TM5 = rf_TM5_clf.predict_proba(X_test)
    rf_pred_class_TM6 = rf_TM6_clf.predict(X_test)
    rf_pred_prob_TM6 = rf_TM6_clf.predict_proba(X_test)
    rf_pred_class_TM8 = rf_TM8_clf.predict(X_test)
    rf_pred_prob_TM8 = rf_TM8_clf.predict_proba(X_test)
    rf_pred_class_TM9 = rf_TM9_clf.predict(X_test)
    rf_pred_prob_TM9 = rf_TM9_clf.predict_proba(X_test)
    rf_pred_class_TMQC = rf_TMQC_clf.predict(X_test)
    rf_pred_prob_TMQC = rf_TMQC_clf.predict_proba(X_test)
    rf_pred_class_TQC = rf_TQC_clf.predict(X_test)
    rf_pred_prob_TQC = rf_TQC_clf.predict_proba(X_test)
    #rf_pred_class_WC13 = rf_WC13_clf.predict(X_test)
    #rf_pred_prob_WC13 = rf_WC13_clf.predict_proba(X_test)

    #svm_pred_class_DM = svm_DM_clf.predict(X_test)
    #svm_pred_prob_DM = svm_DM_clf.predict_proba(X_test)
    svm_pred_class_FI = svm_FI_clf.predict(X_test)
    svm_pred_prob_FI = svm_FI_clf.predict_proba(X_test)
    svm_pred_class_FG = svm_FG_clf.predict(X_test)
    svm_pred_prob_FG = svm_FG_clf.predict_proba(X_test)
    #svm_pred_class_GR = svm_GR_clf.predict(X_test)
    #svm_pred_prob_GR = svm_GR_clf.predict_proba(X_test)
    #svm_pred_class_GR12 = svm_GR12_clf.predict(X_test)
    #svm_pred_prob_GR12 = svm_GR12_clf.predict_proba(X_test)
    svm_pred_class_GR27 = svm_GR27_clf.predict(X_test)
    svm_pred_prob_GR27 = svm_GR27_clf.predict_proba(X_test)
    svm_pred_class_LM = svm_LM_clf.predict(X_test)
    svm_pred_prob_LM = svm_LM_clf.predict_proba(X_test)
    svm_pred_class_LMM = svm_LMM_clf.predict(X_test)
    svm_pred_prob_LMM = svm_LMM_clf.predict_proba(X_test)
    #svm_pred_class_MM14 = svm_MM14_clf.predict(X_test)
    #svm_pred_prob_MM14 = svm_MM14_clf.predict_proba(X_test)
    #svm_pred_class_MM16 = svm_MM16_clf.predict(X_test)
    #svm_pred_prob_MM16 = svm_MM16_clf.predict_proba(X_test)
    svm_pred_class_PC = svm_PC_clf.predict(X_test)
    svm_pred_prob_PC = svm_PC_clf.predict_proba(X_test)
    svm_pred_class_RG12 = svm_RG12_clf.predict(X_test)
    svm_pred_prob_RG12 = svm_RG12_clf.predict_proba(X_test)
    #svm_pred_class_RG19 = svm_RG19_clf.predict(X_test)
    #svm_pred_prob_RG19 = svm_RG19_clf.predict_proba(X_test)
    svm_pred_class_RG2 = svm_RG2_clf.predict(X_test)
    svm_pred_prob_RG2 = svm_RG2_clf.predict_proba(X_test)
    svm_pred_class_RG3 = svm_RG3_clf.predict(X_test)
    svm_pred_prob_RG3 = svm_RG3_clf.predict_proba(X_test)
    svm_pred_class_RGM = svm_RGM_clf.predict(X_test)
    svm_pred_prob_RGM = svm_RGM_clf.predict_proba(X_test)
    svm_pred_class_RGQC = svm_RGQC_clf.predict(X_test)
    svm_pred_prob_RGQC = svm_RGQC_clf.predict_proba(X_test)
    #svm_pred_class_TMSA10 = svm_TMSA10_clf.predict(X_test)
    #svm_pred_prob_TMSA10 = svm_TMSA10_clf.predict_proba(X_test)
    svm_pred_class_T8 = svm_T8_clf.predict(X_test)
    svm_pred_prob_T8 = svm_T8_clf.predict_proba(X_test)
    #svm_pred_class_T9 = svm_T9_clf.predict(X_test)
    #svm_pred_prob_T9 = svm_T9_clf.predict_proba(X_test)
    svm_pred_class_TM10 = svm_TM10_clf.predict(X_test)
    svm_pred_prob_TM10 = svm_TM10_clf.predict_proba(X_test)
    svm_pred_class_TM4 = svm_TM4_clf.predict(X_test)
    svm_pred_prob_TM4 = svm_TM4_clf.predict_proba(X_test)
    svm_pred_class_TM5 = svm_TM5_clf.predict(X_test)
    svm_pred_prob_TM5 = svm_TM5_clf.predict_proba(X_test)
    svm_pred_class_TM6 = svm_TM6_clf.predict(X_test)
    svm_pred_prob_TM6 = svm_TM6_clf.predict_proba(X_test)
    svm_pred_class_TM8 = svm_TM8_clf.predict(X_test)
    svm_pred_prob_TM8 = svm_TM8_clf.predict_proba(X_test)
    svm_pred_class_TM9 = svm_TM9_clf.predict(X_test)
    svm_pred_prob_TM9 = svm_TM9_clf.predict_proba(X_test)
    svm_pred_class_TMQC = svm_TMQC_clf.predict(X_test)
    svm_pred_prob_TMQC = svm_TMQC_clf.predict_proba(X_test)
    svm_pred_class_TQC = svm_TQC_clf.predict(X_test)
    svm_pred_prob_TQC = svm_TQC_clf.predict_proba(X_test)
    #svm_pred_class_WC13 = svm_WC13_clf.predict(X_test)
    #svm_pred_prob_WC13 = svm_WC13_clf.predict_proba(X_test)


    #Below dataframes are generated to store classification result based on predict probabilities
    #This procedure is necessary since this analysis uses one-versus-all classification method
    dnn_prediction = pd.DataFrame(columns=['Prediction'])
    lr_prediction = pd.DataFrame(columns=['Prediction'])
    nb_prediction = pd.DataFrame(columns=['Prediction'])
    rf_prediction = pd.DataFrame(columns=['Prediction'])
    svm_prediction = pd.DataFrame(columns=['Prediction'])

    #Below codes are for aggregating test results from one-versus-all deep neural network training
    for i in range(0, len(y_test)):
        #dnn_DM_index = 0
        dnn_FI_index = 0
        dnn_FG_index = 0
        #dnn_GR_index = 0
        #dnn_GR12_index = 0
        dnn_GR27_index = 0
        dnn_LM_index = 0
        dnn_LMM_index = 0
        #dnn_MM14_index = 0
        #dnn_MM16_index = 0
        dnn_PC_index = 0
        dnn_RG12_index = 0
        #dnn_RG19_index = 0
        dnn_RG2_index = 0
        dnn_RG3_index = 0
        dnn_RGM_index = 0
        dnn_RGQC_index = 0
        #dnn_TMSA10_index = 0
        dnn_T8_index = 0
        #dnn_T9_index = 0
        dnn_TM10_index = 0
        dnn_TM4_index = 0
        dnn_TM5_index = 0
        dnn_TM6_index = 0
        dnn_TM8_index = 0
        dnn_TM9_index = 0
        dnn_TMQC_index = 0
        dnn_TQC_index = 0
        #dnn_WC13_index = 0

        """
        if dnn_pred_class_DM[i] == "Deburring - Manual":
            if dnn_pred_prob_DM[i][0] >= 0.5:
                dnn_DM_index = 0
            else:
                dnn_DM_index = 1
        elif dnn_pred_class_DM[i] == "Others":
            if dnn_pred_prob_DM[i][0] < 0.5:
                dnn_DM_index = 0
            else:
                dnn_DM_index = 1
        """
        if dnn_pred_class_FI[i] == "Final Inspection Q.C.":
            if dnn_pred_prob_FI[i][0] >= 0.5:
                dnn_FI_index = 0
            else:
                dnn_FI_index = 1
        elif dnn_pred_class_FI[i] == "Others":
            if dnn_pred_prob_FI[i][0] < 0.5:
                dnn_FI_index = 0
            else:
                dnn_FI_index = 1
        if dnn_pred_class_FG[i] == "Flat Grinding - Machine 11":
            if dnn_pred_prob_FG[i][0] >= 0.5:
                dnn_FG_index = 0
            else:
                dnn_FG_index = 1
        elif dnn_pred_class_FG[i] == "Others":
            if dnn_pred_prob_FG[i][0] < 0.5:
                dnn_FG_index = 0
            else:
                dnn_FG_index = 1
        """
        if dnn_pred_class_GR[i] == "Grinding Rework":
            if dnn_pred_prob_GR[i][0] >= 0.5:
                dnn_GR_index = 0
            else:
                dnn_GR_index = 1
        elif dnn_pred_class_GR[i] == "Others":
            if dnn_pred_prob_GR[i][0] < 0.5:
                dnn_GR_index = 0
            else:
                dnn_GR_index = 1
        """
        """
        if dnn_pred_class_GR12[i] == "Grinding Rework - Machine 12":
            if dnn_pred_prob_GR12[i][0] >= 0.5:
                dnn_GR12_index = 0
            else:
                dnn_GR12_index = 1
        elif dnn_pred_class_GR12[i] == "Others":
            if dnn_pred_prob_GR12[i][0] < 0.5:
                dnn_GR12_index = 0
            else:
                dnn_GR12_index = 1
        """
        if dnn_pred_class_GR27[i] == "Grinding Rework - Machine 27":
            if dnn_pred_prob_GR27[i][0] >= 0.5:
                dnn_GR27_index = 0
            else:
                dnn_GR27_index = 1
        elif dnn_pred_class_GR27[i] == "Others":
            if dnn_pred_prob_GR27[i][0] < 0.5:
                dnn_GR27_index = 0
            else:
                dnn_GR27_index = 1
        if dnn_pred_class_LM[i] == "Lapping - Machine 1":
            if dnn_pred_prob_LM[i][0] >= 0.5:
                dnn_LM_index = 0
            else:
                dnn_LM_index = 1
        elif dnn_pred_class_LM[i] == "Others":
            if dnn_pred_prob_LM[i][0] < 0.5:
                dnn_LM_index = 0
            else:
                dnn_LM_index = 1
        if dnn_pred_class_LMM[i] == "Laser Marking - Machine 7":
            if dnn_pred_prob_LMM[i][0] >= 0.5:
                dnn_LMM_index = 0
            else:
                dnn_LMM_index = 1
        elif dnn_pred_class_LMM[i] == "Others":
            if dnn_pred_prob_LMM[i][0] < 0.5:
                dnn_LMM_index = 0
            else:
                dnn_LMM_index = 1
        """
        if dnn_pred_class_MM14[i] == "Milling - Machine 14":
            if dnn_pred_prob_MM14[i][0] >= 0.5:
                dnn_MM14_index = 0
            else:
                dnn_MM14_index = 1
        elif dnn_pred_class_MM14[i] == "Others":
            if dnn_pred_prob_MM14[i][0] < 0.5:
                dnn_MM14_index = 0
            else:
                dnn_MM14_index = 1
        """
        """
        if dnn_pred_class_MM16[i] == "Milling - Machine 16":
            if dnn_pred_prob_MM16[i][0] >= 0.5:
                dnn_MM16_index = 0
            else:
                dnn_MM16_index = 1
        elif dnn_pred_class_MM16[i] == "Others":
            if dnn_pred_prob_MM16[i][0] < 0.5:
                dnn_MM16_index = 0
            else:
                dnn_MM16_index = 1
        """
        if dnn_pred_class_PC[i] == "Packing":
            if dnn_pred_prob_PC[i][0] >= 0.5:
                dnn_PC_index = 0
            else:
                dnn_PC_index = 1
        elif dnn_pred_class_PC[i] == "Others":
            if dnn_pred_prob_PC[i][0] < 0.5:
                dnn_PC_index = 0
            else:
                dnn_PC_index = 1
        if dnn_pred_class_RG12[i] == "Round Grinding - Machine 12":
            if dnn_pred_prob_RG12[i][0] >= 0.5:
                dnn_RG12_index = 0
            else:
                dnn_RG12_index = 1
        elif dnn_pred_class_RG12[i] == "Others":
            if dnn_pred_prob_RG12[i][0] < 0.5:
                dnn_RG12_index = 0
            else:
                dnn_RG12_index = 1
        """
        if dnn_pred_class_RG19[i] == "Round Grinding - Machine 19":
            if dnn_pred_prob_RG19[i][0] >= 0.5:
                dnn_RG19_index = 0
            else:
                dnn_RG19_index = 1
        elif dnn_pred_class_RG19[i] == "Others":
            if dnn_pred_prob_RG19[i][0] < 0.5:
                dnn_RG19_index = 0
            else:
                dnn_RG19_index = 1
        """
        if dnn_pred_class_RG2[i] == "Round Grinding - Machine 2":
            if dnn_pred_prob_RG2[i][0] >= 0.5:
                dnn_RG2_index = 0
            else:
                dnn_RG2_index = 1
        elif dnn_pred_class_RG2[i] == "Others":
            if dnn_pred_prob_RG2[i][0] < 0.5:
                dnn_RG2_index = 0
            else:
                dnn_RG2_index = 1
        if dnn_pred_class_RG3[i] == "Round Grinding - Machine 3":
            if dnn_pred_prob_RG3[i][0] >= 0.5:
                dnn_RG3_index = 0
            else:
                dnn_RG3_index = 1
        elif dnn_pred_class_RG3[i] == "Others":
            if dnn_pred_prob_RG3[i][0] < 0.5:
                dnn_RG3_index = 0
            else:
                dnn_RG3_index = 1
        if dnn_pred_class_RGM[i] == "Round Grinding - Manual":
            if dnn_pred_prob_RGM[i][0] >= 0.5:
                dnn_RGM_index = 0
            else:
                dnn_RGM_index = 1
        elif dnn_pred_class_RGM[i] == "Others":
            if dnn_pred_prob_RGM[i][0] < 0.5:
                dnn_RGM_index = 0
            else:
                dnn_RGM_index = 1
        if dnn_pred_class_RGQC[i] == "Round Grinding - Q.C.":
            if dnn_pred_prob_RGQC[i][0] >= 0.5:
                dnn_RGQC_index = 0
            else:
                dnn_RGQC_index = 1
        elif dnn_pred_class_RGQC[i] == "Others":
            if dnn_pred_prob_RGQC[i][0] < 0.5:
                dnn_RGQC_index = 0
            else:
                dnn_RGQC_index = 1
        """
        if dnn_pred_class_TMSA10[i] == "Turn & Mill. & Screw Assem - Machine 10":
            if dnn_pred_prob_TMSA10[i][0] >= 0.5:
                dnn_TMSA10_index = 0
            else:
                dnn_TMSA10_index = 1
        elif dnn_pred_class_TMSA10[i] == "Others":
            if dnn_pred_prob_TMSA10[i][0] < 0.5:
                dnn_TMSA10_index = 0
            else:
                dnn_TMSA10_index = 1
        """
        if dnn_pred_class_T8[i] == "Turning - Machine 8":
            if dnn_pred_prob_T8[i][0] >= 0.5:
                dnn_T8_index = 0
            else:
                dnn_T8_index = 1
        elif dnn_pred_class_T8[i] == "Others":
            if dnn_pred_prob_T8[i][0] < 0.5:
                dnn_T8_index = 0
            else:
                dnn_T8_index = 1
        """
        if dnn_pred_class_T9[i] == "Turning - Machine 9":
            if dnn_pred_prob_T9[i][0] >= 0.5:
                dnn_T9_index = 0
            else:
                dnn_T9_index = 1
        elif dnn_pred_class_T9[i] == "Others":
            if dnn_pred_prob_T9[i][0] < 0.5:
                dnn_T9_index = 0
            else:
                dnn_T9_index = 1
        """
        if dnn_pred_class_TM10[i] == "Turning & Milling - Machine 10":
            if dnn_pred_prob_TM10[i][0] >= 0.5:
                dnn_TM10_index = 0
            else:
                dnn_TM10_index = 1
        elif dnn_pred_class_TM10[i] == "Others":
            if dnn_pred_prob_TM10[i][0] < 0.5:
                dnn_TM10_index = 0
            else:
                dnn_TM10_index = 1
        if dnn_pred_class_TM4[i] == "Turning & Milling - Machine 4":
            if dnn_pred_prob_TM4[i][0] >= 0.5:
                dnn_TM4_index = 0
            else:
                dnn_TM4_index = 1
        elif dnn_pred_class_TM4[i] == "Others":
            if dnn_pred_prob_TM4[i][0] < 0.5:
                dnn_TM4_index = 0
            else:
                dnn_TM4_index = 1
        if dnn_pred_class_TM5[i] == "Turning & Milling - Machine 5":
            if dnn_pred_prob_TM5[i][0] >= 0.5:
                dnn_TM5_index = 0
            else:
                dnn_TM5_index = 1
        elif dnn_pred_class_TM5[i] == "Others":
            if dnn_pred_prob_TM5[i][0] < 0.5:
                dnn_TM5_index = 0
            else:
                dnn_TM5_index = 1
        if dnn_pred_class_TM6[i] == "Turning & Milling - Machine 6":
            if dnn_pred_prob_TM6[i][0] >= 0.5:
                dnn_TM6_index = 0
            else:
                dnn_TM6_index = 1
        elif dnn_pred_class_TM6[i] == "Others":
            if dnn_pred_prob_TM6[i][0] < 0.5:
                dnn_TM6_index = 0
            else:
                dnn_TM6_index = 1
        if dnn_pred_class_TM8[i] == "Turning & Milling - Machine 8":
            if dnn_pred_prob_TM8[i][0] >= 0.5:
                dnn_TM8_index = 0
            else:
                dnn_TM8_index = 1
        elif dnn_pred_class_TM8[i] == "Others":
            if dnn_pred_prob_TM8[i][0] < 0.5:
                dnn_TM8_index = 0
            else:
                dnn_TM8_index = 1
        if dnn_pred_class_TM9[i] == "Turning & Milling - Machine 9":
            if dnn_pred_prob_TM9[i][0] >= 0.5:
                dnn_TM9_index = 0
            else:
                dnn_TM9_index = 1
        elif dnn_pred_class_TM9[i] == "Others":
            if dnn_pred_prob_TM9[i][0] < 0.5:
                dnn_TM9_index = 0
            else:
                dnn_TM9_index = 1
        if dnn_pred_class_TMQC[i] == "Turning & Milling Q.C.":
            if dnn_pred_prob_TMQC[i][0] >= 0.5:
                dnn_TMQC_index = 0
            else:
                dnn_TMQC_index = 1
        elif dnn_pred_class_TMQC[i] == "Others":
            if dnn_pred_prob_TMQC[i][0] < 0.5:
                dnn_TMQC_index = 0
            else:
                dnn_TMQC_index = 1
        if dnn_pred_class_TQC[i] == "Turning Q.C.":
            if dnn_pred_prob_TQC[i][0] >= 0.5:
                dnn_TQC_index = 0
            else:
                dnn_TQC_index = 1
        elif dnn_pred_class_TQC[i] == "Others":
            if dnn_pred_prob_TQC[i][0] < 0.5:
                dnn_TQC_index = 0
            else:
                dnn_TQC_index = 1
        """
        if dnn_pred_class_WC13[i] == "Wire Cut - Machine 13":
            if dnn_pred_prob_WC13[i][0] >= 0.5:
                dnn_WC13_index = 0
            else:
                dnn_WC13_index = 1
        elif dnn_pred_class_WC13[i] == "Others":
            if dnn_pred_prob_WC13[i][0] < 0.5:
                dnn_WC13_index = 0
            else:
                dnn_WC13_index = 1
        """
        #if dnn_pred_prob_DM[i][dnn_DM_index] == max(dnn_pred_prob_FI[i][dnn_FI_index], dnn_pred_prob_FG[i][dnn_FG_index], dnn_pred_prob_GR27[i][dnn_GR27_index], dnn_pred_prob_LM[i][dnn_LM_index], dnn_pred_prob_LMM[i][dnn_LMM_index], dnn_pred_prob_PC[i][dnn_PC_index], dnn_pred_prob_RG12[i][dnn_RG12_index], dnn_pred_prob_RG2[i][dnn_RG2_index], dnn_pred_prob_RG3[i][dnn_RG3_index], dnn_pred_prob_RGM[i][dnn_RGM_index], dnn_pred_prob_RGQC[i][dnn_RGQC_index], dnn_pred_prob_T8[i][dnn_T8_index], dnn_pred_prob_TM10[i][dnn_TM10_index], dnn_pred_prob_TM4[i][dnn_TM4_index], dnn_pred_prob_TM5[i][dnn_TM5_index], dnn_pred_prob_TM6[i][dnn_TM6_index], dnn_pred_prob_TM8[i][dnn_TM8_index], dnn_pred_prob_TM9[i][dnn_TM9_index], dnn_pred_prob_TMQC[i][dnn_TMQC_index], dnn_pred_prob_TQC[i][dnn_TQC_index]):
        #    dnn_prediction.loc[i] = "Deburring - Manual"
        if dnn_pred_prob_FI[i][dnn_FI_index] == max(dnn_pred_prob_FI[i][dnn_FI_index], dnn_pred_prob_FG[i][dnn_FG_index], dnn_pred_prob_GR27[i][dnn_GR27_index], dnn_pred_prob_LM[i][dnn_LM_index], dnn_pred_prob_LMM[i][dnn_LMM_index], dnn_pred_prob_PC[i][dnn_PC_index], dnn_pred_prob_RG12[i][dnn_RG12_index], dnn_pred_prob_RG2[i][dnn_RG2_index], dnn_pred_prob_RG3[i][dnn_RG3_index], dnn_pred_prob_RGM[i][dnn_RGM_index], dnn_pred_prob_RGQC[i][dnn_RGQC_index], dnn_pred_prob_T8[i][dnn_T8_index], dnn_pred_prob_TM10[i][dnn_TM10_index], dnn_pred_prob_TM4[i][dnn_TM4_index], dnn_pred_prob_TM5[i][dnn_TM5_index], dnn_pred_prob_TM6[i][dnn_TM6_index], dnn_pred_prob_TM8[i][dnn_TM8_index], dnn_pred_prob_TM9[i][dnn_TM9_index], dnn_pred_prob_TMQC[i][dnn_TMQC_index], dnn_pred_prob_TQC[i][dnn_TQC_index]):
            dnn_prediction.loc[i] = "Final Inspection Q.C."
        elif dnn_pred_prob_FG[i][dnn_FG_index] == max(dnn_pred_prob_FI[i][dnn_FI_index], dnn_pred_prob_FG[i][dnn_FG_index], dnn_pred_prob_GR27[i][dnn_GR27_index], dnn_pred_prob_LM[i][dnn_LM_index], dnn_pred_prob_LMM[i][dnn_LMM_index], dnn_pred_prob_PC[i][dnn_PC_index], dnn_pred_prob_RG12[i][dnn_RG12_index], dnn_pred_prob_RG2[i][dnn_RG2_index], dnn_pred_prob_RG3[i][dnn_RG3_index], dnn_pred_prob_RGM[i][dnn_RGM_index], dnn_pred_prob_RGQC[i][dnn_RGQC_index], dnn_pred_prob_T8[i][dnn_T8_index], dnn_pred_prob_TM10[i][dnn_TM10_index], dnn_pred_prob_TM4[i][dnn_TM4_index], dnn_pred_prob_TM5[i][dnn_TM5_index], dnn_pred_prob_TM6[i][dnn_TM6_index], dnn_pred_prob_TM8[i][dnn_TM8_index], dnn_pred_prob_TM9[i][dnn_TM9_index], dnn_pred_prob_TMQC[i][dnn_TMQC_index], dnn_pred_prob_TQC[i][dnn_TQC_index]):
            dnn_prediction.loc[i] = "Flat Grinding - Machine 11"
        #elif dnn_pred_prob_GR[i][dnn_GR_index] == max(dnn_pred_prob_FI[i][dnn_FI_index], dnn_pred_prob_FG[i][dnn_FG_index], dnn_pred_prob_GR27[i][dnn_GR27_index], dnn_pred_prob_LM[i][dnn_LM_index], dnn_pred_prob_LMM[i][dnn_LMM_index], dnn_pred_prob_PC[i][dnn_PC_index], dnn_pred_prob_RG12[i][dnn_RG12_index], dnn_pred_prob_RG2[i][dnn_RG2_index], dnn_pred_prob_RG3[i][dnn_RG3_index], dnn_pred_prob_RGM[i][dnn_RGM_index], dnn_pred_prob_RGQC[i][dnn_RGQC_index], dnn_pred_prob_T8[i][dnn_T8_index], dnn_pred_prob_TM10[i][dnn_TM10_index], dnn_pred_prob_TM4[i][dnn_TM4_index], dnn_pred_prob_TM5[i][dnn_TM5_index], dnn_pred_prob_TM6[i][dnn_TM6_index], dnn_pred_prob_TM8[i][dnn_TM8_index], dnn_pred_prob_TM9[i][dnn_TM9_index], dnn_pred_prob_TMQC[i][dnn_TMQC_index], dnn_pred_prob_TQC[i][dnn_TQC_index]):
        #    dnn_prediction.loc[i] = "Grinding Rework"
        #elif dnn_pred_prob_GR12[i][dnn_GR12_index] == max(dnn_pred_prob_FI[i][dnn_FI_index], dnn_pred_prob_FG[i][dnn_FG_index], dnn_pred_prob_GR27[i][dnn_GR27_index], dnn_pred_prob_LM[i][dnn_LM_index], dnn_pred_prob_LMM[i][dnn_LMM_index], dnn_pred_prob_PC[i][dnn_PC_index], dnn_pred_prob_RG12[i][dnn_RG12_index], dnn_pred_prob_RG2[i][dnn_RG2_index], dnn_pred_prob_RG3[i][dnn_RG3_index], dnn_pred_prob_RGM[i][dnn_RGM_index], dnn_pred_prob_RGQC[i][dnn_RGQC_index], dnn_pred_prob_T8[i][dnn_T8_index], dnn_pred_prob_TM10[i][dnn_TM10_index], dnn_pred_prob_TM4[i][dnn_TM4_index], dnn_pred_prob_TM5[i][dnn_TM5_index], dnn_pred_prob_TM6[i][dnn_TM6_index], dnn_pred_prob_TM8[i][dnn_TM8_index], dnn_pred_prob_TM9[i][dnn_TM9_index], dnn_pred_prob_TMQC[i][dnn_TMQC_index], dnn_pred_prob_TQC[i][dnn_TQC_index]):
        #    dnn_prediction.loc[i] = "Grinding Rework - Machine 12"
        elif dnn_pred_prob_GR27[i][dnn_GR27_index] == max(dnn_pred_prob_FI[i][dnn_FI_index], dnn_pred_prob_FG[i][dnn_FG_index], dnn_pred_prob_GR27[i][dnn_GR27_index], dnn_pred_prob_LM[i][dnn_LM_index], dnn_pred_prob_LMM[i][dnn_LMM_index], dnn_pred_prob_PC[i][dnn_PC_index], dnn_pred_prob_RG12[i][dnn_RG12_index], dnn_pred_prob_RG2[i][dnn_RG2_index], dnn_pred_prob_RG3[i][dnn_RG3_index], dnn_pred_prob_RGM[i][dnn_RGM_index], dnn_pred_prob_RGQC[i][dnn_RGQC_index], dnn_pred_prob_T8[i][dnn_T8_index], dnn_pred_prob_TM10[i][dnn_TM10_index], dnn_pred_prob_TM4[i][dnn_TM4_index], dnn_pred_prob_TM5[i][dnn_TM5_index], dnn_pred_prob_TM6[i][dnn_TM6_index], dnn_pred_prob_TM8[i][dnn_TM8_index], dnn_pred_prob_TM9[i][dnn_TM9_index], dnn_pred_prob_TMQC[i][dnn_TMQC_index], dnn_pred_prob_TQC[i][dnn_TQC_index]):
            dnn_prediction.loc[i] = "Grinding Rework - Machine 27"
        elif dnn_pred_prob_LM[i][dnn_LM_index] == max(dnn_pred_prob_FI[i][dnn_FI_index], dnn_pred_prob_FG[i][dnn_FG_index], dnn_pred_prob_GR27[i][dnn_GR27_index], dnn_pred_prob_LM[i][dnn_LM_index], dnn_pred_prob_LMM[i][dnn_LMM_index], dnn_pred_prob_PC[i][dnn_PC_index], dnn_pred_prob_RG12[i][dnn_RG12_index], dnn_pred_prob_RG2[i][dnn_RG2_index], dnn_pred_prob_RG3[i][dnn_RG3_index], dnn_pred_prob_RGM[i][dnn_RGM_index], dnn_pred_prob_RGQC[i][dnn_RGQC_index], dnn_pred_prob_T8[i][dnn_T8_index], dnn_pred_prob_TM10[i][dnn_TM10_index], dnn_pred_prob_TM4[i][dnn_TM4_index], dnn_pred_prob_TM5[i][dnn_TM5_index], dnn_pred_prob_TM6[i][dnn_TM6_index], dnn_pred_prob_TM8[i][dnn_TM8_index], dnn_pred_prob_TM9[i][dnn_TM9_index], dnn_pred_prob_TMQC[i][dnn_TMQC_index], dnn_pred_prob_TQC[i][dnn_TQC_index]):
            dnn_prediction.loc[i] = "Lapping - Machine 1"
        elif dnn_pred_prob_LMM[i][dnn_LMM_index] == max(dnn_pred_prob_FI[i][dnn_FI_index], dnn_pred_prob_FG[i][dnn_FG_index], dnn_pred_prob_GR27[i][dnn_GR27_index], dnn_pred_prob_LM[i][dnn_LM_index], dnn_pred_prob_LMM[i][dnn_LMM_index], dnn_pred_prob_PC[i][dnn_PC_index], dnn_pred_prob_RG12[i][dnn_RG12_index], dnn_pred_prob_RG2[i][dnn_RG2_index], dnn_pred_prob_RG3[i][dnn_RG3_index], dnn_pred_prob_RGM[i][dnn_RGM_index], dnn_pred_prob_RGQC[i][dnn_RGQC_index], dnn_pred_prob_T8[i][dnn_T8_index], dnn_pred_prob_TM10[i][dnn_TM10_index], dnn_pred_prob_TM4[i][dnn_TM4_index], dnn_pred_prob_TM5[i][dnn_TM5_index], dnn_pred_prob_TM6[i][dnn_TM6_index], dnn_pred_prob_TM8[i][dnn_TM8_index], dnn_pred_prob_TM9[i][dnn_TM9_index], dnn_pred_prob_TMQC[i][dnn_TMQC_index], dnn_pred_prob_TQC[i][dnn_TQC_index]):
            dnn_prediction.loc[i] = "Laser Marking - Machine 7"
        #elif dnn_pred_prob_MM14[i][dnn_MM14_index] == max(dnn_pred_prob_FI[i][dnn_FI_index], dnn_pred_prob_FG[i][dnn_FG_index], dnn_pred_prob_GR27[i][dnn_GR27_index], dnn_pred_prob_LM[i][dnn_LM_index], dnn_pred_prob_LMM[i][dnn_LMM_index], dnn_pred_prob_PC[i][dnn_PC_index], dnn_pred_prob_RG12[i][dnn_RG12_index], dnn_pred_prob_RG2[i][dnn_RG2_index], dnn_pred_prob_RG3[i][dnn_RG3_index], dnn_pred_prob_RGM[i][dnn_RGM_index], dnn_pred_prob_RGQC[i][dnn_RGQC_index], dnn_pred_prob_T8[i][dnn_T8_index], dnn_pred_prob_TM10[i][dnn_TM10_index], dnn_pred_prob_TM4[i][dnn_TM4_index], dnn_pred_prob_TM5[i][dnn_TM5_index], dnn_pred_prob_TM6[i][dnn_TM6_index], dnn_pred_prob_TM8[i][dnn_TM8_index], dnn_pred_prob_TM9[i][dnn_TM9_index], dnn_pred_prob_TMQC[i][dnn_TMQC_index], dnn_pred_prob_TQC[i][dnn_TQC_index]):
        #    dnn_prediction.loc[i] = "Milling - Machine 14"
        #elif dnn_pred_prob_MM16[i][dnn_MM16_index] == max(dnn_pred_prob_FI[i][dnn_FI_index], dnn_pred_prob_FG[i][dnn_FG_index], dnn_pred_prob_GR27[i][dnn_GR27_index], dnn_pred_prob_LM[i][dnn_LM_index], dnn_pred_prob_LMM[i][dnn_LMM_index], dnn_pred_prob_PC[i][dnn_PC_index], dnn_pred_prob_RG12[i][dnn_RG12_index], dnn_pred_prob_RG2[i][dnn_RG2_index], dnn_pred_prob_RG3[i][dnn_RG3_index], dnn_pred_prob_RGM[i][dnn_RGM_index], dnn_pred_prob_RGQC[i][dnn_RGQC_index], dnn_pred_prob_T8[i][dnn_T8_index], dnn_pred_prob_TM10[i][dnn_TM10_index], dnn_pred_prob_TM4[i][dnn_TM4_index], dnn_pred_prob_TM5[i][dnn_TM5_index], dnn_pred_prob_TM6[i][dnn_TM6_index], dnn_pred_prob_TM8[i][dnn_TM8_index], dnn_pred_prob_TM9[i][dnn_TM9_index], dnn_pred_prob_TMQC[i][dnn_TMQC_index], dnn_pred_prob_TQC[i][dnn_TQC_index]):
        #    dnn_prediction.loc[i] = "Milling - Machine 16"
        elif dnn_pred_prob_PC[i][dnn_PC_index] == max(dnn_pred_prob_FI[i][dnn_FI_index], dnn_pred_prob_FG[i][dnn_FG_index], dnn_pred_prob_GR27[i][dnn_GR27_index], dnn_pred_prob_LM[i][dnn_LM_index], dnn_pred_prob_LMM[i][dnn_LMM_index], dnn_pred_prob_PC[i][dnn_PC_index], dnn_pred_prob_RG12[i][dnn_RG12_index], dnn_pred_prob_RG2[i][dnn_RG2_index], dnn_pred_prob_RG3[i][dnn_RG3_index], dnn_pred_prob_RGM[i][dnn_RGM_index], dnn_pred_prob_RGQC[i][dnn_RGQC_index], dnn_pred_prob_T8[i][dnn_T8_index], dnn_pred_prob_TM10[i][dnn_TM10_index], dnn_pred_prob_TM4[i][dnn_TM4_index], dnn_pred_prob_TM5[i][dnn_TM5_index], dnn_pred_prob_TM6[i][dnn_TM6_index], dnn_pred_prob_TM8[i][dnn_TM8_index], dnn_pred_prob_TM9[i][dnn_TM9_index], dnn_pred_prob_TMQC[i][dnn_TMQC_index], dnn_pred_prob_TQC[i][dnn_TQC_index]):
            dnn_prediction.loc[i] = "Packing"
        elif dnn_pred_prob_RG12[i][dnn_RG12_index] == max(dnn_pred_prob_FI[i][dnn_FI_index], dnn_pred_prob_FG[i][dnn_FG_index], dnn_pred_prob_GR27[i][dnn_GR27_index], dnn_pred_prob_LM[i][dnn_LM_index], dnn_pred_prob_LMM[i][dnn_LMM_index], dnn_pred_prob_PC[i][dnn_PC_index], dnn_pred_prob_RG12[i][dnn_RG12_index], dnn_pred_prob_RG2[i][dnn_RG2_index], dnn_pred_prob_RG3[i][dnn_RG3_index], dnn_pred_prob_RGM[i][dnn_RGM_index], dnn_pred_prob_RGQC[i][dnn_RGQC_index], dnn_pred_prob_T8[i][dnn_T8_index], dnn_pred_prob_TM10[i][dnn_TM10_index], dnn_pred_prob_TM4[i][dnn_TM4_index], dnn_pred_prob_TM5[i][dnn_TM5_index], dnn_pred_prob_TM6[i][dnn_TM6_index], dnn_pred_prob_TM8[i][dnn_TM8_index], dnn_pred_prob_TM9[i][dnn_TM9_index], dnn_pred_prob_TMQC[i][dnn_TMQC_index], dnn_pred_prob_TQC[i][dnn_TQC_index]):
            dnn_prediction.loc[i] = "Round Grinding - Machine 12"
        #elif dnn_pred_prob_RG19[i][dnn_RG19_index] == max(dnn_pred_prob_FI[i][dnn_FI_index], dnn_pred_prob_FG[i][dnn_FG_index], dnn_pred_prob_GR27[i][dnn_GR27_index], dnn_pred_prob_LM[i][dnn_LM_index], dnn_pred_prob_LMM[i][dnn_LMM_index], dnn_pred_prob_PC[i][dnn_PC_index], dnn_pred_prob_RG12[i][dnn_RG12_index], dnn_pred_prob_RG2[i][dnn_RG2_index], dnn_pred_prob_RG3[i][dnn_RG3_index], dnn_pred_prob_RGM[i][dnn_RGM_index], dnn_pred_prob_RGQC[i][dnn_RGQC_index], dnn_pred_prob_T8[i][dnn_T8_index], dnn_pred_prob_TM10[i][dnn_TM10_index], dnn_pred_prob_TM4[i][dnn_TM4_index], dnn_pred_prob_TM5[i][dnn_TM5_index], dnn_pred_prob_TM6[i][dnn_TM6_index], dnn_pred_prob_TM8[i][dnn_TM8_index], dnn_pred_prob_TM9[i][dnn_TM9_index], dnn_pred_prob_TMQC[i][dnn_TMQC_index], dnn_pred_prob_TQC[i][dnn_TQC_index]):
        #    dnn_prediction.loc[i] = "Round Grinding - Machine 19"
        elif dnn_pred_prob_RG2[i][dnn_RG2_index] == max(dnn_pred_prob_FI[i][dnn_FI_index], dnn_pred_prob_FG[i][dnn_FG_index], dnn_pred_prob_GR27[i][dnn_GR27_index], dnn_pred_prob_LM[i][dnn_LM_index], dnn_pred_prob_LMM[i][dnn_LMM_index], dnn_pred_prob_PC[i][dnn_PC_index], dnn_pred_prob_RG12[i][dnn_RG12_index], dnn_pred_prob_RG2[i][dnn_RG2_index], dnn_pred_prob_RG3[i][dnn_RG3_index], dnn_pred_prob_RGM[i][dnn_RGM_index], dnn_pred_prob_RGQC[i][dnn_RGQC_index], dnn_pred_prob_T8[i][dnn_T8_index], dnn_pred_prob_TM10[i][dnn_TM10_index], dnn_pred_prob_TM4[i][dnn_TM4_index], dnn_pred_prob_TM5[i][dnn_TM5_index], dnn_pred_prob_TM6[i][dnn_TM6_index], dnn_pred_prob_TM8[i][dnn_TM8_index], dnn_pred_prob_TM9[i][dnn_TM9_index], dnn_pred_prob_TMQC[i][dnn_TMQC_index], dnn_pred_prob_TQC[i][dnn_TQC_index]):
            dnn_prediction.loc[i] = "Round Grinding - Machine 2"
        elif dnn_pred_prob_RG3[i][dnn_RG3_index] == max(dnn_pred_prob_FI[i][dnn_FI_index], dnn_pred_prob_FG[i][dnn_FG_index], dnn_pred_prob_GR27[i][dnn_GR27_index], dnn_pred_prob_LM[i][dnn_LM_index], dnn_pred_prob_LMM[i][dnn_LMM_index], dnn_pred_prob_PC[i][dnn_PC_index], dnn_pred_prob_RG12[i][dnn_RG12_index], dnn_pred_prob_RG2[i][dnn_RG2_index], dnn_pred_prob_RG3[i][dnn_RG3_index], dnn_pred_prob_RGM[i][dnn_RGM_index], dnn_pred_prob_RGQC[i][dnn_RGQC_index], dnn_pred_prob_T8[i][dnn_T8_index], dnn_pred_prob_TM10[i][dnn_TM10_index], dnn_pred_prob_TM4[i][dnn_TM4_index], dnn_pred_prob_TM5[i][dnn_TM5_index], dnn_pred_prob_TM6[i][dnn_TM6_index], dnn_pred_prob_TM8[i][dnn_TM8_index], dnn_pred_prob_TM9[i][dnn_TM9_index], dnn_pred_prob_TMQC[i][dnn_TMQC_index], dnn_pred_prob_TQC[i][dnn_TQC_index]):
            dnn_prediction.loc[i] = "Round Grinding - Machine 3"
        elif dnn_pred_prob_RGM[i][dnn_RGM_index] == max(dnn_pred_prob_FI[i][dnn_FI_index], dnn_pred_prob_FG[i][dnn_FG_index], dnn_pred_prob_GR27[i][dnn_GR27_index], dnn_pred_prob_LM[i][dnn_LM_index], dnn_pred_prob_LMM[i][dnn_LMM_index], dnn_pred_prob_PC[i][dnn_PC_index], dnn_pred_prob_RG12[i][dnn_RG12_index], dnn_pred_prob_RG2[i][dnn_RG2_index], dnn_pred_prob_RG3[i][dnn_RG3_index], dnn_pred_prob_RGM[i][dnn_RGM_index], dnn_pred_prob_RGQC[i][dnn_RGQC_index], dnn_pred_prob_T8[i][dnn_T8_index], dnn_pred_prob_TM10[i][dnn_TM10_index], dnn_pred_prob_TM4[i][dnn_TM4_index], dnn_pred_prob_TM5[i][dnn_TM5_index], dnn_pred_prob_TM6[i][dnn_TM6_index], dnn_pred_prob_TM8[i][dnn_TM8_index], dnn_pred_prob_TM9[i][dnn_TM9_index], dnn_pred_prob_TMQC[i][dnn_TMQC_index], dnn_pred_prob_TQC[i][dnn_TQC_index]):
            dnn_prediction.loc[i] = "Round Grinding - Manual"
        elif dnn_pred_prob_RGQC[i][dnn_RGQC_index] == max(dnn_pred_prob_FI[i][dnn_FI_index], dnn_pred_prob_FG[i][dnn_FG_index], dnn_pred_prob_GR27[i][dnn_GR27_index], dnn_pred_prob_LM[i][dnn_LM_index], dnn_pred_prob_LMM[i][dnn_LMM_index], dnn_pred_prob_PC[i][dnn_PC_index], dnn_pred_prob_RG12[i][dnn_RG12_index], dnn_pred_prob_RG2[i][dnn_RG2_index], dnn_pred_prob_RG3[i][dnn_RG3_index], dnn_pred_prob_RGM[i][dnn_RGM_index], dnn_pred_prob_RGQC[i][dnn_RGQC_index], dnn_pred_prob_T8[i][dnn_T8_index], dnn_pred_prob_TM10[i][dnn_TM10_index], dnn_pred_prob_TM4[i][dnn_TM4_index], dnn_pred_prob_TM5[i][dnn_TM5_index], dnn_pred_prob_TM6[i][dnn_TM6_index], dnn_pred_prob_TM8[i][dnn_TM8_index], dnn_pred_prob_TM9[i][dnn_TM9_index], dnn_pred_prob_TMQC[i][dnn_TMQC_index], dnn_pred_prob_TQC[i][dnn_TQC_index]):
            dnn_prediction.loc[i] = "Round Grinding - Q.C."
        #elif dnn_pred_prob_TMSA10[i][dnn_TMSA10_index] == max(dnn_pred_prob_FI[i][dnn_FI_index], dnn_pred_prob_FG[i][dnn_FG_index], dnn_pred_prob_GR27[i][dnn_GR27_index], dnn_pred_prob_LM[i][dnn_LM_index], dnn_pred_prob_LMM[i][dnn_LMM_index], dnn_pred_prob_PC[i][dnn_PC_index], dnn_pred_prob_RG12[i][dnn_RG12_index], dnn_pred_prob_RG2[i][dnn_RG2_index], dnn_pred_prob_RG3[i][dnn_RG3_index], dnn_pred_prob_RGM[i][dnn_RGM_index], dnn_pred_prob_RGQC[i][dnn_RGQC_index], dnn_pred_prob_T8[i][dnn_T8_index], dnn_pred_prob_TM10[i][dnn_TM10_index], dnn_pred_prob_TM4[i][dnn_TM4_index], dnn_pred_prob_TM5[i][dnn_TM5_index], dnn_pred_prob_TM6[i][dnn_TM6_index], dnn_pred_prob_TM8[i][dnn_TM8_index], dnn_pred_prob_TM9[i][dnn_TM9_index], dnn_pred_prob_TMQC[i][dnn_TMQC_index], dnn_pred_prob_TQC[i][dnn_TQC_index]):
        #    dnn_prediction.loc[i] = "Turn & Mill. & Screw Assem - Machine 10"
        elif dnn_pred_prob_T8[i][dnn_T8_index] == max(dnn_pred_prob_FI[i][dnn_FI_index], dnn_pred_prob_FG[i][dnn_FG_index], dnn_pred_prob_GR27[i][dnn_GR27_index], dnn_pred_prob_LM[i][dnn_LM_index], dnn_pred_prob_LMM[i][dnn_LMM_index], dnn_pred_prob_PC[i][dnn_PC_index], dnn_pred_prob_RG12[i][dnn_RG12_index], dnn_pred_prob_RG2[i][dnn_RG2_index], dnn_pred_prob_RG3[i][dnn_RG3_index], dnn_pred_prob_RGM[i][dnn_RGM_index], dnn_pred_prob_RGQC[i][dnn_RGQC_index], dnn_pred_prob_T8[i][dnn_T8_index], dnn_pred_prob_TM10[i][dnn_TM10_index], dnn_pred_prob_TM4[i][dnn_TM4_index], dnn_pred_prob_TM5[i][dnn_TM5_index], dnn_pred_prob_TM6[i][dnn_TM6_index], dnn_pred_prob_TM8[i][dnn_TM8_index], dnn_pred_prob_TM9[i][dnn_TM9_index], dnn_pred_prob_TMQC[i][dnn_TMQC_index], dnn_pred_prob_TQC[i][dnn_TQC_index]):
            dnn_prediction.loc[i] = "Turning - Machine 8"
        #elif dnn_pred_prob_T9[i][dnn_T9_index] == max(dnn_pred_prob_FI[i][dnn_FI_index], dnn_pred_prob_FG[i][dnn_FG_index], dnn_pred_prob_GR27[i][dnn_GR27_index], dnn_pred_prob_LM[i][dnn_LM_index], dnn_pred_prob_LMM[i][dnn_LMM_index], dnn_pred_prob_PC[i][dnn_PC_index], dnn_pred_prob_RG12[i][dnn_RG12_index], dnn_pred_prob_RG2[i][dnn_RG2_index], dnn_pred_prob_RG3[i][dnn_RG3_index], dnn_pred_prob_RGM[i][dnn_RGM_index], dnn_pred_prob_RGQC[i][dnn_RGQC_index], dnn_pred_prob_T8[i][dnn_T8_index], dnn_pred_prob_TM10[i][dnn_TM10_index], dnn_pred_prob_TM4[i][dnn_TM4_index], dnn_pred_prob_TM5[i][dnn_TM5_index], dnn_pred_prob_TM6[i][dnn_TM6_index], dnn_pred_prob_TM8[i][dnn_TM8_index], dnn_pred_prob_TM9[i][dnn_TM9_index], dnn_pred_prob_TMQC[i][dnn_TMQC_index], dnn_pred_prob_TQC[i][dnn_TQC_index]):
        #    dnn_prediction.loc[i] = "Turning - Machine 9"
        elif dnn_pred_prob_TM10[i][dnn_TM10_index] == max(dnn_pred_prob_FI[i][dnn_FI_index], dnn_pred_prob_FG[i][dnn_FG_index], dnn_pred_prob_GR27[i][dnn_GR27_index], dnn_pred_prob_LM[i][dnn_LM_index], dnn_pred_prob_LMM[i][dnn_LMM_index], dnn_pred_prob_PC[i][dnn_PC_index], dnn_pred_prob_RG12[i][dnn_RG12_index], dnn_pred_prob_RG2[i][dnn_RG2_index], dnn_pred_prob_RG3[i][dnn_RG3_index], dnn_pred_prob_RGM[i][dnn_RGM_index], dnn_pred_prob_RGQC[i][dnn_RGQC_index], dnn_pred_prob_T8[i][dnn_T8_index], dnn_pred_prob_TM10[i][dnn_TM10_index], dnn_pred_prob_TM4[i][dnn_TM4_index], dnn_pred_prob_TM5[i][dnn_TM5_index], dnn_pred_prob_TM6[i][dnn_TM6_index], dnn_pred_prob_TM8[i][dnn_TM8_index], dnn_pred_prob_TM9[i][dnn_TM9_index], dnn_pred_prob_TMQC[i][dnn_TMQC_index], dnn_pred_prob_TQC[i][dnn_TQC_index]):
            dnn_prediction.loc[i] = "Turning & Milling - Machine 10"
        elif dnn_pred_prob_TM4[i][dnn_TM4_index] == max(dnn_pred_prob_FI[i][dnn_FI_index], dnn_pred_prob_FG[i][dnn_FG_index], dnn_pred_prob_GR27[i][dnn_GR27_index], dnn_pred_prob_LM[i][dnn_LM_index], dnn_pred_prob_LMM[i][dnn_LMM_index], dnn_pred_prob_PC[i][dnn_PC_index], dnn_pred_prob_RG12[i][dnn_RG12_index], dnn_pred_prob_RG2[i][dnn_RG2_index], dnn_pred_prob_RG3[i][dnn_RG3_index], dnn_pred_prob_RGM[i][dnn_RGM_index], dnn_pred_prob_RGQC[i][dnn_RGQC_index], dnn_pred_prob_T8[i][dnn_T8_index], dnn_pred_prob_TM10[i][dnn_TM10_index], dnn_pred_prob_TM4[i][dnn_TM4_index], dnn_pred_prob_TM5[i][dnn_TM5_index], dnn_pred_prob_TM6[i][dnn_TM6_index], dnn_pred_prob_TM8[i][dnn_TM8_index], dnn_pred_prob_TM9[i][dnn_TM9_index], dnn_pred_prob_TMQC[i][dnn_TMQC_index], dnn_pred_prob_TQC[i][dnn_TQC_index]):
            dnn_prediction.loc[i] = "Turning & Milling - Machine 4"
        elif dnn_pred_prob_TM5[i][dnn_TM5_index] == max(dnn_pred_prob_FI[i][dnn_FI_index], dnn_pred_prob_FG[i][dnn_FG_index], dnn_pred_prob_GR27[i][dnn_GR27_index], dnn_pred_prob_LM[i][dnn_LM_index], dnn_pred_prob_LMM[i][dnn_LMM_index], dnn_pred_prob_PC[i][dnn_PC_index], dnn_pred_prob_RG12[i][dnn_RG12_index], dnn_pred_prob_RG2[i][dnn_RG2_index], dnn_pred_prob_RG3[i][dnn_RG3_index], dnn_pred_prob_RGM[i][dnn_RGM_index], dnn_pred_prob_RGQC[i][dnn_RGQC_index], dnn_pred_prob_T8[i][dnn_T8_index], dnn_pred_prob_TM10[i][dnn_TM10_index], dnn_pred_prob_TM4[i][dnn_TM4_index], dnn_pred_prob_TM5[i][dnn_TM5_index], dnn_pred_prob_TM6[i][dnn_TM6_index], dnn_pred_prob_TM8[i][dnn_TM8_index], dnn_pred_prob_TM9[i][dnn_TM9_index], dnn_pred_prob_TMQC[i][dnn_TMQC_index], dnn_pred_prob_TQC[i][dnn_TQC_index]):
            dnn_prediction.loc[i] = "Turning & Milling - Machine 5"
        elif dnn_pred_prob_TM6[i][dnn_TM6_index] == max(dnn_pred_prob_FI[i][dnn_FI_index], dnn_pred_prob_FG[i][dnn_FG_index], dnn_pred_prob_GR27[i][dnn_GR27_index], dnn_pred_prob_LM[i][dnn_LM_index], dnn_pred_prob_LMM[i][dnn_LMM_index], dnn_pred_prob_PC[i][dnn_PC_index], dnn_pred_prob_RG12[i][dnn_RG12_index], dnn_pred_prob_RG2[i][dnn_RG2_index], dnn_pred_prob_RG3[i][dnn_RG3_index], dnn_pred_prob_RGM[i][dnn_RGM_index], dnn_pred_prob_RGQC[i][dnn_RGQC_index], dnn_pred_prob_T8[i][dnn_T8_index], dnn_pred_prob_TM10[i][dnn_TM10_index], dnn_pred_prob_TM4[i][dnn_TM4_index], dnn_pred_prob_TM5[i][dnn_TM5_index], dnn_pred_prob_TM6[i][dnn_TM6_index], dnn_pred_prob_TM8[i][dnn_TM8_index], dnn_pred_prob_TM9[i][dnn_TM9_index], dnn_pred_prob_TMQC[i][dnn_TMQC_index], dnn_pred_prob_TQC[i][dnn_TQC_index]):
            dnn_prediction.loc[i] = "Turning & Milling - Machine 6"
        elif dnn_pred_prob_TM8[i][dnn_TM8_index] == max(dnn_pred_prob_FI[i][dnn_FI_index], dnn_pred_prob_FG[i][dnn_FG_index], dnn_pred_prob_GR27[i][dnn_GR27_index], dnn_pred_prob_LM[i][dnn_LM_index], dnn_pred_prob_LMM[i][dnn_LMM_index], dnn_pred_prob_PC[i][dnn_PC_index], dnn_pred_prob_RG12[i][dnn_RG12_index], dnn_pred_prob_RG2[i][dnn_RG2_index], dnn_pred_prob_RG3[i][dnn_RG3_index], dnn_pred_prob_RGM[i][dnn_RGM_index], dnn_pred_prob_RGQC[i][dnn_RGQC_index], dnn_pred_prob_T8[i][dnn_T8_index], dnn_pred_prob_TM10[i][dnn_TM10_index], dnn_pred_prob_TM4[i][dnn_TM4_index], dnn_pred_prob_TM5[i][dnn_TM5_index], dnn_pred_prob_TM6[i][dnn_TM6_index], dnn_pred_prob_TM8[i][dnn_TM8_index], dnn_pred_prob_TM9[i][dnn_TM9_index], dnn_pred_prob_TMQC[i][dnn_TMQC_index], dnn_pred_prob_TQC[i][dnn_TQC_index]):
            dnn_prediction.loc[i] = "Turning & Milling - Machine 8"
        elif dnn_pred_prob_TM9[i][dnn_TM9_index] == max(dnn_pred_prob_FI[i][dnn_FI_index], dnn_pred_prob_FG[i][dnn_FG_index], dnn_pred_prob_GR27[i][dnn_GR27_index], dnn_pred_prob_LM[i][dnn_LM_index], dnn_pred_prob_LMM[i][dnn_LMM_index], dnn_pred_prob_PC[i][dnn_PC_index], dnn_pred_prob_RG12[i][dnn_RG12_index], dnn_pred_prob_RG2[i][dnn_RG2_index], dnn_pred_prob_RG3[i][dnn_RG3_index], dnn_pred_prob_RGM[i][dnn_RGM_index], dnn_pred_prob_RGQC[i][dnn_RGQC_index], dnn_pred_prob_T8[i][dnn_T8_index], dnn_pred_prob_TM10[i][dnn_TM10_index], dnn_pred_prob_TM4[i][dnn_TM4_index], dnn_pred_prob_TM5[i][dnn_TM5_index], dnn_pred_prob_TM6[i][dnn_TM6_index], dnn_pred_prob_TM8[i][dnn_TM8_index], dnn_pred_prob_TM9[i][dnn_TM9_index], dnn_pred_prob_TMQC[i][dnn_TMQC_index], dnn_pred_prob_TQC[i][dnn_TQC_index]):
            dnn_prediction.loc[i] = "Turning & Milling - Machine 9"
        elif dnn_pred_prob_TMQC[i][dnn_TMQC_index] == max(dnn_pred_prob_FI[i][dnn_FI_index], dnn_pred_prob_FG[i][dnn_FG_index], dnn_pred_prob_GR27[i][dnn_GR27_index], dnn_pred_prob_LM[i][dnn_LM_index], dnn_pred_prob_LMM[i][dnn_LMM_index], dnn_pred_prob_PC[i][dnn_PC_index], dnn_pred_prob_RG12[i][dnn_RG12_index], dnn_pred_prob_RG2[i][dnn_RG2_index], dnn_pred_prob_RG3[i][dnn_RG3_index], dnn_pred_prob_RGM[i][dnn_RGM_index], dnn_pred_prob_RGQC[i][dnn_RGQC_index], dnn_pred_prob_T8[i][dnn_T8_index], dnn_pred_prob_TM10[i][dnn_TM10_index], dnn_pred_prob_TM4[i][dnn_TM4_index], dnn_pred_prob_TM5[i][dnn_TM5_index], dnn_pred_prob_TM6[i][dnn_TM6_index], dnn_pred_prob_TM8[i][dnn_TM8_index], dnn_pred_prob_TM9[i][dnn_TM9_index], dnn_pred_prob_TMQC[i][dnn_TMQC_index], dnn_pred_prob_TQC[i][dnn_TQC_index]):
            dnn_prediction.loc[i] = "Turning & Milling Q.C."
        elif dnn_pred_prob_TQC[i][dnn_TQC_index] == max(dnn_pred_prob_FI[i][dnn_FI_index], dnn_pred_prob_FG[i][dnn_FG_index], dnn_pred_prob_GR27[i][dnn_GR27_index], dnn_pred_prob_LM[i][dnn_LM_index], dnn_pred_prob_LMM[i][dnn_LMM_index], dnn_pred_prob_PC[i][dnn_PC_index], dnn_pred_prob_RG12[i][dnn_RG12_index], dnn_pred_prob_RG2[i][dnn_RG2_index], dnn_pred_prob_RG3[i][dnn_RG3_index], dnn_pred_prob_RGM[i][dnn_RGM_index], dnn_pred_prob_RGQC[i][dnn_RGQC_index], dnn_pred_prob_T8[i][dnn_T8_index], dnn_pred_prob_TM10[i][dnn_TM10_index], dnn_pred_prob_TM4[i][dnn_TM4_index], dnn_pred_prob_TM5[i][dnn_TM5_index], dnn_pred_prob_TM6[i][dnn_TM6_index], dnn_pred_prob_TM8[i][dnn_TM8_index], dnn_pred_prob_TM9[i][dnn_TM9_index], dnn_pred_prob_TMQC[i][dnn_TMQC_index], dnn_pred_prob_TQC[i][dnn_TQC_index]):
            dnn_prediction.loc[i] = "Turning Q.C."
        #elif dnn_pred_prob_WC13[i][dnn_WC13_index] == max(dnn_pred_prob_FI[i][dnn_FI_index], dnn_pred_prob_FG[i][dnn_FG_index], dnn_pred_prob_GR27[i][dnn_GR27_index], dnn_pred_prob_LM[i][dnn_LM_index], dnn_pred_prob_LMM[i][dnn_LMM_index], dnn_pred_prob_PC[i][dnn_PC_index], dnn_pred_prob_RG12[i][dnn_RG12_index], dnn_pred_prob_RG2[i][dnn_RG2_index], dnn_pred_prob_RG3[i][dnn_RG3_index], dnn_pred_prob_RGM[i][dnn_RGM_index], dnn_pred_prob_RGQC[i][dnn_RGQC_index], dnn_pred_prob_T8[i][dnn_T8_index], dnn_pred_prob_TM10[i][dnn_TM10_index], dnn_pred_prob_TM4[i][dnn_TM4_index], dnn_pred_prob_TM5[i][dnn_TM5_index], dnn_pred_prob_TM6[i][dnn_TM6_index], dnn_pred_prob_TM8[i][dnn_TM8_index], dnn_pred_prob_TM9[i][dnn_TM9_index], dnn_pred_prob_TMQC[i][dnn_TMQC_index], dnn_pred_prob_TQC[i][dnn_TQC_index]):
        #    dnn_prediction.loc[i] = "Wire Cut - Machine 13"


    def get_precision(dnn_conf_matrix):
        dnn_tp_1 = dnn_conf_matrix[0][0]
        dnn_tp_2 = dnn_conf_matrix[1][1]
        dnn_tp_3 = dnn_conf_matrix[2][2]
        dnn_tp_4 = dnn_conf_matrix[3][3]
        dnn_tp_5 = dnn_conf_matrix[4][4]
        dnn_tp_6 = dnn_conf_matrix[5][5]
        dnn_tp_7 = dnn_conf_matrix[6][6]
        dnn_tp_8 = dnn_conf_matrix[7][7]
        dnn_tp_9 = dnn_conf_matrix[8][8]
        dnn_tp_10 = dnn_conf_matrix[9][9]
        dnn_tp_11 = dnn_conf_matrix[10][10]
        dnn_tp_12 = dnn_conf_matrix[11][11]
        dnn_tp_13 = dnn_conf_matrix[12][12]
        dnn_tp_14 = dnn_conf_matrix[13][13]
        dnn_tp_15 = dnn_conf_matrix[14][14]
        dnn_tp_16 = dnn_conf_matrix[15][15]
        dnn_tp_17 = dnn_conf_matrix[16][16]
        dnn_tp_18 = dnn_conf_matrix[17][17]
        dnn_tp_19 = dnn_conf_matrix[18][18]
        dnn_tp_20 = dnn_conf_matrix[19][19]
        #dnn_tp_21 = dnn_conf_matrix[20][20]
        #dnn_tp_22 = dnn_conf_matrix[21][21]
        #dnn_tp_23 = dnn_conf_matrix[22][22]
        #dnn_tp_24 = dnn_conf_matrix[23][23]
        #dnn_tp_25 = dnn_conf_matrix[24][24]
        #dnn_tp_26 = dnn_conf_matrix[25][25]
        #dnn_tp_27 = dnn_conf_matrix[26][26]
        #dnn_tp_28 = dnn_conf_matrix[27][27]
        #dnn_tp_29 = dnn_conf_matrix[28][28]

        dnn_fp_1 = dnn_conf_matrix[1][0] + dnn_conf_matrix[2][0] + dnn_conf_matrix[3][0] + dnn_conf_matrix[4][0] + dnn_conf_matrix[5][0] + dnn_conf_matrix[6][0] + dnn_conf_matrix[7][0] + dnn_conf_matrix[8][0] + dnn_conf_matrix[9][0] + dnn_conf_matrix[10][0] + dnn_conf_matrix[11][0] + dnn_conf_matrix[12][0] + dnn_conf_matrix[13][0] + dnn_conf_matrix[14][0] + dnn_conf_matrix[15][0] + dnn_conf_matrix[16][0] + dnn_conf_matrix[17][0] + dnn_conf_matrix[18][0] + dnn_conf_matrix[19][0]
        dnn_fp_2 = dnn_conf_matrix[0][1] + dnn_conf_matrix[2][1] + dnn_conf_matrix[3][1] + dnn_conf_matrix[4][1] + dnn_conf_matrix[5][1] + dnn_conf_matrix[6][1] + dnn_conf_matrix[7][1] + dnn_conf_matrix[8][1] + dnn_conf_matrix[9][1] + dnn_conf_matrix[10][1] + dnn_conf_matrix[11][1] + dnn_conf_matrix[12][1] + dnn_conf_matrix[13][1] + dnn_conf_matrix[14][1] + dnn_conf_matrix[15][1] + dnn_conf_matrix[16][1] + dnn_conf_matrix[17][1] + dnn_conf_matrix[18][1] + dnn_conf_matrix[19][1]
        dnn_fp_3 = dnn_conf_matrix[0][2] + dnn_conf_matrix[1][2] + dnn_conf_matrix[3][2] + dnn_conf_matrix[4][2] + dnn_conf_matrix[5][2] + dnn_conf_matrix[6][2] + dnn_conf_matrix[7][2] + dnn_conf_matrix[8][2] + dnn_conf_matrix[9][2] + dnn_conf_matrix[10][2] + dnn_conf_matrix[11][2] + dnn_conf_matrix[12][2] + dnn_conf_matrix[13][2] + dnn_conf_matrix[14][2] + dnn_conf_matrix[15][2] + dnn_conf_matrix[16][2] + dnn_conf_matrix[17][2] + dnn_conf_matrix[18][2] + dnn_conf_matrix[19][2]
        dnn_fp_4 = dnn_conf_matrix[0][3] + dnn_conf_matrix[1][3] + dnn_conf_matrix[2][3] + dnn_conf_matrix[4][3] + dnn_conf_matrix[5][3] + dnn_conf_matrix[6][3] + dnn_conf_matrix[7][3] + dnn_conf_matrix[8][3] + dnn_conf_matrix[9][3] + dnn_conf_matrix[10][3] + dnn_conf_matrix[11][3] + dnn_conf_matrix[12][3] + dnn_conf_matrix[13][3] + dnn_conf_matrix[14][3] + dnn_conf_matrix[15][3] + dnn_conf_matrix[16][3] + dnn_conf_matrix[17][3] + dnn_conf_matrix[18][3] + dnn_conf_matrix[19][3]
        dnn_fp_5 = dnn_conf_matrix[0][4] + dnn_conf_matrix[1][4] + dnn_conf_matrix[2][4] + dnn_conf_matrix[3][4] + dnn_conf_matrix[5][4] + dnn_conf_matrix[6][4] + dnn_conf_matrix[7][4] + dnn_conf_matrix[8][4] + dnn_conf_matrix[9][4] + dnn_conf_matrix[10][4] + dnn_conf_matrix[11][4] + dnn_conf_matrix[12][4] + dnn_conf_matrix[13][4] + dnn_conf_matrix[14][4] + dnn_conf_matrix[15][4] + dnn_conf_matrix[16][4] + dnn_conf_matrix[17][4] + dnn_conf_matrix[18][4] + dnn_conf_matrix[19][4]
        dnn_fp_6 = dnn_conf_matrix[0][5] + dnn_conf_matrix[1][5] + dnn_conf_matrix[2][5] + dnn_conf_matrix[3][5] + dnn_conf_matrix[4][5] + dnn_conf_matrix[6][5] + dnn_conf_matrix[7][5] + dnn_conf_matrix[8][5] + dnn_conf_matrix[9][5] + dnn_conf_matrix[10][5] + dnn_conf_matrix[11][5] + dnn_conf_matrix[12][5] + dnn_conf_matrix[13][5] + dnn_conf_matrix[14][5] + dnn_conf_matrix[15][5] + dnn_conf_matrix[16][5] + dnn_conf_matrix[17][5] + dnn_conf_matrix[18][5] + dnn_conf_matrix[19][5]
        dnn_fp_7 = dnn_conf_matrix[0][6] + dnn_conf_matrix[1][6] + dnn_conf_matrix[2][6] + dnn_conf_matrix[3][6] + dnn_conf_matrix[4][6] + dnn_conf_matrix[5][6] + dnn_conf_matrix[7][6] + dnn_conf_matrix[8][6] + dnn_conf_matrix[9][6] + dnn_conf_matrix[10][6] + dnn_conf_matrix[11][6] + dnn_conf_matrix[12][6] + dnn_conf_matrix[13][6] + dnn_conf_matrix[14][6] + dnn_conf_matrix[15][6] + dnn_conf_matrix[16][6] + dnn_conf_matrix[17][6] + dnn_conf_matrix[18][6] + dnn_conf_matrix[19][6]
        dnn_fp_8 = dnn_conf_matrix[0][7] + dnn_conf_matrix[1][7] + dnn_conf_matrix[2][7] + dnn_conf_matrix[3][7] + dnn_conf_matrix[4][7] + dnn_conf_matrix[5][7] + dnn_conf_matrix[6][7] + dnn_conf_matrix[8][7] + dnn_conf_matrix[9][7] + dnn_conf_matrix[10][7] + dnn_conf_matrix[11][7] + dnn_conf_matrix[12][7] + dnn_conf_matrix[13][7] + dnn_conf_matrix[14][7] + dnn_conf_matrix[15][7] + dnn_conf_matrix[16][7] + dnn_conf_matrix[17][7] + dnn_conf_matrix[18][7] + dnn_conf_matrix[19][7]
        dnn_fp_9 = dnn_conf_matrix[0][8] + dnn_conf_matrix[1][8] + dnn_conf_matrix[2][8] + dnn_conf_matrix[3][8] + dnn_conf_matrix[4][8] + dnn_conf_matrix[5][8] + dnn_conf_matrix[6][8] + dnn_conf_matrix[7][8] + dnn_conf_matrix[9][8] + dnn_conf_matrix[10][8] + dnn_conf_matrix[11][8] + dnn_conf_matrix[12][8] + dnn_conf_matrix[13][8] + dnn_conf_matrix[14][8] + dnn_conf_matrix[15][8] + dnn_conf_matrix[16][8] + dnn_conf_matrix[17][8] + dnn_conf_matrix[18][8] + dnn_conf_matrix[19][8]
        dnn_fp_10 = dnn_conf_matrix[0][9] + dnn_conf_matrix[1][9] + dnn_conf_matrix[2][9] + dnn_conf_matrix[3][9] + dnn_conf_matrix[4][9] + dnn_conf_matrix[5][9] + dnn_conf_matrix[6][9] + dnn_conf_matrix[7][9] + dnn_conf_matrix[8][9] + dnn_conf_matrix[10][9] + dnn_conf_matrix[11][9] + dnn_conf_matrix[12][9] + dnn_conf_matrix[13][9] + dnn_conf_matrix[14][9] + dnn_conf_matrix[15][9] + dnn_conf_matrix[16][9] + dnn_conf_matrix[17][9] + dnn_conf_matrix[18][9] + dnn_conf_matrix[19][9]
        dnn_fp_11 = dnn_conf_matrix[0][10] + dnn_conf_matrix[1][10] + dnn_conf_matrix[2][10] + dnn_conf_matrix[3][10] + dnn_conf_matrix[4][10] + dnn_conf_matrix[5][10] + dnn_conf_matrix[6][10] + dnn_conf_matrix[7][10] + dnn_conf_matrix[8][10] + dnn_conf_matrix[9][10] + dnn_conf_matrix[11][10] + dnn_conf_matrix[12][10] + dnn_conf_matrix[13][10] + dnn_conf_matrix[14][10] + dnn_conf_matrix[15][10] + dnn_conf_matrix[16][10] + dnn_conf_matrix[17][10] + dnn_conf_matrix[18][10] + dnn_conf_matrix[19][10]
        dnn_fp_12 = dnn_conf_matrix[0][11] + dnn_conf_matrix[1][11] + dnn_conf_matrix[2][11] + dnn_conf_matrix[3][11] + dnn_conf_matrix[4][11] + dnn_conf_matrix[5][11] + dnn_conf_matrix[6][11] + dnn_conf_matrix[7][11] + dnn_conf_matrix[8][11] + dnn_conf_matrix[9][11] + dnn_conf_matrix[10][11] + dnn_conf_matrix[12][11] + dnn_conf_matrix[13][11] + dnn_conf_matrix[14][11] + dnn_conf_matrix[15][11] + dnn_conf_matrix[16][11] + dnn_conf_matrix[17][11] + dnn_conf_matrix[18][11] + dnn_conf_matrix[19][11]
        dnn_fp_13 = dnn_conf_matrix[0][12] + dnn_conf_matrix[1][12] + dnn_conf_matrix[2][12] + dnn_conf_matrix[3][12] + dnn_conf_matrix[4][12] + dnn_conf_matrix[5][12] + dnn_conf_matrix[6][12] + dnn_conf_matrix[7][12] + dnn_conf_matrix[8][12] + dnn_conf_matrix[9][12] + dnn_conf_matrix[10][12] + dnn_conf_matrix[11][12] + dnn_conf_matrix[13][12] + dnn_conf_matrix[14][12] + dnn_conf_matrix[15][12] + dnn_conf_matrix[16][12] + dnn_conf_matrix[17][12] + dnn_conf_matrix[18][12] + dnn_conf_matrix[19][12]
        dnn_fp_14 = dnn_conf_matrix[0][13] + dnn_conf_matrix[1][13] + dnn_conf_matrix[2][13] + dnn_conf_matrix[3][13] + dnn_conf_matrix[4][13] + dnn_conf_matrix[5][13] + dnn_conf_matrix[6][13] + dnn_conf_matrix[7][13] + dnn_conf_matrix[8][13] + dnn_conf_matrix[9][13] + dnn_conf_matrix[10][13] + dnn_conf_matrix[11][13] + dnn_conf_matrix[12][13] + dnn_conf_matrix[14][13] + dnn_conf_matrix[15][13] + dnn_conf_matrix[16][13] + dnn_conf_matrix[17][13] + dnn_conf_matrix[18][13] + dnn_conf_matrix[19][13]
        dnn_fp_15 = dnn_conf_matrix[0][14] + dnn_conf_matrix[1][14] + dnn_conf_matrix[2][14] + dnn_conf_matrix[3][14] + dnn_conf_matrix[4][14] + dnn_conf_matrix[5][14] + dnn_conf_matrix[6][14] + dnn_conf_matrix[7][14] + dnn_conf_matrix[8][14] + dnn_conf_matrix[9][14] + dnn_conf_matrix[10][14] + dnn_conf_matrix[11][14] + dnn_conf_matrix[12][14] + dnn_conf_matrix[13][14] + dnn_conf_matrix[15][14] + dnn_conf_matrix[16][14] + dnn_conf_matrix[17][14] + dnn_conf_matrix[18][14] + dnn_conf_matrix[19][14]
        dnn_fp_16 = dnn_conf_matrix[0][15] + dnn_conf_matrix[1][15] + dnn_conf_matrix[2][15] + dnn_conf_matrix[3][15] + dnn_conf_matrix[4][15] + dnn_conf_matrix[5][15] + dnn_conf_matrix[6][15] + dnn_conf_matrix[7][15] + dnn_conf_matrix[8][15] + dnn_conf_matrix[9][15] + dnn_conf_matrix[10][15] + dnn_conf_matrix[11][15] + dnn_conf_matrix[12][15] + dnn_conf_matrix[13][15] + dnn_conf_matrix[14][15] + dnn_conf_matrix[16][15] + dnn_conf_matrix[17][15] + dnn_conf_matrix[18][15] + dnn_conf_matrix[19][15]
        dnn_fp_17 = dnn_conf_matrix[0][16] + dnn_conf_matrix[1][16] + dnn_conf_matrix[2][16] + dnn_conf_matrix[3][16] + dnn_conf_matrix[4][16] + dnn_conf_matrix[5][16] + dnn_conf_matrix[6][16] + dnn_conf_matrix[7][16] + dnn_conf_matrix[8][16] + dnn_conf_matrix[9][16] + dnn_conf_matrix[10][16] + dnn_conf_matrix[11][16] + dnn_conf_matrix[12][16] + dnn_conf_matrix[13][16] + dnn_conf_matrix[14][16] + dnn_conf_matrix[15][16] + dnn_conf_matrix[17][16] + dnn_conf_matrix[18][16] + dnn_conf_matrix[19][16]
        dnn_fp_18 = dnn_conf_matrix[0][17] + dnn_conf_matrix[1][17] + dnn_conf_matrix[2][17] + dnn_conf_matrix[3][17] + dnn_conf_matrix[4][17] + dnn_conf_matrix[5][17] + dnn_conf_matrix[6][17] + dnn_conf_matrix[7][17] + dnn_conf_matrix[8][17] + dnn_conf_matrix[9][17] + dnn_conf_matrix[10][17] + dnn_conf_matrix[11][17] + dnn_conf_matrix[12][17] + dnn_conf_matrix[13][17] + dnn_conf_matrix[14][17] + dnn_conf_matrix[15][17] + dnn_conf_matrix[16][17] + dnn_conf_matrix[18][17] + dnn_conf_matrix[19][17]
        dnn_fp_19 = dnn_conf_matrix[0][18] + dnn_conf_matrix[1][18] + dnn_conf_matrix[2][18] + dnn_conf_matrix[3][18] + dnn_conf_matrix[4][18] + dnn_conf_matrix[5][18] + dnn_conf_matrix[6][18] + dnn_conf_matrix[7][18] + dnn_conf_matrix[8][18] + dnn_conf_matrix[9][18] + dnn_conf_matrix[10][18] + dnn_conf_matrix[11][18] + dnn_conf_matrix[12][18] + dnn_conf_matrix[13][18] + dnn_conf_matrix[14][18] + dnn_conf_matrix[15][18] + dnn_conf_matrix[16][18] + dnn_conf_matrix[17][18] + dnn_conf_matrix[19][18]
        dnn_fp_20 = dnn_conf_matrix[0][19] + dnn_conf_matrix[1][19] + dnn_conf_matrix[2][19] + dnn_conf_matrix[3][19] + dnn_conf_matrix[4][19] + dnn_conf_matrix[5][19] + dnn_conf_matrix[6][19] + dnn_conf_matrix[7][19] + dnn_conf_matrix[8][19] + dnn_conf_matrix[9][19] + dnn_conf_matrix[10][19] + dnn_conf_matrix[11][19] + dnn_conf_matrix[12][19] + dnn_conf_matrix[13][19] + dnn_conf_matrix[14][19] + dnn_conf_matrix[15][19] + dnn_conf_matrix[16][19] + dnn_conf_matrix[17][19] + dnn_conf_matrix[18][19]
        #dnn_fp_21 = dnn_conf_matrix[0][20] + dnn_conf_matrix[1][20] + dnn_conf_matrix[2][20] + dnn_conf_matrix[3][20] + dnn_conf_matrix[4][20] + dnn_conf_matrix[5][20] + dnn_conf_matrix[6][20] + dnn_conf_matrix[7][20] + dnn_conf_matrix[8][20] + dnn_conf_matrix[9][20] + dnn_conf_matrix[10][20] + dnn_conf_matrix[11][20] + dnn_conf_matrix[12][20] + dnn_conf_matrix[13][20] + dnn_conf_matrix[14][20] + dnn_conf_matrix[15][20] + dnn_conf_matrix[16][20] + dnn_conf_matrix[17][20] + dnn_conf_matrix[18][20] + dnn_conf_matrix[19][20] + dnn_conf_matrix[21][20] + dnn_conf_matrix[22][20] + dnn_conf_matrix[23][20] + dnn_conf_matrix[24][20] + dnn_conf_matrix[25][20] + dnn_conf_matrix[26][20] + dnn_conf_matrix[27][20] + dnn_conf_matrix[28][20]
        #dnn_fp_22 = dnn_conf_matrix[0][21] + dnn_conf_matrix[1][21] + dnn_conf_matrix[2][21] + dnn_conf_matrix[3][21] + dnn_conf_matrix[4][21] + dnn_conf_matrix[5][21] + dnn_conf_matrix[6][21] + dnn_conf_matrix[7][21] + dnn_conf_matrix[8][21] + dnn_conf_matrix[9][21] + dnn_conf_matrix[10][21] + dnn_conf_matrix[11][21] + dnn_conf_matrix[12][21] + dnn_conf_matrix[13][21] + dnn_conf_matrix[14][21] + dnn_conf_matrix[15][21] + dnn_conf_matrix[16][21] + dnn_conf_matrix[17][21] + dnn_conf_matrix[18][21] + dnn_conf_matrix[19][21] + dnn_conf_matrix[20][21] + dnn_conf_matrix[22][21] + dnn_conf_matrix[23][21] + dnn_conf_matrix[24][21] + dnn_conf_matrix[25][21] + dnn_conf_matrix[26][21] + dnn_conf_matrix[27][21] + dnn_conf_matrix[28][21]
        #dnn_fp_23 = dnn_conf_matrix[0][22] + dnn_conf_matrix[1][22] + dnn_conf_matrix[2][22] + dnn_conf_matrix[3][22] + dnn_conf_matrix[4][22] + dnn_conf_matrix[5][22] + dnn_conf_matrix[6][22] + dnn_conf_matrix[7][22] + dnn_conf_matrix[8][22] + dnn_conf_matrix[9][22] + dnn_conf_matrix[10][22] + dnn_conf_matrix[11][22] + dnn_conf_matrix[12][22] + dnn_conf_matrix[13][22] + dnn_conf_matrix[14][22] + dnn_conf_matrix[15][22] + dnn_conf_matrix[16][22] + dnn_conf_matrix[17][22] + dnn_conf_matrix[18][22] + dnn_conf_matrix[19][22] + dnn_conf_matrix[20][22] + dnn_conf_matrix[21][22] + dnn_conf_matrix[23][22] + dnn_conf_matrix[24][22] + dnn_conf_matrix[25][22] + dnn_conf_matrix[26][22] + dnn_conf_matrix[27][22] + dnn_conf_matrix[28][22]
        #dnn_fp_24 = dnn_conf_matrix[0][23] + dnn_conf_matrix[1][23] + dnn_conf_matrix[2][23] + dnn_conf_matrix[3][23] + dnn_conf_matrix[4][23] + dnn_conf_matrix[5][23] + dnn_conf_matrix[6][23] + dnn_conf_matrix[7][23] + dnn_conf_matrix[8][23] + dnn_conf_matrix[9][23] + dnn_conf_matrix[10][23] + dnn_conf_matrix[11][23] + dnn_conf_matrix[12][23] + dnn_conf_matrix[13][23] + dnn_conf_matrix[14][23] + dnn_conf_matrix[15][23] + dnn_conf_matrix[16][23] + dnn_conf_matrix[17][23] + dnn_conf_matrix[18][23] + dnn_conf_matrix[19][23] + dnn_conf_matrix[20][23] + dnn_conf_matrix[21][23] + dnn_conf_matrix[22][23] + dnn_conf_matrix[24][23] + dnn_conf_matrix[25][23] + dnn_conf_matrix[26][23] + dnn_conf_matrix[27][23] + dnn_conf_matrix[28][23]
        #dnn_fp_25 = dnn_conf_matrix[0][24] + dnn_conf_matrix[1][24] + dnn_conf_matrix[2][24] + dnn_conf_matrix[3][24] + dnn_conf_matrix[4][24] + dnn_conf_matrix[5][24] + dnn_conf_matrix[6][24] + dnn_conf_matrix[7][24] + dnn_conf_matrix[8][24] + dnn_conf_matrix[9][24] + dnn_conf_matrix[10][24] + dnn_conf_matrix[11][24] + dnn_conf_matrix[12][24] + dnn_conf_matrix[13][24] + dnn_conf_matrix[14][24] + dnn_conf_matrix[15][24] + dnn_conf_matrix[16][24] + dnn_conf_matrix[17][24] + dnn_conf_matrix[18][24] + dnn_conf_matrix[19][24] + dnn_conf_matrix[20][24] + dnn_conf_matrix[21][24] + dnn_conf_matrix[22][24] + dnn_conf_matrix[23][24] + dnn_conf_matrix[25][24] + dnn_conf_matrix[26][24] + dnn_conf_matrix[27][24] + dnn_conf_matrix[28][24]
        #dnn_fp_26 = dnn_conf_matrix[0][25] + dnn_conf_matrix[1][25] + dnn_conf_matrix[2][25] + dnn_conf_matrix[3][25] + dnn_conf_matrix[4][25] + dnn_conf_matrix[5][25] + dnn_conf_matrix[6][25] + dnn_conf_matrix[7][25] + dnn_conf_matrix[8][25] + dnn_conf_matrix[9][25] + dnn_conf_matrix[10][25] + dnn_conf_matrix[11][25] + dnn_conf_matrix[12][25] + dnn_conf_matrix[13][25] + dnn_conf_matrix[14][25] + dnn_conf_matrix[15][25] + dnn_conf_matrix[16][25] + dnn_conf_matrix[17][25] + dnn_conf_matrix[18][25] + dnn_conf_matrix[19][25] + dnn_conf_matrix[20][25] + dnn_conf_matrix[21][25] + dnn_conf_matrix[22][25] + dnn_conf_matrix[23][25] + dnn_conf_matrix[24][25] + dnn_conf_matrix[26][25] + dnn_conf_matrix[27][25] + dnn_conf_matrix[28][25]
        #dnn_fp_27 = dnn_conf_matrix[0][26] + dnn_conf_matrix[1][26] + dnn_conf_matrix[2][26] + dnn_conf_matrix[3][26] + dnn_conf_matrix[4][26] + dnn_conf_matrix[5][26] + dnn_conf_matrix[6][26] + dnn_conf_matrix[7][26] + dnn_conf_matrix[8][26] + dnn_conf_matrix[9][26] + dnn_conf_matrix[10][26] + dnn_conf_matrix[11][26] + dnn_conf_matrix[12][26] + dnn_conf_matrix[13][26] + dnn_conf_matrix[14][26] + dnn_conf_matrix[15][26] + dnn_conf_matrix[16][26] + dnn_conf_matrix[17][26] + dnn_conf_matrix[18][26] + dnn_conf_matrix[19][26] + dnn_conf_matrix[20][26] + dnn_conf_matrix[21][26] + dnn_conf_matrix[22][26] + dnn_conf_matrix[23][26] + dnn_conf_matrix[24][26] + dnn_conf_matrix[25][26] + dnn_conf_matrix[27][26] + dnn_conf_matrix[28][26]
        #dnn_fp_28 = dnn_conf_matrix[0][27] + dnn_conf_matrix[1][27] + dnn_conf_matrix[2][27] + dnn_conf_matrix[3][27] + dnn_conf_matrix[4][27] + dnn_conf_matrix[5][27] + dnn_conf_matrix[6][27] + dnn_conf_matrix[7][27] + dnn_conf_matrix[8][27] + dnn_conf_matrix[9][27] + dnn_conf_matrix[10][27] + dnn_conf_matrix[11][27] + dnn_conf_matrix[12][27] + dnn_conf_matrix[13][27] + dnn_conf_matrix[14][27] + dnn_conf_matrix[15][27] + dnn_conf_matrix[16][27] + dnn_conf_matrix[17][27] + dnn_conf_matrix[18][27] + dnn_conf_matrix[19][27] + dnn_conf_matrix[20][27] + dnn_conf_matrix[21][27] + dnn_conf_matrix[22][27] + dnn_conf_matrix[23][27] + dnn_conf_matrix[24][27] + dnn_conf_matrix[25][27] + dnn_conf_matrix[26][27] + dnn_conf_matrix[28][27]
        #dnn_fp_29 = dnn_conf_matrix[0][28] + dnn_conf_matrix[1][28] + dnn_conf_matrix[2][28] + dnn_conf_matrix[3][28] + dnn_conf_matrix[4][28] + dnn_conf_matrix[5][28] + dnn_conf_matrix[6][28] + dnn_conf_matrix[7][28] + dnn_conf_matrix[8][28] + dnn_conf_matrix[9][28] + dnn_conf_matrix[10][28] + dnn_conf_matrix[11][28] + dnn_conf_matrix[12][28] + dnn_conf_matrix[13][28] + dnn_conf_matrix[14][28] + dnn_conf_matrix[15][28] + dnn_conf_matrix[16][28] + dnn_conf_matrix[17][28] + dnn_conf_matrix[18][28] + dnn_conf_matrix[19][28] + dnn_conf_matrix[20][28] + dnn_conf_matrix[21][28] + dnn_conf_matrix[22][28] + dnn_conf_matrix[23][28] + dnn_conf_matrix[24][28] + dnn_conf_matrix[25][28] + dnn_conf_matrix[26][28] + dnn_conf_matrix[27][28]

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
        if dnn_tp_6 + dnn_fp_6 == 0:
            dnn_precision_6 = 0
        else:
            dnn_precision_6 = dnn_tp_6 / (dnn_tp_6 + dnn_fp_6)
        if dnn_tp_7 + dnn_fp_7 == 0:
            dnn_precision_7 = 0
        else:
            dnn_precision_7 = dnn_tp_7 / (dnn_tp_7 + dnn_fp_7)
        if dnn_tp_8 + dnn_fp_8 == 0:
            dnn_precision_8 = 0
        else:
            dnn_precision_8 = dnn_tp_8 / (dnn_tp_8 + dnn_fp_8)
        if dnn_tp_9 + dnn_fp_9 == 0:
            dnn_precision_9 = 0
        else:
            dnn_precision_9 = dnn_tp_9 / (dnn_tp_9 + dnn_fp_9)
        if dnn_tp_10 + dnn_fp_10 == 0:
            dnn_precision_10 = 0
        else:
            dnn_precision_10 = dnn_tp_10 / (dnn_tp_10 + dnn_fp_10)
        if dnn_tp_11 + dnn_fp_11 == 0:
            dnn_precision_11 = 0
        else:
            dnn_precision_11 = dnn_tp_11 / (dnn_tp_11 + dnn_fp_11)
        if dnn_tp_12 + dnn_fp_12 == 0:
            dnn_precision_12 = 0
        else:
            dnn_precision_12 = dnn_tp_12 / (dnn_tp_12 + dnn_fp_12)
        if dnn_tp_13 + dnn_fp_13 == 0:
            dnn_precision_13 = 0
        else:
            dnn_precision_13 = dnn_tp_13 / (dnn_tp_13 + dnn_fp_13)
        if dnn_tp_14 + dnn_fp_14 == 0:
            dnn_precision_14 = 0
        else:
            dnn_precision_14 = dnn_tp_14 / (dnn_tp_14 + dnn_fp_14)
        if dnn_tp_15 + dnn_fp_15 == 0:
            dnn_precision_15 = 0
        else:
            dnn_precision_15 = dnn_tp_15 / (dnn_tp_15 + dnn_fp_15)
        if dnn_tp_16 + dnn_fp_16 == 0:
            dnn_precision_16 = 0
        else:
            dnn_precision_16 = dnn_tp_16 / (dnn_tp_16 + dnn_fp_16)
        if dnn_tp_17 + dnn_fp_17 == 0:
            dnn_precision_17 = 0
        else:
            dnn_precision_17 = dnn_tp_17 / (dnn_tp_17 + dnn_fp_17)
        if dnn_tp_18 + dnn_fp_18 == 0:
            dnn_precision_18 = 0
        else:
            dnn_precision_18 = dnn_tp_18 / (dnn_tp_18 + dnn_fp_18)
        if dnn_tp_19 + dnn_fp_19 == 0:
            dnn_precision_19 = 0
        else:
            dnn_precision_19 = dnn_tp_19 / (dnn_tp_19 + dnn_fp_19)
        if dnn_tp_20 + dnn_fp_20 == 0:
            dnn_precision_20 = 0
        else:
            dnn_precision_20 = dnn_tp_20 / (dnn_tp_20 + dnn_fp_20)
        '''
        if dnn_tp_21 + dnn_fp_21 == 0:
            dnn_precision_21 = 0
        else:
            dnn_precision_21 = dnn_tp_21 / (dnn_tp_21 + dnn_fp_21)
        if dnn_tp_22 + dnn_fp_22 == 0:
            dnn_precision_22 = 0
        else:
            dnn_precision_22 = dnn_tp_22 / (dnn_tp_22 + dnn_fp_22)
        if dnn_tp_23 + dnn_fp_23 == 0:
            dnn_precision_23 = 0
        else:
            dnn_precision_23 = dnn_tp_23 / (dnn_tp_23 + dnn_fp_23)
        if dnn_tp_24 + dnn_fp_24 == 0:
            dnn_precision_24 = 0
        else:
            dnn_precision_24 = dnn_tp_24 / (dnn_tp_24 + dnn_fp_24)
        if dnn_tp_25 + dnn_fp_25 == 0:
            dnn_precision_25 = 0
        else:
            dnn_precision_25 = dnn_tp_25 / (dnn_tp_25 + dnn_fp_25)
        if dnn_tp_26 + dnn_fp_26 == 0:
            dnn_precision_26 = 0
        else:
            dnn_precision_26 = dnn_tp_26 / (dnn_tp_26 + dnn_fp_26)
        if dnn_tp_27 + dnn_fp_27 == 0:
            dnn_precision_27 = 0
        else:
            dnn_precision_27 = dnn_tp_27 / (dnn_tp_27 + dnn_fp_27)
        if dnn_tp_28 + dnn_fp_28 == 0:
            dnn_precision_28 = 0
        else:
            dnn_precision_28 = dnn_tp_28 / (dnn_tp_28 + dnn_fp_28)
        if dnn_tp_29 + dnn_fp_29 == 0:
            dnn_precision_29 = 0
        else:
            dnn_precision_29 = dnn_tp_29 / (dnn_tp_29 + dnn_fp_29)
        '''
        dnn_precision_avg = (dnn_precision_1 + dnn_precision_2 + dnn_precision_3 + dnn_precision_4 + dnn_precision_5 + dnn_precision_6 + dnn_precision_7 + dnn_precision_8 + dnn_precision_9 + dnn_precision_10 + dnn_precision_11 + dnn_precision_12 + dnn_precision_13 + dnn_precision_14 + dnn_precision_15 + dnn_precision_16 + dnn_precision_17 + dnn_precision_18 + dnn_precision_19 + dnn_precision_20) / 20
        return dnn_precision_avg


    def get_recall_pen_1(dnn_conf_matrix):
        dnn_tp_1 = dnn_conf_matrix[0][0]
        dnn_tp_2 = dnn_conf_matrix[1][1]
        dnn_tp_3 = dnn_conf_matrix[2][2]
        dnn_tp_4 = dnn_conf_matrix[3][3]
        dnn_tp_5 = dnn_conf_matrix[4][4]
        dnn_tp_6 = dnn_conf_matrix[5][5]
        dnn_tp_7 = dnn_conf_matrix[6][6]
        dnn_tp_8 = dnn_conf_matrix[7][7]
        dnn_tp_9 = dnn_conf_matrix[8][8]
        dnn_tp_10 = dnn_conf_matrix[9][9]
        dnn_tp_11 = dnn_conf_matrix[10][10]
        dnn_tp_12 = dnn_conf_matrix[11][11]
        dnn_tp_13 = dnn_conf_matrix[12][12]
        dnn_tp_14 = dnn_conf_matrix[13][13]
        dnn_tp_15 = dnn_conf_matrix[14][14]
        dnn_tp_16 = dnn_conf_matrix[15][15]
        dnn_tp_17 = dnn_conf_matrix[16][16]
        dnn_tp_18 = dnn_conf_matrix[17][17]
        dnn_tp_19 = dnn_conf_matrix[18][18]
        dnn_tp_20 = dnn_conf_matrix[19][19]
        #dnn_tp_21 = dnn_conf_matrix[20][20]
        #dnn_tp_22 = dnn_conf_matrix[21][21]
        #dnn_tp_23 = dnn_conf_matrix[22][22]
        #dnn_tp_24 = dnn_conf_matrix[23][23]
        #dnn_tp_25 = dnn_conf_matrix[24][24]
        #dnn_tp_26 = dnn_conf_matrix[25][25]
        #dnn_tp_27 = dnn_conf_matrix[26][26]
        #dnn_tp_28 = dnn_conf_matrix[27][27]
        #dnn_tp_29 = dnn_conf_matrix[28][28]

        dnn_fn_1 = dnn_conf_matrix[0][1] + dnn_conf_matrix[0][2] + dnn_conf_matrix[0][3] + dnn_conf_matrix[0][4] + dnn_conf_matrix[0][5] + dnn_conf_matrix[0][6] + dnn_conf_matrix[0][7] + dnn_conf_matrix[0][8] + dnn_conf_matrix[0][9] + dnn_conf_matrix[0][10] + dnn_conf_matrix[0][11] + dnn_conf_matrix[0][12] + dnn_conf_matrix[0][13] + dnn_conf_matrix[0][14] + dnn_conf_matrix[0][15] + dnn_conf_matrix[0][16] + dnn_conf_matrix[0][17] + dnn_conf_matrix[0][18] + dnn_conf_matrix[0][19]
        dnn_fn_2 = dnn_conf_matrix[1][0] + dnn_conf_matrix[1][2] + dnn_conf_matrix[1][3] + dnn_conf_matrix[1][4] + dnn_conf_matrix[1][5] + dnn_conf_matrix[1][6] + dnn_conf_matrix[1][7] + dnn_conf_matrix[1][8] + dnn_conf_matrix[1][9] + dnn_conf_matrix[1][10] + dnn_conf_matrix[1][11] + dnn_conf_matrix[1][12] + dnn_conf_matrix[1][13] + dnn_conf_matrix[1][14] + dnn_conf_matrix[1][15] + dnn_conf_matrix[1][16] + dnn_conf_matrix[1][17] + dnn_conf_matrix[1][18] + dnn_conf_matrix[1][19]
        dnn_fn_3 = dnn_conf_matrix[2][0] + dnn_conf_matrix[2][1] + dnn_conf_matrix[2][3] + dnn_conf_matrix[2][4] + dnn_conf_matrix[2][5] + dnn_conf_matrix[2][6] + dnn_conf_matrix[2][7] + dnn_conf_matrix[2][8] + dnn_conf_matrix[2][9] + dnn_conf_matrix[2][10] + dnn_conf_matrix[2][11] + dnn_conf_matrix[2][12] + dnn_conf_matrix[2][13] + dnn_conf_matrix[2][14] + dnn_conf_matrix[2][15] + dnn_conf_matrix[2][16] + dnn_conf_matrix[2][17] + dnn_conf_matrix[2][18] + dnn_conf_matrix[2][19]
        dnn_fn_4 = dnn_conf_matrix[3][0] + dnn_conf_matrix[3][1] + dnn_conf_matrix[3][2] + dnn_conf_matrix[3][4] + dnn_conf_matrix[3][5] + dnn_conf_matrix[3][6] + dnn_conf_matrix[3][7] + dnn_conf_matrix[3][8] + dnn_conf_matrix[3][9] + dnn_conf_matrix[3][10] + dnn_conf_matrix[3][11] + dnn_conf_matrix[3][12] + dnn_conf_matrix[3][13] + dnn_conf_matrix[3][14] + dnn_conf_matrix[3][15] + dnn_conf_matrix[3][16] + dnn_conf_matrix[3][17] + dnn_conf_matrix[3][18] + dnn_conf_matrix[3][19]
        dnn_fn_5 = dnn_conf_matrix[4][0] + dnn_conf_matrix[4][1] + dnn_conf_matrix[4][2] + dnn_conf_matrix[4][3] + dnn_conf_matrix[4][5] + dnn_conf_matrix[4][6] + dnn_conf_matrix[4][7] + dnn_conf_matrix[4][8] + dnn_conf_matrix[4][9] + dnn_conf_matrix[4][10] + dnn_conf_matrix[4][11] + dnn_conf_matrix[4][12] + dnn_conf_matrix[4][13] + dnn_conf_matrix[4][14] + dnn_conf_matrix[4][15] + dnn_conf_matrix[4][16] + dnn_conf_matrix[4][17] + dnn_conf_matrix[4][18] + dnn_conf_matrix[4][19]
        dnn_fn_6 = dnn_conf_matrix[5][0] + dnn_conf_matrix[5][1] + dnn_conf_matrix[5][2] + dnn_conf_matrix[5][3] + dnn_conf_matrix[5][4] + dnn_conf_matrix[5][6] + dnn_conf_matrix[5][7] + dnn_conf_matrix[5][8] + dnn_conf_matrix[5][9] + dnn_conf_matrix[5][10] + dnn_conf_matrix[5][11] + dnn_conf_matrix[5][12] + dnn_conf_matrix[5][13] + dnn_conf_matrix[5][14] + dnn_conf_matrix[5][15] + dnn_conf_matrix[5][16] + dnn_conf_matrix[5][17] + dnn_conf_matrix[5][18] + dnn_conf_matrix[5][19]
        dnn_fn_7 = dnn_conf_matrix[6][0] + dnn_conf_matrix[6][1] + dnn_conf_matrix[6][2] + dnn_conf_matrix[6][3] + dnn_conf_matrix[6][4] + dnn_conf_matrix[6][5] + dnn_conf_matrix[6][7] + dnn_conf_matrix[6][8] + dnn_conf_matrix[6][9] + dnn_conf_matrix[6][10] + dnn_conf_matrix[6][11] + dnn_conf_matrix[6][12] + dnn_conf_matrix[6][13] + dnn_conf_matrix[6][14] + dnn_conf_matrix[6][15] + dnn_conf_matrix[6][16] + dnn_conf_matrix[6][17] + dnn_conf_matrix[6][18] + dnn_conf_matrix[6][19]
        dnn_fn_8 = dnn_conf_matrix[7][0] + dnn_conf_matrix[7][1] + dnn_conf_matrix[7][2] + dnn_conf_matrix[7][3] + dnn_conf_matrix[7][4] + dnn_conf_matrix[7][5] + dnn_conf_matrix[7][6] + dnn_conf_matrix[7][8] + dnn_conf_matrix[7][9] + dnn_conf_matrix[7][10] + dnn_conf_matrix[7][11] + dnn_conf_matrix[7][12] + dnn_conf_matrix[7][13] + dnn_conf_matrix[7][14] + dnn_conf_matrix[7][15] + dnn_conf_matrix[7][16] + dnn_conf_matrix[7][17] + dnn_conf_matrix[7][18] + dnn_conf_matrix[7][19]
        dnn_fn_9 = dnn_conf_matrix[8][0] + dnn_conf_matrix[8][1] + dnn_conf_matrix[8][2] + dnn_conf_matrix[8][3] + dnn_conf_matrix[8][4] + dnn_conf_matrix[8][5] + dnn_conf_matrix[8][6] + dnn_conf_matrix[8][7] + dnn_conf_matrix[8][9] + dnn_conf_matrix[8][10] + dnn_conf_matrix[8][11] + dnn_conf_matrix[8][12] + dnn_conf_matrix[8][13] + dnn_conf_matrix[8][14] + dnn_conf_matrix[8][15] + dnn_conf_matrix[8][16] + dnn_conf_matrix[8][17] + dnn_conf_matrix[8][18] + dnn_conf_matrix[8][19]
        dnn_fn_10 = dnn_conf_matrix[9][0] + dnn_conf_matrix[9][1] + dnn_conf_matrix[9][2] + dnn_conf_matrix[9][3] + dnn_conf_matrix[9][4] + dnn_conf_matrix[9][5] + dnn_conf_matrix[9][6] + dnn_conf_matrix[9][7] + dnn_conf_matrix[9][8] + dnn_conf_matrix[9][10] + dnn_conf_matrix[9][11] + dnn_conf_matrix[9][12] + dnn_conf_matrix[9][13] + dnn_conf_matrix[9][14] + dnn_conf_matrix[9][15] + dnn_conf_matrix[9][16] + dnn_conf_matrix[9][17] + dnn_conf_matrix[9][18] + dnn_conf_matrix[9][19]
        dnn_fn_11 = dnn_conf_matrix[10][0] + dnn_conf_matrix[10][1] + dnn_conf_matrix[10][2] + dnn_conf_matrix[10][3] + dnn_conf_matrix[10][4] + dnn_conf_matrix[10][5] + dnn_conf_matrix[10][6] + dnn_conf_matrix[10][7] + dnn_conf_matrix[10][8] + dnn_conf_matrix[10][9] + dnn_conf_matrix[10][11] + dnn_conf_matrix[10][12] + dnn_conf_matrix[10][13] + dnn_conf_matrix[10][14] + dnn_conf_matrix[10][15] + dnn_conf_matrix[10][16] + dnn_conf_matrix[10][17] + dnn_conf_matrix[10][18] + dnn_conf_matrix[10][19]
        dnn_fn_12 = dnn_conf_matrix[11][0] + dnn_conf_matrix[11][1] + dnn_conf_matrix[11][2] + dnn_conf_matrix[11][3] + dnn_conf_matrix[11][4] + dnn_conf_matrix[11][5] + dnn_conf_matrix[11][6] + dnn_conf_matrix[11][7] + dnn_conf_matrix[11][8] + dnn_conf_matrix[11][9] + dnn_conf_matrix[11][10] + dnn_conf_matrix[11][12] + dnn_conf_matrix[11][13] + dnn_conf_matrix[11][14] + dnn_conf_matrix[11][15] + dnn_conf_matrix[11][16] + dnn_conf_matrix[11][17] + dnn_conf_matrix[11][18] + dnn_conf_matrix[11][19]
        dnn_fn_13 = dnn_conf_matrix[12][0] + dnn_conf_matrix[12][1] + dnn_conf_matrix[12][2] + dnn_conf_matrix[12][3] + dnn_conf_matrix[12][4] + dnn_conf_matrix[12][5] + dnn_conf_matrix[12][6] + dnn_conf_matrix[12][7] + dnn_conf_matrix[12][8] + dnn_conf_matrix[12][9] + dnn_conf_matrix[12][10] + dnn_conf_matrix[12][11] + dnn_conf_matrix[12][13] + dnn_conf_matrix[12][14] + dnn_conf_matrix[12][15] + dnn_conf_matrix[12][16] + dnn_conf_matrix[12][17] + dnn_conf_matrix[12][18] + dnn_conf_matrix[12][19]
        dnn_fn_14 = dnn_conf_matrix[13][0] + dnn_conf_matrix[13][1] + dnn_conf_matrix[13][2] + dnn_conf_matrix[13][3] + dnn_conf_matrix[13][4] + dnn_conf_matrix[13][5] + dnn_conf_matrix[13][6] + dnn_conf_matrix[13][7] + dnn_conf_matrix[13][8] + dnn_conf_matrix[13][9] + dnn_conf_matrix[13][10] + dnn_conf_matrix[13][11] + dnn_conf_matrix[13][12] + dnn_conf_matrix[13][14] + dnn_conf_matrix[13][15] + dnn_conf_matrix[13][16] + dnn_conf_matrix[13][17] + dnn_conf_matrix[13][18] + dnn_conf_matrix[13][19]
        dnn_fn_15 = dnn_conf_matrix[14][0] + dnn_conf_matrix[14][1] + dnn_conf_matrix[14][2] + dnn_conf_matrix[14][3] + dnn_conf_matrix[14][4] + dnn_conf_matrix[14][5] + dnn_conf_matrix[14][6] + dnn_conf_matrix[14][7] + dnn_conf_matrix[14][8] + dnn_conf_matrix[14][9] + dnn_conf_matrix[14][10] + dnn_conf_matrix[14][11] + dnn_conf_matrix[14][12] + dnn_conf_matrix[14][13] + dnn_conf_matrix[14][15] + dnn_conf_matrix[14][16] + dnn_conf_matrix[14][17] + dnn_conf_matrix[14][18] + dnn_conf_matrix[14][19]
        dnn_fn_16 = dnn_conf_matrix[15][0] + dnn_conf_matrix[15][1] + dnn_conf_matrix[15][2] + dnn_conf_matrix[15][3] + dnn_conf_matrix[15][4] + dnn_conf_matrix[15][5] + dnn_conf_matrix[15][6] + dnn_conf_matrix[15][7] + dnn_conf_matrix[15][8] + dnn_conf_matrix[15][9] + dnn_conf_matrix[15][10] + dnn_conf_matrix[15][11] + dnn_conf_matrix[15][12] + dnn_conf_matrix[15][13] + dnn_conf_matrix[15][14] + dnn_conf_matrix[15][16] + dnn_conf_matrix[15][17] + dnn_conf_matrix[15][18] + dnn_conf_matrix[15][19]
        dnn_fn_17 = dnn_conf_matrix[16][0] + dnn_conf_matrix[16][1] + dnn_conf_matrix[16][2] + dnn_conf_matrix[16][3] + dnn_conf_matrix[16][4] + dnn_conf_matrix[16][5] + dnn_conf_matrix[16][6] + dnn_conf_matrix[16][7] + dnn_conf_matrix[16][8] + dnn_conf_matrix[16][9] + dnn_conf_matrix[16][10] + dnn_conf_matrix[16][11] + dnn_conf_matrix[16][12] + dnn_conf_matrix[16][13] + dnn_conf_matrix[16][14] + dnn_conf_matrix[16][15] + dnn_conf_matrix[16][17] + dnn_conf_matrix[16][18] + dnn_conf_matrix[16][19]
        dnn_fn_18 = dnn_conf_matrix[17][0] + dnn_conf_matrix[17][1] + dnn_conf_matrix[17][2] + dnn_conf_matrix[17][3] + dnn_conf_matrix[17][4] + dnn_conf_matrix[17][5] + dnn_conf_matrix[17][6] + dnn_conf_matrix[17][7] + dnn_conf_matrix[17][8] + dnn_conf_matrix[17][9] + dnn_conf_matrix[17][10] + dnn_conf_matrix[17][11] + dnn_conf_matrix[17][12] + dnn_conf_matrix[17][13] + dnn_conf_matrix[17][14] + dnn_conf_matrix[17][15] + dnn_conf_matrix[17][16] + dnn_conf_matrix[17][18] + dnn_conf_matrix[17][19]
        dnn_fn_19 = dnn_conf_matrix[18][0] + dnn_conf_matrix[18][1] + dnn_conf_matrix[18][2] + dnn_conf_matrix[18][3] + dnn_conf_matrix[18][4] + dnn_conf_matrix[18][5] + dnn_conf_matrix[18][6] + dnn_conf_matrix[18][7] + dnn_conf_matrix[18][8] + dnn_conf_matrix[18][9] + dnn_conf_matrix[18][10] + dnn_conf_matrix[18][11] + dnn_conf_matrix[18][12] + dnn_conf_matrix[18][13] + dnn_conf_matrix[18][14] + dnn_conf_matrix[18][15] + dnn_conf_matrix[18][16] + dnn_conf_matrix[18][17] + dnn_conf_matrix[18][19]
        dnn_fn_20 = dnn_conf_matrix[19][0] + dnn_conf_matrix[19][1] + dnn_conf_matrix[19][2] + dnn_conf_matrix[19][3] + dnn_conf_matrix[19][4] + dnn_conf_matrix[19][5] + dnn_conf_matrix[19][6] + dnn_conf_matrix[19][7] + dnn_conf_matrix[19][8] + dnn_conf_matrix[19][9] + dnn_conf_matrix[19][10] + dnn_conf_matrix[19][11] + dnn_conf_matrix[19][12] + dnn_conf_matrix[19][13] + dnn_conf_matrix[19][14] + dnn_conf_matrix[19][15] + dnn_conf_matrix[19][16] + dnn_conf_matrix[19][17] + dnn_conf_matrix[19][18]
        #dnn_fn_21 = dnn_conf_matrix[20][0] + dnn_conf_matrix[20][1] + dnn_conf_matrix[20][2] + dnn_conf_matrix[20][3] + dnn_conf_matrix[20][4] + dnn_conf_matrix[20][5] + dnn_conf_matrix[20][6] + dnn_conf_matrix[20][7] + dnn_conf_matrix[20][8] + dnn_conf_matrix[20][9] + dnn_conf_matrix[20][10] + dnn_conf_matrix[20][11] + dnn_conf_matrix[20][12] + dnn_conf_matrix[20][13] + dnn_conf_matrix[20][14] + dnn_conf_matrix[20][15] + dnn_conf_matrix[20][16] + dnn_conf_matrix[20][17] + dnn_conf_matrix[20][18] + dnn_conf_matrix[20][19] + dnn_conf_matrix[20][21] + dnn_conf_matrix[20][22] + dnn_conf_matrix[20][23] + dnn_conf_matrix[20][24] + dnn_conf_matrix[20][25] + dnn_conf_matrix[20][26] + dnn_conf_matrix[20][27] + dnn_conf_matrix[20][28]
        #dnn_fn_22 = dnn_conf_matrix[21][0] + dnn_conf_matrix[21][1] + dnn_conf_matrix[21][2] + dnn_conf_matrix[21][3] + dnn_conf_matrix[21][4] + dnn_conf_matrix[21][5] + dnn_conf_matrix[21][6] + dnn_conf_matrix[21][7] + dnn_conf_matrix[21][8] + dnn_conf_matrix[21][9] + dnn_conf_matrix[21][10] + dnn_conf_matrix[21][11] + dnn_conf_matrix[21][12] + dnn_conf_matrix[21][13] + dnn_conf_matrix[21][14] + dnn_conf_matrix[21][15] + dnn_conf_matrix[21][16] + dnn_conf_matrix[21][17] + dnn_conf_matrix[21][18] + dnn_conf_matrix[21][19] + dnn_conf_matrix[21][20] + dnn_conf_matrix[21][22] + dnn_conf_matrix[21][23] + dnn_conf_matrix[21][24] + dnn_conf_matrix[21][25] + dnn_conf_matrix[21][26] + dnn_conf_matrix[21][27] + dnn_conf_matrix[21][28]
        #dnn_fn_23 = dnn_conf_matrix[22][0] + dnn_conf_matrix[22][1] + dnn_conf_matrix[22][2] + dnn_conf_matrix[22][3] + dnn_conf_matrix[22][4] + dnn_conf_matrix[22][5] + dnn_conf_matrix[22][6] + dnn_conf_matrix[22][7] + dnn_conf_matrix[22][8] + dnn_conf_matrix[22][9] + dnn_conf_matrix[22][10] + dnn_conf_matrix[22][11] + dnn_conf_matrix[22][12] + dnn_conf_matrix[22][13] + dnn_conf_matrix[22][14] + dnn_conf_matrix[22][15] + dnn_conf_matrix[22][16] + dnn_conf_matrix[22][17] + dnn_conf_matrix[22][18] + dnn_conf_matrix[22][19] + dnn_conf_matrix[22][20] + dnn_conf_matrix[22][21] + dnn_conf_matrix[22][23] + dnn_conf_matrix[22][24] + dnn_conf_matrix[22][25] + dnn_conf_matrix[22][26] + dnn_conf_matrix[22][27] + dnn_conf_matrix[22][28]
        #dnn_fn_24 = dnn_conf_matrix[23][0] + dnn_conf_matrix[23][1] + dnn_conf_matrix[23][2] + dnn_conf_matrix[23][3] + dnn_conf_matrix[23][4] + dnn_conf_matrix[23][5] + dnn_conf_matrix[23][6] + dnn_conf_matrix[23][7] + dnn_conf_matrix[23][8] + dnn_conf_matrix[23][9] + dnn_conf_matrix[23][10] + dnn_conf_matrix[23][11] + dnn_conf_matrix[23][12] + dnn_conf_matrix[23][13] + dnn_conf_matrix[23][14] + dnn_conf_matrix[23][15] + dnn_conf_matrix[23][16] + dnn_conf_matrix[23][17] + dnn_conf_matrix[23][18] + dnn_conf_matrix[23][19] + dnn_conf_matrix[23][20] + dnn_conf_matrix[23][21] + dnn_conf_matrix[23][22] + dnn_conf_matrix[23][24] + dnn_conf_matrix[23][25] + dnn_conf_matrix[23][26] + dnn_conf_matrix[23][27] + dnn_conf_matrix[23][28]
        #dnn_fn_25 = dnn_conf_matrix[24][0] + dnn_conf_matrix[24][1] + dnn_conf_matrix[24][2] + dnn_conf_matrix[24][3] + dnn_conf_matrix[24][4] + dnn_conf_matrix[24][5] + dnn_conf_matrix[24][6] + dnn_conf_matrix[24][7] + dnn_conf_matrix[24][8] + dnn_conf_matrix[24][9] + dnn_conf_matrix[24][10] + dnn_conf_matrix[24][11] + dnn_conf_matrix[24][12] + dnn_conf_matrix[24][13] + dnn_conf_matrix[24][14] + dnn_conf_matrix[24][15] + dnn_conf_matrix[24][16] + dnn_conf_matrix[24][17] + dnn_conf_matrix[24][18] + dnn_conf_matrix[24][19] + dnn_conf_matrix[24][20] + dnn_conf_matrix[24][21] + dnn_conf_matrix[24][22] + dnn_conf_matrix[24][23] + dnn_conf_matrix[24][25] + dnn_conf_matrix[24][26] + dnn_conf_matrix[24][27] + dnn_conf_matrix[24][28]
        #dnn_fn_26 = dnn_conf_matrix[25][0] + dnn_conf_matrix[25][1] + dnn_conf_matrix[25][2] + dnn_conf_matrix[25][3] + dnn_conf_matrix[25][4] + dnn_conf_matrix[25][5] + dnn_conf_matrix[25][6] + dnn_conf_matrix[25][7] + dnn_conf_matrix[25][8] + dnn_conf_matrix[25][9] + dnn_conf_matrix[25][10] + dnn_conf_matrix[25][11] + dnn_conf_matrix[25][12] + dnn_conf_matrix[25][13] + dnn_conf_matrix[25][14] + dnn_conf_matrix[25][15] + dnn_conf_matrix[25][16] + dnn_conf_matrix[25][17] + dnn_conf_matrix[25][18] + dnn_conf_matrix[25][19] + dnn_conf_matrix[25][20] + dnn_conf_matrix[25][21] + dnn_conf_matrix[25][22] + dnn_conf_matrix[25][23] + dnn_conf_matrix[25][24] + dnn_conf_matrix[25][26] + dnn_conf_matrix[25][27] + dnn_conf_matrix[25][28]
        #dnn_fn_27 = dnn_conf_matrix[26][0] + dnn_conf_matrix[26][1] + dnn_conf_matrix[26][2] + dnn_conf_matrix[26][3] + dnn_conf_matrix[26][4] + dnn_conf_matrix[26][5] + dnn_conf_matrix[26][6] + dnn_conf_matrix[26][7] + dnn_conf_matrix[26][8] + dnn_conf_matrix[26][9] + dnn_conf_matrix[26][10] + dnn_conf_matrix[26][11] + dnn_conf_matrix[26][12] + dnn_conf_matrix[26][13] + dnn_conf_matrix[26][14] + dnn_conf_matrix[26][15] + dnn_conf_matrix[26][16] + dnn_conf_matrix[26][17] + dnn_conf_matrix[26][18] + dnn_conf_matrix[26][19] + dnn_conf_matrix[26][20] + dnn_conf_matrix[26][21] + dnn_conf_matrix[26][22] + dnn_conf_matrix[26][23] + dnn_conf_matrix[26][24] + dnn_conf_matrix[26][25] + dnn_conf_matrix[26][27] + dnn_conf_matrix[26][28]
        #dnn_fn_28 = dnn_conf_matrix[27][0] + dnn_conf_matrix[27][1] + dnn_conf_matrix[27][2] + dnn_conf_matrix[27][3] + dnn_conf_matrix[27][4] + dnn_conf_matrix[27][5] + dnn_conf_matrix[27][6] + dnn_conf_matrix[27][7] + dnn_conf_matrix[27][8] + dnn_conf_matrix[27][9] + dnn_conf_matrix[27][10] + dnn_conf_matrix[27][11] + dnn_conf_matrix[27][12] + dnn_conf_matrix[27][13] + dnn_conf_matrix[27][14] + dnn_conf_matrix[27][15] + dnn_conf_matrix[27][16] + dnn_conf_matrix[27][17] + dnn_conf_matrix[27][18] + dnn_conf_matrix[27][19] + dnn_conf_matrix[27][20] + dnn_conf_matrix[27][21] + dnn_conf_matrix[27][22] + dnn_conf_matrix[27][23] + dnn_conf_matrix[27][24] + dnn_conf_matrix[27][25] + dnn_conf_matrix[27][26] + dnn_conf_matrix[27][28]
        #dnn_fn_29 = dnn_conf_matrix[28][0] + dnn_conf_matrix[28][1] + dnn_conf_matrix[28][2] + dnn_conf_matrix[28][3] + dnn_conf_matrix[28][4] + dnn_conf_matrix[28][5] + dnn_conf_matrix[28][6] + dnn_conf_matrix[28][7] + dnn_conf_matrix[28][8] + dnn_conf_matrix[28][9] + dnn_conf_matrix[28][10] + dnn_conf_matrix[28][11] + dnn_conf_matrix[28][12] + dnn_conf_matrix[28][13] + dnn_conf_matrix[28][14] + dnn_conf_matrix[28][15] + dnn_conf_matrix[28][16] + dnn_conf_matrix[28][17] + dnn_conf_matrix[28][18] + dnn_conf_matrix[28][19] + dnn_conf_matrix[28][20] + dnn_conf_matrix[28][21] + dnn_conf_matrix[28][22] + dnn_conf_matrix[28][23] + dnn_conf_matrix[28][24] + dnn_conf_matrix[28][25] + dnn_conf_matrix[28][26] + dnn_conf_matrix[28][27]

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
            dnn_recall_5 = dnn_tp_5 / (dnn_tp_5 + dnn_fn_5)
        if dnn_tp_6 + dnn_fn_6 == 0:
            dnn_recall_6 = 0
        else:
            dnn_recall_6 = dnn_tp_6 / (dnn_tp_6 + dnn_fn_6)
        if dnn_tp_7 + dnn_fn_7 == 0:
            dnn_recall_7 = 0
        else:
            dnn_recall_7 = dnn_tp_7 / (dnn_tp_7 + dnn_fn_7)
        if dnn_tp_8 + dnn_fn_8 == 0:
            dnn_recall_8 = 0
        else:
            dnn_recall_8 = dnn_tp_8 / (dnn_tp_8 + dnn_fn_8)
        if dnn_tp_9 + dnn_fn_9 == 0:
            dnn_recall_9 = 0
        else:
            dnn_recall_9 = dnn_tp_9 / (dnn_tp_9 + dnn_fn_9)
        if dnn_tp_10 + dnn_fn_10 == 0:
            dnn_recall_10 = 0
        else:
            dnn_recall_10 = dnn_tp_10 / (dnn_tp_10 + dnn_fn_10)
        if dnn_tp_11 + dnn_fn_11 == 0:
            dnn_recall_11 = 0
        else:
            dnn_recall_11 = dnn_tp_11 / (dnn_tp_11 + dnn_fn_11)
        if dnn_tp_12 + dnn_fn_12 == 0:
            dnn_recall_12 = 0
        else:
            dnn_recall_12 = dnn_tp_12 / (dnn_tp_12 + dnn_fn_12)
        if dnn_tp_13 + dnn_fn_13 == 0:
            dnn_recall_13 = 0
        else:
            dnn_recall_13 = dnn_tp_13 / (dnn_tp_13 + dnn_fn_13)
        if dnn_tp_14 + dnn_fn_14 == 0:
            dnn_recall_14 = 0
        else:
            dnn_recall_14 = dnn_tp_14 / (dnn_tp_14 + dnn_fn_14)
        if dnn_tp_15 + dnn_fn_15 == 0:
            dnn_recall_15 = 0
        else:
            dnn_recall_15 = dnn_tp_15 / (dnn_tp_15 + dnn_fn_15)
        if dnn_tp_16 + dnn_fn_16 == 0:
            dnn_recall_16 = 0
        else:
            dnn_recall_16 = dnn_tp_16 / (dnn_tp_16 + dnn_fn_16)
        if dnn_tp_17 + dnn_fn_17 == 0:
            dnn_recall_17 = 0
        else:
            dnn_recall_17 = dnn_tp_17 / (dnn_tp_17 + dnn_fn_17)
        if dnn_tp_18 + dnn_fn_18 == 0:
            dnn_recall_18 = 0
        else:
            dnn_recall_18 = dnn_tp_18 / (dnn_tp_18 + dnn_fn_18)
        if dnn_tp_19 + dnn_fn_19 == 0:
            dnn_recall_19 = 0
        else:
            dnn_recall_19 = dnn_tp_19 / (dnn_tp_19 + dnn_fn_19)
        if dnn_tp_20 + dnn_fn_20 == 0:
            dnn_recall_20 = 0
        else:
            dnn_recall_20 = dnn_tp_20 / (dnn_tp_20 + dnn_fn_20)
        '''
        if dnn_tp_21 + dnn_fn_21 == 0:
            dnn_recall_21 = 0
        else:
            dnn_recall_21 = dnn_tp_21 / (dnn_tp_21 + dnn_fn_21)
        if dnn_tp_22 + dnn_fn_22 == 0:
            dnn_recall_22 = 0
        else:
            dnn_recall_22 = dnn_tp_22 / (dnn_tp_22 + dnn_fn_22)
        if dnn_tp_23 + dnn_fn_23 == 0:
            dnn_recall_23 = 0
        else:
            dnn_recall_23 = dnn_tp_23 / (dnn_tp_23 + dnn_fn_23)
        if dnn_tp_24 + dnn_fn_24 == 0:
            dnn_recall_24 = 0
        else:
            dnn_recall_24 = dnn_tp_24 / (dnn_tp_24 + dnn_fn_24)
        if dnn_tp_25 + dnn_fn_25 == 0:
            dnn_recall_25 = 0
        else:
            dnn_recall_25 = dnn_tp_25 / (dnn_tp_25 + dnn_fn_25)
        if dnn_tp_26 + dnn_fn_26 == 0:
            dnn_recall_26 = 0
        else:
            dnn_recall_26 = dnn_tp_26 / (dnn_tp_26 + dnn_fn_26)
        if dnn_tp_27 + dnn_fn_27 == 0:
            dnn_recall_27 = 0
        else:
            dnn_recall_27 = dnn_tp_27 / (dnn_tp_27 + dnn_fn_27)
        if dnn_tp_28 + dnn_fn_28 == 0:
            dnn_recall_28 = 0
        else:
            dnn_recall_28 = dnn_tp_28 / (dnn_tp_28 + dnn_fn_28)
        if dnn_tp_29 + dnn_fn_29 == 0:
            dnn_recall_29 = 0
        else:
            dnn_recall_29 = dnn_tp_29 / (dnn_tp_29 + dnn_fn_29)
        '''
        dnn_recall_avg_pen_1 = (
                                 dnn_recall_1 + dnn_recall_2 + dnn_recall_3 + dnn_recall_4 + dnn_recall_5 + dnn_recall_6 + dnn_recall_7 + dnn_recall_8 + dnn_recall_9 + dnn_recall_10 + dnn_recall_11 + dnn_recall_12 + dnn_recall_13 + dnn_recall_14 + dnn_recall_15 + dnn_recall_16 + dnn_recall_17 + dnn_recall_18 + dnn_recall_19 + dnn_recall_20) / (20+1-1)
        return dnn_recall_avg_pen_1

    def get_recall_pen_5(dnn_conf_matrix):
        dnn_tp_1 = dnn_conf_matrix[0][0]
        dnn_tp_2 = dnn_conf_matrix[1][1]
        dnn_tp_3 = dnn_conf_matrix[2][2]
        dnn_tp_4 = dnn_conf_matrix[3][3]
        dnn_tp_5 = dnn_conf_matrix[4][4]
        dnn_tp_6 = dnn_conf_matrix[5][5]
        dnn_tp_7 = dnn_conf_matrix[6][6]
        dnn_tp_8 = dnn_conf_matrix[7][7]
        dnn_tp_9 = dnn_conf_matrix[8][8]
        dnn_tp_10 = dnn_conf_matrix[9][9]
        dnn_tp_11 = dnn_conf_matrix[10][10]
        dnn_tp_12 = dnn_conf_matrix[11][11]
        dnn_tp_13 = dnn_conf_matrix[12][12]
        dnn_tp_14 = dnn_conf_matrix[13][13]
        dnn_tp_15 = dnn_conf_matrix[14][14]
        dnn_tp_16 = dnn_conf_matrix[15][15]
        dnn_tp_17 = dnn_conf_matrix[16][16]
        dnn_tp_18 = dnn_conf_matrix[17][17]
        dnn_tp_19 = dnn_conf_matrix[18][18]
        dnn_tp_20 = dnn_conf_matrix[19][19]
        #dnn_tp_21 = dnn_conf_matrix[20][20]
        #dnn_tp_22 = dnn_conf_matrix[21][21]
        #dnn_tp_23 = dnn_conf_matrix[22][22]
        #dnn_tp_24 = dnn_conf_matrix[23][23]
        #dnn_tp_25 = dnn_conf_matrix[24][24]
        #dnn_tp_26 = dnn_conf_matrix[25][25]
        #dnn_tp_27 = dnn_conf_matrix[26][26]
        #dnn_tp_28 = dnn_conf_matrix[27][27]
        #dnn_tp_29 = dnn_conf_matrix[28][28]

        dnn_fn_1 = dnn_conf_matrix[0][1] + dnn_conf_matrix[0][2] + dnn_conf_matrix[0][3] + dnn_conf_matrix[0][4] + dnn_conf_matrix[0][5] + dnn_conf_matrix[0][6] + dnn_conf_matrix[0][7] + dnn_conf_matrix[0][8] + dnn_conf_matrix[0][9] + dnn_conf_matrix[0][10] + dnn_conf_matrix[0][11] + dnn_conf_matrix[0][12] + dnn_conf_matrix[0][13] + dnn_conf_matrix[0][14] + dnn_conf_matrix[0][15] + dnn_conf_matrix[0][16] + dnn_conf_matrix[0][17] + dnn_conf_matrix[0][18] + dnn_conf_matrix[0][19]
        dnn_fn_2 = dnn_conf_matrix[1][0] + dnn_conf_matrix[1][2] + dnn_conf_matrix[1][3] + dnn_conf_matrix[1][4] + dnn_conf_matrix[1][5] + dnn_conf_matrix[1][6] + dnn_conf_matrix[1][7] + dnn_conf_matrix[1][8] + dnn_conf_matrix[1][9] + dnn_conf_matrix[1][10] + dnn_conf_matrix[1][11] + dnn_conf_matrix[1][12] + dnn_conf_matrix[1][13] + dnn_conf_matrix[1][14] + dnn_conf_matrix[1][15] + dnn_conf_matrix[1][16] + dnn_conf_matrix[1][17] + dnn_conf_matrix[1][18] + dnn_conf_matrix[1][19]
        dnn_fn_3 = dnn_conf_matrix[2][0] + dnn_conf_matrix[2][1] + dnn_conf_matrix[2][3] + dnn_conf_matrix[2][4] + dnn_conf_matrix[2][5] + dnn_conf_matrix[2][6] + dnn_conf_matrix[2][7] + dnn_conf_matrix[2][8] + dnn_conf_matrix[2][9] + dnn_conf_matrix[2][10] + dnn_conf_matrix[2][11] + dnn_conf_matrix[2][12] + dnn_conf_matrix[2][13] + dnn_conf_matrix[2][14] + dnn_conf_matrix[2][15] + dnn_conf_matrix[2][16] + dnn_conf_matrix[2][17] + dnn_conf_matrix[2][18] + dnn_conf_matrix[2][19]
        dnn_fn_4 = dnn_conf_matrix[3][0] + dnn_conf_matrix[3][1] + dnn_conf_matrix[3][2] + dnn_conf_matrix[3][4] + dnn_conf_matrix[3][5] + dnn_conf_matrix[3][6] + dnn_conf_matrix[3][7] + dnn_conf_matrix[3][8] + dnn_conf_matrix[3][9] + dnn_conf_matrix[3][10] + dnn_conf_matrix[3][11] + dnn_conf_matrix[3][12] + dnn_conf_matrix[3][13] + dnn_conf_matrix[3][14] + dnn_conf_matrix[3][15] + dnn_conf_matrix[3][16] + dnn_conf_matrix[3][17] + dnn_conf_matrix[3][18] + dnn_conf_matrix[3][19]
        dnn_fn_5 = dnn_conf_matrix[4][0] + dnn_conf_matrix[4][1] + dnn_conf_matrix[4][2] + dnn_conf_matrix[4][3] + dnn_conf_matrix[4][5] + dnn_conf_matrix[4][6] + dnn_conf_matrix[4][7] + dnn_conf_matrix[4][8] + dnn_conf_matrix[4][9] + dnn_conf_matrix[4][10] + dnn_conf_matrix[4][11] + dnn_conf_matrix[4][12] + dnn_conf_matrix[4][13] + dnn_conf_matrix[4][14] + dnn_conf_matrix[4][15] + dnn_conf_matrix[4][16] + dnn_conf_matrix[4][17] + dnn_conf_matrix[4][18] + dnn_conf_matrix[4][19]
        dnn_fn_6 = dnn_conf_matrix[5][0] + dnn_conf_matrix[5][1] + dnn_conf_matrix[5][2] + dnn_conf_matrix[5][3] + dnn_conf_matrix[5][4] + dnn_conf_matrix[5][6] + dnn_conf_matrix[5][7] + dnn_conf_matrix[5][8] + dnn_conf_matrix[5][9] + dnn_conf_matrix[5][10] + dnn_conf_matrix[5][11] + dnn_conf_matrix[5][12] + dnn_conf_matrix[5][13] + dnn_conf_matrix[5][14] + dnn_conf_matrix[5][15] + dnn_conf_matrix[5][16] + dnn_conf_matrix[5][17] + dnn_conf_matrix[5][18] + dnn_conf_matrix[5][19]
        dnn_fn_7 = dnn_conf_matrix[6][0] + dnn_conf_matrix[6][1] + dnn_conf_matrix[6][2] + dnn_conf_matrix[6][3] + dnn_conf_matrix[6][4] + dnn_conf_matrix[6][5] + dnn_conf_matrix[6][7] + dnn_conf_matrix[6][8] + dnn_conf_matrix[6][9] + dnn_conf_matrix[6][10] + dnn_conf_matrix[6][11] + dnn_conf_matrix[6][12] + dnn_conf_matrix[6][13] + dnn_conf_matrix[6][14] + dnn_conf_matrix[6][15] + dnn_conf_matrix[6][16] + dnn_conf_matrix[6][17] + dnn_conf_matrix[6][18] + dnn_conf_matrix[6][19]
        dnn_fn_8 = dnn_conf_matrix[7][0] + dnn_conf_matrix[7][1] + dnn_conf_matrix[7][2] + dnn_conf_matrix[7][3] + dnn_conf_matrix[7][4] + dnn_conf_matrix[7][5] + dnn_conf_matrix[7][6] + dnn_conf_matrix[7][8] + dnn_conf_matrix[7][9] + dnn_conf_matrix[7][10] + dnn_conf_matrix[7][11] + dnn_conf_matrix[7][12] + dnn_conf_matrix[7][13] + dnn_conf_matrix[7][14] + dnn_conf_matrix[7][15] + dnn_conf_matrix[7][16] + dnn_conf_matrix[7][17] + dnn_conf_matrix[7][18] + dnn_conf_matrix[7][19]
        dnn_fn_9 = dnn_conf_matrix[8][0] + dnn_conf_matrix[8][1] + dnn_conf_matrix[8][2] + dnn_conf_matrix[8][3] + dnn_conf_matrix[8][4] + dnn_conf_matrix[8][5] + dnn_conf_matrix[8][6] + dnn_conf_matrix[8][7] + dnn_conf_matrix[8][9] + dnn_conf_matrix[8][10] + dnn_conf_matrix[8][11] + dnn_conf_matrix[8][12] + dnn_conf_matrix[8][13] + dnn_conf_matrix[8][14] + dnn_conf_matrix[8][15] + dnn_conf_matrix[8][16] + dnn_conf_matrix[8][17] + dnn_conf_matrix[8][18] + dnn_conf_matrix[8][19]
        dnn_fn_10 = dnn_conf_matrix[9][0] + dnn_conf_matrix[9][1] + dnn_conf_matrix[9][2] + dnn_conf_matrix[9][3] + dnn_conf_matrix[9][4] + dnn_conf_matrix[9][5] + dnn_conf_matrix[9][6] + dnn_conf_matrix[9][7] + dnn_conf_matrix[9][8] + dnn_conf_matrix[9][10] + dnn_conf_matrix[9][11] + dnn_conf_matrix[9][12] + dnn_conf_matrix[9][13] + dnn_conf_matrix[9][14] + dnn_conf_matrix[9][15] + dnn_conf_matrix[9][16] + dnn_conf_matrix[9][17] + dnn_conf_matrix[9][18] + dnn_conf_matrix[9][19]
        dnn_fn_11 = dnn_conf_matrix[10][0] + dnn_conf_matrix[10][1] + dnn_conf_matrix[10][2] + dnn_conf_matrix[10][3] + dnn_conf_matrix[10][4] + dnn_conf_matrix[10][5] + dnn_conf_matrix[10][6] + dnn_conf_matrix[10][7] + dnn_conf_matrix[10][8] + dnn_conf_matrix[10][9] + dnn_conf_matrix[10][11] + dnn_conf_matrix[10][12] + dnn_conf_matrix[10][13] + dnn_conf_matrix[10][14] + dnn_conf_matrix[10][15] + dnn_conf_matrix[10][16] + dnn_conf_matrix[10][17] + dnn_conf_matrix[10][18] + dnn_conf_matrix[10][19]
        dnn_fn_12 = dnn_conf_matrix[11][0] + dnn_conf_matrix[11][1] + dnn_conf_matrix[11][2] + dnn_conf_matrix[11][3] + dnn_conf_matrix[11][4] + dnn_conf_matrix[11][5] + dnn_conf_matrix[11][6] + dnn_conf_matrix[11][7] + dnn_conf_matrix[11][8] + dnn_conf_matrix[11][9] + dnn_conf_matrix[11][10] + dnn_conf_matrix[11][12] + dnn_conf_matrix[11][13] + dnn_conf_matrix[11][14] + dnn_conf_matrix[11][15] + dnn_conf_matrix[11][16] + dnn_conf_matrix[11][17] + dnn_conf_matrix[11][18] + dnn_conf_matrix[11][19]
        dnn_fn_13 = dnn_conf_matrix[12][0] + dnn_conf_matrix[12][1] + dnn_conf_matrix[12][2] + dnn_conf_matrix[12][3] + dnn_conf_matrix[12][4] + dnn_conf_matrix[12][5] + dnn_conf_matrix[12][6] + dnn_conf_matrix[12][7] + dnn_conf_matrix[12][8] + dnn_conf_matrix[12][9] + dnn_conf_matrix[12][10] + dnn_conf_matrix[12][11] + dnn_conf_matrix[12][13] + dnn_conf_matrix[12][14] + dnn_conf_matrix[12][15] + dnn_conf_matrix[12][16] + dnn_conf_matrix[12][17] + dnn_conf_matrix[12][18] + dnn_conf_matrix[12][19]
        dnn_fn_14 = dnn_conf_matrix[13][0] + dnn_conf_matrix[13][1] + dnn_conf_matrix[13][2] + dnn_conf_matrix[13][3] + dnn_conf_matrix[13][4] + dnn_conf_matrix[13][5] + dnn_conf_matrix[13][6] + dnn_conf_matrix[13][7] + dnn_conf_matrix[13][8] + dnn_conf_matrix[13][9] + dnn_conf_matrix[13][10] + dnn_conf_matrix[13][11] + dnn_conf_matrix[13][12] + dnn_conf_matrix[13][14] + dnn_conf_matrix[13][15] + dnn_conf_matrix[13][16] + dnn_conf_matrix[13][17] + dnn_conf_matrix[13][18] + dnn_conf_matrix[13][19]
        dnn_fn_15 = dnn_conf_matrix[14][0] + dnn_conf_matrix[14][1] + dnn_conf_matrix[14][2] + dnn_conf_matrix[14][3] + dnn_conf_matrix[14][4] + dnn_conf_matrix[14][5] + dnn_conf_matrix[14][6] + dnn_conf_matrix[14][7] + dnn_conf_matrix[14][8] + dnn_conf_matrix[14][9] + dnn_conf_matrix[14][10] + dnn_conf_matrix[14][11] + dnn_conf_matrix[14][12] + dnn_conf_matrix[14][13] + dnn_conf_matrix[14][15] + dnn_conf_matrix[14][16] + dnn_conf_matrix[14][17] + dnn_conf_matrix[14][18] + dnn_conf_matrix[14][19]
        dnn_fn_16 = dnn_conf_matrix[15][0] + dnn_conf_matrix[15][1] + dnn_conf_matrix[15][2] + dnn_conf_matrix[15][3] + dnn_conf_matrix[15][4] + dnn_conf_matrix[15][5] + dnn_conf_matrix[15][6] + dnn_conf_matrix[15][7] + dnn_conf_matrix[15][8] + dnn_conf_matrix[15][9] + dnn_conf_matrix[15][10] + dnn_conf_matrix[15][11] + dnn_conf_matrix[15][12] + dnn_conf_matrix[15][13] + dnn_conf_matrix[15][14] + dnn_conf_matrix[15][16] + dnn_conf_matrix[15][17] + dnn_conf_matrix[15][18] + dnn_conf_matrix[15][19]
        dnn_fn_17 = dnn_conf_matrix[16][0] + dnn_conf_matrix[16][1] + dnn_conf_matrix[16][2] + dnn_conf_matrix[16][3] + dnn_conf_matrix[16][4] + dnn_conf_matrix[16][5] + dnn_conf_matrix[16][6] + dnn_conf_matrix[16][7] + dnn_conf_matrix[16][8] + dnn_conf_matrix[16][9] + dnn_conf_matrix[16][10] + dnn_conf_matrix[16][11] + dnn_conf_matrix[16][12] + dnn_conf_matrix[16][13] + dnn_conf_matrix[16][14] + dnn_conf_matrix[16][15] + dnn_conf_matrix[16][17] + dnn_conf_matrix[16][18] + dnn_conf_matrix[16][19]
        dnn_fn_18 = dnn_conf_matrix[17][0] + dnn_conf_matrix[17][1] + dnn_conf_matrix[17][2] + dnn_conf_matrix[17][3] + dnn_conf_matrix[17][4] + dnn_conf_matrix[17][5] + dnn_conf_matrix[17][6] + dnn_conf_matrix[17][7] + dnn_conf_matrix[17][8] + dnn_conf_matrix[17][9] + dnn_conf_matrix[17][10] + dnn_conf_matrix[17][11] + dnn_conf_matrix[17][12] + dnn_conf_matrix[17][13] + dnn_conf_matrix[17][14] + dnn_conf_matrix[17][15] + dnn_conf_matrix[17][16] + dnn_conf_matrix[17][18] + dnn_conf_matrix[17][19]
        dnn_fn_19 = dnn_conf_matrix[18][0] + dnn_conf_matrix[18][1] + dnn_conf_matrix[18][2] + dnn_conf_matrix[18][3] + dnn_conf_matrix[18][4] + dnn_conf_matrix[18][5] + dnn_conf_matrix[18][6] + dnn_conf_matrix[18][7] + dnn_conf_matrix[18][8] + dnn_conf_matrix[18][9] + dnn_conf_matrix[18][10] + dnn_conf_matrix[18][11] + dnn_conf_matrix[18][12] + dnn_conf_matrix[18][13] + dnn_conf_matrix[18][14] + dnn_conf_matrix[18][15] + dnn_conf_matrix[18][16] + dnn_conf_matrix[18][17] + dnn_conf_matrix[18][19]
        dnn_fn_20 = dnn_conf_matrix[19][0] + dnn_conf_matrix[19][1] + dnn_conf_matrix[19][2] + dnn_conf_matrix[19][3] + dnn_conf_matrix[19][4] + dnn_conf_matrix[19][5] + dnn_conf_matrix[19][6] + dnn_conf_matrix[19][7] + dnn_conf_matrix[19][8] + dnn_conf_matrix[19][9] + dnn_conf_matrix[19][10] + dnn_conf_matrix[19][11] + dnn_conf_matrix[19][12] + dnn_conf_matrix[19][13] + dnn_conf_matrix[19][14] + dnn_conf_matrix[19][15] + dnn_conf_matrix[19][16] + dnn_conf_matrix[19][17] + dnn_conf_matrix[19][18]
        #dnn_fn_21 = dnn_conf_matrix[20][0] + dnn_conf_matrix[20][1] + dnn_conf_matrix[20][2] + dnn_conf_matrix[20][3] + dnn_conf_matrix[20][4] + dnn_conf_matrix[20][5] + dnn_conf_matrix[20][6] + dnn_conf_matrix[20][7] + dnn_conf_matrix[20][8] + dnn_conf_matrix[20][9] + dnn_conf_matrix[20][10] + dnn_conf_matrix[20][11] + dnn_conf_matrix[20][12] + dnn_conf_matrix[20][13] + dnn_conf_matrix[20][14] + dnn_conf_matrix[20][15] + dnn_conf_matrix[20][16] + dnn_conf_matrix[20][17] + dnn_conf_matrix[20][18] + dnn_conf_matrix[20][19] + dnn_conf_matrix[20][21] + dnn_conf_matrix[20][22] + dnn_conf_matrix[20][23] + dnn_conf_matrix[20][24] + dnn_conf_matrix[20][25] + dnn_conf_matrix[20][26] + dnn_conf_matrix[20][27] + dnn_conf_matrix[20][28]
        #dnn_fn_22 = dnn_conf_matrix[21][0] + dnn_conf_matrix[21][1] + dnn_conf_matrix[21][2] + dnn_conf_matrix[21][3] + dnn_conf_matrix[21][4] + dnn_conf_matrix[21][5] + dnn_conf_matrix[21][6] + dnn_conf_matrix[21][7] + dnn_conf_matrix[21][8] + dnn_conf_matrix[21][9] + dnn_conf_matrix[21][10] + dnn_conf_matrix[21][11] + dnn_conf_matrix[21][12] + dnn_conf_matrix[21][13] + dnn_conf_matrix[21][14] + dnn_conf_matrix[21][15] + dnn_conf_matrix[21][16] + dnn_conf_matrix[21][17] + dnn_conf_matrix[21][18] + dnn_conf_matrix[21][19] + dnn_conf_matrix[21][20] + dnn_conf_matrix[21][22] + dnn_conf_matrix[21][23] + dnn_conf_matrix[21][24] + dnn_conf_matrix[21][25] + dnn_conf_matrix[21][26] + dnn_conf_matrix[21][27] + dnn_conf_matrix[21][28]
        #dnn_fn_23 = dnn_conf_matrix[22][0] + dnn_conf_matrix[22][1] + dnn_conf_matrix[22][2] + dnn_conf_matrix[22][3] + dnn_conf_matrix[22][4] + dnn_conf_matrix[22][5] + dnn_conf_matrix[22][6] + dnn_conf_matrix[22][7] + dnn_conf_matrix[22][8] + dnn_conf_matrix[22][9] + dnn_conf_matrix[22][10] + dnn_conf_matrix[22][11] + dnn_conf_matrix[22][12] + dnn_conf_matrix[22][13] + dnn_conf_matrix[22][14] + dnn_conf_matrix[22][15] + dnn_conf_matrix[22][16] + dnn_conf_matrix[22][17] + dnn_conf_matrix[22][18] + dnn_conf_matrix[22][19] + dnn_conf_matrix[22][20] + dnn_conf_matrix[22][21] + dnn_conf_matrix[22][23] + dnn_conf_matrix[22][24] + dnn_conf_matrix[22][25] + dnn_conf_matrix[22][26] + dnn_conf_matrix[22][27] + dnn_conf_matrix[22][28]
        #dnn_fn_24 = dnn_conf_matrix[23][0] + dnn_conf_matrix[23][1] + dnn_conf_matrix[23][2] + dnn_conf_matrix[23][3] + dnn_conf_matrix[23][4] + dnn_conf_matrix[23][5] + dnn_conf_matrix[23][6] + dnn_conf_matrix[23][7] + dnn_conf_matrix[23][8] + dnn_conf_matrix[23][9] + dnn_conf_matrix[23][10] + dnn_conf_matrix[23][11] + dnn_conf_matrix[23][12] + dnn_conf_matrix[23][13] + dnn_conf_matrix[23][14] + dnn_conf_matrix[23][15] + dnn_conf_matrix[23][16] + dnn_conf_matrix[23][17] + dnn_conf_matrix[23][18] + dnn_conf_matrix[23][19] + dnn_conf_matrix[23][20] + dnn_conf_matrix[23][21] + dnn_conf_matrix[23][22] + dnn_conf_matrix[23][24] + dnn_conf_matrix[23][25] + dnn_conf_matrix[23][26] + dnn_conf_matrix[23][27] + dnn_conf_matrix[23][28]
        #dnn_fn_25 = dnn_conf_matrix[24][0] + dnn_conf_matrix[24][1] + dnn_conf_matrix[24][2] + dnn_conf_matrix[24][3] + dnn_conf_matrix[24][4] + dnn_conf_matrix[24][5] + dnn_conf_matrix[24][6] + dnn_conf_matrix[24][7] + dnn_conf_matrix[24][8] + dnn_conf_matrix[24][9] + dnn_conf_matrix[24][10] + dnn_conf_matrix[24][11] + dnn_conf_matrix[24][12] + dnn_conf_matrix[24][13] + dnn_conf_matrix[24][14] + dnn_conf_matrix[24][15] + dnn_conf_matrix[24][16] + dnn_conf_matrix[24][17] + dnn_conf_matrix[24][18] + dnn_conf_matrix[24][19] + dnn_conf_matrix[24][20] + dnn_conf_matrix[24][21] + dnn_conf_matrix[24][22] + dnn_conf_matrix[24][23] + dnn_conf_matrix[24][25] + dnn_conf_matrix[24][26] + dnn_conf_matrix[24][27] + dnn_conf_matrix[24][28]
        #dnn_fn_26 = dnn_conf_matrix[25][0] + dnn_conf_matrix[25][1] + dnn_conf_matrix[25][2] + dnn_conf_matrix[25][3] + dnn_conf_matrix[25][4] + dnn_conf_matrix[25][5] + dnn_conf_matrix[25][6] + dnn_conf_matrix[25][7] + dnn_conf_matrix[25][8] + dnn_conf_matrix[25][9] + dnn_conf_matrix[25][10] + dnn_conf_matrix[25][11] + dnn_conf_matrix[25][12] + dnn_conf_matrix[25][13] + dnn_conf_matrix[25][14] + dnn_conf_matrix[25][15] + dnn_conf_matrix[25][16] + dnn_conf_matrix[25][17] + dnn_conf_matrix[25][18] + dnn_conf_matrix[25][19] + dnn_conf_matrix[25][20] + dnn_conf_matrix[25][21] + dnn_conf_matrix[25][22] + dnn_conf_matrix[25][23] + dnn_conf_matrix[25][24] + dnn_conf_matrix[25][26] + dnn_conf_matrix[25][27] + dnn_conf_matrix[25][28]
        #dnn_fn_27 = dnn_conf_matrix[26][0] + dnn_conf_matrix[26][1] + dnn_conf_matrix[26][2] + dnn_conf_matrix[26][3] + dnn_conf_matrix[26][4] + dnn_conf_matrix[26][5] + dnn_conf_matrix[26][6] + dnn_conf_matrix[26][7] + dnn_conf_matrix[26][8] + dnn_conf_matrix[26][9] + dnn_conf_matrix[26][10] + dnn_conf_matrix[26][11] + dnn_conf_matrix[26][12] + dnn_conf_matrix[26][13] + dnn_conf_matrix[26][14] + dnn_conf_matrix[26][15] + dnn_conf_matrix[26][16] + dnn_conf_matrix[26][17] + dnn_conf_matrix[26][18] + dnn_conf_matrix[26][19] + dnn_conf_matrix[26][20] + dnn_conf_matrix[26][21] + dnn_conf_matrix[26][22] + dnn_conf_matrix[26][23] + dnn_conf_matrix[26][24] + dnn_conf_matrix[26][25] + dnn_conf_matrix[26][27] + dnn_conf_matrix[26][28]
        #dnn_fn_28 = dnn_conf_matrix[27][0] + dnn_conf_matrix[27][1] + dnn_conf_matrix[27][2] + dnn_conf_matrix[27][3] + dnn_conf_matrix[27][4] + dnn_conf_matrix[27][5] + dnn_conf_matrix[27][6] + dnn_conf_matrix[27][7] + dnn_conf_matrix[27][8] + dnn_conf_matrix[27][9] + dnn_conf_matrix[27][10] + dnn_conf_matrix[27][11] + dnn_conf_matrix[27][12] + dnn_conf_matrix[27][13] + dnn_conf_matrix[27][14] + dnn_conf_matrix[27][15] + dnn_conf_matrix[27][16] + dnn_conf_matrix[27][17] + dnn_conf_matrix[27][18] + dnn_conf_matrix[27][19] + dnn_conf_matrix[27][20] + dnn_conf_matrix[27][21] + dnn_conf_matrix[27][22] + dnn_conf_matrix[27][23] + dnn_conf_matrix[27][24] + dnn_conf_matrix[27][25] + dnn_conf_matrix[27][26] + dnn_conf_matrix[27][28]
        #dnn_fn_29 = dnn_conf_matrix[28][0] + dnn_conf_matrix[28][1] + dnn_conf_matrix[28][2] + dnn_conf_matrix[28][3] + dnn_conf_matrix[28][4] + dnn_conf_matrix[28][5] + dnn_conf_matrix[28][6] + dnn_conf_matrix[28][7] + dnn_conf_matrix[28][8] + dnn_conf_matrix[28][9] + dnn_conf_matrix[28][10] + dnn_conf_matrix[28][11] + dnn_conf_matrix[28][12] + dnn_conf_matrix[28][13] + dnn_conf_matrix[28][14] + dnn_conf_matrix[28][15] + dnn_conf_matrix[28][16] + dnn_conf_matrix[28][17] + dnn_conf_matrix[28][18] + dnn_conf_matrix[28][19] + dnn_conf_matrix[28][20] + dnn_conf_matrix[28][21] + dnn_conf_matrix[28][22] + dnn_conf_matrix[28][23] + dnn_conf_matrix[28][24] + dnn_conf_matrix[28][25] + dnn_conf_matrix[28][26] + dnn_conf_matrix[28][27]

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
            dnn_recall_5 = dnn_tp_5 / (dnn_tp_5 + dnn_fn_5)
        if dnn_tp_6 + dnn_fn_6 == 0:
            dnn_recall_6 = 0
        else:
            dnn_recall_6 = dnn_tp_6 / (dnn_tp_6 + dnn_fn_6)
        if dnn_tp_7 + dnn_fn_7 == 0:
            dnn_recall_7 = 0
        else:
            dnn_recall_7 = dnn_tp_7 / (dnn_tp_7 + dnn_fn_7)
        if dnn_tp_8 + dnn_fn_8 == 0:
            dnn_recall_8 = 0
        else:
            dnn_recall_8 = dnn_tp_8 / (dnn_tp_8 + dnn_fn_8)
        if dnn_tp_9 + dnn_fn_9 == 0:
            dnn_recall_9 = 0
        else:
            dnn_recall_9 = dnn_tp_9 / (dnn_tp_9 + dnn_fn_9)
        if dnn_tp_10 + dnn_fn_10 == 0:
            dnn_recall_10 = 0
        else:
            dnn_recall_10 = dnn_tp_10 / (dnn_tp_10 + dnn_fn_10)
        if dnn_tp_11 + dnn_fn_11 == 0:
            dnn_recall_11 = 0
        else:
            dnn_recall_11 = dnn_tp_11 / (dnn_tp_11 + dnn_fn_11)
        if dnn_tp_12 + dnn_fn_12 == 0:
            dnn_recall_12 = 0
        else:
            dnn_recall_12 = dnn_tp_12 / (dnn_tp_12 + dnn_fn_12)
        if dnn_tp_13 + dnn_fn_13 == 0:
            dnn_recall_13 = 0
        else:
            dnn_recall_13 = dnn_tp_13 / (dnn_tp_13 + dnn_fn_13)
        if dnn_tp_14 + dnn_fn_14 == 0:
            dnn_recall_14 = 0
        else:
            dnn_recall_14 = dnn_tp_14 / (dnn_tp_14 + dnn_fn_14)
        if dnn_tp_15 + dnn_fn_15 == 0:
            dnn_recall_15 = 0
        else:
            dnn_recall_15 = dnn_tp_15 / (dnn_tp_15 + dnn_fn_15)
        if dnn_tp_16 + dnn_fn_16 == 0:
            dnn_recall_16 = 0
        else:
            dnn_recall_16 = dnn_tp_16 / (dnn_tp_16 + dnn_fn_16)
        if dnn_tp_17 + dnn_fn_17 == 0:
            dnn_recall_17 = 0
        else:
            dnn_recall_17 = dnn_tp_17 / (dnn_tp_17 + dnn_fn_17)
        if dnn_tp_18 + dnn_fn_18 == 0:
            dnn_recall_18 = 0
        else:
            dnn_recall_18 = dnn_tp_18 / (dnn_tp_18 + dnn_fn_18)
        if dnn_tp_19 + dnn_fn_19 == 0:
            dnn_recall_19 = 0
        else:
            dnn_recall_19 = dnn_tp_19 / (dnn_tp_19 + dnn_fn_19)
        if dnn_tp_20 + dnn_fn_20 == 0:
            dnn_recall_20 = 0
        else:
            dnn_recall_20 = dnn_tp_20 / (dnn_tp_20 + dnn_fn_20)
        '''
        if dnn_tp_21 + dnn_fn_21 == 0:
            dnn_recall_21 = 0
        else:
            dnn_recall_21 = dnn_tp_21 / (dnn_tp_21 + dnn_fn_21)
        if dnn_tp_22 + dnn_fn_22 == 0:
            dnn_recall_22 = 0
        else:
            dnn_recall_22 = dnn_tp_22 / (dnn_tp_22 + dnn_fn_22)
        if dnn_tp_23 + dnn_fn_23 == 0:
            dnn_recall_23 = 0
        else:
            dnn_recall_23 = dnn_tp_23 / (dnn_tp_23 + dnn_fn_23)
        if dnn_tp_24 + dnn_fn_24 == 0:
            dnn_recall_24 = 0
        else:
            dnn_recall_24 = dnn_tp_24 / (dnn_tp_24 + dnn_fn_24)
        if dnn_tp_25 + dnn_fn_25 == 0:
            dnn_recall_25 = 0
        else:
            dnn_recall_25 = dnn_tp_25 / (dnn_tp_25 + dnn_fn_25)
        if dnn_tp_26 + dnn_fn_26 == 0:
            dnn_recall_26 = 0
        else:
            dnn_recall_26 = dnn_tp_26 / (dnn_tp_26 + dnn_fn_26)
        if dnn_tp_27 + dnn_fn_27 == 0:
            dnn_recall_27 = 0
        else:
            dnn_recall_27 = dnn_tp_27 / (dnn_tp_27 + dnn_fn_27)
        if dnn_tp_28 + dnn_fn_28 == 0:
            dnn_recall_28 = 0
        else:
            dnn_recall_28 = dnn_tp_28 / (dnn_tp_28 + dnn_fn_28)
        if dnn_tp_29 + dnn_fn_29 == 0:
            dnn_recall_29 = 0
        else:
            dnn_recall_29 = dnn_tp_29 / (dnn_tp_29 + dnn_fn_29)
        '''
        dnn_recall_avg_pen_5 = (
                                 dnn_recall_1 + dnn_recall_2 + dnn_recall_3 + dnn_recall_4 + dnn_recall_5 + dnn_recall_6 + dnn_recall_7 + dnn_recall_8 + dnn_recall_9 + dnn_recall_10 + dnn_recall_11 + dnn_recall_12 + dnn_recall_13 + dnn_recall_14 + dnn_recall_15 + dnn_recall_16 + dnn_recall_17 + dnn_recall_18 + dnn_recall_19 + (5*dnn_recall_20)) / (20+5-1)
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
    dnn_ovr_accuracy = (dnn_conf_matrix[0][0] + dnn_conf_matrix[1][1] + dnn_conf_matrix[2][2] + dnn_conf_matrix[3][3] + dnn_conf_matrix[4][4] + dnn_conf_matrix[5][5] + dnn_conf_matrix[6][6] + dnn_conf_matrix[7][7] + dnn_conf_matrix[8][8] + dnn_conf_matrix[9][9] + dnn_conf_matrix[10][10] + dnn_conf_matrix[11][11] + dnn_conf_matrix[12][12] + dnn_conf_matrix[13][13] + dnn_conf_matrix[14][14] + dnn_conf_matrix[15][15] + dnn_conf_matrix[16][16] + dnn_conf_matrix[17][17] + dnn_conf_matrix[18][18] + dnn_conf_matrix[19][19]) / (
                sum(dnn_conf_matrix[0]) + sum(dnn_conf_matrix[1]) + sum(dnn_conf_matrix[2]) + sum(dnn_conf_matrix[3]) + sum(dnn_conf_matrix[4]) + sum(dnn_conf_matrix[5]) + sum(dnn_conf_matrix[6]) + sum(dnn_conf_matrix[7]) + sum(dnn_conf_matrix[8]) + sum(dnn_conf_matrix[9]) + sum(dnn_conf_matrix[10]) + sum(dnn_conf_matrix[11]) + sum(dnn_conf_matrix[12]) + sum(dnn_conf_matrix[13]) + sum(dnn_conf_matrix[14]) + sum(dnn_conf_matrix[15]) + sum(dnn_conf_matrix[16]) + sum(dnn_conf_matrix[17]) + sum(dnn_conf_matrix[18]) + sum(dnn_conf_matrix[19]))
    print("dnn_f1 score of pen 1 is:")
    print(dnn_f1_score_pen_1)
    print("dnn_f1 score of pen 5 is:")
    print(dnn_f1_score_pen_5)
    print("dnn_overall accuracy is:")
    print(dnn_ovr_accuracy)
    dnn_conf_matrix = pd.DataFrame(dnn_conf_matrix)
    dnn_conf_matrix.to_csv('conf_matrix_' + imb_technique + '_dnn_production_' + str(nsplits) + 'foldcv_' + str(repeat+1) + '.csv', header=False, index=False)  # First repetition
    #dnn_conf_matrix.to_csv('conf_matrix_' + imb_technique + '_penalty_' + str(penalty) + '_dnn_production_' + str(nsplits) + 'foldcv_' + str(repeat+6) + '.csv', header=False, index=False)  # First repetition
    dnn_f1_score_pen_1_kfoldcv[repeat] = dnn_f1_score_pen_1
    dnn_f1_score_pen_5_kfoldcv[repeat] = dnn_f1_score_pen_5
    dnn_ovr_accuracy_kfoldcv[repeat] = dnn_ovr_accuracy

    for i in range(0, len(y_test)):
        #lr_DM_index = 0
        lr_FI_index = 0
        lr_FG_index = 0
        #lr_GR_index = 0
        #lr_GR12_index = 0
        lr_GR27_index = 0
        lr_LM_index = 0
        lr_LMM_index = 0
        #lr_MM14_index = 0
        #lr_MM16_index = 0
        lr_PC_index = 0
        lr_RG12_index = 0
        #lr_RG19_index = 0
        lr_RG2_index = 0
        lr_RG3_index = 0
        lr_RGM_index = 0
        lr_RGQC_index = 0
        #lr_TMSA10_index = 0
        lr_T8_index = 0
        #lr_T9_index = 0
        lr_TM10_index = 0
        lr_TM4_index = 0
        lr_TM5_index = 0
        lr_TM6_index = 0
        lr_TM8_index = 0
        lr_TM9_index = 0
        lr_TMQC_index = 0
        lr_TQC_index = 0
        #lr_WC13_index = 0

        """
        if lr_pred_class_DM[i] == "Deburring - Manual":
            if lr_pred_prob_DM[i][0] >= 0.5:
                lr_DM_index = 0
            else:
                lr_DM_index = 1
        elif lr_pred_class_DM[i] == "Others":
            if lr_pred_prob_DM[i][0] < 0.5:
                lr_DM_index = 0
            else:
                lr_DM_index = 1
        """
        if lr_pred_class_FI[i] == "Final Inspection Q.C.":
            if lr_pred_prob_FI[i][0] >= 0.5:
                lr_FI_index = 0
            else:
                lr_FI_index = 1
        elif lr_pred_class_FI[i] == "Others":
            if lr_pred_prob_FI[i][0] < 0.5:
                lr_FI_index = 0
            else:
                lr_FI_index = 1
        if lr_pred_class_FG[i] == "Flat Grinding - Machine 11":
            if lr_pred_prob_FG[i][0] >= 0.5:
                lr_FG_index = 0
            else:
                lr_FG_index = 1
        elif lr_pred_class_FG[i] == "Others":
            if lr_pred_prob_FG[i][0] < 0.5:
                lr_FG_index = 0
            else:
                lr_FG_index = 1
        """
        if lr_pred_class_GR[i] == "Grinding Rework":
            if lr_pred_prob_GR[i][0] >= 0.5:
                lr_GR_index = 0
            else:
                lr_GR_index = 1
        elif lr_pred_class_GR[i] == "Others":
            if lr_pred_prob_GR[i][0] < 0.5:
                lr_GR_index = 0
            else:
                lr_GR_index = 1
        """
        """
        if lr_pred_class_GR12[i] == "Grinding Rework - Machine 12":
            if lr_pred_prob_GR12[i][0] >= 0.5:
                lr_GR12_index = 0
            else:
                lr_GR12_index = 1
        elif lr_pred_class_GR12[i] == "Others":
            if lr_pred_prob_GR12[i][0] < 0.5:
                lr_GR12_index = 0
            else:
                lr_GR12_index = 1
        """
        if lr_pred_class_GR27[i] == "Grinding Rework - Machine 27":
            if lr_pred_prob_GR27[i][0] >= 0.5:
                lr_GR27_index = 0
            else:
                lr_GR27_index = 1
        elif lr_pred_class_GR27[i] == "Others":
            if lr_pred_prob_GR27[i][0] < 0.5:
                lr_GR27_index = 0
            else:
                lr_GR27_index = 1
        if lr_pred_class_LM[i] == "Lapping - Machine 1":
            if lr_pred_prob_LM[i][0] >= 0.5:
                lr_LM_index = 0
            else:
                lr_LM_index = 1
        elif lr_pred_class_LM[i] == "Others":
            if lr_pred_prob_LM[i][0] < 0.5:
                lr_LM_index = 0
            else:
                lr_LM_index = 1
        if lr_pred_class_LMM[i] == "Laser Marking - Machine 7":
            if lr_pred_prob_LMM[i][0] >= 0.5:
                lr_LMM_index = 0
            else:
                lr_LMM_index = 1
        elif lr_pred_class_LMM[i] == "Others":
            if lr_pred_prob_LMM[i][0] < 0.5:
                lr_LMM_index = 0
            else:
                lr_LMM_index = 1
        """
        if lr_pred_class_MM14[i] == "Milling - Machine 14":
            if lr_pred_prob_MM14[i][0] >= 0.5:
                lr_MM14_index = 0
            else:
                lr_MM14_index = 1
        elif lr_pred_class_MM14[i] == "Others":
            if lr_pred_prob_MM14[i][0] < 0.5:
                lr_MM14_index = 0
            else:
                lr_MM14_index = 1
        """
        """
        if lr_pred_class_MM16[i] == "Milling - Machine 16":
            if lr_pred_prob_MM16[i][0] >= 0.5:
                lr_MM16_index = 0
            else:
                lr_MM16_index = 1
        elif lr_pred_class_MM16[i] == "Others":
            if lr_pred_prob_MM16[i][0] < 0.5:
                lr_MM16_index = 0
            else:
                lr_MM16_index = 1
        """
        if lr_pred_class_PC[i] == "Packing":
            if lr_pred_prob_PC[i][0] >= 0.5:
                lr_PC_index = 0
            else:
                lr_PC_index = 1
        elif lr_pred_class_PC[i] == "Others":
            if lr_pred_prob_PC[i][0] < 0.5:
                lr_PC_index = 0
            else:
                lr_PC_index = 1
        if lr_pred_class_RG12[i] == "Round Grinding - Machine 12":
            if lr_pred_prob_RG12[i][0] >= 0.5:
                lr_RG12_index = 0
            else:
                lr_RG12_index = 1
        elif lr_pred_class_RG12[i] == "Others":
            if lr_pred_prob_RG12[i][0] < 0.5:
                lr_RG12_index = 0
            else:
                lr_RG12_index = 1
        """
        if lr_pred_class_RG19[i] == "Round Grinding - Machine 19":
            if lr_pred_prob_RG19[i][0] >= 0.5:
                lr_RG19_index = 0
            else:
                lr_RG19_index = 1
        elif lr_pred_class_RG19[i] == "Others":
            if lr_pred_prob_RG19[i][0] < 0.5:
                lr_RG19_index = 0
            else:
                lr_RG19_index = 1
        """
        if lr_pred_class_RG2[i] == "Round Grinding - Machine 2":
            if lr_pred_prob_RG2[i][0] >= 0.5:
                lr_RG2_index = 0
            else:
                lr_RG2_index = 1
        elif lr_pred_class_RG2[i] == "Others":
            if lr_pred_prob_RG2[i][0] < 0.5:
                lr_RG2_index = 0
            else:
                lr_RG2_index = 1
        if lr_pred_class_RG3[i] == "Round Grinding - Machine 3":
            if lr_pred_prob_RG3[i][0] >= 0.5:
                lr_RG3_index = 0
            else:
                lr_RG3_index = 1
        elif lr_pred_class_RG3[i] == "Others":
            if lr_pred_prob_RG3[i][0] < 0.5:
                lr_RG3_index = 0
            else:
                lr_RG3_index = 1
        if lr_pred_class_RGM[i] == "Round Grinding - Manual":
            if lr_pred_prob_RGM[i][0] >= 0.5:
                lr_RGM_index = 0
            else:
                lr_RGM_index = 1
        elif lr_pred_class_RGM[i] == "Others":
            if lr_pred_prob_RGM[i][0] < 0.5:
                lr_RGM_index = 0
            else:
                lr_RGM_index = 1
        if lr_pred_class_RGQC[i] == "Round Grinding - Q.C.":
            if lr_pred_prob_RGQC[i][0] >= 0.5:
                lr_RGQC_index = 0
            else:
                lr_RGQC_index = 1
        elif lr_pred_class_RGQC[i] == "Others":
            if lr_pred_prob_RGQC[i][0] < 0.5:
                lr_RGQC_index = 0
            else:
                lr_RGQC_index = 1
        """
        if lr_pred_class_TMSA10[i] == "Turn & Mill. & Screw Assem - Machine 10":
            if lr_pred_prob_TMSA10[i][0] >= 0.5:
                lr_TMSA10_index = 0
            else:
                lr_TMSA10_index = 1
        elif lr_pred_class_TMSA10[i] == "Others":
            if lr_pred_prob_TMSA10[i][0] < 0.5:
                lr_TMSA10_index = 0
            else:
                lr_TMSA10_index = 1
        """
        if lr_pred_class_T8[i] == "Turning - Machine 8":
            if lr_pred_prob_T8[i][0] >= 0.5:
                lr_T8_index = 0
            else:
                lr_T8_index = 1
        elif lr_pred_class_T8[i] == "Others":
            if lr_pred_prob_T8[i][0] < 0.5:
                lr_T8_index = 0
            else:
                lr_T8_index = 1
        """
        if lr_pred_class_T9[i] == "Turning - Machine 9":
            if lr_pred_prob_T9[i][0] >= 0.5:
                lr_T9_index = 0
            else:
                lr_T9_index = 1
        elif lr_pred_class_T9[i] == "Others":
            if lr_pred_prob_T9[i][0] < 0.5:
                lr_T9_index = 0
            else:
                lr_T9_index = 1
        """
        if lr_pred_class_TM10[i] == "Turning & Milling - Machine 10":
            if lr_pred_prob_TM10[i][0] >= 0.5:
                lr_TM10_index = 0
            else:
                lr_TM10_index = 1
        elif lr_pred_class_TM10[i] == "Others":
            if lr_pred_prob_TM10[i][0] < 0.5:
                lr_TM10_index = 0
            else:
                lr_TM10_index = 1
        if lr_pred_class_TM4[i] == "Turning & Milling - Machine 4":
            if lr_pred_prob_TM4[i][0] >= 0.5:
                lr_TM4_index = 0
            else:
                lr_TM4_index = 1
        elif lr_pred_class_TM4[i] == "Others":
            if lr_pred_prob_TM4[i][0] < 0.5:
                lr_TM4_index = 0
            else:
                lr_TM4_index = 1
        if lr_pred_class_TM5[i] == "Turning & Milling - Machine 5":
            if lr_pred_prob_TM5[i][0] >= 0.5:
                lr_TM5_index = 0
            else:
                lr_TM5_index = 1
        elif lr_pred_class_TM5[i] == "Others":
            if lr_pred_prob_TM5[i][0] < 0.5:
                lr_TM5_index = 0
            else:
                lr_TM5_index = 1
        if lr_pred_class_TM6[i] == "Turning & Milling - Machine 6":
            if lr_pred_prob_TM6[i][0] >= 0.5:
                lr_TM6_index = 0
            else:
                lr_TM6_index = 1
        elif lr_pred_class_TM6[i] == "Others":
            if lr_pred_prob_TM6[i][0] < 0.5:
                lr_TM6_index = 0
            else:
                lr_TM6_index = 1
        if lr_pred_class_TM8[i] == "Turning & Milling - Machine 8":
            if lr_pred_prob_TM8[i][0] >= 0.5:
                lr_TM8_index = 0
            else:
                lr_TM8_index = 1
        elif lr_pred_class_TM8[i] == "Others":
            if lr_pred_prob_TM8[i][0] < 0.5:
                lr_TM8_index = 0
            else:
                lr_TM8_index = 1
        if lr_pred_class_TM9[i] == "Turning & Milling - Machine 9":
            if lr_pred_prob_TM9[i][0] >= 0.5:
                lr_TM9_index = 0
            else:
                lr_TM9_index = 1
        elif lr_pred_class_TM9[i] == "Others":
            if lr_pred_prob_TM9[i][0] < 0.5:
                lr_TM9_index = 0
            else:
                lr_TM9_index = 1
        if lr_pred_class_TMQC[i] == "Turning & Milling Q.C.":
            if lr_pred_prob_TMQC[i][0] >= 0.5:
                lr_TMQC_index = 0
            else:
                lr_TMQC_index = 1
        elif lr_pred_class_TMQC[i] == "Others":
            if lr_pred_prob_TMQC[i][0] < 0.5:
                lr_TMQC_index = 0
            else:
                lr_TMQC_index = 1
        if lr_pred_class_TQC[i] == "Turning Q.C.":
            if lr_pred_prob_TQC[i][0] >= 0.5:
                lr_TQC_index = 0
            else:
                lr_TQC_index = 1
        elif lr_pred_class_TQC[i] == "Others":
            if lr_pred_prob_TQC[i][0] < 0.5:
                lr_TQC_index = 0
            else:
                lr_TQC_index = 1
        """
        if lr_pred_class_WC13[i] == "Wire Cut - Machine 13":
            if lr_pred_prob_WC13[i][0] >= 0.5:
                lr_WC13_index = 0
            else:
                lr_WC13_index = 1
        elif lr_pred_class_WC13[i] == "Others":
            if lr_pred_prob_WC13[i][0] < 0.5:
                lr_WC13_index = 0
            else:
                lr_WC13_index = 1
        """
        #if lr_pred_prob_DM[i][lr_DM_index] == max(lr_pred_prob_FI[i][lr_FI_index], lr_pred_prob_FG[i][lr_FG_index], lr_pred_prob_GR27[i][lr_GR27_index], lr_pred_prob_LM[i][lr_LM_index], lr_pred_prob_LMM[i][lr_LMM_index], lr_pred_prob_PC[i][lr_PC_index], lr_pred_prob_RG12[i][lr_RG12_index], lr_pred_prob_RG2[i][lr_RG2_index], lr_pred_prob_RG3[i][lr_RG3_index], lr_pred_prob_RGM[i][lr_RGM_index], lr_pred_prob_RGQC[i][lr_RGQC_index], lr_pred_prob_T8[i][lr_T8_index], lr_pred_prob_TM10[i][lr_TM10_index], lr_pred_prob_TM4[i][lr_TM4_index], lr_pred_prob_TM5[i][lr_TM5_index], lr_pred_prob_TM6[i][lr_TM6_index], lr_pred_prob_TM8[i][lr_TM8_index], lr_pred_prob_TM9[i][lr_TM9_index], lr_pred_prob_TMQC[i][lr_TMQC_index], lr_pred_prob_TQC[i][lr_TQC_index]):
        #    lr_prediction.loc[i] = "Deburring - Manual"
        if lr_pred_prob_FI[i][lr_FI_index] == max(lr_pred_prob_FI[i][lr_FI_index], lr_pred_prob_FG[i][lr_FG_index], lr_pred_prob_GR27[i][lr_GR27_index], lr_pred_prob_LM[i][lr_LM_index], lr_pred_prob_LMM[i][lr_LMM_index], lr_pred_prob_PC[i][lr_PC_index], lr_pred_prob_RG12[i][lr_RG12_index], lr_pred_prob_RG2[i][lr_RG2_index], lr_pred_prob_RG3[i][lr_RG3_index], lr_pred_prob_RGM[i][lr_RGM_index], lr_pred_prob_RGQC[i][lr_RGQC_index], lr_pred_prob_T8[i][lr_T8_index], lr_pred_prob_TM10[i][lr_TM10_index], lr_pred_prob_TM4[i][lr_TM4_index], lr_pred_prob_TM5[i][lr_TM5_index], lr_pred_prob_TM6[i][lr_TM6_index], lr_pred_prob_TM8[i][lr_TM8_index], lr_pred_prob_TM9[i][lr_TM9_index], lr_pred_prob_TMQC[i][lr_TMQC_index], lr_pred_prob_TQC[i][lr_TQC_index]):
            lr_prediction.loc[i] = "Final Inspection Q.C."
        elif lr_pred_prob_FG[i][lr_FG_index] == max(lr_pred_prob_FI[i][lr_FI_index], lr_pred_prob_FG[i][lr_FG_index], lr_pred_prob_GR27[i][lr_GR27_index], lr_pred_prob_LM[i][lr_LM_index], lr_pred_prob_LMM[i][lr_LMM_index], lr_pred_prob_PC[i][lr_PC_index], lr_pred_prob_RG12[i][lr_RG12_index], lr_pred_prob_RG2[i][lr_RG2_index], lr_pred_prob_RG3[i][lr_RG3_index], lr_pred_prob_RGM[i][lr_RGM_index], lr_pred_prob_RGQC[i][lr_RGQC_index], lr_pred_prob_T8[i][lr_T8_index], lr_pred_prob_TM10[i][lr_TM10_index], lr_pred_prob_TM4[i][lr_TM4_index], lr_pred_prob_TM5[i][lr_TM5_index], lr_pred_prob_TM6[i][lr_TM6_index], lr_pred_prob_TM8[i][lr_TM8_index], lr_pred_prob_TM9[i][lr_TM9_index], lr_pred_prob_TMQC[i][lr_TMQC_index], lr_pred_prob_TQC[i][lr_TQC_index]):
            lr_prediction.loc[i] = "Flat Grinding - Machine 11"
        #elif lr_pred_prob_GR[i][lr_GR_index] == max(lr_pred_prob_FI[i][lr_FI_index], lr_pred_prob_FG[i][lr_FG_index], lr_pred_prob_GR27[i][lr_GR27_index], lr_pred_prob_LM[i][lr_LM_index], lr_pred_prob_LMM[i][lr_LMM_index], lr_pred_prob_PC[i][lr_PC_index], lr_pred_prob_RG12[i][lr_RG12_index], lr_pred_prob_RG2[i][lr_RG2_index], lr_pred_prob_RG3[i][lr_RG3_index], lr_pred_prob_RGM[i][lr_RGM_index], lr_pred_prob_RGQC[i][lr_RGQC_index], lr_pred_prob_T8[i][lr_T8_index], lr_pred_prob_TM10[i][lr_TM10_index], lr_pred_prob_TM4[i][lr_TM4_index], lr_pred_prob_TM5[i][lr_TM5_index], lr_pred_prob_TM6[i][lr_TM6_index], lr_pred_prob_TM8[i][lr_TM8_index], lr_pred_prob_TM9[i][lr_TM9_index], lr_pred_prob_TMQC[i][lr_TMQC_index], lr_pred_prob_TQC[i][lr_TQC_index]):
        #    lr_prediction.loc[i] = "Grinding Rework"
        #elif lr_pred_prob_GR12[i][lr_GR12_index] == max(lr_pred_prob_FI[i][lr_FI_index], lr_pred_prob_FG[i][lr_FG_index], lr_pred_prob_GR27[i][lr_GR27_index], lr_pred_prob_LM[i][lr_LM_index], lr_pred_prob_LMM[i][lr_LMM_index], lr_pred_prob_PC[i][lr_PC_index], lr_pred_prob_RG12[i][lr_RG12_index], lr_pred_prob_RG2[i][lr_RG2_index], lr_pred_prob_RG3[i][lr_RG3_index], lr_pred_prob_RGM[i][lr_RGM_index], lr_pred_prob_RGQC[i][lr_RGQC_index], lr_pred_prob_T8[i][lr_T8_index], lr_pred_prob_TM10[i][lr_TM10_index], lr_pred_prob_TM4[i][lr_TM4_index], lr_pred_prob_TM5[i][lr_TM5_index], lr_pred_prob_TM6[i][lr_TM6_index], lr_pred_prob_TM8[i][lr_TM8_index], lr_pred_prob_TM9[i][lr_TM9_index], lr_pred_prob_TMQC[i][lr_TMQC_index], lr_pred_prob_TQC[i][lr_TQC_index]):
        #    lr_prediction.loc[i] = "Grinding Rework - Machine 12"
        elif lr_pred_prob_GR27[i][lr_GR27_index] == max(lr_pred_prob_FI[i][lr_FI_index], lr_pred_prob_FG[i][lr_FG_index], lr_pred_prob_GR27[i][lr_GR27_index], lr_pred_prob_LM[i][lr_LM_index], lr_pred_prob_LMM[i][lr_LMM_index], lr_pred_prob_PC[i][lr_PC_index], lr_pred_prob_RG12[i][lr_RG12_index], lr_pred_prob_RG2[i][lr_RG2_index], lr_pred_prob_RG3[i][lr_RG3_index], lr_pred_prob_RGM[i][lr_RGM_index], lr_pred_prob_RGQC[i][lr_RGQC_index], lr_pred_prob_T8[i][lr_T8_index], lr_pred_prob_TM10[i][lr_TM10_index], lr_pred_prob_TM4[i][lr_TM4_index], lr_pred_prob_TM5[i][lr_TM5_index], lr_pred_prob_TM6[i][lr_TM6_index], lr_pred_prob_TM8[i][lr_TM8_index], lr_pred_prob_TM9[i][lr_TM9_index], lr_pred_prob_TMQC[i][lr_TMQC_index], lr_pred_prob_TQC[i][lr_TQC_index]):
            lr_prediction.loc[i] = "Grinding Rework - Machine 27"
        elif lr_pred_prob_LM[i][lr_LM_index] == max(lr_pred_prob_FI[i][lr_FI_index], lr_pred_prob_FG[i][lr_FG_index], lr_pred_prob_GR27[i][lr_GR27_index], lr_pred_prob_LM[i][lr_LM_index], lr_pred_prob_LMM[i][lr_LMM_index], lr_pred_prob_PC[i][lr_PC_index], lr_pred_prob_RG12[i][lr_RG12_index], lr_pred_prob_RG2[i][lr_RG2_index], lr_pred_prob_RG3[i][lr_RG3_index], lr_pred_prob_RGM[i][lr_RGM_index], lr_pred_prob_RGQC[i][lr_RGQC_index], lr_pred_prob_T8[i][lr_T8_index], lr_pred_prob_TM10[i][lr_TM10_index], lr_pred_prob_TM4[i][lr_TM4_index], lr_pred_prob_TM5[i][lr_TM5_index], lr_pred_prob_TM6[i][lr_TM6_index], lr_pred_prob_TM8[i][lr_TM8_index], lr_pred_prob_TM9[i][lr_TM9_index], lr_pred_prob_TMQC[i][lr_TMQC_index], lr_pred_prob_TQC[i][lr_TQC_index]):
            lr_prediction.loc[i] = "Lapping - Machine 1"
        elif lr_pred_prob_LMM[i][lr_LMM_index] == max(lr_pred_prob_FI[i][lr_FI_index], lr_pred_prob_FG[i][lr_FG_index], lr_pred_prob_GR27[i][lr_GR27_index], lr_pred_prob_LM[i][lr_LM_index], lr_pred_prob_LMM[i][lr_LMM_index], lr_pred_prob_PC[i][lr_PC_index], lr_pred_prob_RG12[i][lr_RG12_index], lr_pred_prob_RG2[i][lr_RG2_index], lr_pred_prob_RG3[i][lr_RG3_index], lr_pred_prob_RGM[i][lr_RGM_index], lr_pred_prob_RGQC[i][lr_RGQC_index], lr_pred_prob_T8[i][lr_T8_index], lr_pred_prob_TM10[i][lr_TM10_index], lr_pred_prob_TM4[i][lr_TM4_index], lr_pred_prob_TM5[i][lr_TM5_index], lr_pred_prob_TM6[i][lr_TM6_index], lr_pred_prob_TM8[i][lr_TM8_index], lr_pred_prob_TM9[i][lr_TM9_index], lr_pred_prob_TMQC[i][lr_TMQC_index], lr_pred_prob_TQC[i][lr_TQC_index]):
            lr_prediction.loc[i] = "Laser Marking - Machine 7"
        #elif lr_pred_prob_MM14[i][lr_MM14_index] == max(lr_pred_prob_FI[i][lr_FI_index], lr_pred_prob_FG[i][lr_FG_index], lr_pred_prob_GR27[i][lr_GR27_index], lr_pred_prob_LM[i][lr_LM_index], lr_pred_prob_LMM[i][lr_LMM_index], lr_pred_prob_PC[i][lr_PC_index], lr_pred_prob_RG12[i][lr_RG12_index], lr_pred_prob_RG2[i][lr_RG2_index], lr_pred_prob_RG3[i][lr_RG3_index], lr_pred_prob_RGM[i][lr_RGM_index], lr_pred_prob_RGQC[i][lr_RGQC_index], lr_pred_prob_T8[i][lr_T8_index], lr_pred_prob_TM10[i][lr_TM10_index], lr_pred_prob_TM4[i][lr_TM4_index], lr_pred_prob_TM5[i][lr_TM5_index], lr_pred_prob_TM6[i][lr_TM6_index], lr_pred_prob_TM8[i][lr_TM8_index], lr_pred_prob_TM9[i][lr_TM9_index], lr_pred_prob_TMQC[i][lr_TMQC_index], lr_pred_prob_TQC[i][lr_TQC_index]):
        #    lr_prediction.loc[i] = "Milling - Machine 14"
        #elif lr_pred_prob_MM16[i][lr_MM16_index] == max(lr_pred_prob_FI[i][lr_FI_index], lr_pred_prob_FG[i][lr_FG_index], lr_pred_prob_GR27[i][lr_GR27_index], lr_pred_prob_LM[i][lr_LM_index], lr_pred_prob_LMM[i][lr_LMM_index], lr_pred_prob_PC[i][lr_PC_index], lr_pred_prob_RG12[i][lr_RG12_index], lr_pred_prob_RG2[i][lr_RG2_index], lr_pred_prob_RG3[i][lr_RG3_index], lr_pred_prob_RGM[i][lr_RGM_index], lr_pred_prob_RGQC[i][lr_RGQC_index], lr_pred_prob_T8[i][lr_T8_index], lr_pred_prob_TM10[i][lr_TM10_index], lr_pred_prob_TM4[i][lr_TM4_index], lr_pred_prob_TM5[i][lr_TM5_index], lr_pred_prob_TM6[i][lr_TM6_index], lr_pred_prob_TM8[i][lr_TM8_index], lr_pred_prob_TM9[i][lr_TM9_index], lr_pred_prob_TMQC[i][lr_TMQC_index], lr_pred_prob_TQC[i][lr_TQC_index]):
        #    lr_prediction.loc[i] = "Milling - Machine 16"
        elif lr_pred_prob_PC[i][lr_PC_index] == max(lr_pred_prob_FI[i][lr_FI_index], lr_pred_prob_FG[i][lr_FG_index], lr_pred_prob_GR27[i][lr_GR27_index], lr_pred_prob_LM[i][lr_LM_index], lr_pred_prob_LMM[i][lr_LMM_index], lr_pred_prob_PC[i][lr_PC_index], lr_pred_prob_RG12[i][lr_RG12_index], lr_pred_prob_RG2[i][lr_RG2_index], lr_pred_prob_RG3[i][lr_RG3_index], lr_pred_prob_RGM[i][lr_RGM_index], lr_pred_prob_RGQC[i][lr_RGQC_index], lr_pred_prob_T8[i][lr_T8_index], lr_pred_prob_TM10[i][lr_TM10_index], lr_pred_prob_TM4[i][lr_TM4_index], lr_pred_prob_TM5[i][lr_TM5_index], lr_pred_prob_TM6[i][lr_TM6_index], lr_pred_prob_TM8[i][lr_TM8_index], lr_pred_prob_TM9[i][lr_TM9_index], lr_pred_prob_TMQC[i][lr_TMQC_index], lr_pred_prob_TQC[i][lr_TQC_index]):
            lr_prediction.loc[i] = "Packing"
        elif lr_pred_prob_RG12[i][lr_RG12_index] == max(lr_pred_prob_FI[i][lr_FI_index], lr_pred_prob_FG[i][lr_FG_index], lr_pred_prob_GR27[i][lr_GR27_index], lr_pred_prob_LM[i][lr_LM_index], lr_pred_prob_LMM[i][lr_LMM_index], lr_pred_prob_PC[i][lr_PC_index], lr_pred_prob_RG12[i][lr_RG12_index], lr_pred_prob_RG2[i][lr_RG2_index], lr_pred_prob_RG3[i][lr_RG3_index], lr_pred_prob_RGM[i][lr_RGM_index], lr_pred_prob_RGQC[i][lr_RGQC_index], lr_pred_prob_T8[i][lr_T8_index], lr_pred_prob_TM10[i][lr_TM10_index], lr_pred_prob_TM4[i][lr_TM4_index], lr_pred_prob_TM5[i][lr_TM5_index], lr_pred_prob_TM6[i][lr_TM6_index], lr_pred_prob_TM8[i][lr_TM8_index], lr_pred_prob_TM9[i][lr_TM9_index], lr_pred_prob_TMQC[i][lr_TMQC_index], lr_pred_prob_TQC[i][lr_TQC_index]):
            lr_prediction.loc[i] = "Round Grinding - Machine 12"
        #elif lr_pred_prob_RG19[i][lr_RG19_index] == max(lr_pred_prob_FI[i][lr_FI_index], lr_pred_prob_FG[i][lr_FG_index], lr_pred_prob_GR27[i][lr_GR27_index], lr_pred_prob_LM[i][lr_LM_index], lr_pred_prob_LMM[i][lr_LMM_index], lr_pred_prob_PC[i][lr_PC_index], lr_pred_prob_RG12[i][lr_RG12_index], lr_pred_prob_RG2[i][lr_RG2_index], lr_pred_prob_RG3[i][lr_RG3_index], lr_pred_prob_RGM[i][lr_RGM_index], lr_pred_prob_RGQC[i][lr_RGQC_index], lr_pred_prob_T8[i][lr_T8_index], lr_pred_prob_TM10[i][lr_TM10_index], lr_pred_prob_TM4[i][lr_TM4_index], lr_pred_prob_TM5[i][lr_TM5_index], lr_pred_prob_TM6[i][lr_TM6_index], lr_pred_prob_TM8[i][lr_TM8_index], lr_pred_prob_TM9[i][lr_TM9_index], lr_pred_prob_TMQC[i][lr_TMQC_index], lr_pred_prob_TQC[i][lr_TQC_index]):
        #    lr_prediction.loc[i] = "Round Grinding - Machine 19"
        elif lr_pred_prob_RG2[i][lr_RG2_index] == max(lr_pred_prob_FI[i][lr_FI_index], lr_pred_prob_FG[i][lr_FG_index], lr_pred_prob_GR27[i][lr_GR27_index], lr_pred_prob_LM[i][lr_LM_index], lr_pred_prob_LMM[i][lr_LMM_index], lr_pred_prob_PC[i][lr_PC_index], lr_pred_prob_RG12[i][lr_RG12_index], lr_pred_prob_RG2[i][lr_RG2_index], lr_pred_prob_RG3[i][lr_RG3_index], lr_pred_prob_RGM[i][lr_RGM_index], lr_pred_prob_RGQC[i][lr_RGQC_index], lr_pred_prob_T8[i][lr_T8_index], lr_pred_prob_TM10[i][lr_TM10_index], lr_pred_prob_TM4[i][lr_TM4_index], lr_pred_prob_TM5[i][lr_TM5_index], lr_pred_prob_TM6[i][lr_TM6_index], lr_pred_prob_TM8[i][lr_TM8_index], lr_pred_prob_TM9[i][lr_TM9_index], lr_pred_prob_TMQC[i][lr_TMQC_index], lr_pred_prob_TQC[i][lr_TQC_index]):
            lr_prediction.loc[i] = "Round Grinding - Machine 2"
        elif lr_pred_prob_RG3[i][lr_RG3_index] == max(lr_pred_prob_FI[i][lr_FI_index], lr_pred_prob_FG[i][lr_FG_index], lr_pred_prob_GR27[i][lr_GR27_index], lr_pred_prob_LM[i][lr_LM_index], lr_pred_prob_LMM[i][lr_LMM_index], lr_pred_prob_PC[i][lr_PC_index], lr_pred_prob_RG12[i][lr_RG12_index], lr_pred_prob_RG2[i][lr_RG2_index], lr_pred_prob_RG3[i][lr_RG3_index], lr_pred_prob_RGM[i][lr_RGM_index], lr_pred_prob_RGQC[i][lr_RGQC_index], lr_pred_prob_T8[i][lr_T8_index], lr_pred_prob_TM10[i][lr_TM10_index], lr_pred_prob_TM4[i][lr_TM4_index], lr_pred_prob_TM5[i][lr_TM5_index], lr_pred_prob_TM6[i][lr_TM6_index], lr_pred_prob_TM8[i][lr_TM8_index], lr_pred_prob_TM9[i][lr_TM9_index], lr_pred_prob_TMQC[i][lr_TMQC_index], lr_pred_prob_TQC[i][lr_TQC_index]):
            lr_prediction.loc[i] = "Round Grinding - Machine 3"
        elif lr_pred_prob_RGM[i][lr_RGM_index] == max(lr_pred_prob_FI[i][lr_FI_index], lr_pred_prob_FG[i][lr_FG_index], lr_pred_prob_GR27[i][lr_GR27_index], lr_pred_prob_LM[i][lr_LM_index], lr_pred_prob_LMM[i][lr_LMM_index], lr_pred_prob_PC[i][lr_PC_index], lr_pred_prob_RG12[i][lr_RG12_index], lr_pred_prob_RG2[i][lr_RG2_index], lr_pred_prob_RG3[i][lr_RG3_index], lr_pred_prob_RGM[i][lr_RGM_index], lr_pred_prob_RGQC[i][lr_RGQC_index], lr_pred_prob_T8[i][lr_T8_index], lr_pred_prob_TM10[i][lr_TM10_index], lr_pred_prob_TM4[i][lr_TM4_index], lr_pred_prob_TM5[i][lr_TM5_index], lr_pred_prob_TM6[i][lr_TM6_index], lr_pred_prob_TM8[i][lr_TM8_index], lr_pred_prob_TM9[i][lr_TM9_index], lr_pred_prob_TMQC[i][lr_TMQC_index], lr_pred_prob_TQC[i][lr_TQC_index]):
            lr_prediction.loc[i] = "Round Grinding - Manual"
        elif lr_pred_prob_RGQC[i][lr_RGQC_index] == max(lr_pred_prob_FI[i][lr_FI_index], lr_pred_prob_FG[i][lr_FG_index], lr_pred_prob_GR27[i][lr_GR27_index], lr_pred_prob_LM[i][lr_LM_index], lr_pred_prob_LMM[i][lr_LMM_index], lr_pred_prob_PC[i][lr_PC_index], lr_pred_prob_RG12[i][lr_RG12_index], lr_pred_prob_RG2[i][lr_RG2_index], lr_pred_prob_RG3[i][lr_RG3_index], lr_pred_prob_RGM[i][lr_RGM_index], lr_pred_prob_RGQC[i][lr_RGQC_index], lr_pred_prob_T8[i][lr_T8_index], lr_pred_prob_TM10[i][lr_TM10_index], lr_pred_prob_TM4[i][lr_TM4_index], lr_pred_prob_TM5[i][lr_TM5_index], lr_pred_prob_TM6[i][lr_TM6_index], lr_pred_prob_TM8[i][lr_TM8_index], lr_pred_prob_TM9[i][lr_TM9_index], lr_pred_prob_TMQC[i][lr_TMQC_index], lr_pred_prob_TQC[i][lr_TQC_index]):
            lr_prediction.loc[i] = "Round Grinding - Q.C."
        #elif lr_pred_prob_TMSA10[i][lr_TMSA10_index] == max(lr_pred_prob_FI[i][lr_FI_index], lr_pred_prob_FG[i][lr_FG_index], lr_pred_prob_GR27[i][lr_GR27_index], lr_pred_prob_LM[i][lr_LM_index], lr_pred_prob_LMM[i][lr_LMM_index], lr_pred_prob_PC[i][lr_PC_index], lr_pred_prob_RG12[i][lr_RG12_index], lr_pred_prob_RG2[i][lr_RG2_index], lr_pred_prob_RG3[i][lr_RG3_index], lr_pred_prob_RGM[i][lr_RGM_index], lr_pred_prob_RGQC[i][lr_RGQC_index], lr_pred_prob_T8[i][lr_T8_index], lr_pred_prob_TM10[i][lr_TM10_index], lr_pred_prob_TM4[i][lr_TM4_index], lr_pred_prob_TM5[i][lr_TM5_index], lr_pred_prob_TM6[i][lr_TM6_index], lr_pred_prob_TM8[i][lr_TM8_index], lr_pred_prob_TM9[i][lr_TM9_index], lr_pred_prob_TMQC[i][lr_TMQC_index], lr_pred_prob_TQC[i][lr_TQC_index]):
        #    lr_prediction.loc[i] = "Turn & Mill. & Screw Assem - Machine 10"
        elif lr_pred_prob_T8[i][lr_T8_index] == max(lr_pred_prob_FI[i][lr_FI_index], lr_pred_prob_FG[i][lr_FG_index], lr_pred_prob_GR27[i][lr_GR27_index], lr_pred_prob_LM[i][lr_LM_index], lr_pred_prob_LMM[i][lr_LMM_index], lr_pred_prob_PC[i][lr_PC_index], lr_pred_prob_RG12[i][lr_RG12_index], lr_pred_prob_RG2[i][lr_RG2_index], lr_pred_prob_RG3[i][lr_RG3_index], lr_pred_prob_RGM[i][lr_RGM_index], lr_pred_prob_RGQC[i][lr_RGQC_index], lr_pred_prob_T8[i][lr_T8_index], lr_pred_prob_TM10[i][lr_TM10_index], lr_pred_prob_TM4[i][lr_TM4_index], lr_pred_prob_TM5[i][lr_TM5_index], lr_pred_prob_TM6[i][lr_TM6_index], lr_pred_prob_TM8[i][lr_TM8_index], lr_pred_prob_TM9[i][lr_TM9_index], lr_pred_prob_TMQC[i][lr_TMQC_index], lr_pred_prob_TQC[i][lr_TQC_index]):
            lr_prediction.loc[i] = "Turning - Machine 8"
        #elif lr_pred_prob_T9[i][lr_T9_index] == max(lr_pred_prob_FI[i][lr_FI_index], lr_pred_prob_FG[i][lr_FG_index], lr_pred_prob_GR27[i][lr_GR27_index], lr_pred_prob_LM[i][lr_LM_index], lr_pred_prob_LMM[i][lr_LMM_index], lr_pred_prob_PC[i][lr_PC_index], lr_pred_prob_RG12[i][lr_RG12_index], lr_pred_prob_RG2[i][lr_RG2_index], lr_pred_prob_RG3[i][lr_RG3_index], lr_pred_prob_RGM[i][lr_RGM_index], lr_pred_prob_RGQC[i][lr_RGQC_index], lr_pred_prob_T8[i][lr_T8_index], lr_pred_prob_TM10[i][lr_TM10_index], lr_pred_prob_TM4[i][lr_TM4_index], lr_pred_prob_TM5[i][lr_TM5_index], lr_pred_prob_TM6[i][lr_TM6_index], lr_pred_prob_TM8[i][lr_TM8_index], lr_pred_prob_TM9[i][lr_TM9_index], lr_pred_prob_TMQC[i][lr_TMQC_index], lr_pred_prob_TQC[i][lr_TQC_index]):
        #    lr_prediction.loc[i] = "Turning - Machine 9"
        elif lr_pred_prob_TM10[i][lr_TM10_index] == max(lr_pred_prob_FI[i][lr_FI_index], lr_pred_prob_FG[i][lr_FG_index], lr_pred_prob_GR27[i][lr_GR27_index], lr_pred_prob_LM[i][lr_LM_index], lr_pred_prob_LMM[i][lr_LMM_index], lr_pred_prob_PC[i][lr_PC_index], lr_pred_prob_RG12[i][lr_RG12_index], lr_pred_prob_RG2[i][lr_RG2_index], lr_pred_prob_RG3[i][lr_RG3_index], lr_pred_prob_RGM[i][lr_RGM_index], lr_pred_prob_RGQC[i][lr_RGQC_index], lr_pred_prob_T8[i][lr_T8_index], lr_pred_prob_TM10[i][lr_TM10_index], lr_pred_prob_TM4[i][lr_TM4_index], lr_pred_prob_TM5[i][lr_TM5_index], lr_pred_prob_TM6[i][lr_TM6_index], lr_pred_prob_TM8[i][lr_TM8_index], lr_pred_prob_TM9[i][lr_TM9_index], lr_pred_prob_TMQC[i][lr_TMQC_index], lr_pred_prob_TQC[i][lr_TQC_index]):
            lr_prediction.loc[i] = "Turning & Milling - Machine 10"
        elif lr_pred_prob_TM4[i][lr_TM4_index] == max(lr_pred_prob_FI[i][lr_FI_index], lr_pred_prob_FG[i][lr_FG_index], lr_pred_prob_GR27[i][lr_GR27_index], lr_pred_prob_LM[i][lr_LM_index], lr_pred_prob_LMM[i][lr_LMM_index], lr_pred_prob_PC[i][lr_PC_index], lr_pred_prob_RG12[i][lr_RG12_index], lr_pred_prob_RG2[i][lr_RG2_index], lr_pred_prob_RG3[i][lr_RG3_index], lr_pred_prob_RGM[i][lr_RGM_index], lr_pred_prob_RGQC[i][lr_RGQC_index], lr_pred_prob_T8[i][lr_T8_index], lr_pred_prob_TM10[i][lr_TM10_index], lr_pred_prob_TM4[i][lr_TM4_index], lr_pred_prob_TM5[i][lr_TM5_index], lr_pred_prob_TM6[i][lr_TM6_index], lr_pred_prob_TM8[i][lr_TM8_index], lr_pred_prob_TM9[i][lr_TM9_index], lr_pred_prob_TMQC[i][lr_TMQC_index], lr_pred_prob_TQC[i][lr_TQC_index]):
            lr_prediction.loc[i] = "Turning & Milling - Machine 4"
        elif lr_pred_prob_TM5[i][lr_TM5_index] == max(lr_pred_prob_FI[i][lr_FI_index], lr_pred_prob_FG[i][lr_FG_index], lr_pred_prob_GR27[i][lr_GR27_index], lr_pred_prob_LM[i][lr_LM_index], lr_pred_prob_LMM[i][lr_LMM_index], lr_pred_prob_PC[i][lr_PC_index], lr_pred_prob_RG12[i][lr_RG12_index], lr_pred_prob_RG2[i][lr_RG2_index], lr_pred_prob_RG3[i][lr_RG3_index], lr_pred_prob_RGM[i][lr_RGM_index], lr_pred_prob_RGQC[i][lr_RGQC_index], lr_pred_prob_T8[i][lr_T8_index], lr_pred_prob_TM10[i][lr_TM10_index], lr_pred_prob_TM4[i][lr_TM4_index], lr_pred_prob_TM5[i][lr_TM5_index], lr_pred_prob_TM6[i][lr_TM6_index], lr_pred_prob_TM8[i][lr_TM8_index], lr_pred_prob_TM9[i][lr_TM9_index], lr_pred_prob_TMQC[i][lr_TMQC_index], lr_pred_prob_TQC[i][lr_TQC_index]):
            lr_prediction.loc[i] = "Turning & Milling - Machine 5"
        elif lr_pred_prob_TM6[i][lr_TM6_index] == max(lr_pred_prob_FI[i][lr_FI_index], lr_pred_prob_FG[i][lr_FG_index], lr_pred_prob_GR27[i][lr_GR27_index], lr_pred_prob_LM[i][lr_LM_index], lr_pred_prob_LMM[i][lr_LMM_index], lr_pred_prob_PC[i][lr_PC_index], lr_pred_prob_RG12[i][lr_RG12_index], lr_pred_prob_RG2[i][lr_RG2_index], lr_pred_prob_RG3[i][lr_RG3_index], lr_pred_prob_RGM[i][lr_RGM_index], lr_pred_prob_RGQC[i][lr_RGQC_index], lr_pred_prob_T8[i][lr_T8_index], lr_pred_prob_TM10[i][lr_TM10_index], lr_pred_prob_TM4[i][lr_TM4_index], lr_pred_prob_TM5[i][lr_TM5_index], lr_pred_prob_TM6[i][lr_TM6_index], lr_pred_prob_TM8[i][lr_TM8_index], lr_pred_prob_TM9[i][lr_TM9_index], lr_pred_prob_TMQC[i][lr_TMQC_index], lr_pred_prob_TQC[i][lr_TQC_index]):
            lr_prediction.loc[i] = "Turning & Milling - Machine 6"
        elif lr_pred_prob_TM8[i][lr_TM8_index] == max(lr_pred_prob_FI[i][lr_FI_index], lr_pred_prob_FG[i][lr_FG_index], lr_pred_prob_GR27[i][lr_GR27_index], lr_pred_prob_LM[i][lr_LM_index], lr_pred_prob_LMM[i][lr_LMM_index], lr_pred_prob_PC[i][lr_PC_index], lr_pred_prob_RG12[i][lr_RG12_index], lr_pred_prob_RG2[i][lr_RG2_index], lr_pred_prob_RG3[i][lr_RG3_index], lr_pred_prob_RGM[i][lr_RGM_index], lr_pred_prob_RGQC[i][lr_RGQC_index], lr_pred_prob_T8[i][lr_T8_index], lr_pred_prob_TM10[i][lr_TM10_index], lr_pred_prob_TM4[i][lr_TM4_index], lr_pred_prob_TM5[i][lr_TM5_index], lr_pred_prob_TM6[i][lr_TM6_index], lr_pred_prob_TM8[i][lr_TM8_index], lr_pred_prob_TM9[i][lr_TM9_index], lr_pred_prob_TMQC[i][lr_TMQC_index], lr_pred_prob_TQC[i][lr_TQC_index]):
            lr_prediction.loc[i] = "Turning & Milling - Machine 8"
        elif lr_pred_prob_TM9[i][lr_TM9_index] == max(lr_pred_prob_FI[i][lr_FI_index], lr_pred_prob_FG[i][lr_FG_index], lr_pred_prob_GR27[i][lr_GR27_index], lr_pred_prob_LM[i][lr_LM_index], lr_pred_prob_LMM[i][lr_LMM_index], lr_pred_prob_PC[i][lr_PC_index], lr_pred_prob_RG12[i][lr_RG12_index], lr_pred_prob_RG2[i][lr_RG2_index], lr_pred_prob_RG3[i][lr_RG3_index], lr_pred_prob_RGM[i][lr_RGM_index], lr_pred_prob_RGQC[i][lr_RGQC_index], lr_pred_prob_T8[i][lr_T8_index], lr_pred_prob_TM10[i][lr_TM10_index], lr_pred_prob_TM4[i][lr_TM4_index], lr_pred_prob_TM5[i][lr_TM5_index], lr_pred_prob_TM6[i][lr_TM6_index], lr_pred_prob_TM8[i][lr_TM8_index], lr_pred_prob_TM9[i][lr_TM9_index], lr_pred_prob_TMQC[i][lr_TMQC_index], lr_pred_prob_TQC[i][lr_TQC_index]):
            lr_prediction.loc[i] = "Turning & Milling - Machine 9"
        elif lr_pred_prob_TMQC[i][lr_TMQC_index] == max(lr_pred_prob_FI[i][lr_FI_index], lr_pred_prob_FG[i][lr_FG_index], lr_pred_prob_GR27[i][lr_GR27_index], lr_pred_prob_LM[i][lr_LM_index], lr_pred_prob_LMM[i][lr_LMM_index], lr_pred_prob_PC[i][lr_PC_index], lr_pred_prob_RG12[i][lr_RG12_index], lr_pred_prob_RG2[i][lr_RG2_index], lr_pred_prob_RG3[i][lr_RG3_index], lr_pred_prob_RGM[i][lr_RGM_index], lr_pred_prob_RGQC[i][lr_RGQC_index], lr_pred_prob_T8[i][lr_T8_index], lr_pred_prob_TM10[i][lr_TM10_index], lr_pred_prob_TM4[i][lr_TM4_index], lr_pred_prob_TM5[i][lr_TM5_index], lr_pred_prob_TM6[i][lr_TM6_index], lr_pred_prob_TM8[i][lr_TM8_index], lr_pred_prob_TM9[i][lr_TM9_index], lr_pred_prob_TMQC[i][lr_TMQC_index], lr_pred_prob_TQC[i][lr_TQC_index]):
            lr_prediction.loc[i] = "Turning & Milling Q.C."
        elif lr_pred_prob_TQC[i][lr_TQC_index] == max(lr_pred_prob_FI[i][lr_FI_index], lr_pred_prob_FG[i][lr_FG_index], lr_pred_prob_GR27[i][lr_GR27_index], lr_pred_prob_LM[i][lr_LM_index], lr_pred_prob_LMM[i][lr_LMM_index], lr_pred_prob_PC[i][lr_PC_index], lr_pred_prob_RG12[i][lr_RG12_index], lr_pred_prob_RG2[i][lr_RG2_index], lr_pred_prob_RG3[i][lr_RG3_index], lr_pred_prob_RGM[i][lr_RGM_index], lr_pred_prob_RGQC[i][lr_RGQC_index], lr_pred_prob_T8[i][lr_T8_index], lr_pred_prob_TM10[i][lr_TM10_index], lr_pred_prob_TM4[i][lr_TM4_index], lr_pred_prob_TM5[i][lr_TM5_index], lr_pred_prob_TM6[i][lr_TM6_index], lr_pred_prob_TM8[i][lr_TM8_index], lr_pred_prob_TM9[i][lr_TM9_index], lr_pred_prob_TMQC[i][lr_TMQC_index], lr_pred_prob_TQC[i][lr_TQC_index]):
            lr_prediction.loc[i] = "Turning Q.C."
        #elif lr_pred_prob_WC13[i][lr_WC13_index] == max(lr_pred_prob_FI[i][lr_FI_index], lr_pred_prob_FG[i][lr_FG_index], lr_pred_prob_GR27[i][lr_GR27_index], lr_pred_prob_LM[i][lr_LM_index], lr_pred_prob_LMM[i][lr_LMM_index], lr_pred_prob_PC[i][lr_PC_index], lr_pred_prob_RG12[i][lr_RG12_index], lr_pred_prob_RG2[i][lr_RG2_index], lr_pred_prob_RG3[i][lr_RG3_index], lr_pred_prob_RGM[i][lr_RGM_index], lr_pred_prob_RGQC[i][lr_RGQC_index], lr_pred_prob_T8[i][lr_T8_index], lr_pred_prob_TM10[i][lr_TM10_index], lr_pred_prob_TM4[i][lr_TM4_index], lr_pred_prob_TM5[i][lr_TM5_index], lr_pred_prob_TM6[i][lr_TM6_index], lr_pred_prob_TM8[i][lr_TM8_index], lr_pred_prob_TM9[i][lr_TM9_index], lr_pred_prob_TMQC[i][lr_TMQC_index], lr_pred_prob_TQC[i][lr_TQC_index]):
        #    lr_prediction.loc[i] = "Wire Cut - Machine 13"


    def get_precision(lr_conf_matrix):
        lr_tp_1 = lr_conf_matrix[0][0]
        lr_tp_2 = lr_conf_matrix[1][1]
        lr_tp_3 = lr_conf_matrix[2][2]
        lr_tp_4 = lr_conf_matrix[3][3]
        lr_tp_5 = lr_conf_matrix[4][4]
        lr_tp_6 = lr_conf_matrix[5][5]
        lr_tp_7 = lr_conf_matrix[6][6]
        lr_tp_8 = lr_conf_matrix[7][7]
        lr_tp_9 = lr_conf_matrix[8][8]
        lr_tp_10 = lr_conf_matrix[9][9]
        lr_tp_11 = lr_conf_matrix[10][10]
        lr_tp_12 = lr_conf_matrix[11][11]
        lr_tp_13 = lr_conf_matrix[12][12]
        lr_tp_14 = lr_conf_matrix[13][13]
        lr_tp_15 = lr_conf_matrix[14][14]
        lr_tp_16 = lr_conf_matrix[15][15]
        lr_tp_17 = lr_conf_matrix[16][16]
        lr_tp_18 = lr_conf_matrix[17][17]
        lr_tp_19 = lr_conf_matrix[18][18]
        lr_tp_20 = lr_conf_matrix[19][19]
        #lr_tp_21 = lr_conf_matrix[20][20]
        #lr_tp_22 = lr_conf_matrix[21][21]
        #lr_tp_23 = lr_conf_matrix[22][22]
        #lr_tp_24 = lr_conf_matrix[23][23]
        #lr_tp_25 = lr_conf_matrix[24][24]
        #lr_tp_26 = lr_conf_matrix[25][25]
        #lr_tp_27 = lr_conf_matrix[26][26]
        #lr_tp_28 = lr_conf_matrix[27][27]
        #lr_tp_29 = lr_conf_matrix[28][28]

        lr_fp_1 = lr_conf_matrix[1][0] + lr_conf_matrix[2][0] + lr_conf_matrix[3][0] + lr_conf_matrix[4][0] + lr_conf_matrix[5][0] + lr_conf_matrix[6][0] + lr_conf_matrix[7][0] + lr_conf_matrix[8][0] + lr_conf_matrix[9][0] + lr_conf_matrix[10][0] + lr_conf_matrix[11][0] + lr_conf_matrix[12][0] + lr_conf_matrix[13][0] + lr_conf_matrix[14][0] + lr_conf_matrix[15][0] + lr_conf_matrix[16][0] + lr_conf_matrix[17][0] + lr_conf_matrix[18][0] + lr_conf_matrix[19][0]
        lr_fp_2 = lr_conf_matrix[0][1] + lr_conf_matrix[2][1] + lr_conf_matrix[3][1] + lr_conf_matrix[4][1] + lr_conf_matrix[5][1] + lr_conf_matrix[6][1] + lr_conf_matrix[7][1] + lr_conf_matrix[8][1] + lr_conf_matrix[9][1] + lr_conf_matrix[10][1] + lr_conf_matrix[11][1] + lr_conf_matrix[12][1] + lr_conf_matrix[13][1] + lr_conf_matrix[14][1] + lr_conf_matrix[15][1] + lr_conf_matrix[16][1] + lr_conf_matrix[17][1] + lr_conf_matrix[18][1] + lr_conf_matrix[19][1]
        lr_fp_3 = lr_conf_matrix[0][2] + lr_conf_matrix[1][2] + lr_conf_matrix[3][2] + lr_conf_matrix[4][2] + lr_conf_matrix[5][2] + lr_conf_matrix[6][2] + lr_conf_matrix[7][2] + lr_conf_matrix[8][2] + lr_conf_matrix[9][2] + lr_conf_matrix[10][2] + lr_conf_matrix[11][2] + lr_conf_matrix[12][2] + lr_conf_matrix[13][2] + lr_conf_matrix[14][2] + lr_conf_matrix[15][2] + lr_conf_matrix[16][2] + lr_conf_matrix[17][2] + lr_conf_matrix[18][2] + lr_conf_matrix[19][2]
        lr_fp_4 = lr_conf_matrix[0][3] + lr_conf_matrix[1][3] + lr_conf_matrix[2][3] + lr_conf_matrix[4][3] + lr_conf_matrix[5][3] + lr_conf_matrix[6][3] + lr_conf_matrix[7][3] + lr_conf_matrix[8][3] + lr_conf_matrix[9][3] + lr_conf_matrix[10][3] + lr_conf_matrix[11][3] + lr_conf_matrix[12][3] + lr_conf_matrix[13][3] + lr_conf_matrix[14][3] + lr_conf_matrix[15][3] + lr_conf_matrix[16][3] + lr_conf_matrix[17][3] + lr_conf_matrix[18][3] + lr_conf_matrix[19][3]
        lr_fp_5 = lr_conf_matrix[0][4] + lr_conf_matrix[1][4] + lr_conf_matrix[2][4] + lr_conf_matrix[3][4] + lr_conf_matrix[5][4] + lr_conf_matrix[6][4] + lr_conf_matrix[7][4] + lr_conf_matrix[8][4] + lr_conf_matrix[9][4] + lr_conf_matrix[10][4] + lr_conf_matrix[11][4] + lr_conf_matrix[12][4] + lr_conf_matrix[13][4] + lr_conf_matrix[14][4] + lr_conf_matrix[15][4] + lr_conf_matrix[16][4] + lr_conf_matrix[17][4] + lr_conf_matrix[18][4] + lr_conf_matrix[19][4]
        lr_fp_6 = lr_conf_matrix[0][5] + lr_conf_matrix[1][5] + lr_conf_matrix[2][5] + lr_conf_matrix[3][5] + lr_conf_matrix[4][5] + lr_conf_matrix[6][5] + lr_conf_matrix[7][5] + lr_conf_matrix[8][5] + lr_conf_matrix[9][5] + lr_conf_matrix[10][5] + lr_conf_matrix[11][5] + lr_conf_matrix[12][5] + lr_conf_matrix[13][5] + lr_conf_matrix[14][5] + lr_conf_matrix[15][5] + lr_conf_matrix[16][5] + lr_conf_matrix[17][5] + lr_conf_matrix[18][5] + lr_conf_matrix[19][5]
        lr_fp_7 = lr_conf_matrix[0][6] + lr_conf_matrix[1][6] + lr_conf_matrix[2][6] + lr_conf_matrix[3][6] + lr_conf_matrix[4][6] + lr_conf_matrix[5][6] + lr_conf_matrix[7][6] + lr_conf_matrix[8][6] + lr_conf_matrix[9][6] + lr_conf_matrix[10][6] + lr_conf_matrix[11][6] + lr_conf_matrix[12][6] + lr_conf_matrix[13][6] + lr_conf_matrix[14][6] + lr_conf_matrix[15][6] + lr_conf_matrix[16][6] + lr_conf_matrix[17][6] + lr_conf_matrix[18][6] + lr_conf_matrix[19][6]
        lr_fp_8 = lr_conf_matrix[0][7] + lr_conf_matrix[1][7] + lr_conf_matrix[2][7] + lr_conf_matrix[3][7] + lr_conf_matrix[4][7] + lr_conf_matrix[5][7] + lr_conf_matrix[6][7] + lr_conf_matrix[8][7] + lr_conf_matrix[9][7] + lr_conf_matrix[10][7] + lr_conf_matrix[11][7] + lr_conf_matrix[12][7] + lr_conf_matrix[13][7] + lr_conf_matrix[14][7] + lr_conf_matrix[15][7] + lr_conf_matrix[16][7] + lr_conf_matrix[17][7] + lr_conf_matrix[18][7] + lr_conf_matrix[19][7]
        lr_fp_9 = lr_conf_matrix[0][8] + lr_conf_matrix[1][8] + lr_conf_matrix[2][8] + lr_conf_matrix[3][8] + lr_conf_matrix[4][8] + lr_conf_matrix[5][8] + lr_conf_matrix[6][8] + lr_conf_matrix[7][8] + lr_conf_matrix[9][8] + lr_conf_matrix[10][8] + lr_conf_matrix[11][8] + lr_conf_matrix[12][8] + lr_conf_matrix[13][8] + lr_conf_matrix[14][8] + lr_conf_matrix[15][8] + lr_conf_matrix[16][8] + lr_conf_matrix[17][8] + lr_conf_matrix[18][8] + lr_conf_matrix[19][8]
        lr_fp_10 = lr_conf_matrix[0][9] + lr_conf_matrix[1][9] + lr_conf_matrix[2][9] + lr_conf_matrix[3][9] + lr_conf_matrix[4][9] + lr_conf_matrix[5][9] + lr_conf_matrix[6][9] + lr_conf_matrix[7][9] + lr_conf_matrix[8][9] + lr_conf_matrix[10][9] + lr_conf_matrix[11][9] + lr_conf_matrix[12][9] + lr_conf_matrix[13][9] + lr_conf_matrix[14][9] + lr_conf_matrix[15][9] + lr_conf_matrix[16][9] + lr_conf_matrix[17][9] + lr_conf_matrix[18][9] + lr_conf_matrix[19][9]
        lr_fp_11 = lr_conf_matrix[0][10] + lr_conf_matrix[1][10] + lr_conf_matrix[2][10] + lr_conf_matrix[3][10] + lr_conf_matrix[4][10] + lr_conf_matrix[5][10] + lr_conf_matrix[6][10] + lr_conf_matrix[7][10] + lr_conf_matrix[8][10] + lr_conf_matrix[9][10] + lr_conf_matrix[11][10] + lr_conf_matrix[12][10] + lr_conf_matrix[13][10] + lr_conf_matrix[14][10] + lr_conf_matrix[15][10] + lr_conf_matrix[16][10] + lr_conf_matrix[17][10] + lr_conf_matrix[18][10] + lr_conf_matrix[19][10]
        lr_fp_12 = lr_conf_matrix[0][11] + lr_conf_matrix[1][11] + lr_conf_matrix[2][11] + lr_conf_matrix[3][11] + lr_conf_matrix[4][11] + lr_conf_matrix[5][11] + lr_conf_matrix[6][11] + lr_conf_matrix[7][11] + lr_conf_matrix[8][11] + lr_conf_matrix[9][11] + lr_conf_matrix[10][11] + lr_conf_matrix[12][11] + lr_conf_matrix[13][11] + lr_conf_matrix[14][11] + lr_conf_matrix[15][11] + lr_conf_matrix[16][11] + lr_conf_matrix[17][11] + lr_conf_matrix[18][11] + lr_conf_matrix[19][11]
        lr_fp_13 = lr_conf_matrix[0][12] + lr_conf_matrix[1][12] + lr_conf_matrix[2][12] + lr_conf_matrix[3][12] + lr_conf_matrix[4][12] + lr_conf_matrix[5][12] + lr_conf_matrix[6][12] + lr_conf_matrix[7][12] + lr_conf_matrix[8][12] + lr_conf_matrix[9][12] + lr_conf_matrix[10][12] + lr_conf_matrix[11][12] + lr_conf_matrix[13][12] + lr_conf_matrix[14][12] + lr_conf_matrix[15][12] + lr_conf_matrix[16][12] + lr_conf_matrix[17][12] + lr_conf_matrix[18][12] + lr_conf_matrix[19][12]
        lr_fp_14 = lr_conf_matrix[0][13] + lr_conf_matrix[1][13] + lr_conf_matrix[2][13] + lr_conf_matrix[3][13] + lr_conf_matrix[4][13] + lr_conf_matrix[5][13] + lr_conf_matrix[6][13] + lr_conf_matrix[7][13] + lr_conf_matrix[8][13] + lr_conf_matrix[9][13] + lr_conf_matrix[10][13] + lr_conf_matrix[11][13] + lr_conf_matrix[12][13] + lr_conf_matrix[14][13] + lr_conf_matrix[15][13] + lr_conf_matrix[16][13] + lr_conf_matrix[17][13] + lr_conf_matrix[18][13] + lr_conf_matrix[19][13]
        lr_fp_15 = lr_conf_matrix[0][14] + lr_conf_matrix[1][14] + lr_conf_matrix[2][14] + lr_conf_matrix[3][14] + lr_conf_matrix[4][14] + lr_conf_matrix[5][14] + lr_conf_matrix[6][14] + lr_conf_matrix[7][14] + lr_conf_matrix[8][14] + lr_conf_matrix[9][14] + lr_conf_matrix[10][14] + lr_conf_matrix[11][14] + lr_conf_matrix[12][14] + lr_conf_matrix[13][14] + lr_conf_matrix[15][14] + lr_conf_matrix[16][14] + lr_conf_matrix[17][14] + lr_conf_matrix[18][14] + lr_conf_matrix[19][14]
        lr_fp_16 = lr_conf_matrix[0][15] + lr_conf_matrix[1][15] + lr_conf_matrix[2][15] + lr_conf_matrix[3][15] + lr_conf_matrix[4][15] + lr_conf_matrix[5][15] + lr_conf_matrix[6][15] + lr_conf_matrix[7][15] + lr_conf_matrix[8][15] + lr_conf_matrix[9][15] + lr_conf_matrix[10][15] + lr_conf_matrix[11][15] + lr_conf_matrix[12][15] + lr_conf_matrix[13][15] + lr_conf_matrix[14][15] + lr_conf_matrix[16][15] + lr_conf_matrix[17][15] + lr_conf_matrix[18][15] + lr_conf_matrix[19][15]
        lr_fp_17 = lr_conf_matrix[0][16] + lr_conf_matrix[1][16] + lr_conf_matrix[2][16] + lr_conf_matrix[3][16] + lr_conf_matrix[4][16] + lr_conf_matrix[5][16] + lr_conf_matrix[6][16] + lr_conf_matrix[7][16] + lr_conf_matrix[8][16] + lr_conf_matrix[9][16] + lr_conf_matrix[10][16] + lr_conf_matrix[11][16] + lr_conf_matrix[12][16] + lr_conf_matrix[13][16] + lr_conf_matrix[14][16] + lr_conf_matrix[15][16] + lr_conf_matrix[17][16] + lr_conf_matrix[18][16] + lr_conf_matrix[19][16]
        lr_fp_18 = lr_conf_matrix[0][17] + lr_conf_matrix[1][17] + lr_conf_matrix[2][17] + lr_conf_matrix[3][17] + lr_conf_matrix[4][17] + lr_conf_matrix[5][17] + lr_conf_matrix[6][17] + lr_conf_matrix[7][17] + lr_conf_matrix[8][17] + lr_conf_matrix[9][17] + lr_conf_matrix[10][17] + lr_conf_matrix[11][17] + lr_conf_matrix[12][17] + lr_conf_matrix[13][17] + lr_conf_matrix[14][17] + lr_conf_matrix[15][17] + lr_conf_matrix[16][17] + lr_conf_matrix[18][17] + lr_conf_matrix[19][17]
        lr_fp_19 = lr_conf_matrix[0][18] + lr_conf_matrix[1][18] + lr_conf_matrix[2][18] + lr_conf_matrix[3][18] + lr_conf_matrix[4][18] + lr_conf_matrix[5][18] + lr_conf_matrix[6][18] + lr_conf_matrix[7][18] + lr_conf_matrix[8][18] + lr_conf_matrix[9][18] + lr_conf_matrix[10][18] + lr_conf_matrix[11][18] + lr_conf_matrix[12][18] + lr_conf_matrix[13][18] + lr_conf_matrix[14][18] + lr_conf_matrix[15][18] + lr_conf_matrix[16][18] + lr_conf_matrix[17][18] + lr_conf_matrix[19][18]
        lr_fp_20 = lr_conf_matrix[0][19] + lr_conf_matrix[1][19] + lr_conf_matrix[2][19] + lr_conf_matrix[3][19] + lr_conf_matrix[4][19] + lr_conf_matrix[5][19] + lr_conf_matrix[6][19] + lr_conf_matrix[7][19] + lr_conf_matrix[8][19] + lr_conf_matrix[9][19] + lr_conf_matrix[10][19] + lr_conf_matrix[11][19] + lr_conf_matrix[12][19] + lr_conf_matrix[13][19] + lr_conf_matrix[14][19] + lr_conf_matrix[15][19] + lr_conf_matrix[16][19] + lr_conf_matrix[17][19] + lr_conf_matrix[18][19]
        #lr_fp_21 = lr_conf_matrix[0][20] + lr_conf_matrix[1][20] + lr_conf_matrix[2][20] + lr_conf_matrix[3][20] + lr_conf_matrix[4][20] + lr_conf_matrix[5][20] + lr_conf_matrix[6][20] + lr_conf_matrix[7][20] + lr_conf_matrix[8][20] + lr_conf_matrix[9][20] + lr_conf_matrix[10][20] + lr_conf_matrix[11][20] + lr_conf_matrix[12][20] + lr_conf_matrix[13][20] + lr_conf_matrix[14][20] + lr_conf_matrix[15][20] + lr_conf_matrix[16][20] + lr_conf_matrix[17][20] + lr_conf_matrix[18][20] + lr_conf_matrix[19][20] + lr_conf_matrix[21][20] + lr_conf_matrix[22][20] + lr_conf_matrix[23][20] + lr_conf_matrix[24][20] + lr_conf_matrix[25][20] + lr_conf_matrix[26][20] + lr_conf_matrix[27][20] + lr_conf_matrix[28][20]
        #lr_fp_22 = lr_conf_matrix[0][21] + lr_conf_matrix[1][21] + lr_conf_matrix[2][21] + lr_conf_matrix[3][21] + lr_conf_matrix[4][21] + lr_conf_matrix[5][21] + lr_conf_matrix[6][21] + lr_conf_matrix[7][21] + lr_conf_matrix[8][21] + lr_conf_matrix[9][21] + lr_conf_matrix[10][21] + lr_conf_matrix[11][21] + lr_conf_matrix[12][21] + lr_conf_matrix[13][21] + lr_conf_matrix[14][21] + lr_conf_matrix[15][21] + lr_conf_matrix[16][21] + lr_conf_matrix[17][21] + lr_conf_matrix[18][21] + lr_conf_matrix[19][21] + lr_conf_matrix[20][21] + lr_conf_matrix[22][21] + lr_conf_matrix[23][21] + lr_conf_matrix[24][21] + lr_conf_matrix[25][21] + lr_conf_matrix[26][21] + lr_conf_matrix[27][21] + lr_conf_matrix[28][21]
        #lr_fp_23 = lr_conf_matrix[0][22] + lr_conf_matrix[1][22] + lr_conf_matrix[2][22] + lr_conf_matrix[3][22] + lr_conf_matrix[4][22] + lr_conf_matrix[5][22] + lr_conf_matrix[6][22] + lr_conf_matrix[7][22] + lr_conf_matrix[8][22] + lr_conf_matrix[9][22] + lr_conf_matrix[10][22] + lr_conf_matrix[11][22] + lr_conf_matrix[12][22] + lr_conf_matrix[13][22] + lr_conf_matrix[14][22] + lr_conf_matrix[15][22] + lr_conf_matrix[16][22] + lr_conf_matrix[17][22] + lr_conf_matrix[18][22] + lr_conf_matrix[19][22] + lr_conf_matrix[20][22] + lr_conf_matrix[21][22] + lr_conf_matrix[23][22] + lr_conf_matrix[24][22] + lr_conf_matrix[25][22] + lr_conf_matrix[26][22] + lr_conf_matrix[27][22] + lr_conf_matrix[28][22]
        #lr_fp_24 = lr_conf_matrix[0][23] + lr_conf_matrix[1][23] + lr_conf_matrix[2][23] + lr_conf_matrix[3][23] + lr_conf_matrix[4][23] + lr_conf_matrix[5][23] + lr_conf_matrix[6][23] + lr_conf_matrix[7][23] + lr_conf_matrix[8][23] + lr_conf_matrix[9][23] + lr_conf_matrix[10][23] + lr_conf_matrix[11][23] + lr_conf_matrix[12][23] + lr_conf_matrix[13][23] + lr_conf_matrix[14][23] + lr_conf_matrix[15][23] + lr_conf_matrix[16][23] + lr_conf_matrix[17][23] + lr_conf_matrix[18][23] + lr_conf_matrix[19][23] + lr_conf_matrix[20][23] + lr_conf_matrix[21][23] + lr_conf_matrix[22][23] + lr_conf_matrix[24][23] + lr_conf_matrix[25][23] + lr_conf_matrix[26][23] + lr_conf_matrix[27][23] + lr_conf_matrix[28][23]
        #lr_fp_25 = lr_conf_matrix[0][24] + lr_conf_matrix[1][24] + lr_conf_matrix[2][24] + lr_conf_matrix[3][24] + lr_conf_matrix[4][24] + lr_conf_matrix[5][24] + lr_conf_matrix[6][24] + lr_conf_matrix[7][24] + lr_conf_matrix[8][24] + lr_conf_matrix[9][24] + lr_conf_matrix[10][24] + lr_conf_matrix[11][24] + lr_conf_matrix[12][24] + lr_conf_matrix[13][24] + lr_conf_matrix[14][24] + lr_conf_matrix[15][24] + lr_conf_matrix[16][24] + lr_conf_matrix[17][24] + lr_conf_matrix[18][24] + lr_conf_matrix[19][24] + lr_conf_matrix[20][24] + lr_conf_matrix[21][24] + lr_conf_matrix[22][24] + lr_conf_matrix[23][24] + lr_conf_matrix[25][24] + lr_conf_matrix[26][24] + lr_conf_matrix[27][24] + lr_conf_matrix[28][24]
        #lr_fp_26 = lr_conf_matrix[0][25] + lr_conf_matrix[1][25] + lr_conf_matrix[2][25] + lr_conf_matrix[3][25] + lr_conf_matrix[4][25] + lr_conf_matrix[5][25] + lr_conf_matrix[6][25] + lr_conf_matrix[7][25] + lr_conf_matrix[8][25] + lr_conf_matrix[9][25] + lr_conf_matrix[10][25] + lr_conf_matrix[11][25] + lr_conf_matrix[12][25] + lr_conf_matrix[13][25] + lr_conf_matrix[14][25] + lr_conf_matrix[15][25] + lr_conf_matrix[16][25] + lr_conf_matrix[17][25] + lr_conf_matrix[18][25] + lr_conf_matrix[19][25] + lr_conf_matrix[20][25] + lr_conf_matrix[21][25] + lr_conf_matrix[22][25] + lr_conf_matrix[23][25] + lr_conf_matrix[24][25] + lr_conf_matrix[26][25] + lr_conf_matrix[27][25] + lr_conf_matrix[28][25]
        #lr_fp_27 = lr_conf_matrix[0][26] + lr_conf_matrix[1][26] + lr_conf_matrix[2][26] + lr_conf_matrix[3][26] + lr_conf_matrix[4][26] + lr_conf_matrix[5][26] + lr_conf_matrix[6][26] + lr_conf_matrix[7][26] + lr_conf_matrix[8][26] + lr_conf_matrix[9][26] + lr_conf_matrix[10][26] + lr_conf_matrix[11][26] + lr_conf_matrix[12][26] + lr_conf_matrix[13][26] + lr_conf_matrix[14][26] + lr_conf_matrix[15][26] + lr_conf_matrix[16][26] + lr_conf_matrix[17][26] + lr_conf_matrix[18][26] + lr_conf_matrix[19][26] + lr_conf_matrix[20][26] + lr_conf_matrix[21][26] + lr_conf_matrix[22][26] + lr_conf_matrix[23][26] + lr_conf_matrix[24][26] + lr_conf_matrix[25][26] + lr_conf_matrix[27][26] + lr_conf_matrix[28][26]
        #lr_fp_28 = lr_conf_matrix[0][27] + lr_conf_matrix[1][27] + lr_conf_matrix[2][27] + lr_conf_matrix[3][27] + lr_conf_matrix[4][27] + lr_conf_matrix[5][27] + lr_conf_matrix[6][27] + lr_conf_matrix[7][27] + lr_conf_matrix[8][27] + lr_conf_matrix[9][27] + lr_conf_matrix[10][27] + lr_conf_matrix[11][27] + lr_conf_matrix[12][27] + lr_conf_matrix[13][27] + lr_conf_matrix[14][27] + lr_conf_matrix[15][27] + lr_conf_matrix[16][27] + lr_conf_matrix[17][27] + lr_conf_matrix[18][27] + lr_conf_matrix[19][27] + lr_conf_matrix[20][27] + lr_conf_matrix[21][27] + lr_conf_matrix[22][27] + lr_conf_matrix[23][27] + lr_conf_matrix[24][27] + lr_conf_matrix[25][27] + lr_conf_matrix[26][27] + lr_conf_matrix[28][27]
        #lr_fp_29 = lr_conf_matrix[0][28] + lr_conf_matrix[1][28] + lr_conf_matrix[2][28] + lr_conf_matrix[3][28] + lr_conf_matrix[4][28] + lr_conf_matrix[5][28] + lr_conf_matrix[6][28] + lr_conf_matrix[7][28] + lr_conf_matrix[8][28] + lr_conf_matrix[9][28] + lr_conf_matrix[10][28] + lr_conf_matrix[11][28] + lr_conf_matrix[12][28] + lr_conf_matrix[13][28] + lr_conf_matrix[14][28] + lr_conf_matrix[15][28] + lr_conf_matrix[16][28] + lr_conf_matrix[17][28] + lr_conf_matrix[18][28] + lr_conf_matrix[19][28] + lr_conf_matrix[20][28] + lr_conf_matrix[21][28] + lr_conf_matrix[22][28] + lr_conf_matrix[23][28] + lr_conf_matrix[24][28] + lr_conf_matrix[25][28] + lr_conf_matrix[26][28] + lr_conf_matrix[27][28]

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
        if lr_tp_6 + lr_fp_6 == 0:
            lr_precision_6 = 0
        else:
            lr_precision_6 = lr_tp_6 / (lr_tp_6 + lr_fp_6)
        if lr_tp_7 + lr_fp_7 == 0:
            lr_precision_7 = 0
        else:
            lr_precision_7 = lr_tp_7 / (lr_tp_7 + lr_fp_7)
        if lr_tp_8 + lr_fp_8 == 0:
            lr_precision_8 = 0
        else:
            lr_precision_8 = lr_tp_8 / (lr_tp_8 + lr_fp_8)
        if lr_tp_9 + lr_fp_9 == 0:
            lr_precision_9 = 0
        else:
            lr_precision_9 = lr_tp_9 / (lr_tp_9 + lr_fp_9)
        if lr_tp_10 + lr_fp_10 == 0:
            lr_precision_10 = 0
        else:
            lr_precision_10 = lr_tp_10 / (lr_tp_10 + lr_fp_10)
        if lr_tp_11 + lr_fp_11 == 0:
            lr_precision_11 = 0
        else:
            lr_precision_11 = lr_tp_11 / (lr_tp_11 + lr_fp_11)
        if lr_tp_12 + lr_fp_12 == 0:
            lr_precision_12 = 0
        else:
            lr_precision_12 = lr_tp_12 / (lr_tp_12 + lr_fp_12)
        if lr_tp_13 + lr_fp_13 == 0:
            lr_precision_13 = 0
        else:
            lr_precision_13 = lr_tp_13 / (lr_tp_13 + lr_fp_13)
        if lr_tp_14 + lr_fp_14 == 0:
            lr_precision_14 = 0
        else:
            lr_precision_14 = lr_tp_14 / (lr_tp_14 + lr_fp_14)
        if lr_tp_15 + lr_fp_15 == 0:
            lr_precision_15 = 0
        else:
            lr_precision_15 = lr_tp_15 / (lr_tp_15 + lr_fp_15)
        if lr_tp_16 + lr_fp_16 == 0:
            lr_precision_16 = 0
        else:
            lr_precision_16 = lr_tp_16 / (lr_tp_16 + lr_fp_16)
        if lr_tp_17 + lr_fp_17 == 0:
            lr_precision_17 = 0
        else:
            lr_precision_17 = lr_tp_17 / (lr_tp_17 + lr_fp_17)
        if lr_tp_18 + lr_fp_18 == 0:
            lr_precision_18 = 0
        else:
            lr_precision_18 = lr_tp_18 / (lr_tp_18 + lr_fp_18)
        if lr_tp_19 + lr_fp_19 == 0:
            lr_precision_19 = 0
        else:
            lr_precision_19 = lr_tp_19 / (lr_tp_19 + lr_fp_19)
        if lr_tp_20 + lr_fp_20 == 0:
            lr_precision_20 = 0
        else:
            lr_precision_20 = lr_tp_20 / (lr_tp_20 + lr_fp_20)
        '''
        if lr_tp_21 + lr_fp_21 == 0:
            lr_precision_21 = 0
        else:
            lr_precision_21 = lr_tp_21 / (lr_tp_21 + lr_fp_21)
        if lr_tp_22 + lr_fp_22 == 0:
            lr_precision_22 = 0
        else:
            lr_precision_22 = lr_tp_22 / (lr_tp_22 + lr_fp_22)
        if lr_tp_23 + lr_fp_23 == 0:
            lr_precision_23 = 0
        else:
            lr_precision_23 = lr_tp_23 / (lr_tp_23 + lr_fp_23)
        if lr_tp_24 + lr_fp_24 == 0:
            lr_precision_24 = 0
        else:
            lr_precision_24 = lr_tp_24 / (lr_tp_24 + lr_fp_24)
        if lr_tp_25 + lr_fp_25 == 0:
            lr_precision_25 = 0
        else:
            lr_precision_25 = lr_tp_25 / (lr_tp_25 + lr_fp_25)
        if lr_tp_26 + lr_fp_26 == 0:
            lr_precision_26 = 0
        else:
            lr_precision_26 = lr_tp_26 / (lr_tp_26 + lr_fp_26)
        if lr_tp_27 + lr_fp_27 == 0:
            lr_precision_27 = 0
        else:
            lr_precision_27 = lr_tp_27 / (lr_tp_27 + lr_fp_27)
        if lr_tp_28 + lr_fp_28 == 0:
            lr_precision_28 = 0
        else:
            lr_precision_28 = lr_tp_28 / (lr_tp_28 + lr_fp_28)
        if lr_tp_29 + lr_fp_29 == 0:
            lr_precision_29 = 0
        else:
            lr_precision_29 = lr_tp_29 / (lr_tp_29 + lr_fp_29)
        '''
        lr_precision_avg = (lr_precision_1 + lr_precision_2 + lr_precision_3 + lr_precision_4 + lr_precision_5 + lr_precision_6 + lr_precision_7 + lr_precision_8 + lr_precision_9 + lr_precision_10 + lr_precision_11 + lr_precision_12 + lr_precision_13 + lr_precision_14 + lr_precision_15 + lr_precision_16 + lr_precision_17 + lr_precision_18 + lr_precision_19 + lr_precision_20) / 20
        return lr_precision_avg


    def get_recall_pen_1(lr_conf_matrix):
        lr_tp_1 = lr_conf_matrix[0][0]
        lr_tp_2 = lr_conf_matrix[1][1]
        lr_tp_3 = lr_conf_matrix[2][2]
        lr_tp_4 = lr_conf_matrix[3][3]
        lr_tp_5 = lr_conf_matrix[4][4]
        lr_tp_6 = lr_conf_matrix[5][5]
        lr_tp_7 = lr_conf_matrix[6][6]
        lr_tp_8 = lr_conf_matrix[7][7]
        lr_tp_9 = lr_conf_matrix[8][8]
        lr_tp_10 = lr_conf_matrix[9][9]
        lr_tp_11 = lr_conf_matrix[10][10]
        lr_tp_12 = lr_conf_matrix[11][11]
        lr_tp_13 = lr_conf_matrix[12][12]
        lr_tp_14 = lr_conf_matrix[13][13]
        lr_tp_15 = lr_conf_matrix[14][14]
        lr_tp_16 = lr_conf_matrix[15][15]
        lr_tp_17 = lr_conf_matrix[16][16]
        lr_tp_18 = lr_conf_matrix[17][17]
        lr_tp_19 = lr_conf_matrix[18][18]
        lr_tp_20 = lr_conf_matrix[19][19]
        #lr_tp_21 = lr_conf_matrix[20][20]
        #lr_tp_22 = lr_conf_matrix[21][21]
        #lr_tp_23 = lr_conf_matrix[22][22]
        #lr_tp_24 = lr_conf_matrix[23][23]
        #lr_tp_25 = lr_conf_matrix[24][24]
        #lr_tp_26 = lr_conf_matrix[25][25]
        #lr_tp_27 = lr_conf_matrix[26][26]
        #lr_tp_28 = lr_conf_matrix[27][27]
        #lr_tp_29 = lr_conf_matrix[28][28]

        lr_fn_1 = lr_conf_matrix[0][1] + lr_conf_matrix[0][2] + lr_conf_matrix[0][3] + lr_conf_matrix[0][4] + lr_conf_matrix[0][5] + lr_conf_matrix[0][6] + lr_conf_matrix[0][7] + lr_conf_matrix[0][8] + lr_conf_matrix[0][9] + lr_conf_matrix[0][10] + lr_conf_matrix[0][11] + lr_conf_matrix[0][12] + lr_conf_matrix[0][13] + lr_conf_matrix[0][14] + lr_conf_matrix[0][15] + lr_conf_matrix[0][16] + lr_conf_matrix[0][17] + lr_conf_matrix[0][18] + lr_conf_matrix[0][19]
        lr_fn_2 = lr_conf_matrix[1][0] + lr_conf_matrix[1][2] + lr_conf_matrix[1][3] + lr_conf_matrix[1][4] + lr_conf_matrix[1][5] + lr_conf_matrix[1][6] + lr_conf_matrix[1][7] + lr_conf_matrix[1][8] + lr_conf_matrix[1][9] + lr_conf_matrix[1][10] + lr_conf_matrix[1][11] + lr_conf_matrix[1][12] + lr_conf_matrix[1][13] + lr_conf_matrix[1][14] + lr_conf_matrix[1][15] + lr_conf_matrix[1][16] + lr_conf_matrix[1][17] + lr_conf_matrix[1][18] + lr_conf_matrix[1][19]
        lr_fn_3 = lr_conf_matrix[2][0] + lr_conf_matrix[2][1] + lr_conf_matrix[2][3] + lr_conf_matrix[2][4] + lr_conf_matrix[2][5] + lr_conf_matrix[2][6] + lr_conf_matrix[2][7] + lr_conf_matrix[2][8] + lr_conf_matrix[2][9] + lr_conf_matrix[2][10] + lr_conf_matrix[2][11] + lr_conf_matrix[2][12] + lr_conf_matrix[2][13] + lr_conf_matrix[2][14] + lr_conf_matrix[2][15] + lr_conf_matrix[2][16] + lr_conf_matrix[2][17] + lr_conf_matrix[2][18] + lr_conf_matrix[2][19]
        lr_fn_4 = lr_conf_matrix[3][0] + lr_conf_matrix[3][1] + lr_conf_matrix[3][2] + lr_conf_matrix[3][4] + lr_conf_matrix[3][5] + lr_conf_matrix[3][6] + lr_conf_matrix[3][7] + lr_conf_matrix[3][8] + lr_conf_matrix[3][9] + lr_conf_matrix[3][10] + lr_conf_matrix[3][11] + lr_conf_matrix[3][12] + lr_conf_matrix[3][13] + lr_conf_matrix[3][14] + lr_conf_matrix[3][15] + lr_conf_matrix[3][16] + lr_conf_matrix[3][17] + lr_conf_matrix[3][18] + lr_conf_matrix[3][19]
        lr_fn_5 = lr_conf_matrix[4][0] + lr_conf_matrix[4][1] + lr_conf_matrix[4][2] + lr_conf_matrix[4][3] + lr_conf_matrix[4][5] + lr_conf_matrix[4][6] + lr_conf_matrix[4][7] + lr_conf_matrix[4][8] + lr_conf_matrix[4][9] + lr_conf_matrix[4][10] + lr_conf_matrix[4][11] + lr_conf_matrix[4][12] + lr_conf_matrix[4][13] + lr_conf_matrix[4][14] + lr_conf_matrix[4][15] + lr_conf_matrix[4][16] + lr_conf_matrix[4][17] + lr_conf_matrix[4][18] + lr_conf_matrix[4][19]
        lr_fn_6 = lr_conf_matrix[5][0] + lr_conf_matrix[5][1] + lr_conf_matrix[5][2] + lr_conf_matrix[5][3] + lr_conf_matrix[5][4] + lr_conf_matrix[5][6] + lr_conf_matrix[5][7] + lr_conf_matrix[5][8] + lr_conf_matrix[5][9] + lr_conf_matrix[5][10] + lr_conf_matrix[5][11] + lr_conf_matrix[5][12] + lr_conf_matrix[5][13] + lr_conf_matrix[5][14] + lr_conf_matrix[5][15] + lr_conf_matrix[5][16] + lr_conf_matrix[5][17] + lr_conf_matrix[5][18] + lr_conf_matrix[5][19]
        lr_fn_7 = lr_conf_matrix[6][0] + lr_conf_matrix[6][1] + lr_conf_matrix[6][2] + lr_conf_matrix[6][3] + lr_conf_matrix[6][4] + lr_conf_matrix[6][5] + lr_conf_matrix[6][7] + lr_conf_matrix[6][8] + lr_conf_matrix[6][9] + lr_conf_matrix[6][10] + lr_conf_matrix[6][11] + lr_conf_matrix[6][12] + lr_conf_matrix[6][13] + lr_conf_matrix[6][14] + lr_conf_matrix[6][15] + lr_conf_matrix[6][16] + lr_conf_matrix[6][17] + lr_conf_matrix[6][18] + lr_conf_matrix[6][19]
        lr_fn_8 = lr_conf_matrix[7][0] + lr_conf_matrix[7][1] + lr_conf_matrix[7][2] + lr_conf_matrix[7][3] + lr_conf_matrix[7][4] + lr_conf_matrix[7][5] + lr_conf_matrix[7][6] + lr_conf_matrix[7][8] + lr_conf_matrix[7][9] + lr_conf_matrix[7][10] + lr_conf_matrix[7][11] + lr_conf_matrix[7][12] + lr_conf_matrix[7][13] + lr_conf_matrix[7][14] + lr_conf_matrix[7][15] + lr_conf_matrix[7][16] + lr_conf_matrix[7][17] + lr_conf_matrix[7][18] + lr_conf_matrix[7][19]
        lr_fn_9 = lr_conf_matrix[8][0] + lr_conf_matrix[8][1] + lr_conf_matrix[8][2] + lr_conf_matrix[8][3] + lr_conf_matrix[8][4] + lr_conf_matrix[8][5] + lr_conf_matrix[8][6] + lr_conf_matrix[8][7] + lr_conf_matrix[8][9] + lr_conf_matrix[8][10] + lr_conf_matrix[8][11] + lr_conf_matrix[8][12] + lr_conf_matrix[8][13] + lr_conf_matrix[8][14] + lr_conf_matrix[8][15] + lr_conf_matrix[8][16] + lr_conf_matrix[8][17] + lr_conf_matrix[8][18] + lr_conf_matrix[8][19]
        lr_fn_10 = lr_conf_matrix[9][0] + lr_conf_matrix[9][1] + lr_conf_matrix[9][2] + lr_conf_matrix[9][3] + lr_conf_matrix[9][4] + lr_conf_matrix[9][5] + lr_conf_matrix[9][6] + lr_conf_matrix[9][7] + lr_conf_matrix[9][8] + lr_conf_matrix[9][10] + lr_conf_matrix[9][11] + lr_conf_matrix[9][12] + lr_conf_matrix[9][13] + lr_conf_matrix[9][14] + lr_conf_matrix[9][15] + lr_conf_matrix[9][16] + lr_conf_matrix[9][17] + lr_conf_matrix[9][18] + lr_conf_matrix[9][19]
        lr_fn_11 = lr_conf_matrix[10][0] + lr_conf_matrix[10][1] + lr_conf_matrix[10][2] + lr_conf_matrix[10][3] + lr_conf_matrix[10][4] + lr_conf_matrix[10][5] + lr_conf_matrix[10][6] + lr_conf_matrix[10][7] + lr_conf_matrix[10][8] + lr_conf_matrix[10][9] + lr_conf_matrix[10][11] + lr_conf_matrix[10][12] + lr_conf_matrix[10][13] + lr_conf_matrix[10][14] + lr_conf_matrix[10][15] + lr_conf_matrix[10][16] + lr_conf_matrix[10][17] + lr_conf_matrix[10][18] + lr_conf_matrix[10][19]
        lr_fn_12 = lr_conf_matrix[11][0] + lr_conf_matrix[11][1] + lr_conf_matrix[11][2] + lr_conf_matrix[11][3] + lr_conf_matrix[11][4] + lr_conf_matrix[11][5] + lr_conf_matrix[11][6] + lr_conf_matrix[11][7] + lr_conf_matrix[11][8] + lr_conf_matrix[11][9] + lr_conf_matrix[11][10] + lr_conf_matrix[11][12] + lr_conf_matrix[11][13] + lr_conf_matrix[11][14] + lr_conf_matrix[11][15] + lr_conf_matrix[11][16] + lr_conf_matrix[11][17] + lr_conf_matrix[11][18] + lr_conf_matrix[11][19]
        lr_fn_13 = lr_conf_matrix[12][0] + lr_conf_matrix[12][1] + lr_conf_matrix[12][2] + lr_conf_matrix[12][3] + lr_conf_matrix[12][4] + lr_conf_matrix[12][5] + lr_conf_matrix[12][6] + lr_conf_matrix[12][7] + lr_conf_matrix[12][8] + lr_conf_matrix[12][9] + lr_conf_matrix[12][10] + lr_conf_matrix[12][11] + lr_conf_matrix[12][13] + lr_conf_matrix[12][14] + lr_conf_matrix[12][15] + lr_conf_matrix[12][16] + lr_conf_matrix[12][17] + lr_conf_matrix[12][18] + lr_conf_matrix[12][19]
        lr_fn_14 = lr_conf_matrix[13][0] + lr_conf_matrix[13][1] + lr_conf_matrix[13][2] + lr_conf_matrix[13][3] + lr_conf_matrix[13][4] + lr_conf_matrix[13][5] + lr_conf_matrix[13][6] + lr_conf_matrix[13][7] + lr_conf_matrix[13][8] + lr_conf_matrix[13][9] + lr_conf_matrix[13][10] + lr_conf_matrix[13][11] + lr_conf_matrix[13][12] + lr_conf_matrix[13][14] + lr_conf_matrix[13][15] + lr_conf_matrix[13][16] + lr_conf_matrix[13][17] + lr_conf_matrix[13][18] + lr_conf_matrix[13][19]
        lr_fn_15 = lr_conf_matrix[14][0] + lr_conf_matrix[14][1] + lr_conf_matrix[14][2] + lr_conf_matrix[14][3] + lr_conf_matrix[14][4] + lr_conf_matrix[14][5] + lr_conf_matrix[14][6] + lr_conf_matrix[14][7] + lr_conf_matrix[14][8] + lr_conf_matrix[14][9] + lr_conf_matrix[14][10] + lr_conf_matrix[14][11] + lr_conf_matrix[14][12] + lr_conf_matrix[14][13] + lr_conf_matrix[14][15] + lr_conf_matrix[14][16] + lr_conf_matrix[14][17] + lr_conf_matrix[14][18] + lr_conf_matrix[14][19]
        lr_fn_16 = lr_conf_matrix[15][0] + lr_conf_matrix[15][1] + lr_conf_matrix[15][2] + lr_conf_matrix[15][3] + lr_conf_matrix[15][4] + lr_conf_matrix[15][5] + lr_conf_matrix[15][6] + lr_conf_matrix[15][7] + lr_conf_matrix[15][8] + lr_conf_matrix[15][9] + lr_conf_matrix[15][10] + lr_conf_matrix[15][11] + lr_conf_matrix[15][12] + lr_conf_matrix[15][13] + lr_conf_matrix[15][14] + lr_conf_matrix[15][16] + lr_conf_matrix[15][17] + lr_conf_matrix[15][18] + lr_conf_matrix[15][19]
        lr_fn_17 = lr_conf_matrix[16][0] + lr_conf_matrix[16][1] + lr_conf_matrix[16][2] + lr_conf_matrix[16][3] + lr_conf_matrix[16][4] + lr_conf_matrix[16][5] + lr_conf_matrix[16][6] + lr_conf_matrix[16][7] + lr_conf_matrix[16][8] + lr_conf_matrix[16][9] + lr_conf_matrix[16][10] + lr_conf_matrix[16][11] + lr_conf_matrix[16][12] + lr_conf_matrix[16][13] + lr_conf_matrix[16][14] + lr_conf_matrix[16][15] + lr_conf_matrix[16][17] + lr_conf_matrix[16][18] + lr_conf_matrix[16][19]
        lr_fn_18 = lr_conf_matrix[17][0] + lr_conf_matrix[17][1] + lr_conf_matrix[17][2] + lr_conf_matrix[17][3] + lr_conf_matrix[17][4] + lr_conf_matrix[17][5] + lr_conf_matrix[17][6] + lr_conf_matrix[17][7] + lr_conf_matrix[17][8] + lr_conf_matrix[17][9] + lr_conf_matrix[17][10] + lr_conf_matrix[17][11] + lr_conf_matrix[17][12] + lr_conf_matrix[17][13] + lr_conf_matrix[17][14] + lr_conf_matrix[17][15] + lr_conf_matrix[17][16] + lr_conf_matrix[17][18] + lr_conf_matrix[17][19]
        lr_fn_19 = lr_conf_matrix[18][0] + lr_conf_matrix[18][1] + lr_conf_matrix[18][2] + lr_conf_matrix[18][3] + lr_conf_matrix[18][4] + lr_conf_matrix[18][5] + lr_conf_matrix[18][6] + lr_conf_matrix[18][7] + lr_conf_matrix[18][8] + lr_conf_matrix[18][9] + lr_conf_matrix[18][10] + lr_conf_matrix[18][11] + lr_conf_matrix[18][12] + lr_conf_matrix[18][13] + lr_conf_matrix[18][14] + lr_conf_matrix[18][15] + lr_conf_matrix[18][16] + lr_conf_matrix[18][17] + lr_conf_matrix[18][19]
        lr_fn_20 = lr_conf_matrix[19][0] + lr_conf_matrix[19][1] + lr_conf_matrix[19][2] + lr_conf_matrix[19][3] + lr_conf_matrix[19][4] + lr_conf_matrix[19][5] + lr_conf_matrix[19][6] + lr_conf_matrix[19][7] + lr_conf_matrix[19][8] + lr_conf_matrix[19][9] + lr_conf_matrix[19][10] + lr_conf_matrix[19][11] + lr_conf_matrix[19][12] + lr_conf_matrix[19][13] + lr_conf_matrix[19][14] + lr_conf_matrix[19][15] + lr_conf_matrix[19][16] + lr_conf_matrix[19][17] + lr_conf_matrix[19][18]
        #lr_fn_21 = lr_conf_matrix[20][0] + lr_conf_matrix[20][1] + lr_conf_matrix[20][2] + lr_conf_matrix[20][3] + lr_conf_matrix[20][4] + lr_conf_matrix[20][5] + lr_conf_matrix[20][6] + lr_conf_matrix[20][7] + lr_conf_matrix[20][8] + lr_conf_matrix[20][9] + lr_conf_matrix[20][10] + lr_conf_matrix[20][11] + lr_conf_matrix[20][12] + lr_conf_matrix[20][13] + lr_conf_matrix[20][14] + lr_conf_matrix[20][15] + lr_conf_matrix[20][16] + lr_conf_matrix[20][17] + lr_conf_matrix[20][18] + lr_conf_matrix[20][19] + lr_conf_matrix[20][21] + lr_conf_matrix[20][22] + lr_conf_matrix[20][23] + lr_conf_matrix[20][24] + lr_conf_matrix[20][25] + lr_conf_matrix[20][26] + lr_conf_matrix[20][27] + lr_conf_matrix[20][28]
        #lr_fn_22 = lr_conf_matrix[21][0] + lr_conf_matrix[21][1] + lr_conf_matrix[21][2] + lr_conf_matrix[21][3] + lr_conf_matrix[21][4] + lr_conf_matrix[21][5] + lr_conf_matrix[21][6] + lr_conf_matrix[21][7] + lr_conf_matrix[21][8] + lr_conf_matrix[21][9] + lr_conf_matrix[21][10] + lr_conf_matrix[21][11] + lr_conf_matrix[21][12] + lr_conf_matrix[21][13] + lr_conf_matrix[21][14] + lr_conf_matrix[21][15] + lr_conf_matrix[21][16] + lr_conf_matrix[21][17] + lr_conf_matrix[21][18] + lr_conf_matrix[21][19] + lr_conf_matrix[21][20] + lr_conf_matrix[21][22] + lr_conf_matrix[21][23] + lr_conf_matrix[21][24] + lr_conf_matrix[21][25] + lr_conf_matrix[21][26] + lr_conf_matrix[21][27] + lr_conf_matrix[21][28]
        #lr_fn_23 = lr_conf_matrix[22][0] + lr_conf_matrix[22][1] + lr_conf_matrix[22][2] + lr_conf_matrix[22][3] + lr_conf_matrix[22][4] + lr_conf_matrix[22][5] + lr_conf_matrix[22][6] + lr_conf_matrix[22][7] + lr_conf_matrix[22][8] + lr_conf_matrix[22][9] + lr_conf_matrix[22][10] + lr_conf_matrix[22][11] + lr_conf_matrix[22][12] + lr_conf_matrix[22][13] + lr_conf_matrix[22][14] + lr_conf_matrix[22][15] + lr_conf_matrix[22][16] + lr_conf_matrix[22][17] + lr_conf_matrix[22][18] + lr_conf_matrix[22][19] + lr_conf_matrix[22][20] + lr_conf_matrix[22][21] + lr_conf_matrix[22][23] + lr_conf_matrix[22][24] + lr_conf_matrix[22][25] + lr_conf_matrix[22][26] + lr_conf_matrix[22][27] + lr_conf_matrix[22][28]
        #lr_fn_24 = lr_conf_matrix[23][0] + lr_conf_matrix[23][1] + lr_conf_matrix[23][2] + lr_conf_matrix[23][3] + lr_conf_matrix[23][4] + lr_conf_matrix[23][5] + lr_conf_matrix[23][6] + lr_conf_matrix[23][7] + lr_conf_matrix[23][8] + lr_conf_matrix[23][9] + lr_conf_matrix[23][10] + lr_conf_matrix[23][11] + lr_conf_matrix[23][12] + lr_conf_matrix[23][13] + lr_conf_matrix[23][14] + lr_conf_matrix[23][15] + lr_conf_matrix[23][16] + lr_conf_matrix[23][17] + lr_conf_matrix[23][18] + lr_conf_matrix[23][19] + lr_conf_matrix[23][20] + lr_conf_matrix[23][21] + lr_conf_matrix[23][22] + lr_conf_matrix[23][24] + lr_conf_matrix[23][25] + lr_conf_matrix[23][26] + lr_conf_matrix[23][27] + lr_conf_matrix[23][28]
        #lr_fn_25 = lr_conf_matrix[24][0] + lr_conf_matrix[24][1] + lr_conf_matrix[24][2] + lr_conf_matrix[24][3] + lr_conf_matrix[24][4] + lr_conf_matrix[24][5] + lr_conf_matrix[24][6] + lr_conf_matrix[24][7] + lr_conf_matrix[24][8] + lr_conf_matrix[24][9] + lr_conf_matrix[24][10] + lr_conf_matrix[24][11] + lr_conf_matrix[24][12] + lr_conf_matrix[24][13] + lr_conf_matrix[24][14] + lr_conf_matrix[24][15] + lr_conf_matrix[24][16] + lr_conf_matrix[24][17] + lr_conf_matrix[24][18] + lr_conf_matrix[24][19] + lr_conf_matrix[24][20] + lr_conf_matrix[24][21] + lr_conf_matrix[24][22] + lr_conf_matrix[24][23] + lr_conf_matrix[24][25] + lr_conf_matrix[24][26] + lr_conf_matrix[24][27] + lr_conf_matrix[24][28]
        #lr_fn_26 = lr_conf_matrix[25][0] + lr_conf_matrix[25][1] + lr_conf_matrix[25][2] + lr_conf_matrix[25][3] + lr_conf_matrix[25][4] + lr_conf_matrix[25][5] + lr_conf_matrix[25][6] + lr_conf_matrix[25][7] + lr_conf_matrix[25][8] + lr_conf_matrix[25][9] + lr_conf_matrix[25][10] + lr_conf_matrix[25][11] + lr_conf_matrix[25][12] + lr_conf_matrix[25][13] + lr_conf_matrix[25][14] + lr_conf_matrix[25][15] + lr_conf_matrix[25][16] + lr_conf_matrix[25][17] + lr_conf_matrix[25][18] + lr_conf_matrix[25][19] + lr_conf_matrix[25][20] + lr_conf_matrix[25][21] + lr_conf_matrix[25][22] + lr_conf_matrix[25][23] + lr_conf_matrix[25][24] + lr_conf_matrix[25][26] + lr_conf_matrix[25][27] + lr_conf_matrix[25][28]
        #lr_fn_27 = lr_conf_matrix[26][0] + lr_conf_matrix[26][1] + lr_conf_matrix[26][2] + lr_conf_matrix[26][3] + lr_conf_matrix[26][4] + lr_conf_matrix[26][5] + lr_conf_matrix[26][6] + lr_conf_matrix[26][7] + lr_conf_matrix[26][8] + lr_conf_matrix[26][9] + lr_conf_matrix[26][10] + lr_conf_matrix[26][11] + lr_conf_matrix[26][12] + lr_conf_matrix[26][13] + lr_conf_matrix[26][14] + lr_conf_matrix[26][15] + lr_conf_matrix[26][16] + lr_conf_matrix[26][17] + lr_conf_matrix[26][18] + lr_conf_matrix[26][19] + lr_conf_matrix[26][20] + lr_conf_matrix[26][21] + lr_conf_matrix[26][22] + lr_conf_matrix[26][23] + lr_conf_matrix[26][24] + lr_conf_matrix[26][25] + lr_conf_matrix[26][27] + lr_conf_matrix[26][28]
        #lr_fn_28 = lr_conf_matrix[27][0] + lr_conf_matrix[27][1] + lr_conf_matrix[27][2] + lr_conf_matrix[27][3] + lr_conf_matrix[27][4] + lr_conf_matrix[27][5] + lr_conf_matrix[27][6] + lr_conf_matrix[27][7] + lr_conf_matrix[27][8] + lr_conf_matrix[27][9] + lr_conf_matrix[27][10] + lr_conf_matrix[27][11] + lr_conf_matrix[27][12] + lr_conf_matrix[27][13] + lr_conf_matrix[27][14] + lr_conf_matrix[27][15] + lr_conf_matrix[27][16] + lr_conf_matrix[27][17] + lr_conf_matrix[27][18] + lr_conf_matrix[27][19] + lr_conf_matrix[27][20] + lr_conf_matrix[27][21] + lr_conf_matrix[27][22] + lr_conf_matrix[27][23] + lr_conf_matrix[27][24] + lr_conf_matrix[27][25] + lr_conf_matrix[27][26] + lr_conf_matrix[27][28]
        #lr_fn_29 = lr_conf_matrix[28][0] + lr_conf_matrix[28][1] + lr_conf_matrix[28][2] + lr_conf_matrix[28][3] + lr_conf_matrix[28][4] + lr_conf_matrix[28][5] + lr_conf_matrix[28][6] + lr_conf_matrix[28][7] + lr_conf_matrix[28][8] + lr_conf_matrix[28][9] + lr_conf_matrix[28][10] + lr_conf_matrix[28][11] + lr_conf_matrix[28][12] + lr_conf_matrix[28][13] + lr_conf_matrix[28][14] + lr_conf_matrix[28][15] + lr_conf_matrix[28][16] + lr_conf_matrix[28][17] + lr_conf_matrix[28][18] + lr_conf_matrix[28][19] + lr_conf_matrix[28][20] + lr_conf_matrix[28][21] + lr_conf_matrix[28][22] + lr_conf_matrix[28][23] + lr_conf_matrix[28][24] + lr_conf_matrix[28][25] + lr_conf_matrix[28][26] + lr_conf_matrix[28][27]

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
            lr_recall_5 = lr_tp_5 / (lr_tp_5 + lr_fn_5)
        if lr_tp_6 + lr_fn_6 == 0:
            lr_recall_6 = 0
        else:
            lr_recall_6 = lr_tp_6 / (lr_tp_6 + lr_fn_6)
        if lr_tp_7 + lr_fn_7 == 0:
            lr_recall_7 = 0
        else:
            lr_recall_7 = lr_tp_7 / (lr_tp_7 + lr_fn_7)
        if lr_tp_8 + lr_fn_8 == 0:
            lr_recall_8 = 0
        else:
            lr_recall_8 = lr_tp_8 / (lr_tp_8 + lr_fn_8)
        if lr_tp_9 + lr_fn_9 == 0:
            lr_recall_9 = 0
        else:
            lr_recall_9 = lr_tp_9 / (lr_tp_9 + lr_fn_9)
        if lr_tp_10 + lr_fn_10 == 0:
            lr_recall_10 = 0
        else:
            lr_recall_10 = lr_tp_10 / (lr_tp_10 + lr_fn_10)
        if lr_tp_11 + lr_fn_11 == 0:
            lr_recall_11 = 0
        else:
            lr_recall_11 = lr_tp_11 / (lr_tp_11 + lr_fn_11)
        if lr_tp_12 + lr_fn_12 == 0:
            lr_recall_12 = 0
        else:
            lr_recall_12 = lr_tp_12 / (lr_tp_12 + lr_fn_12)
        if lr_tp_13 + lr_fn_13 == 0:
            lr_recall_13 = 0
        else:
            lr_recall_13 = lr_tp_13 / (lr_tp_13 + lr_fn_13)
        if lr_tp_14 + lr_fn_14 == 0:
            lr_recall_14 = 0
        else:
            lr_recall_14 = lr_tp_14 / (lr_tp_14 + lr_fn_14)
        if lr_tp_15 + lr_fn_15 == 0:
            lr_recall_15 = 0
        else:
            lr_recall_15 = lr_tp_15 / (lr_tp_15 + lr_fn_15)
        if lr_tp_16 + lr_fn_16 == 0:
            lr_recall_16 = 0
        else:
            lr_recall_16 = lr_tp_16 / (lr_tp_16 + lr_fn_16)
        if lr_tp_17 + lr_fn_17 == 0:
            lr_recall_17 = 0
        else:
            lr_recall_17 = lr_tp_17 / (lr_tp_17 + lr_fn_17)
        if lr_tp_18 + lr_fn_18 == 0:
            lr_recall_18 = 0
        else:
            lr_recall_18 = lr_tp_18 / (lr_tp_18 + lr_fn_18)
        if lr_tp_19 + lr_fn_19 == 0:
            lr_recall_19 = 0
        else:
            lr_recall_19 = lr_tp_19 / (lr_tp_19 + lr_fn_19)
        if lr_tp_20 + lr_fn_20 == 0:
            lr_recall_20 = 0
        else:
            lr_recall_20 = lr_tp_20 / (lr_tp_20 + lr_fn_20)
        '''
        if lr_tp_21 + lr_fn_21 == 0:
            lr_recall_21 = 0
        else:
            lr_recall_21 = lr_tp_21 / (lr_tp_21 + lr_fn_21)
        if lr_tp_22 + lr_fn_22 == 0:
            lr_recall_22 = 0
        else:
            lr_recall_22 = lr_tp_22 / (lr_tp_22 + lr_fn_22)
        if lr_tp_23 + lr_fn_23 == 0:
            lr_recall_23 = 0
        else:
            lr_recall_23 = lr_tp_23 / (lr_tp_23 + lr_fn_23)
        if lr_tp_24 + lr_fn_24 == 0:
            lr_recall_24 = 0
        else:
            lr_recall_24 = lr_tp_24 / (lr_tp_24 + lr_fn_24)
        if lr_tp_25 + lr_fn_25 == 0:
            lr_recall_25 = 0
        else:
            lr_recall_25 = lr_tp_25 / (lr_tp_25 + lr_fn_25)
        if lr_tp_26 + lr_fn_26 == 0:
            lr_recall_26 = 0
        else:
            lr_recall_26 = lr_tp_26 / (lr_tp_26 + lr_fn_26)
        if lr_tp_27 + lr_fn_27 == 0:
            lr_recall_27 = 0
        else:
            lr_recall_27 = lr_tp_27 / (lr_tp_27 + lr_fn_27)
        if lr_tp_28 + lr_fn_28 == 0:
            lr_recall_28 = 0
        else:
            lr_recall_28 = lr_tp_28 / (lr_tp_28 + lr_fn_28)
        if lr_tp_29 + lr_fn_29 == 0:
            lr_recall_29 = 0
        else:
            lr_recall_29 = lr_tp_29 / (lr_tp_29 + lr_fn_29)
        '''
        lr_recall_avg_pen_1 = (
                                 lr_recall_1 + lr_recall_2 + lr_recall_3 + lr_recall_4 + lr_recall_5 + lr_recall_6 + lr_recall_7 + lr_recall_8 + lr_recall_9 + lr_recall_10 + lr_recall_11 + lr_recall_12 + lr_recall_13 + lr_recall_14 + lr_recall_15 + lr_recall_16 + lr_recall_17 + lr_recall_18 + lr_recall_19 + lr_recall_20) / (20+1-1)
        return lr_recall_avg_pen_1

    def get_recall_pen_5(lr_conf_matrix):
        lr_tp_1 = lr_conf_matrix[0][0]
        lr_tp_2 = lr_conf_matrix[1][1]
        lr_tp_3 = lr_conf_matrix[2][2]
        lr_tp_4 = lr_conf_matrix[3][3]
        lr_tp_5 = lr_conf_matrix[4][4]
        lr_tp_6 = lr_conf_matrix[5][5]
        lr_tp_7 = lr_conf_matrix[6][6]
        lr_tp_8 = lr_conf_matrix[7][7]
        lr_tp_9 = lr_conf_matrix[8][8]
        lr_tp_10 = lr_conf_matrix[9][9]
        lr_tp_11 = lr_conf_matrix[10][10]
        lr_tp_12 = lr_conf_matrix[11][11]
        lr_tp_13 = lr_conf_matrix[12][12]
        lr_tp_14 = lr_conf_matrix[13][13]
        lr_tp_15 = lr_conf_matrix[14][14]
        lr_tp_16 = lr_conf_matrix[15][15]
        lr_tp_17 = lr_conf_matrix[16][16]
        lr_tp_18 = lr_conf_matrix[17][17]
        lr_tp_19 = lr_conf_matrix[18][18]
        lr_tp_20 = lr_conf_matrix[19][19]
        #lr_tp_21 = lr_conf_matrix[20][20]
        #lr_tp_22 = lr_conf_matrix[21][21]
        #lr_tp_23 = lr_conf_matrix[22][22]
        #lr_tp_24 = lr_conf_matrix[23][23]
        #lr_tp_25 = lr_conf_matrix[24][24]
        #lr_tp_26 = lr_conf_matrix[25][25]
        #lr_tp_27 = lr_conf_matrix[26][26]
        #lr_tp_28 = lr_conf_matrix[27][27]
        #lr_tp_29 = lr_conf_matrix[28][28]

        lr_fn_1 = lr_conf_matrix[0][1] + lr_conf_matrix[0][2] + lr_conf_matrix[0][3] + lr_conf_matrix[0][4] + lr_conf_matrix[0][5] + lr_conf_matrix[0][6] + lr_conf_matrix[0][7] + lr_conf_matrix[0][8] + lr_conf_matrix[0][9] + lr_conf_matrix[0][10] + lr_conf_matrix[0][11] + lr_conf_matrix[0][12] + lr_conf_matrix[0][13] + lr_conf_matrix[0][14] + lr_conf_matrix[0][15] + lr_conf_matrix[0][16] + lr_conf_matrix[0][17] + lr_conf_matrix[0][18] + lr_conf_matrix[0][19]
        lr_fn_2 = lr_conf_matrix[1][0] + lr_conf_matrix[1][2] + lr_conf_matrix[1][3] + lr_conf_matrix[1][4] + lr_conf_matrix[1][5] + lr_conf_matrix[1][6] + lr_conf_matrix[1][7] + lr_conf_matrix[1][8] + lr_conf_matrix[1][9] + lr_conf_matrix[1][10] + lr_conf_matrix[1][11] + lr_conf_matrix[1][12] + lr_conf_matrix[1][13] + lr_conf_matrix[1][14] + lr_conf_matrix[1][15] + lr_conf_matrix[1][16] + lr_conf_matrix[1][17] + lr_conf_matrix[1][18] + lr_conf_matrix[1][19]
        lr_fn_3 = lr_conf_matrix[2][0] + lr_conf_matrix[2][1] + lr_conf_matrix[2][3] + lr_conf_matrix[2][4] + lr_conf_matrix[2][5] + lr_conf_matrix[2][6] + lr_conf_matrix[2][7] + lr_conf_matrix[2][8] + lr_conf_matrix[2][9] + lr_conf_matrix[2][10] + lr_conf_matrix[2][11] + lr_conf_matrix[2][12] + lr_conf_matrix[2][13] + lr_conf_matrix[2][14] + lr_conf_matrix[2][15] + lr_conf_matrix[2][16] + lr_conf_matrix[2][17] + lr_conf_matrix[2][18] + lr_conf_matrix[2][19]
        lr_fn_4 = lr_conf_matrix[3][0] + lr_conf_matrix[3][1] + lr_conf_matrix[3][2] + lr_conf_matrix[3][4] + lr_conf_matrix[3][5] + lr_conf_matrix[3][6] + lr_conf_matrix[3][7] + lr_conf_matrix[3][8] + lr_conf_matrix[3][9] + lr_conf_matrix[3][10] + lr_conf_matrix[3][11] + lr_conf_matrix[3][12] + lr_conf_matrix[3][13] + lr_conf_matrix[3][14] + lr_conf_matrix[3][15] + lr_conf_matrix[3][16] + lr_conf_matrix[3][17] + lr_conf_matrix[3][18] + lr_conf_matrix[3][19]
        lr_fn_5 = lr_conf_matrix[4][0] + lr_conf_matrix[4][1] + lr_conf_matrix[4][2] + lr_conf_matrix[4][3] + lr_conf_matrix[4][5] + lr_conf_matrix[4][6] + lr_conf_matrix[4][7] + lr_conf_matrix[4][8] + lr_conf_matrix[4][9] + lr_conf_matrix[4][10] + lr_conf_matrix[4][11] + lr_conf_matrix[4][12] + lr_conf_matrix[4][13] + lr_conf_matrix[4][14] + lr_conf_matrix[4][15] + lr_conf_matrix[4][16] + lr_conf_matrix[4][17] + lr_conf_matrix[4][18] + lr_conf_matrix[4][19]
        lr_fn_6 = lr_conf_matrix[5][0] + lr_conf_matrix[5][1] + lr_conf_matrix[5][2] + lr_conf_matrix[5][3] + lr_conf_matrix[5][4] + lr_conf_matrix[5][6] + lr_conf_matrix[5][7] + lr_conf_matrix[5][8] + lr_conf_matrix[5][9] + lr_conf_matrix[5][10] + lr_conf_matrix[5][11] + lr_conf_matrix[5][12] + lr_conf_matrix[5][13] + lr_conf_matrix[5][14] + lr_conf_matrix[5][15] + lr_conf_matrix[5][16] + lr_conf_matrix[5][17] + lr_conf_matrix[5][18] + lr_conf_matrix[5][19]
        lr_fn_7 = lr_conf_matrix[6][0] + lr_conf_matrix[6][1] + lr_conf_matrix[6][2] + lr_conf_matrix[6][3] + lr_conf_matrix[6][4] + lr_conf_matrix[6][5] + lr_conf_matrix[6][7] + lr_conf_matrix[6][8] + lr_conf_matrix[6][9] + lr_conf_matrix[6][10] + lr_conf_matrix[6][11] + lr_conf_matrix[6][12] + lr_conf_matrix[6][13] + lr_conf_matrix[6][14] + lr_conf_matrix[6][15] + lr_conf_matrix[6][16] + lr_conf_matrix[6][17] + lr_conf_matrix[6][18] + lr_conf_matrix[6][19]
        lr_fn_8 = lr_conf_matrix[7][0] + lr_conf_matrix[7][1] + lr_conf_matrix[7][2] + lr_conf_matrix[7][3] + lr_conf_matrix[7][4] + lr_conf_matrix[7][5] + lr_conf_matrix[7][6] + lr_conf_matrix[7][8] + lr_conf_matrix[7][9] + lr_conf_matrix[7][10] + lr_conf_matrix[7][11] + lr_conf_matrix[7][12] + lr_conf_matrix[7][13] + lr_conf_matrix[7][14] + lr_conf_matrix[7][15] + lr_conf_matrix[7][16] + lr_conf_matrix[7][17] + lr_conf_matrix[7][18] + lr_conf_matrix[7][19]
        lr_fn_9 = lr_conf_matrix[8][0] + lr_conf_matrix[8][1] + lr_conf_matrix[8][2] + lr_conf_matrix[8][3] + lr_conf_matrix[8][4] + lr_conf_matrix[8][5] + lr_conf_matrix[8][6] + lr_conf_matrix[8][7] + lr_conf_matrix[8][9] + lr_conf_matrix[8][10] + lr_conf_matrix[8][11] + lr_conf_matrix[8][12] + lr_conf_matrix[8][13] + lr_conf_matrix[8][14] + lr_conf_matrix[8][15] + lr_conf_matrix[8][16] + lr_conf_matrix[8][17] + lr_conf_matrix[8][18] + lr_conf_matrix[8][19]
        lr_fn_10 = lr_conf_matrix[9][0] + lr_conf_matrix[9][1] + lr_conf_matrix[9][2] + lr_conf_matrix[9][3] + lr_conf_matrix[9][4] + lr_conf_matrix[9][5] + lr_conf_matrix[9][6] + lr_conf_matrix[9][7] + lr_conf_matrix[9][8] + lr_conf_matrix[9][10] + lr_conf_matrix[9][11] + lr_conf_matrix[9][12] + lr_conf_matrix[9][13] + lr_conf_matrix[9][14] + lr_conf_matrix[9][15] + lr_conf_matrix[9][16] + lr_conf_matrix[9][17] + lr_conf_matrix[9][18] + lr_conf_matrix[9][19]
        lr_fn_11 = lr_conf_matrix[10][0] + lr_conf_matrix[10][1] + lr_conf_matrix[10][2] + lr_conf_matrix[10][3] + lr_conf_matrix[10][4] + lr_conf_matrix[10][5] + lr_conf_matrix[10][6] + lr_conf_matrix[10][7] + lr_conf_matrix[10][8] + lr_conf_matrix[10][9] + lr_conf_matrix[10][11] + lr_conf_matrix[10][12] + lr_conf_matrix[10][13] + lr_conf_matrix[10][14] + lr_conf_matrix[10][15] + lr_conf_matrix[10][16] + lr_conf_matrix[10][17] + lr_conf_matrix[10][18] + lr_conf_matrix[10][19]
        lr_fn_12 = lr_conf_matrix[11][0] + lr_conf_matrix[11][1] + lr_conf_matrix[11][2] + lr_conf_matrix[11][3] + lr_conf_matrix[11][4] + lr_conf_matrix[11][5] + lr_conf_matrix[11][6] + lr_conf_matrix[11][7] + lr_conf_matrix[11][8] + lr_conf_matrix[11][9] + lr_conf_matrix[11][10] + lr_conf_matrix[11][12] + lr_conf_matrix[11][13] + lr_conf_matrix[11][14] + lr_conf_matrix[11][15] + lr_conf_matrix[11][16] + lr_conf_matrix[11][17] + lr_conf_matrix[11][18] + lr_conf_matrix[11][19]
        lr_fn_13 = lr_conf_matrix[12][0] + lr_conf_matrix[12][1] + lr_conf_matrix[12][2] + lr_conf_matrix[12][3] + lr_conf_matrix[12][4] + lr_conf_matrix[12][5] + lr_conf_matrix[12][6] + lr_conf_matrix[12][7] + lr_conf_matrix[12][8] + lr_conf_matrix[12][9] + lr_conf_matrix[12][10] + lr_conf_matrix[12][11] + lr_conf_matrix[12][13] + lr_conf_matrix[12][14] + lr_conf_matrix[12][15] + lr_conf_matrix[12][16] + lr_conf_matrix[12][17] + lr_conf_matrix[12][18] + lr_conf_matrix[12][19]
        lr_fn_14 = lr_conf_matrix[13][0] + lr_conf_matrix[13][1] + lr_conf_matrix[13][2] + lr_conf_matrix[13][3] + lr_conf_matrix[13][4] + lr_conf_matrix[13][5] + lr_conf_matrix[13][6] + lr_conf_matrix[13][7] + lr_conf_matrix[13][8] + lr_conf_matrix[13][9] + lr_conf_matrix[13][10] + lr_conf_matrix[13][11] + lr_conf_matrix[13][12] + lr_conf_matrix[13][14] + lr_conf_matrix[13][15] + lr_conf_matrix[13][16] + lr_conf_matrix[13][17] + lr_conf_matrix[13][18] + lr_conf_matrix[13][19]
        lr_fn_15 = lr_conf_matrix[14][0] + lr_conf_matrix[14][1] + lr_conf_matrix[14][2] + lr_conf_matrix[14][3] + lr_conf_matrix[14][4] + lr_conf_matrix[14][5] + lr_conf_matrix[14][6] + lr_conf_matrix[14][7] + lr_conf_matrix[14][8] + lr_conf_matrix[14][9] + lr_conf_matrix[14][10] + lr_conf_matrix[14][11] + lr_conf_matrix[14][12] + lr_conf_matrix[14][13] + lr_conf_matrix[14][15] + lr_conf_matrix[14][16] + lr_conf_matrix[14][17] + lr_conf_matrix[14][18] + lr_conf_matrix[14][19]
        lr_fn_16 = lr_conf_matrix[15][0] + lr_conf_matrix[15][1] + lr_conf_matrix[15][2] + lr_conf_matrix[15][3] + lr_conf_matrix[15][4] + lr_conf_matrix[15][5] + lr_conf_matrix[15][6] + lr_conf_matrix[15][7] + lr_conf_matrix[15][8] + lr_conf_matrix[15][9] + lr_conf_matrix[15][10] + lr_conf_matrix[15][11] + lr_conf_matrix[15][12] + lr_conf_matrix[15][13] + lr_conf_matrix[15][14] + lr_conf_matrix[15][16] + lr_conf_matrix[15][17] + lr_conf_matrix[15][18] + lr_conf_matrix[15][19]
        lr_fn_17 = lr_conf_matrix[16][0] + lr_conf_matrix[16][1] + lr_conf_matrix[16][2] + lr_conf_matrix[16][3] + lr_conf_matrix[16][4] + lr_conf_matrix[16][5] + lr_conf_matrix[16][6] + lr_conf_matrix[16][7] + lr_conf_matrix[16][8] + lr_conf_matrix[16][9] + lr_conf_matrix[16][10] + lr_conf_matrix[16][11] + lr_conf_matrix[16][12] + lr_conf_matrix[16][13] + lr_conf_matrix[16][14] + lr_conf_matrix[16][15] + lr_conf_matrix[16][17] + lr_conf_matrix[16][18] + lr_conf_matrix[16][19]
        lr_fn_18 = lr_conf_matrix[17][0] + lr_conf_matrix[17][1] + lr_conf_matrix[17][2] + lr_conf_matrix[17][3] + lr_conf_matrix[17][4] + lr_conf_matrix[17][5] + lr_conf_matrix[17][6] + lr_conf_matrix[17][7] + lr_conf_matrix[17][8] + lr_conf_matrix[17][9] + lr_conf_matrix[17][10] + lr_conf_matrix[17][11] + lr_conf_matrix[17][12] + lr_conf_matrix[17][13] + lr_conf_matrix[17][14] + lr_conf_matrix[17][15] + lr_conf_matrix[17][16] + lr_conf_matrix[17][18] + lr_conf_matrix[17][19]
        lr_fn_19 = lr_conf_matrix[18][0] + lr_conf_matrix[18][1] + lr_conf_matrix[18][2] + lr_conf_matrix[18][3] + lr_conf_matrix[18][4] + lr_conf_matrix[18][5] + lr_conf_matrix[18][6] + lr_conf_matrix[18][7] + lr_conf_matrix[18][8] + lr_conf_matrix[18][9] + lr_conf_matrix[18][10] + lr_conf_matrix[18][11] + lr_conf_matrix[18][12] + lr_conf_matrix[18][13] + lr_conf_matrix[18][14] + lr_conf_matrix[18][15] + lr_conf_matrix[18][16] + lr_conf_matrix[18][17] + lr_conf_matrix[18][19]
        lr_fn_20 = lr_conf_matrix[19][0] + lr_conf_matrix[19][1] + lr_conf_matrix[19][2] + lr_conf_matrix[19][3] + lr_conf_matrix[19][4] + lr_conf_matrix[19][5] + lr_conf_matrix[19][6] + lr_conf_matrix[19][7] + lr_conf_matrix[19][8] + lr_conf_matrix[19][9] + lr_conf_matrix[19][10] + lr_conf_matrix[19][11] + lr_conf_matrix[19][12] + lr_conf_matrix[19][13] + lr_conf_matrix[19][14] + lr_conf_matrix[19][15] + lr_conf_matrix[19][16] + lr_conf_matrix[19][17] + lr_conf_matrix[19][18]
        #lr_fn_21 = lr_conf_matrix[20][0] + lr_conf_matrix[20][1] + lr_conf_matrix[20][2] + lr_conf_matrix[20][3] + lr_conf_matrix[20][4] + lr_conf_matrix[20][5] + lr_conf_matrix[20][6] + lr_conf_matrix[20][7] + lr_conf_matrix[20][8] + lr_conf_matrix[20][9] + lr_conf_matrix[20][10] + lr_conf_matrix[20][11] + lr_conf_matrix[20][12] + lr_conf_matrix[20][13] + lr_conf_matrix[20][14] + lr_conf_matrix[20][15] + lr_conf_matrix[20][16] + lr_conf_matrix[20][17] + lr_conf_matrix[20][18] + lr_conf_matrix[20][19] + lr_conf_matrix[20][21] + lr_conf_matrix[20][22] + lr_conf_matrix[20][23] + lr_conf_matrix[20][24] + lr_conf_matrix[20][25] + lr_conf_matrix[20][26] + lr_conf_matrix[20][27] + lr_conf_matrix[20][28]
        #lr_fn_22 = lr_conf_matrix[21][0] + lr_conf_matrix[21][1] + lr_conf_matrix[21][2] + lr_conf_matrix[21][3] + lr_conf_matrix[21][4] + lr_conf_matrix[21][5] + lr_conf_matrix[21][6] + lr_conf_matrix[21][7] + lr_conf_matrix[21][8] + lr_conf_matrix[21][9] + lr_conf_matrix[21][10] + lr_conf_matrix[21][11] + lr_conf_matrix[21][12] + lr_conf_matrix[21][13] + lr_conf_matrix[21][14] + lr_conf_matrix[21][15] + lr_conf_matrix[21][16] + lr_conf_matrix[21][17] + lr_conf_matrix[21][18] + lr_conf_matrix[21][19] + lr_conf_matrix[21][20] + lr_conf_matrix[21][22] + lr_conf_matrix[21][23] + lr_conf_matrix[21][24] + lr_conf_matrix[21][25] + lr_conf_matrix[21][26] + lr_conf_matrix[21][27] + lr_conf_matrix[21][28]
        #lr_fn_23 = lr_conf_matrix[22][0] + lr_conf_matrix[22][1] + lr_conf_matrix[22][2] + lr_conf_matrix[22][3] + lr_conf_matrix[22][4] + lr_conf_matrix[22][5] + lr_conf_matrix[22][6] + lr_conf_matrix[22][7] + lr_conf_matrix[22][8] + lr_conf_matrix[22][9] + lr_conf_matrix[22][10] + lr_conf_matrix[22][11] + lr_conf_matrix[22][12] + lr_conf_matrix[22][13] + lr_conf_matrix[22][14] + lr_conf_matrix[22][15] + lr_conf_matrix[22][16] + lr_conf_matrix[22][17] + lr_conf_matrix[22][18] + lr_conf_matrix[22][19] + lr_conf_matrix[22][20] + lr_conf_matrix[22][21] + lr_conf_matrix[22][23] + lr_conf_matrix[22][24] + lr_conf_matrix[22][25] + lr_conf_matrix[22][26] + lr_conf_matrix[22][27] + lr_conf_matrix[22][28]
        #lr_fn_24 = lr_conf_matrix[23][0] + lr_conf_matrix[23][1] + lr_conf_matrix[23][2] + lr_conf_matrix[23][3] + lr_conf_matrix[23][4] + lr_conf_matrix[23][5] + lr_conf_matrix[23][6] + lr_conf_matrix[23][7] + lr_conf_matrix[23][8] + lr_conf_matrix[23][9] + lr_conf_matrix[23][10] + lr_conf_matrix[23][11] + lr_conf_matrix[23][12] + lr_conf_matrix[23][13] + lr_conf_matrix[23][14] + lr_conf_matrix[23][15] + lr_conf_matrix[23][16] + lr_conf_matrix[23][17] + lr_conf_matrix[23][18] + lr_conf_matrix[23][19] + lr_conf_matrix[23][20] + lr_conf_matrix[23][21] + lr_conf_matrix[23][22] + lr_conf_matrix[23][24] + lr_conf_matrix[23][25] + lr_conf_matrix[23][26] + lr_conf_matrix[23][27] + lr_conf_matrix[23][28]
        #lr_fn_25 = lr_conf_matrix[24][0] + lr_conf_matrix[24][1] + lr_conf_matrix[24][2] + lr_conf_matrix[24][3] + lr_conf_matrix[24][4] + lr_conf_matrix[24][5] + lr_conf_matrix[24][6] + lr_conf_matrix[24][7] + lr_conf_matrix[24][8] + lr_conf_matrix[24][9] + lr_conf_matrix[24][10] + lr_conf_matrix[24][11] + lr_conf_matrix[24][12] + lr_conf_matrix[24][13] + lr_conf_matrix[24][14] + lr_conf_matrix[24][15] + lr_conf_matrix[24][16] + lr_conf_matrix[24][17] + lr_conf_matrix[24][18] + lr_conf_matrix[24][19] + lr_conf_matrix[24][20] + lr_conf_matrix[24][21] + lr_conf_matrix[24][22] + lr_conf_matrix[24][23] + lr_conf_matrix[24][25] + lr_conf_matrix[24][26] + lr_conf_matrix[24][27] + lr_conf_matrix[24][28]
        #lr_fn_26 = lr_conf_matrix[25][0] + lr_conf_matrix[25][1] + lr_conf_matrix[25][2] + lr_conf_matrix[25][3] + lr_conf_matrix[25][4] + lr_conf_matrix[25][5] + lr_conf_matrix[25][6] + lr_conf_matrix[25][7] + lr_conf_matrix[25][8] + lr_conf_matrix[25][9] + lr_conf_matrix[25][10] + lr_conf_matrix[25][11] + lr_conf_matrix[25][12] + lr_conf_matrix[25][13] + lr_conf_matrix[25][14] + lr_conf_matrix[25][15] + lr_conf_matrix[25][16] + lr_conf_matrix[25][17] + lr_conf_matrix[25][18] + lr_conf_matrix[25][19] + lr_conf_matrix[25][20] + lr_conf_matrix[25][21] + lr_conf_matrix[25][22] + lr_conf_matrix[25][23] + lr_conf_matrix[25][24] + lr_conf_matrix[25][26] + lr_conf_matrix[25][27] + lr_conf_matrix[25][28]
        #lr_fn_27 = lr_conf_matrix[26][0] + lr_conf_matrix[26][1] + lr_conf_matrix[26][2] + lr_conf_matrix[26][3] + lr_conf_matrix[26][4] + lr_conf_matrix[26][5] + lr_conf_matrix[26][6] + lr_conf_matrix[26][7] + lr_conf_matrix[26][8] + lr_conf_matrix[26][9] + lr_conf_matrix[26][10] + lr_conf_matrix[26][11] + lr_conf_matrix[26][12] + lr_conf_matrix[26][13] + lr_conf_matrix[26][14] + lr_conf_matrix[26][15] + lr_conf_matrix[26][16] + lr_conf_matrix[26][17] + lr_conf_matrix[26][18] + lr_conf_matrix[26][19] + lr_conf_matrix[26][20] + lr_conf_matrix[26][21] + lr_conf_matrix[26][22] + lr_conf_matrix[26][23] + lr_conf_matrix[26][24] + lr_conf_matrix[26][25] + lr_conf_matrix[26][27] + lr_conf_matrix[26][28]
        #lr_fn_28 = lr_conf_matrix[27][0] + lr_conf_matrix[27][1] + lr_conf_matrix[27][2] + lr_conf_matrix[27][3] + lr_conf_matrix[27][4] + lr_conf_matrix[27][5] + lr_conf_matrix[27][6] + lr_conf_matrix[27][7] + lr_conf_matrix[27][8] + lr_conf_matrix[27][9] + lr_conf_matrix[27][10] + lr_conf_matrix[27][11] + lr_conf_matrix[27][12] + lr_conf_matrix[27][13] + lr_conf_matrix[27][14] + lr_conf_matrix[27][15] + lr_conf_matrix[27][16] + lr_conf_matrix[27][17] + lr_conf_matrix[27][18] + lr_conf_matrix[27][19] + lr_conf_matrix[27][20] + lr_conf_matrix[27][21] + lr_conf_matrix[27][22] + lr_conf_matrix[27][23] + lr_conf_matrix[27][24] + lr_conf_matrix[27][25] + lr_conf_matrix[27][26] + lr_conf_matrix[27][28]
        #lr_fn_29 = lr_conf_matrix[28][0] + lr_conf_matrix[28][1] + lr_conf_matrix[28][2] + lr_conf_matrix[28][3] + lr_conf_matrix[28][4] + lr_conf_matrix[28][5] + lr_conf_matrix[28][6] + lr_conf_matrix[28][7] + lr_conf_matrix[28][8] + lr_conf_matrix[28][9] + lr_conf_matrix[28][10] + lr_conf_matrix[28][11] + lr_conf_matrix[28][12] + lr_conf_matrix[28][13] + lr_conf_matrix[28][14] + lr_conf_matrix[28][15] + lr_conf_matrix[28][16] + lr_conf_matrix[28][17] + lr_conf_matrix[28][18] + lr_conf_matrix[28][19] + lr_conf_matrix[28][20] + lr_conf_matrix[28][21] + lr_conf_matrix[28][22] + lr_conf_matrix[28][23] + lr_conf_matrix[28][24] + lr_conf_matrix[28][25] + lr_conf_matrix[28][26] + lr_conf_matrix[28][27]

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
            lr_recall_5 = lr_tp_5 / (lr_tp_5 + lr_fn_5)
        if lr_tp_6 + lr_fn_6 == 0:
            lr_recall_6 = 0
        else:
            lr_recall_6 = lr_tp_6 / (lr_tp_6 + lr_fn_6)
        if lr_tp_7 + lr_fn_7 == 0:
            lr_recall_7 = 0
        else:
            lr_recall_7 = lr_tp_7 / (lr_tp_7 + lr_fn_7)
        if lr_tp_8 + lr_fn_8 == 0:
            lr_recall_8 = 0
        else:
            lr_recall_8 = lr_tp_8 / (lr_tp_8 + lr_fn_8)
        if lr_tp_9 + lr_fn_9 == 0:
            lr_recall_9 = 0
        else:
            lr_recall_9 = lr_tp_9 / (lr_tp_9 + lr_fn_9)
        if lr_tp_10 + lr_fn_10 == 0:
            lr_recall_10 = 0
        else:
            lr_recall_10 = lr_tp_10 / (lr_tp_10 + lr_fn_10)
        if lr_tp_11 + lr_fn_11 == 0:
            lr_recall_11 = 0
        else:
            lr_recall_11 = lr_tp_11 / (lr_tp_11 + lr_fn_11)
        if lr_tp_12 + lr_fn_12 == 0:
            lr_recall_12 = 0
        else:
            lr_recall_12 = lr_tp_12 / (lr_tp_12 + lr_fn_12)
        if lr_tp_13 + lr_fn_13 == 0:
            lr_recall_13 = 0
        else:
            lr_recall_13 = lr_tp_13 / (lr_tp_13 + lr_fn_13)
        if lr_tp_14 + lr_fn_14 == 0:
            lr_recall_14 = 0
        else:
            lr_recall_14 = lr_tp_14 / (lr_tp_14 + lr_fn_14)
        if lr_tp_15 + lr_fn_15 == 0:
            lr_recall_15 = 0
        else:
            lr_recall_15 = lr_tp_15 / (lr_tp_15 + lr_fn_15)
        if lr_tp_16 + lr_fn_16 == 0:
            lr_recall_16 = 0
        else:
            lr_recall_16 = lr_tp_16 / (lr_tp_16 + lr_fn_16)
        if lr_tp_17 + lr_fn_17 == 0:
            lr_recall_17 = 0
        else:
            lr_recall_17 = lr_tp_17 / (lr_tp_17 + lr_fn_17)
        if lr_tp_18 + lr_fn_18 == 0:
            lr_recall_18 = 0
        else:
            lr_recall_18 = lr_tp_18 / (lr_tp_18 + lr_fn_18)
        if lr_tp_19 + lr_fn_19 == 0:
            lr_recall_19 = 0
        else:
            lr_recall_19 = lr_tp_19 / (lr_tp_19 + lr_fn_19)
        if lr_tp_20 + lr_fn_20 == 0:
            lr_recall_20 = 0
        else:
            lr_recall_20 = lr_tp_20 / (lr_tp_20 + lr_fn_20)
        '''
        if lr_tp_21 + lr_fn_21 == 0:
            lr_recall_21 = 0
        else:
            lr_recall_21 = lr_tp_21 / (lr_tp_21 + lr_fn_21)
        if lr_tp_22 + lr_fn_22 == 0:
            lr_recall_22 = 0
        else:
            lr_recall_22 = lr_tp_22 / (lr_tp_22 + lr_fn_22)
        if lr_tp_23 + lr_fn_23 == 0:
            lr_recall_23 = 0
        else:
            lr_recall_23 = lr_tp_23 / (lr_tp_23 + lr_fn_23)
        if lr_tp_24 + lr_fn_24 == 0:
            lr_recall_24 = 0
        else:
            lr_recall_24 = lr_tp_24 / (lr_tp_24 + lr_fn_24)
        if lr_tp_25 + lr_fn_25 == 0:
            lr_recall_25 = 0
        else:
            lr_recall_25 = lr_tp_25 / (lr_tp_25 + lr_fn_25)
        if lr_tp_26 + lr_fn_26 == 0:
            lr_recall_26 = 0
        else:
            lr_recall_26 = lr_tp_26 / (lr_tp_26 + lr_fn_26)
        if lr_tp_27 + lr_fn_27 == 0:
            lr_recall_27 = 0
        else:
            lr_recall_27 = lr_tp_27 / (lr_tp_27 + lr_fn_27)
        if lr_tp_28 + lr_fn_28 == 0:
            lr_recall_28 = 0
        else:
            lr_recall_28 = lr_tp_28 / (lr_tp_28 + lr_fn_28)
        if lr_tp_29 + lr_fn_29 == 0:
            lr_recall_29 = 0
        else:
            lr_recall_29 = lr_tp_29 / (lr_tp_29 + lr_fn_29)
        '''
        lr_recall_avg_pen_5 = (
                                 lr_recall_1 + lr_recall_2 + lr_recall_3 + lr_recall_4 + lr_recall_5 + lr_recall_6 + lr_recall_7 + lr_recall_8 + lr_recall_9 + lr_recall_10 + lr_recall_11 + lr_recall_12 + lr_recall_13 + lr_recall_14 + lr_recall_15 + lr_recall_16 + lr_recall_17 + lr_recall_18 + lr_recall_19 + (5*lr_recall_20)) / (20+5-1)
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
    lr_ovr_accuracy = (lr_conf_matrix[0][0] + lr_conf_matrix[1][1] + lr_conf_matrix[2][2] + lr_conf_matrix[3][3] + lr_conf_matrix[4][4] + lr_conf_matrix[5][5] + lr_conf_matrix[6][6] + lr_conf_matrix[7][7] + lr_conf_matrix[8][8] + lr_conf_matrix[9][9] + lr_conf_matrix[10][10] + lr_conf_matrix[11][11] + lr_conf_matrix[12][12] + lr_conf_matrix[13][13] + lr_conf_matrix[14][14] + lr_conf_matrix[15][15] + lr_conf_matrix[16][16] + lr_conf_matrix[17][17] + lr_conf_matrix[18][18] + lr_conf_matrix[19][19]) / (
                sum(lr_conf_matrix[0]) + sum(lr_conf_matrix[1]) + sum(lr_conf_matrix[2]) + sum(lr_conf_matrix[3]) + sum(lr_conf_matrix[4]) + sum(lr_conf_matrix[5]) + sum(lr_conf_matrix[6]) + sum(lr_conf_matrix[7]) + sum(lr_conf_matrix[8]) + sum(lr_conf_matrix[9]) + sum(lr_conf_matrix[10]) + sum(lr_conf_matrix[11]) + sum(lr_conf_matrix[12]) + sum(lr_conf_matrix[13]) + sum(lr_conf_matrix[14]) + sum(lr_conf_matrix[15]) + sum(lr_conf_matrix[16]) + sum(lr_conf_matrix[17]) + sum(lr_conf_matrix[18]) + sum(lr_conf_matrix[19]))
    print("lr_f1 score of pen 1 is:")
    print(lr_f1_score_pen_1)
    print("lr_f1 score of pen 5 is:")
    print(lr_f1_score_pen_5)
    print("lr_overall accuracy is:")
    print(lr_ovr_accuracy)
    lr_conf_matrix = pd.DataFrame(lr_conf_matrix)
    lr_conf_matrix.to_csv('conf_matrix_'+imb_technique+'_'+ str(nsplits) + '_lr_production_foldcv_' + str(repeat+1) + '.csv',header=False,index=False) #Second repetition
    #lr_conf_matrix.to_csv('conf_matrix_'+imb_technique+'_penalty_'+ str(nsplits) + '_lr_production_' + str(nsplits)+ 'foldcv_' + str(repeat+6) + '.csv',header=False,index=False) #Second repetition
    lr_f1_score_pen_1_kfoldcv[repeat] = lr_f1_score_pen_1
    lr_f1_score_pen_5_kfoldcv[repeat] = lr_f1_score_pen_5
    lr_ovr_accuracy_kfoldcv[repeat] = lr_ovr_accuracy


    for i in range(0, len(y_test)):
        #nb_DM_index = 0
        nb_FI_index = 0
        nb_FG_index = 0
        #nb_GR_index = 0
        #nb_GR12_index = 0
        nb_GR27_index = 0
        nb_LM_index = 0
        nb_LMM_index = 0
        #nb_MM14_index = 0
        #nb_MM16_index = 0
        nb_PC_index = 0
        nb_RG12_index = 0
        #nb_RG19_index = 0
        nb_RG2_index = 0
        nb_RG3_index = 0
        nb_RGM_index = 0
        nb_RGQC_index = 0
        #nb_TMSA10_index = 0
        nb_T8_index = 0
        #nb_T9_index = 0
        nb_TM10_index = 0
        nb_TM4_index = 0
        nb_TM5_index = 0
        nb_TM6_index = 0
        nb_TM8_index = 0
        nb_TM9_index = 0
        nb_TMQC_index = 0
        nb_TQC_index = 0
        #nb_WC13_index = 0

        """
        if nb_pred_class_DM[i] == "Deburring - Manual":
            if nb_pred_prob_DM[i][0] >= 0.5:
                nb_DM_index = 0
            else:
                nb_DM_index = 1
        elif nb_pred_class_DM[i] == "Others":
            if nb_pred_prob_DM[i][0] < 0.5:
                nb_DM_index = 0
            else:
                nb_DM_index = 1
        """
        if nb_pred_class_FI[i] == "Final Inspection Q.C.":
            if nb_pred_prob_FI[i][0] >= 0.5:
                nb_FI_index = 0
            else:
                nb_FI_index = 1
        elif nb_pred_class_FI[i] == "Others":
            if nb_pred_prob_FI[i][0] < 0.5:
                nb_FI_index = 0
            else:
                nb_FI_index = 1
        if nb_pred_class_FG[i] == "Flat Grinding - Machine 11":
            if nb_pred_prob_FG[i][0] >= 0.5:
                nb_FG_index = 0
            else:
                nb_FG_index = 1
        elif nb_pred_class_FG[i] == "Others":
            if nb_pred_prob_FG[i][0] < 0.5:
                nb_FG_index = 0
            else:
                nb_FG_index = 1
        """
        if nb_pred_class_GR[i] == "Grinding Rework":
            if nb_pred_prob_GR[i][0] >= 0.5:
                nb_GR_index = 0
            else:
                nb_GR_index = 1
        elif nb_pred_class_GR[i] == "Others":
            if nb_pred_prob_GR[i][0] < 0.5:
                nb_GR_index = 0
            else:
                nb_GR_index = 1
        """
        """
        if nb_pred_class_GR12[i] == "Grinding Rework - Machine 12":
            if nb_pred_prob_GR12[i][0] >= 0.5:
                nb_GR12_index = 0
            else:
                nb_GR12_index = 1
        elif nb_pred_class_GR12[i] == "Others":
            if nb_pred_prob_GR12[i][0] < 0.5:
                nb_GR12_index = 0
            else:
                nb_GR12_index = 1
        """
        if nb_pred_class_GR27[i] == "Grinding Rework - Machine 27":
            if nb_pred_prob_GR27[i][0] >= 0.5:
                nb_GR27_index = 0
            else:
                nb_GR27_index = 1
        elif nb_pred_class_GR27[i] == "Others":
            if nb_pred_prob_GR27[i][0] < 0.5:
                nb_GR27_index = 0
            else:
                nb_GR27_index = 1
        if nb_pred_class_LM[i] == "Lapping - Machine 1":
            if nb_pred_prob_LM[i][0] >= 0.5:
                nb_LM_index = 0
            else:
                nb_LM_index = 1
        elif nb_pred_class_LM[i] == "Others":
            if nb_pred_prob_LM[i][0] < 0.5:
                nb_LM_index = 0
            else:
                nb_LM_index = 1
        if nb_pred_class_LMM[i] == "Laser Marking - Machine 7":
            if nb_pred_prob_LMM[i][0] >= 0.5:
                nb_LMM_index = 0
            else:
                nb_LMM_index = 1
        elif nb_pred_class_LMM[i] == "Others":
            if nb_pred_prob_LMM[i][0] < 0.5:
                nb_LMM_index = 0
            else:
                nb_LMM_index = 1
        """
        if nb_pred_class_MM14[i] == "Milling - Machine 14":
            if nb_pred_prob_MM14[i][0] >= 0.5:
                nb_MM14_index = 0
            else:
                nb_MM14_index = 1
        elif nb_pred_class_MM14[i] == "Others":
            if nb_pred_prob_MM14[i][0] < 0.5:
                nb_MM14_index = 0
            else:
                nb_MM14_index = 1
        """
        """
        if nb_pred_class_MM16[i] == "Milling - Machine 16":
            if nb_pred_prob_MM16[i][0] >= 0.5:
                nb_MM16_index = 0
            else:
                nb_MM16_index = 1
        elif nb_pred_class_MM16[i] == "Others":
            if nb_pred_prob_MM16[i][0] < 0.5:
                nb_MM16_index = 0
            else:
                nb_MM16_index = 1
        """
        if nb_pred_class_PC[i] == "Packing":
            if nb_pred_prob_PC[i][0] >= 0.5:
                nb_PC_index = 0
            else:
                nb_PC_index = 1
        elif nb_pred_class_PC[i] == "Others":
            if nb_pred_prob_PC[i][0] < 0.5:
                nb_PC_index = 0
            else:
                nb_PC_index = 1
        if nb_pred_class_RG12[i] == "Round Grinding - Machine 12":
            if nb_pred_prob_RG12[i][0] >= 0.5:
                nb_RG12_index = 0
            else:
                nb_RG12_index = 1
        elif nb_pred_class_RG12[i] == "Others":
            if nb_pred_prob_RG12[i][0] < 0.5:
                nb_RG12_index = 0
            else:
                nb_RG12_index = 1
        """
        if nb_pred_class_RG19[i] == "Round Grinding - Machine 19":
            if nb_pred_prob_RG19[i][0] >= 0.5:
                nb_RG19_index = 0
            else:
                nb_RG19_index = 1
        elif nb_pred_class_RG19[i] == "Others":
            if nb_pred_prob_RG19[i][0] < 0.5:
                nb_RG19_index = 0
            else:
                nb_RG19_index = 1
        """
        if nb_pred_class_RG2[i] == "Round Grinding - Machine 2":
            if nb_pred_prob_RG2[i][0] >= 0.5:
                nb_RG2_index = 0
            else:
                nb_RG2_index = 1
        elif nb_pred_class_RG2[i] == "Others":
            if nb_pred_prob_RG2[i][0] < 0.5:
                nb_RG2_index = 0
            else:
                nb_RG2_index = 1
        if nb_pred_class_RG3[i] == "Round Grinding - Machine 3":
            if nb_pred_prob_RG3[i][0] >= 0.5:
                nb_RG3_index = 0
            else:
                nb_RG3_index = 1
        elif nb_pred_class_RG3[i] == "Others":
            if nb_pred_prob_RG3[i][0] < 0.5:
                nb_RG3_index = 0
            else:
                nb_RG3_index = 1
        if nb_pred_class_RGM[i] == "Round Grinding - Manual":
            if nb_pred_prob_RGM[i][0] >= 0.5:
                nb_RGM_index = 0
            else:
                nb_RGM_index = 1
        elif nb_pred_class_RGM[i] == "Others":
            if nb_pred_prob_RGM[i][0] < 0.5:
                nb_RGM_index = 0
            else:
                nb_RGM_index = 1
        if nb_pred_class_RGQC[i] == "Round Grinding - Q.C.":
            if nb_pred_prob_RGQC[i][0] >= 0.5:
                nb_RGQC_index = 0
            else:
                nb_RGQC_index = 1
        elif nb_pred_class_RGQC[i] == "Others":
            if nb_pred_prob_RGQC[i][0] < 0.5:
                nb_RGQC_index = 0
            else:
                nb_RGQC_index = 1
        """
        if nb_pred_class_TMSA10[i] == "Turn & Mill. & Screw Assem - Machine 10":
            if nb_pred_prob_TMSA10[i][0] >= 0.5:
                nb_TMSA10_index = 0
            else:
                nb_TMSA10_index = 1
        elif nb_pred_class_TMSA10[i] == "Others":
            if nb_pred_prob_TMSA10[i][0] < 0.5:
                nb_TMSA10_index = 0
            else:
                nb_TMSA10_index = 1
        """
        if nb_pred_class_T8[i] == "Turning - Machine 8":
            if nb_pred_prob_T8[i][0] >= 0.5:
                nb_T8_index = 0
            else:
                nb_T8_index = 1
        elif nb_pred_class_T8[i] == "Others":
            if nb_pred_prob_T8[i][0] < 0.5:
                nb_T8_index = 0
            else:
                nb_T8_index = 1
        """
        if nb_pred_class_T9[i] == "Turning - Machine 9":
            if nb_pred_prob_T9[i][0] >= 0.5:
                nb_T9_index = 0
            else:
                nb_T9_index = 1
        elif nb_pred_class_T9[i] == "Others":
            if nb_pred_prob_T9[i][0] < 0.5:
                nb_T9_index = 0
            else:
                nb_T9_index = 1
        """
        if nb_pred_class_TM10[i] == "Turning & Milling - Machine 10":
            if nb_pred_prob_TM10[i][0] >= 0.5:
                nb_TM10_index = 0
            else:
                nb_TM10_index = 1
        elif nb_pred_class_TM10[i] == "Others":
            if nb_pred_prob_TM10[i][0] < 0.5:
                nb_TM10_index = 0
            else:
                nb_TM10_index = 1
        if nb_pred_class_TM4[i] == "Turning & Milling - Machine 4":
            if nb_pred_prob_TM4[i][0] >= 0.5:
                nb_TM4_index = 0
            else:
                nb_TM4_index = 1
        elif nb_pred_class_TM4[i] == "Others":
            if nb_pred_prob_TM4[i][0] < 0.5:
                nb_TM4_index = 0
            else:
                nb_TM4_index = 1
        if nb_pred_class_TM5[i] == "Turning & Milling - Machine 5":
            if nb_pred_prob_TM5[i][0] >= 0.5:
                nb_TM5_index = 0
            else:
                nb_TM5_index = 1
        elif nb_pred_class_TM5[i] == "Others":
            if nb_pred_prob_TM5[i][0] < 0.5:
                nb_TM5_index = 0
            else:
                nb_TM5_index = 1
        if nb_pred_class_TM6[i] == "Turning & Milling - Machine 6":
            if nb_pred_prob_TM6[i][0] >= 0.5:
                nb_TM6_index = 0
            else:
                nb_TM6_index = 1
        elif nb_pred_class_TM6[i] == "Others":
            if nb_pred_prob_TM6[i][0] < 0.5:
                nb_TM6_index = 0
            else:
                nb_TM6_index = 1
        if nb_pred_class_TM8[i] == "Turning & Milling - Machine 8":
            if nb_pred_prob_TM8[i][0] >= 0.5:
                nb_TM8_index = 0
            else:
                nb_TM8_index = 1
        elif nb_pred_class_TM8[i] == "Others":
            if nb_pred_prob_TM8[i][0] < 0.5:
                nb_TM8_index = 0
            else:
                nb_TM8_index = 1
        if nb_pred_class_TM9[i] == "Turning & Milling - Machine 9":
            if nb_pred_prob_TM9[i][0] >= 0.5:
                nb_TM9_index = 0
            else:
                nb_TM9_index = 1
        elif nb_pred_class_TM9[i] == "Others":
            if nb_pred_prob_TM9[i][0] < 0.5:
                nb_TM9_index = 0
            else:
                nb_TM9_index = 1
        if nb_pred_class_TMQC[i] == "Turning & Milling Q.C.":
            if nb_pred_prob_TMQC[i][0] >= 0.5:
                nb_TMQC_index = 0
            else:
                nb_TMQC_index = 1
        elif nb_pred_class_TMQC[i] == "Others":
            if nb_pred_prob_TMQC[i][0] < 0.5:
                nb_TMQC_index = 0
            else:
                nb_TMQC_index = 1
        if nb_pred_class_TQC[i] == "Turning Q.C.":
            if nb_pred_prob_TQC[i][0] >= 0.5:
                nb_TQC_index = 0
            else:
                nb_TQC_index = 1
        elif nb_pred_class_TQC[i] == "Others":
            if nb_pred_prob_TQC[i][0] < 0.5:
                nb_TQC_index = 0
            else:
                nb_TQC_index = 1
        """
        if nb_pred_class_WC13[i] == "Wire Cut - Machine 13":
            if nb_pred_prob_WC13[i][0] >= 0.5:
                nb_WC13_index = 0
            else:
                nb_WC13_index = 1
        elif nb_pred_class_WC13[i] == "Others":
            if nb_pred_prob_WC13[i][0] < 0.5:
                nb_WC13_index = 0
            else:
                nb_WC13_index = 1
        """
        #if nb_pred_prob_DM[i][nb_DM_index] == max(nb_pred_prob_FI[i][nb_FI_index], nb_pred_prob_FG[i][nb_FG_index], nb_pred_prob_GR27[i][nb_GR27_index], nb_pred_prob_LM[i][nb_LM_index], nb_pred_prob_LMM[i][nb_LMM_index], nb_pred_prob_PC[i][nb_PC_index], nb_pred_prob_RG12[i][nb_RG12_index], nb_pred_prob_RG2[i][nb_RG2_index], nb_pred_prob_RG3[i][nb_RG3_index], nb_pred_prob_RGM[i][nb_RGM_index], nb_pred_prob_RGQC[i][nb_RGQC_index], nb_pred_prob_T8[i][nb_T8_index], nb_pred_prob_TM10[i][nb_TM10_index], nb_pred_prob_TM4[i][nb_TM4_index], nb_pred_prob_TM5[i][nb_TM5_index], nb_pred_prob_TM6[i][nb_TM6_index], nb_pred_prob_TM8[i][nb_TM8_index], nb_pred_prob_TM9[i][nb_TM9_index], nb_pred_prob_TMQC[i][nb_TMQC_index], nb_pred_prob_TQC[i][nb_TQC_index]):
        #    nb_prediction.loc[i] = "Deburring - Manual"
        if nb_pred_prob_FI[i][nb_FI_index] == max(nb_pred_prob_FI[i][nb_FI_index], nb_pred_prob_FG[i][nb_FG_index], nb_pred_prob_GR27[i][nb_GR27_index], nb_pred_prob_LM[i][nb_LM_index], nb_pred_prob_LMM[i][nb_LMM_index], nb_pred_prob_PC[i][nb_PC_index], nb_pred_prob_RG12[i][nb_RG12_index], nb_pred_prob_RG2[i][nb_RG2_index], nb_pred_prob_RG3[i][nb_RG3_index], nb_pred_prob_RGM[i][nb_RGM_index], nb_pred_prob_RGQC[i][nb_RGQC_index], nb_pred_prob_T8[i][nb_T8_index], nb_pred_prob_TM10[i][nb_TM10_index], nb_pred_prob_TM4[i][nb_TM4_index], nb_pred_prob_TM5[i][nb_TM5_index], nb_pred_prob_TM6[i][nb_TM6_index], nb_pred_prob_TM8[i][nb_TM8_index], nb_pred_prob_TM9[i][nb_TM9_index], nb_pred_prob_TMQC[i][nb_TMQC_index], nb_pred_prob_TQC[i][nb_TQC_index]):
            nb_prediction.loc[i] = "Final Inspection Q.C."
        elif nb_pred_prob_FG[i][nb_FG_index] == max(nb_pred_prob_FI[i][nb_FI_index], nb_pred_prob_FG[i][nb_FG_index], nb_pred_prob_GR27[i][nb_GR27_index], nb_pred_prob_LM[i][nb_LM_index], nb_pred_prob_LMM[i][nb_LMM_index], nb_pred_prob_PC[i][nb_PC_index], nb_pred_prob_RG12[i][nb_RG12_index], nb_pred_prob_RG2[i][nb_RG2_index], nb_pred_prob_RG3[i][nb_RG3_index], nb_pred_prob_RGM[i][nb_RGM_index], nb_pred_prob_RGQC[i][nb_RGQC_index], nb_pred_prob_T8[i][nb_T8_index], nb_pred_prob_TM10[i][nb_TM10_index], nb_pred_prob_TM4[i][nb_TM4_index], nb_pred_prob_TM5[i][nb_TM5_index], nb_pred_prob_TM6[i][nb_TM6_index], nb_pred_prob_TM8[i][nb_TM8_index], nb_pred_prob_TM9[i][nb_TM9_index], nb_pred_prob_TMQC[i][nb_TMQC_index], nb_pred_prob_TQC[i][nb_TQC_index]):
            nb_prediction.loc[i] = "Flat Grinding - Machine 11"
        #elif nb_pred_prob_GR[i][nb_GR_index] == max(nb_pred_prob_FI[i][nb_FI_index], nb_pred_prob_FG[i][nb_FG_index], nb_pred_prob_GR27[i][nb_GR27_index], nb_pred_prob_LM[i][nb_LM_index], nb_pred_prob_LMM[i][nb_LMM_index], nb_pred_prob_PC[i][nb_PC_index], nb_pred_prob_RG12[i][nb_RG12_index], nb_pred_prob_RG2[i][nb_RG2_index], nb_pred_prob_RG3[i][nb_RG3_index], nb_pred_prob_RGM[i][nb_RGM_index], nb_pred_prob_RGQC[i][nb_RGQC_index], nb_pred_prob_T8[i][nb_T8_index], nb_pred_prob_TM10[i][nb_TM10_index], nb_pred_prob_TM4[i][nb_TM4_index], nb_pred_prob_TM5[i][nb_TM5_index], nb_pred_prob_TM6[i][nb_TM6_index], nb_pred_prob_TM8[i][nb_TM8_index], nb_pred_prob_TM9[i][nb_TM9_index], nb_pred_prob_TMQC[i][nb_TMQC_index], nb_pred_prob_TQC[i][nb_TQC_index]):
        #    nb_prediction.loc[i] = "Grinding Rework"
        #elif nb_pred_prob_GR12[i][nb_GR12_index] == max(nb_pred_prob_FI[i][nb_FI_index], nb_pred_prob_FG[i][nb_FG_index], nb_pred_prob_GR27[i][nb_GR27_index], nb_pred_prob_LM[i][nb_LM_index], nb_pred_prob_LMM[i][nb_LMM_index], nb_pred_prob_PC[i][nb_PC_index], nb_pred_prob_RG12[i][nb_RG12_index], nb_pred_prob_RG2[i][nb_RG2_index], nb_pred_prob_RG3[i][nb_RG3_index], nb_pred_prob_RGM[i][nb_RGM_index], nb_pred_prob_RGQC[i][nb_RGQC_index], nb_pred_prob_T8[i][nb_T8_index], nb_pred_prob_TM10[i][nb_TM10_index], nb_pred_prob_TM4[i][nb_TM4_index], nb_pred_prob_TM5[i][nb_TM5_index], nb_pred_prob_TM6[i][nb_TM6_index], nb_pred_prob_TM8[i][nb_TM8_index], nb_pred_prob_TM9[i][nb_TM9_index], nb_pred_prob_TMQC[i][nb_TMQC_index], nb_pred_prob_TQC[i][nb_TQC_index]):
        #    nb_prediction.loc[i] = "Grinding Rework - Machine 12"
        elif nb_pred_prob_GR27[i][nb_GR27_index] == max(nb_pred_prob_FI[i][nb_FI_index], nb_pred_prob_FG[i][nb_FG_index], nb_pred_prob_GR27[i][nb_GR27_index], nb_pred_prob_LM[i][nb_LM_index], nb_pred_prob_LMM[i][nb_LMM_index], nb_pred_prob_PC[i][nb_PC_index], nb_pred_prob_RG12[i][nb_RG12_index], nb_pred_prob_RG2[i][nb_RG2_index], nb_pred_prob_RG3[i][nb_RG3_index], nb_pred_prob_RGM[i][nb_RGM_index], nb_pred_prob_RGQC[i][nb_RGQC_index], nb_pred_prob_T8[i][nb_T8_index], nb_pred_prob_TM10[i][nb_TM10_index], nb_pred_prob_TM4[i][nb_TM4_index], nb_pred_prob_TM5[i][nb_TM5_index], nb_pred_prob_TM6[i][nb_TM6_index], nb_pred_prob_TM8[i][nb_TM8_index], nb_pred_prob_TM9[i][nb_TM9_index], nb_pred_prob_TMQC[i][nb_TMQC_index], nb_pred_prob_TQC[i][nb_TQC_index]):
            nb_prediction.loc[i] = "Grinding Rework - Machine 27"
        elif nb_pred_prob_LM[i][nb_LM_index] == max(nb_pred_prob_FI[i][nb_FI_index], nb_pred_prob_FG[i][nb_FG_index], nb_pred_prob_GR27[i][nb_GR27_index], nb_pred_prob_LM[i][nb_LM_index], nb_pred_prob_LMM[i][nb_LMM_index], nb_pred_prob_PC[i][nb_PC_index], nb_pred_prob_RG12[i][nb_RG12_index], nb_pred_prob_RG2[i][nb_RG2_index], nb_pred_prob_RG3[i][nb_RG3_index], nb_pred_prob_RGM[i][nb_RGM_index], nb_pred_prob_RGQC[i][nb_RGQC_index], nb_pred_prob_T8[i][nb_T8_index], nb_pred_prob_TM10[i][nb_TM10_index], nb_pred_prob_TM4[i][nb_TM4_index], nb_pred_prob_TM5[i][nb_TM5_index], nb_pred_prob_TM6[i][nb_TM6_index], nb_pred_prob_TM8[i][nb_TM8_index], nb_pred_prob_TM9[i][nb_TM9_index], nb_pred_prob_TMQC[i][nb_TMQC_index], nb_pred_prob_TQC[i][nb_TQC_index]):
            nb_prediction.loc[i] = "Lapping - Machine 1"
        elif nb_pred_prob_LMM[i][nb_LMM_index] == max(nb_pred_prob_FI[i][nb_FI_index], nb_pred_prob_FG[i][nb_FG_index], nb_pred_prob_GR27[i][nb_GR27_index], nb_pred_prob_LM[i][nb_LM_index], nb_pred_prob_LMM[i][nb_LMM_index], nb_pred_prob_PC[i][nb_PC_index], nb_pred_prob_RG12[i][nb_RG12_index], nb_pred_prob_RG2[i][nb_RG2_index], nb_pred_prob_RG3[i][nb_RG3_index], nb_pred_prob_RGM[i][nb_RGM_index], nb_pred_prob_RGQC[i][nb_RGQC_index], nb_pred_prob_T8[i][nb_T8_index], nb_pred_prob_TM10[i][nb_TM10_index], nb_pred_prob_TM4[i][nb_TM4_index], nb_pred_prob_TM5[i][nb_TM5_index], nb_pred_prob_TM6[i][nb_TM6_index], nb_pred_prob_TM8[i][nb_TM8_index], nb_pred_prob_TM9[i][nb_TM9_index], nb_pred_prob_TMQC[i][nb_TMQC_index], nb_pred_prob_TQC[i][nb_TQC_index]):
            nb_prediction.loc[i] = "Laser Marking - Machine 7"
        #elif nb_pred_prob_MM14[i][nb_MM14_index] == max(nb_pred_prob_FI[i][nb_FI_index], nb_pred_prob_FG[i][nb_FG_index], nb_pred_prob_GR27[i][nb_GR27_index], nb_pred_prob_LM[i][nb_LM_index], nb_pred_prob_LMM[i][nb_LMM_index], nb_pred_prob_PC[i][nb_PC_index], nb_pred_prob_RG12[i][nb_RG12_index], nb_pred_prob_RG2[i][nb_RG2_index], nb_pred_prob_RG3[i][nb_RG3_index], nb_pred_prob_RGM[i][nb_RGM_index], nb_pred_prob_RGQC[i][nb_RGQC_index], nb_pred_prob_T8[i][nb_T8_index], nb_pred_prob_TM10[i][nb_TM10_index], nb_pred_prob_TM4[i][nb_TM4_index], nb_pred_prob_TM5[i][nb_TM5_index], nb_pred_prob_TM6[i][nb_TM6_index], nb_pred_prob_TM8[i][nb_TM8_index], nb_pred_prob_TM9[i][nb_TM9_index], nb_pred_prob_TMQC[i][nb_TMQC_index], nb_pred_prob_TQC[i][nb_TQC_index]):
        #    nb_prediction.loc[i] = "Milling - Machine 14"
        #elif nb_pred_prob_MM16[i][nb_MM16_index] == max(nb_pred_prob_FI[i][nb_FI_index], nb_pred_prob_FG[i][nb_FG_index], nb_pred_prob_GR27[i][nb_GR27_index], nb_pred_prob_LM[i][nb_LM_index], nb_pred_prob_LMM[i][nb_LMM_index], nb_pred_prob_PC[i][nb_PC_index], nb_pred_prob_RG12[i][nb_RG12_index], nb_pred_prob_RG2[i][nb_RG2_index], nb_pred_prob_RG3[i][nb_RG3_index], nb_pred_prob_RGM[i][nb_RGM_index], nb_pred_prob_RGQC[i][nb_RGQC_index], nb_pred_prob_T8[i][nb_T8_index], nb_pred_prob_TM10[i][nb_TM10_index], nb_pred_prob_TM4[i][nb_TM4_index], nb_pred_prob_TM5[i][nb_TM5_index], nb_pred_prob_TM6[i][nb_TM6_index], nb_pred_prob_TM8[i][nb_TM8_index], nb_pred_prob_TM9[i][nb_TM9_index], nb_pred_prob_TMQC[i][nb_TMQC_index], nb_pred_prob_TQC[i][nb_TQC_index]):
        #    nb_prediction.loc[i] = "Milling - Machine 16"
        elif nb_pred_prob_PC[i][nb_PC_index] == max(nb_pred_prob_FI[i][nb_FI_index], nb_pred_prob_FG[i][nb_FG_index], nb_pred_prob_GR27[i][nb_GR27_index], nb_pred_prob_LM[i][nb_LM_index], nb_pred_prob_LMM[i][nb_LMM_index], nb_pred_prob_PC[i][nb_PC_index], nb_pred_prob_RG12[i][nb_RG12_index], nb_pred_prob_RG2[i][nb_RG2_index], nb_pred_prob_RG3[i][nb_RG3_index], nb_pred_prob_RGM[i][nb_RGM_index], nb_pred_prob_RGQC[i][nb_RGQC_index], nb_pred_prob_T8[i][nb_T8_index], nb_pred_prob_TM10[i][nb_TM10_index], nb_pred_prob_TM4[i][nb_TM4_index], nb_pred_prob_TM5[i][nb_TM5_index], nb_pred_prob_TM6[i][nb_TM6_index], nb_pred_prob_TM8[i][nb_TM8_index], nb_pred_prob_TM9[i][nb_TM9_index], nb_pred_prob_TMQC[i][nb_TMQC_index], nb_pred_prob_TQC[i][nb_TQC_index]):
            nb_prediction.loc[i] = "Packing"
        elif nb_pred_prob_RG12[i][nb_RG12_index] == max(nb_pred_prob_FI[i][nb_FI_index], nb_pred_prob_FG[i][nb_FG_index], nb_pred_prob_GR27[i][nb_GR27_index], nb_pred_prob_LM[i][nb_LM_index], nb_pred_prob_LMM[i][nb_LMM_index], nb_pred_prob_PC[i][nb_PC_index], nb_pred_prob_RG12[i][nb_RG12_index], nb_pred_prob_RG2[i][nb_RG2_index], nb_pred_prob_RG3[i][nb_RG3_index], nb_pred_prob_RGM[i][nb_RGM_index], nb_pred_prob_RGQC[i][nb_RGQC_index], nb_pred_prob_T8[i][nb_T8_index], nb_pred_prob_TM10[i][nb_TM10_index], nb_pred_prob_TM4[i][nb_TM4_index], nb_pred_prob_TM5[i][nb_TM5_index], nb_pred_prob_TM6[i][nb_TM6_index], nb_pred_prob_TM8[i][nb_TM8_index], nb_pred_prob_TM9[i][nb_TM9_index], nb_pred_prob_TMQC[i][nb_TMQC_index], nb_pred_prob_TQC[i][nb_TQC_index]):
            nb_prediction.loc[i] = "Round Grinding - Machine 12"
        #elif nb_pred_prob_RG19[i][nb_RG19_index] == max(nb_pred_prob_FI[i][nb_FI_index], nb_pred_prob_FG[i][nb_FG_index], nb_pred_prob_GR27[i][nb_GR27_index], nb_pred_prob_LM[i][nb_LM_index], nb_pred_prob_LMM[i][nb_LMM_index], nb_pred_prob_PC[i][nb_PC_index], nb_pred_prob_RG12[i][nb_RG12_index], nb_pred_prob_RG2[i][nb_RG2_index], nb_pred_prob_RG3[i][nb_RG3_index], nb_pred_prob_RGM[i][nb_RGM_index], nb_pred_prob_RGQC[i][nb_RGQC_index], nb_pred_prob_T8[i][nb_T8_index], nb_pred_prob_TM10[i][nb_TM10_index], nb_pred_prob_TM4[i][nb_TM4_index], nb_pred_prob_TM5[i][nb_TM5_index], nb_pred_prob_TM6[i][nb_TM6_index], nb_pred_prob_TM8[i][nb_TM8_index], nb_pred_prob_TM9[i][nb_TM9_index], nb_pred_prob_TMQC[i][nb_TMQC_index], nb_pred_prob_TQC[i][nb_TQC_index]):
        #    nb_prediction.loc[i] = "Round Grinding - Machine 19"
        elif nb_pred_prob_RG2[i][nb_RG2_index] == max(nb_pred_prob_FI[i][nb_FI_index], nb_pred_prob_FG[i][nb_FG_index], nb_pred_prob_GR27[i][nb_GR27_index], nb_pred_prob_LM[i][nb_LM_index], nb_pred_prob_LMM[i][nb_LMM_index], nb_pred_prob_PC[i][nb_PC_index], nb_pred_prob_RG12[i][nb_RG12_index], nb_pred_prob_RG2[i][nb_RG2_index], nb_pred_prob_RG3[i][nb_RG3_index], nb_pred_prob_RGM[i][nb_RGM_index], nb_pred_prob_RGQC[i][nb_RGQC_index], nb_pred_prob_T8[i][nb_T8_index], nb_pred_prob_TM10[i][nb_TM10_index], nb_pred_prob_TM4[i][nb_TM4_index], nb_pred_prob_TM5[i][nb_TM5_index], nb_pred_prob_TM6[i][nb_TM6_index], nb_pred_prob_TM8[i][nb_TM8_index], nb_pred_prob_TM9[i][nb_TM9_index], nb_pred_prob_TMQC[i][nb_TMQC_index], nb_pred_prob_TQC[i][nb_TQC_index]):
            nb_prediction.loc[i] = "Round Grinding - Machine 2"
        elif nb_pred_prob_RG3[i][nb_RG3_index] == max(nb_pred_prob_FI[i][nb_FI_index], nb_pred_prob_FG[i][nb_FG_index], nb_pred_prob_GR27[i][nb_GR27_index], nb_pred_prob_LM[i][nb_LM_index], nb_pred_prob_LMM[i][nb_LMM_index], nb_pred_prob_PC[i][nb_PC_index], nb_pred_prob_RG12[i][nb_RG12_index], nb_pred_prob_RG2[i][nb_RG2_index], nb_pred_prob_RG3[i][nb_RG3_index], nb_pred_prob_RGM[i][nb_RGM_index], nb_pred_prob_RGQC[i][nb_RGQC_index], nb_pred_prob_T8[i][nb_T8_index], nb_pred_prob_TM10[i][nb_TM10_index], nb_pred_prob_TM4[i][nb_TM4_index], nb_pred_prob_TM5[i][nb_TM5_index], nb_pred_prob_TM6[i][nb_TM6_index], nb_pred_prob_TM8[i][nb_TM8_index], nb_pred_prob_TM9[i][nb_TM9_index], nb_pred_prob_TMQC[i][nb_TMQC_index], nb_pred_prob_TQC[i][nb_TQC_index]):
            nb_prediction.loc[i] = "Round Grinding - Machine 3"
        elif nb_pred_prob_RGM[i][nb_RGM_index] == max(nb_pred_prob_FI[i][nb_FI_index], nb_pred_prob_FG[i][nb_FG_index], nb_pred_prob_GR27[i][nb_GR27_index], nb_pred_prob_LM[i][nb_LM_index], nb_pred_prob_LMM[i][nb_LMM_index], nb_pred_prob_PC[i][nb_PC_index], nb_pred_prob_RG12[i][nb_RG12_index], nb_pred_prob_RG2[i][nb_RG2_index], nb_pred_prob_RG3[i][nb_RG3_index], nb_pred_prob_RGM[i][nb_RGM_index], nb_pred_prob_RGQC[i][nb_RGQC_index], nb_pred_prob_T8[i][nb_T8_index], nb_pred_prob_TM10[i][nb_TM10_index], nb_pred_prob_TM4[i][nb_TM4_index], nb_pred_prob_TM5[i][nb_TM5_index], nb_pred_prob_TM6[i][nb_TM6_index], nb_pred_prob_TM8[i][nb_TM8_index], nb_pred_prob_TM9[i][nb_TM9_index], nb_pred_prob_TMQC[i][nb_TMQC_index], nb_pred_prob_TQC[i][nb_TQC_index]):
            nb_prediction.loc[i] = "Round Grinding - Manual"
        elif nb_pred_prob_RGQC[i][nb_RGQC_index] == max(nb_pred_prob_FI[i][nb_FI_index], nb_pred_prob_FG[i][nb_FG_index], nb_pred_prob_GR27[i][nb_GR27_index], nb_pred_prob_LM[i][nb_LM_index], nb_pred_prob_LMM[i][nb_LMM_index], nb_pred_prob_PC[i][nb_PC_index], nb_pred_prob_RG12[i][nb_RG12_index], nb_pred_prob_RG2[i][nb_RG2_index], nb_pred_prob_RG3[i][nb_RG3_index], nb_pred_prob_RGM[i][nb_RGM_index], nb_pred_prob_RGQC[i][nb_RGQC_index], nb_pred_prob_T8[i][nb_T8_index], nb_pred_prob_TM10[i][nb_TM10_index], nb_pred_prob_TM4[i][nb_TM4_index], nb_pred_prob_TM5[i][nb_TM5_index], nb_pred_prob_TM6[i][nb_TM6_index], nb_pred_prob_TM8[i][nb_TM8_index], nb_pred_prob_TM9[i][nb_TM9_index], nb_pred_prob_TMQC[i][nb_TMQC_index], nb_pred_prob_TQC[i][nb_TQC_index]):
            nb_prediction.loc[i] = "Round Grinding - Q.C."
        #elif nb_pred_prob_TMSA10[i][nb_TMSA10_index] == max(nb_pred_prob_FI[i][nb_FI_index], nb_pred_prob_FG[i][nb_FG_index], nb_pred_prob_GR27[i][nb_GR27_index], nb_pred_prob_LM[i][nb_LM_index], nb_pred_prob_LMM[i][nb_LMM_index], nb_pred_prob_PC[i][nb_PC_index], nb_pred_prob_RG12[i][nb_RG12_index], nb_pred_prob_RG2[i][nb_RG2_index], nb_pred_prob_RG3[i][nb_RG3_index], nb_pred_prob_RGM[i][nb_RGM_index], nb_pred_prob_RGQC[i][nb_RGQC_index], nb_pred_prob_T8[i][nb_T8_index], nb_pred_prob_TM10[i][nb_TM10_index], nb_pred_prob_TM4[i][nb_TM4_index], nb_pred_prob_TM5[i][nb_TM5_index], nb_pred_prob_TM6[i][nb_TM6_index], nb_pred_prob_TM8[i][nb_TM8_index], nb_pred_prob_TM9[i][nb_TM9_index], nb_pred_prob_TMQC[i][nb_TMQC_index], nb_pred_prob_TQC[i][nb_TQC_index]):
        #    nb_prediction.loc[i] = "Turn & Mill. & Screw Assem - Machine 10"
        elif nb_pred_prob_T8[i][nb_T8_index] == max(nb_pred_prob_FI[i][nb_FI_index], nb_pred_prob_FG[i][nb_FG_index], nb_pred_prob_GR27[i][nb_GR27_index], nb_pred_prob_LM[i][nb_LM_index], nb_pred_prob_LMM[i][nb_LMM_index], nb_pred_prob_PC[i][nb_PC_index], nb_pred_prob_RG12[i][nb_RG12_index], nb_pred_prob_RG2[i][nb_RG2_index], nb_pred_prob_RG3[i][nb_RG3_index], nb_pred_prob_RGM[i][nb_RGM_index], nb_pred_prob_RGQC[i][nb_RGQC_index], nb_pred_prob_T8[i][nb_T8_index], nb_pred_prob_TM10[i][nb_TM10_index], nb_pred_prob_TM4[i][nb_TM4_index], nb_pred_prob_TM5[i][nb_TM5_index], nb_pred_prob_TM6[i][nb_TM6_index], nb_pred_prob_TM8[i][nb_TM8_index], nb_pred_prob_TM9[i][nb_TM9_index], nb_pred_prob_TMQC[i][nb_TMQC_index], nb_pred_prob_TQC[i][nb_TQC_index]):
            nb_prediction.loc[i] = "Turning - Machine 8"
        #elif nb_pred_prob_T9[i][nb_T9_index] == max(nb_pred_prob_FI[i][nb_FI_index], nb_pred_prob_FG[i][nb_FG_index], nb_pred_prob_GR27[i][nb_GR27_index], nb_pred_prob_LM[i][nb_LM_index], nb_pred_prob_LMM[i][nb_LMM_index], nb_pred_prob_PC[i][nb_PC_index], nb_pred_prob_RG12[i][nb_RG12_index], nb_pred_prob_RG2[i][nb_RG2_index], nb_pred_prob_RG3[i][nb_RG3_index], nb_pred_prob_RGM[i][nb_RGM_index], nb_pred_prob_RGQC[i][nb_RGQC_index], nb_pred_prob_T8[i][nb_T8_index], nb_pred_prob_TM10[i][nb_TM10_index], nb_pred_prob_TM4[i][nb_TM4_index], nb_pred_prob_TM5[i][nb_TM5_index], nb_pred_prob_TM6[i][nb_TM6_index], nb_pred_prob_TM8[i][nb_TM8_index], nb_pred_prob_TM9[i][nb_TM9_index], nb_pred_prob_TMQC[i][nb_TMQC_index], nb_pred_prob_TQC[i][nb_TQC_index]):
        #    nb_prediction.loc[i] = "Turning - Machine 9"
        elif nb_pred_prob_TM10[i][nb_TM10_index] == max(nb_pred_prob_FI[i][nb_FI_index], nb_pred_prob_FG[i][nb_FG_index], nb_pred_prob_GR27[i][nb_GR27_index], nb_pred_prob_LM[i][nb_LM_index], nb_pred_prob_LMM[i][nb_LMM_index], nb_pred_prob_PC[i][nb_PC_index], nb_pred_prob_RG12[i][nb_RG12_index], nb_pred_prob_RG2[i][nb_RG2_index], nb_pred_prob_RG3[i][nb_RG3_index], nb_pred_prob_RGM[i][nb_RGM_index], nb_pred_prob_RGQC[i][nb_RGQC_index], nb_pred_prob_T8[i][nb_T8_index], nb_pred_prob_TM10[i][nb_TM10_index], nb_pred_prob_TM4[i][nb_TM4_index], nb_pred_prob_TM5[i][nb_TM5_index], nb_pred_prob_TM6[i][nb_TM6_index], nb_pred_prob_TM8[i][nb_TM8_index], nb_pred_prob_TM9[i][nb_TM9_index], nb_pred_prob_TMQC[i][nb_TMQC_index], nb_pred_prob_TQC[i][nb_TQC_index]):
            nb_prediction.loc[i] = "Turning & Milling - Machine 10"
        elif nb_pred_prob_TM4[i][nb_TM4_index] == max(nb_pred_prob_FI[i][nb_FI_index], nb_pred_prob_FG[i][nb_FG_index], nb_pred_prob_GR27[i][nb_GR27_index], nb_pred_prob_LM[i][nb_LM_index], nb_pred_prob_LMM[i][nb_LMM_index], nb_pred_prob_PC[i][nb_PC_index], nb_pred_prob_RG12[i][nb_RG12_index], nb_pred_prob_RG2[i][nb_RG2_index], nb_pred_prob_RG3[i][nb_RG3_index], nb_pred_prob_RGM[i][nb_RGM_index], nb_pred_prob_RGQC[i][nb_RGQC_index], nb_pred_prob_T8[i][nb_T8_index], nb_pred_prob_TM10[i][nb_TM10_index], nb_pred_prob_TM4[i][nb_TM4_index], nb_pred_prob_TM5[i][nb_TM5_index], nb_pred_prob_TM6[i][nb_TM6_index], nb_pred_prob_TM8[i][nb_TM8_index], nb_pred_prob_TM9[i][nb_TM9_index], nb_pred_prob_TMQC[i][nb_TMQC_index], nb_pred_prob_TQC[i][nb_TQC_index]):
            nb_prediction.loc[i] = "Turning & Milling - Machine 4"
        elif nb_pred_prob_TM5[i][nb_TM5_index] == max(nb_pred_prob_FI[i][nb_FI_index], nb_pred_prob_FG[i][nb_FG_index], nb_pred_prob_GR27[i][nb_GR27_index], nb_pred_prob_LM[i][nb_LM_index], nb_pred_prob_LMM[i][nb_LMM_index], nb_pred_prob_PC[i][nb_PC_index], nb_pred_prob_RG12[i][nb_RG12_index], nb_pred_prob_RG2[i][nb_RG2_index], nb_pred_prob_RG3[i][nb_RG3_index], nb_pred_prob_RGM[i][nb_RGM_index], nb_pred_prob_RGQC[i][nb_RGQC_index], nb_pred_prob_T8[i][nb_T8_index], nb_pred_prob_TM10[i][nb_TM10_index], nb_pred_prob_TM4[i][nb_TM4_index], nb_pred_prob_TM5[i][nb_TM5_index], nb_pred_prob_TM6[i][nb_TM6_index], nb_pred_prob_TM8[i][nb_TM8_index], nb_pred_prob_TM9[i][nb_TM9_index], nb_pred_prob_TMQC[i][nb_TMQC_index], nb_pred_prob_TQC[i][nb_TQC_index]):
            nb_prediction.loc[i] = "Turning & Milling - Machine 5"
        elif nb_pred_prob_TM6[i][nb_TM6_index] == max(nb_pred_prob_FI[i][nb_FI_index], nb_pred_prob_FG[i][nb_FG_index], nb_pred_prob_GR27[i][nb_GR27_index], nb_pred_prob_LM[i][nb_LM_index], nb_pred_prob_LMM[i][nb_LMM_index], nb_pred_prob_PC[i][nb_PC_index], nb_pred_prob_RG12[i][nb_RG12_index], nb_pred_prob_RG2[i][nb_RG2_index], nb_pred_prob_RG3[i][nb_RG3_index], nb_pred_prob_RGM[i][nb_RGM_index], nb_pred_prob_RGQC[i][nb_RGQC_index], nb_pred_prob_T8[i][nb_T8_index], nb_pred_prob_TM10[i][nb_TM10_index], nb_pred_prob_TM4[i][nb_TM4_index], nb_pred_prob_TM5[i][nb_TM5_index], nb_pred_prob_TM6[i][nb_TM6_index], nb_pred_prob_TM8[i][nb_TM8_index], nb_pred_prob_TM9[i][nb_TM9_index], nb_pred_prob_TMQC[i][nb_TMQC_index], nb_pred_prob_TQC[i][nb_TQC_index]):
            nb_prediction.loc[i] = "Turning & Milling - Machine 6"
        elif nb_pred_prob_TM8[i][nb_TM8_index] == max(nb_pred_prob_FI[i][nb_FI_index], nb_pred_prob_FG[i][nb_FG_index], nb_pred_prob_GR27[i][nb_GR27_index], nb_pred_prob_LM[i][nb_LM_index], nb_pred_prob_LMM[i][nb_LMM_index], nb_pred_prob_PC[i][nb_PC_index], nb_pred_prob_RG12[i][nb_RG12_index], nb_pred_prob_RG2[i][nb_RG2_index], nb_pred_prob_RG3[i][nb_RG3_index], nb_pred_prob_RGM[i][nb_RGM_index], nb_pred_prob_RGQC[i][nb_RGQC_index], nb_pred_prob_T8[i][nb_T8_index], nb_pred_prob_TM10[i][nb_TM10_index], nb_pred_prob_TM4[i][nb_TM4_index], nb_pred_prob_TM5[i][nb_TM5_index], nb_pred_prob_TM6[i][nb_TM6_index], nb_pred_prob_TM8[i][nb_TM8_index], nb_pred_prob_TM9[i][nb_TM9_index], nb_pred_prob_TMQC[i][nb_TMQC_index], nb_pred_prob_TQC[i][nb_TQC_index]):
            nb_prediction.loc[i] = "Turning & Milling - Machine 8"
        elif nb_pred_prob_TM9[i][nb_TM9_index] == max(nb_pred_prob_FI[i][nb_FI_index], nb_pred_prob_FG[i][nb_FG_index], nb_pred_prob_GR27[i][nb_GR27_index], nb_pred_prob_LM[i][nb_LM_index], nb_pred_prob_LMM[i][nb_LMM_index], nb_pred_prob_PC[i][nb_PC_index], nb_pred_prob_RG12[i][nb_RG12_index], nb_pred_prob_RG2[i][nb_RG2_index], nb_pred_prob_RG3[i][nb_RG3_index], nb_pred_prob_RGM[i][nb_RGM_index], nb_pred_prob_RGQC[i][nb_RGQC_index], nb_pred_prob_T8[i][nb_T8_index], nb_pred_prob_TM10[i][nb_TM10_index], nb_pred_prob_TM4[i][nb_TM4_index], nb_pred_prob_TM5[i][nb_TM5_index], nb_pred_prob_TM6[i][nb_TM6_index], nb_pred_prob_TM8[i][nb_TM8_index], nb_pred_prob_TM9[i][nb_TM9_index], nb_pred_prob_TMQC[i][nb_TMQC_index], nb_pred_prob_TQC[i][nb_TQC_index]):
            nb_prediction.loc[i] = "Turning & Milling - Machine 9"
        elif nb_pred_prob_TMQC[i][nb_TMQC_index] == max(nb_pred_prob_FI[i][nb_FI_index], nb_pred_prob_FG[i][nb_FG_index], nb_pred_prob_GR27[i][nb_GR27_index], nb_pred_prob_LM[i][nb_LM_index], nb_pred_prob_LMM[i][nb_LMM_index], nb_pred_prob_PC[i][nb_PC_index], nb_pred_prob_RG12[i][nb_RG12_index], nb_pred_prob_RG2[i][nb_RG2_index], nb_pred_prob_RG3[i][nb_RG3_index], nb_pred_prob_RGM[i][nb_RGM_index], nb_pred_prob_RGQC[i][nb_RGQC_index], nb_pred_prob_T8[i][nb_T8_index], nb_pred_prob_TM10[i][nb_TM10_index], nb_pred_prob_TM4[i][nb_TM4_index], nb_pred_prob_TM5[i][nb_TM5_index], nb_pred_prob_TM6[i][nb_TM6_index], nb_pred_prob_TM8[i][nb_TM8_index], nb_pred_prob_TM9[i][nb_TM9_index], nb_pred_prob_TMQC[i][nb_TMQC_index], nb_pred_prob_TQC[i][nb_TQC_index]):
            nb_prediction.loc[i] = "Turning & Milling Q.C."
        elif nb_pred_prob_TQC[i][nb_TQC_index] == max(nb_pred_prob_FI[i][nb_FI_index], nb_pred_prob_FG[i][nb_FG_index], nb_pred_prob_GR27[i][nb_GR27_index], nb_pred_prob_LM[i][nb_LM_index], nb_pred_prob_LMM[i][nb_LMM_index], nb_pred_prob_PC[i][nb_PC_index], nb_pred_prob_RG12[i][nb_RG12_index], nb_pred_prob_RG2[i][nb_RG2_index], nb_pred_prob_RG3[i][nb_RG3_index], nb_pred_prob_RGM[i][nb_RGM_index], nb_pred_prob_RGQC[i][nb_RGQC_index], nb_pred_prob_T8[i][nb_T8_index], nb_pred_prob_TM10[i][nb_TM10_index], nb_pred_prob_TM4[i][nb_TM4_index], nb_pred_prob_TM5[i][nb_TM5_index], nb_pred_prob_TM6[i][nb_TM6_index], nb_pred_prob_TM8[i][nb_TM8_index], nb_pred_prob_TM9[i][nb_TM9_index], nb_pred_prob_TMQC[i][nb_TMQC_index], nb_pred_prob_TQC[i][nb_TQC_index]):
            nb_prediction.loc[i] = "Turning Q.C."
        #elif nb_pred_prob_WC13[i][nb_WC13_index] == max(nb_pred_prob_FI[i][nb_FI_index], nb_pred_prob_FG[i][nb_FG_index], nb_pred_prob_GR27[i][nb_GR27_index], nb_pred_prob_LM[i][nb_LM_index], nb_pred_prob_LMM[i][nb_LMM_index], nb_pred_prob_PC[i][nb_PC_index], nb_pred_prob_RG12[i][nb_RG12_index], nb_pred_prob_RG2[i][nb_RG2_index], nb_pred_prob_RG3[i][nb_RG3_index], nb_pred_prob_RGM[i][nb_RGM_index], nb_pred_prob_RGQC[i][nb_RGQC_index], nb_pred_prob_T8[i][nb_T8_index], nb_pred_prob_TM10[i][nb_TM10_index], nb_pred_prob_TM4[i][nb_TM4_index], nb_pred_prob_TM5[i][nb_TM5_index], nb_pred_prob_TM6[i][nb_TM6_index], nb_pred_prob_TM8[i][nb_TM8_index], nb_pred_prob_TM9[i][nb_TM9_index], nb_pred_prob_TMQC[i][nb_TMQC_index], nb_pred_prob_TQC[i][nb_TQC_index]):
        #    nb_prediction.loc[i] = "Wire Cut - Machine 13"


    def get_precision(nb_conf_matrix):
        nb_tp_1 = nb_conf_matrix[0][0]
        nb_tp_2 = nb_conf_matrix[1][1]
        nb_tp_3 = nb_conf_matrix[2][2]
        nb_tp_4 = nb_conf_matrix[3][3]
        nb_tp_5 = nb_conf_matrix[4][4]
        nb_tp_6 = nb_conf_matrix[5][5]
        nb_tp_7 = nb_conf_matrix[6][6]
        nb_tp_8 = nb_conf_matrix[7][7]
        nb_tp_9 = nb_conf_matrix[8][8]
        nb_tp_10 = nb_conf_matrix[9][9]
        nb_tp_11 = nb_conf_matrix[10][10]
        nb_tp_12 = nb_conf_matrix[11][11]
        nb_tp_13 = nb_conf_matrix[12][12]
        nb_tp_14 = nb_conf_matrix[13][13]
        nb_tp_15 = nb_conf_matrix[14][14]
        nb_tp_16 = nb_conf_matrix[15][15]
        nb_tp_17 = nb_conf_matrix[16][16]
        nb_tp_18 = nb_conf_matrix[17][17]
        nb_tp_19 = nb_conf_matrix[18][18]
        nb_tp_20 = nb_conf_matrix[19][19]
        #nb_tp_21 = nb_conf_matrix[20][20]
        #nb_tp_22 = nb_conf_matrix[21][21]
        #nb_tp_23 = nb_conf_matrix[22][22]
        #nb_tp_24 = nb_conf_matrix[23][23]
        #nb_tp_25 = nb_conf_matrix[24][24]
        #nb_tp_26 = nb_conf_matrix[25][25]
        #nb_tp_27 = nb_conf_matrix[26][26]
        #nb_tp_28 = nb_conf_matrix[27][27]
        #nb_tp_29 = nb_conf_matrix[28][28]

        nb_fp_1 = nb_conf_matrix[1][0] + nb_conf_matrix[2][0] + nb_conf_matrix[3][0] + nb_conf_matrix[4][0] + nb_conf_matrix[5][0] + nb_conf_matrix[6][0] + nb_conf_matrix[7][0] + nb_conf_matrix[8][0] + nb_conf_matrix[9][0] + nb_conf_matrix[10][0] + nb_conf_matrix[11][0] + nb_conf_matrix[12][0] + nb_conf_matrix[13][0] + nb_conf_matrix[14][0] + nb_conf_matrix[15][0] + nb_conf_matrix[16][0] + nb_conf_matrix[17][0] + nb_conf_matrix[18][0] + nb_conf_matrix[19][0]
        nb_fp_2 = nb_conf_matrix[0][1] + nb_conf_matrix[2][1] + nb_conf_matrix[3][1] + nb_conf_matrix[4][1] + nb_conf_matrix[5][1] + nb_conf_matrix[6][1] + nb_conf_matrix[7][1] + nb_conf_matrix[8][1] + nb_conf_matrix[9][1] + nb_conf_matrix[10][1] + nb_conf_matrix[11][1] + nb_conf_matrix[12][1] + nb_conf_matrix[13][1] + nb_conf_matrix[14][1] + nb_conf_matrix[15][1] + nb_conf_matrix[16][1] + nb_conf_matrix[17][1] + nb_conf_matrix[18][1] + nb_conf_matrix[19][1]
        nb_fp_3 = nb_conf_matrix[0][2] + nb_conf_matrix[1][2] + nb_conf_matrix[3][2] + nb_conf_matrix[4][2] + nb_conf_matrix[5][2] + nb_conf_matrix[6][2] + nb_conf_matrix[7][2] + nb_conf_matrix[8][2] + nb_conf_matrix[9][2] + nb_conf_matrix[10][2] + nb_conf_matrix[11][2] + nb_conf_matrix[12][2] + nb_conf_matrix[13][2] + nb_conf_matrix[14][2] + nb_conf_matrix[15][2] + nb_conf_matrix[16][2] + nb_conf_matrix[17][2] + nb_conf_matrix[18][2] + nb_conf_matrix[19][2]
        nb_fp_4 = nb_conf_matrix[0][3] + nb_conf_matrix[1][3] + nb_conf_matrix[2][3] + nb_conf_matrix[4][3] + nb_conf_matrix[5][3] + nb_conf_matrix[6][3] + nb_conf_matrix[7][3] + nb_conf_matrix[8][3] + nb_conf_matrix[9][3] + nb_conf_matrix[10][3] + nb_conf_matrix[11][3] + nb_conf_matrix[12][3] + nb_conf_matrix[13][3] + nb_conf_matrix[14][3] + nb_conf_matrix[15][3] + nb_conf_matrix[16][3] + nb_conf_matrix[17][3] + nb_conf_matrix[18][3] + nb_conf_matrix[19][3]
        nb_fp_5 = nb_conf_matrix[0][4] + nb_conf_matrix[1][4] + nb_conf_matrix[2][4] + nb_conf_matrix[3][4] + nb_conf_matrix[5][4] + nb_conf_matrix[6][4] + nb_conf_matrix[7][4] + nb_conf_matrix[8][4] + nb_conf_matrix[9][4] + nb_conf_matrix[10][4] + nb_conf_matrix[11][4] + nb_conf_matrix[12][4] + nb_conf_matrix[13][4] + nb_conf_matrix[14][4] + nb_conf_matrix[15][4] + nb_conf_matrix[16][4] + nb_conf_matrix[17][4] + nb_conf_matrix[18][4] + nb_conf_matrix[19][4]
        nb_fp_6 = nb_conf_matrix[0][5] + nb_conf_matrix[1][5] + nb_conf_matrix[2][5] + nb_conf_matrix[3][5] + nb_conf_matrix[4][5] + nb_conf_matrix[6][5] + nb_conf_matrix[7][5] + nb_conf_matrix[8][5] + nb_conf_matrix[9][5] + nb_conf_matrix[10][5] + nb_conf_matrix[11][5] + nb_conf_matrix[12][5] + nb_conf_matrix[13][5] + nb_conf_matrix[14][5] + nb_conf_matrix[15][5] + nb_conf_matrix[16][5] + nb_conf_matrix[17][5] + nb_conf_matrix[18][5] + nb_conf_matrix[19][5]
        nb_fp_7 = nb_conf_matrix[0][6] + nb_conf_matrix[1][6] + nb_conf_matrix[2][6] + nb_conf_matrix[3][6] + nb_conf_matrix[4][6] + nb_conf_matrix[5][6] + nb_conf_matrix[7][6] + nb_conf_matrix[8][6] + nb_conf_matrix[9][6] + nb_conf_matrix[10][6] + nb_conf_matrix[11][6] + nb_conf_matrix[12][6] + nb_conf_matrix[13][6] + nb_conf_matrix[14][6] + nb_conf_matrix[15][6] + nb_conf_matrix[16][6] + nb_conf_matrix[17][6] + nb_conf_matrix[18][6] + nb_conf_matrix[19][6]
        nb_fp_8 = nb_conf_matrix[0][7] + nb_conf_matrix[1][7] + nb_conf_matrix[2][7] + nb_conf_matrix[3][7] + nb_conf_matrix[4][7] + nb_conf_matrix[5][7] + nb_conf_matrix[6][7] + nb_conf_matrix[8][7] + nb_conf_matrix[9][7] + nb_conf_matrix[10][7] + nb_conf_matrix[11][7] + nb_conf_matrix[12][7] + nb_conf_matrix[13][7] + nb_conf_matrix[14][7] + nb_conf_matrix[15][7] + nb_conf_matrix[16][7] + nb_conf_matrix[17][7] + nb_conf_matrix[18][7] + nb_conf_matrix[19][7]
        nb_fp_9 = nb_conf_matrix[0][8] + nb_conf_matrix[1][8] + nb_conf_matrix[2][8] + nb_conf_matrix[3][8] + nb_conf_matrix[4][8] + nb_conf_matrix[5][8] + nb_conf_matrix[6][8] + nb_conf_matrix[7][8] + nb_conf_matrix[9][8] + nb_conf_matrix[10][8] + nb_conf_matrix[11][8] + nb_conf_matrix[12][8] + nb_conf_matrix[13][8] + nb_conf_matrix[14][8] + nb_conf_matrix[15][8] + nb_conf_matrix[16][8] + nb_conf_matrix[17][8] + nb_conf_matrix[18][8] + nb_conf_matrix[19][8]
        nb_fp_10 = nb_conf_matrix[0][9] + nb_conf_matrix[1][9] + nb_conf_matrix[2][9] + nb_conf_matrix[3][9] + nb_conf_matrix[4][9] + nb_conf_matrix[5][9] + nb_conf_matrix[6][9] + nb_conf_matrix[7][9] + nb_conf_matrix[8][9] + nb_conf_matrix[10][9] + nb_conf_matrix[11][9] + nb_conf_matrix[12][9] + nb_conf_matrix[13][9] + nb_conf_matrix[14][9] + nb_conf_matrix[15][9] + nb_conf_matrix[16][9] + nb_conf_matrix[17][9] + nb_conf_matrix[18][9] + nb_conf_matrix[19][9]
        nb_fp_11 = nb_conf_matrix[0][10] + nb_conf_matrix[1][10] + nb_conf_matrix[2][10] + nb_conf_matrix[3][10] + nb_conf_matrix[4][10] + nb_conf_matrix[5][10] + nb_conf_matrix[6][10] + nb_conf_matrix[7][10] + nb_conf_matrix[8][10] + nb_conf_matrix[9][10] + nb_conf_matrix[11][10] + nb_conf_matrix[12][10] + nb_conf_matrix[13][10] + nb_conf_matrix[14][10] + nb_conf_matrix[15][10] + nb_conf_matrix[16][10] + nb_conf_matrix[17][10] + nb_conf_matrix[18][10] + nb_conf_matrix[19][10]
        nb_fp_12 = nb_conf_matrix[0][11] + nb_conf_matrix[1][11] + nb_conf_matrix[2][11] + nb_conf_matrix[3][11] + nb_conf_matrix[4][11] + nb_conf_matrix[5][11] + nb_conf_matrix[6][11] + nb_conf_matrix[7][11] + nb_conf_matrix[8][11] + nb_conf_matrix[9][11] + nb_conf_matrix[10][11] + nb_conf_matrix[12][11] + nb_conf_matrix[13][11] + nb_conf_matrix[14][11] + nb_conf_matrix[15][11] + nb_conf_matrix[16][11] + nb_conf_matrix[17][11] + nb_conf_matrix[18][11] + nb_conf_matrix[19][11]
        nb_fp_13 = nb_conf_matrix[0][12] + nb_conf_matrix[1][12] + nb_conf_matrix[2][12] + nb_conf_matrix[3][12] + nb_conf_matrix[4][12] + nb_conf_matrix[5][12] + nb_conf_matrix[6][12] + nb_conf_matrix[7][12] + nb_conf_matrix[8][12] + nb_conf_matrix[9][12] + nb_conf_matrix[10][12] + nb_conf_matrix[11][12] + nb_conf_matrix[13][12] + nb_conf_matrix[14][12] + nb_conf_matrix[15][12] + nb_conf_matrix[16][12] + nb_conf_matrix[17][12] + nb_conf_matrix[18][12] + nb_conf_matrix[19][12]
        nb_fp_14 = nb_conf_matrix[0][13] + nb_conf_matrix[1][13] + nb_conf_matrix[2][13] + nb_conf_matrix[3][13] + nb_conf_matrix[4][13] + nb_conf_matrix[5][13] + nb_conf_matrix[6][13] + nb_conf_matrix[7][13] + nb_conf_matrix[8][13] + nb_conf_matrix[9][13] + nb_conf_matrix[10][13] + nb_conf_matrix[11][13] + nb_conf_matrix[12][13] + nb_conf_matrix[14][13] + nb_conf_matrix[15][13] + nb_conf_matrix[16][13] + nb_conf_matrix[17][13] + nb_conf_matrix[18][13] + nb_conf_matrix[19][13]
        nb_fp_15 = nb_conf_matrix[0][14] + nb_conf_matrix[1][14] + nb_conf_matrix[2][14] + nb_conf_matrix[3][14] + nb_conf_matrix[4][14] + nb_conf_matrix[5][14] + nb_conf_matrix[6][14] + nb_conf_matrix[7][14] + nb_conf_matrix[8][14] + nb_conf_matrix[9][14] + nb_conf_matrix[10][14] + nb_conf_matrix[11][14] + nb_conf_matrix[12][14] + nb_conf_matrix[13][14] + nb_conf_matrix[15][14] + nb_conf_matrix[16][14] + nb_conf_matrix[17][14] + nb_conf_matrix[18][14] + nb_conf_matrix[19][14]
        nb_fp_16 = nb_conf_matrix[0][15] + nb_conf_matrix[1][15] + nb_conf_matrix[2][15] + nb_conf_matrix[3][15] + nb_conf_matrix[4][15] + nb_conf_matrix[5][15] + nb_conf_matrix[6][15] + nb_conf_matrix[7][15] + nb_conf_matrix[8][15] + nb_conf_matrix[9][15] + nb_conf_matrix[10][15] + nb_conf_matrix[11][15] + nb_conf_matrix[12][15] + nb_conf_matrix[13][15] + nb_conf_matrix[14][15] + nb_conf_matrix[16][15] + nb_conf_matrix[17][15] + nb_conf_matrix[18][15] + nb_conf_matrix[19][15]
        nb_fp_17 = nb_conf_matrix[0][16] + nb_conf_matrix[1][16] + nb_conf_matrix[2][16] + nb_conf_matrix[3][16] + nb_conf_matrix[4][16] + nb_conf_matrix[5][16] + nb_conf_matrix[6][16] + nb_conf_matrix[7][16] + nb_conf_matrix[8][16] + nb_conf_matrix[9][16] + nb_conf_matrix[10][16] + nb_conf_matrix[11][16] + nb_conf_matrix[12][16] + nb_conf_matrix[13][16] + nb_conf_matrix[14][16] + nb_conf_matrix[15][16] + nb_conf_matrix[17][16] + nb_conf_matrix[18][16] + nb_conf_matrix[19][16]
        nb_fp_18 = nb_conf_matrix[0][17] + nb_conf_matrix[1][17] + nb_conf_matrix[2][17] + nb_conf_matrix[3][17] + nb_conf_matrix[4][17] + nb_conf_matrix[5][17] + nb_conf_matrix[6][17] + nb_conf_matrix[7][17] + nb_conf_matrix[8][17] + nb_conf_matrix[9][17] + nb_conf_matrix[10][17] + nb_conf_matrix[11][17] + nb_conf_matrix[12][17] + nb_conf_matrix[13][17] + nb_conf_matrix[14][17] + nb_conf_matrix[15][17] + nb_conf_matrix[16][17] + nb_conf_matrix[18][17] + nb_conf_matrix[19][17]
        nb_fp_19 = nb_conf_matrix[0][18] + nb_conf_matrix[1][18] + nb_conf_matrix[2][18] + nb_conf_matrix[3][18] + nb_conf_matrix[4][18] + nb_conf_matrix[5][18] + nb_conf_matrix[6][18] + nb_conf_matrix[7][18] + nb_conf_matrix[8][18] + nb_conf_matrix[9][18] + nb_conf_matrix[10][18] + nb_conf_matrix[11][18] + nb_conf_matrix[12][18] + nb_conf_matrix[13][18] + nb_conf_matrix[14][18] + nb_conf_matrix[15][18] + nb_conf_matrix[16][18] + nb_conf_matrix[17][18] + nb_conf_matrix[19][18]
        nb_fp_20 = nb_conf_matrix[0][19] + nb_conf_matrix[1][19] + nb_conf_matrix[2][19] + nb_conf_matrix[3][19] + nb_conf_matrix[4][19] + nb_conf_matrix[5][19] + nb_conf_matrix[6][19] + nb_conf_matrix[7][19] + nb_conf_matrix[8][19] + nb_conf_matrix[9][19] + nb_conf_matrix[10][19] + nb_conf_matrix[11][19] + nb_conf_matrix[12][19] + nb_conf_matrix[13][19] + nb_conf_matrix[14][19] + nb_conf_matrix[15][19] + nb_conf_matrix[16][19] + nb_conf_matrix[17][19] + nb_conf_matrix[18][19]
        #nb_fp_21 = nb_conf_matrix[0][20] + nb_conf_matrix[1][20] + nb_conf_matrix[2][20] + nb_conf_matrix[3][20] + nb_conf_matrix[4][20] + nb_conf_matrix[5][20] + nb_conf_matrix[6][20] + nb_conf_matrix[7][20] + nb_conf_matrix[8][20] + nb_conf_matrix[9][20] + nb_conf_matrix[10][20] + nb_conf_matrix[11][20] + nb_conf_matrix[12][20] + nb_conf_matrix[13][20] + nb_conf_matrix[14][20] + nb_conf_matrix[15][20] + nb_conf_matrix[16][20] + nb_conf_matrix[17][20] + nb_conf_matrix[18][20] + nb_conf_matrix[19][20] + nb_conf_matrix[21][20] + nb_conf_matrix[22][20] + nb_conf_matrix[23][20] + nb_conf_matrix[24][20] + nb_conf_matrix[25][20] + nb_conf_matrix[26][20] + nb_conf_matrix[27][20] + nb_conf_matrix[28][20]
        #nb_fp_22 = nb_conf_matrix[0][21] + nb_conf_matrix[1][21] + nb_conf_matrix[2][21] + nb_conf_matrix[3][21] + nb_conf_matrix[4][21] + nb_conf_matrix[5][21] + nb_conf_matrix[6][21] + nb_conf_matrix[7][21] + nb_conf_matrix[8][21] + nb_conf_matrix[9][21] + nb_conf_matrix[10][21] + nb_conf_matrix[11][21] + nb_conf_matrix[12][21] + nb_conf_matrix[13][21] + nb_conf_matrix[14][21] + nb_conf_matrix[15][21] + nb_conf_matrix[16][21] + nb_conf_matrix[17][21] + nb_conf_matrix[18][21] + nb_conf_matrix[19][21] + nb_conf_matrix[20][21] + nb_conf_matrix[22][21] + nb_conf_matrix[23][21] + nb_conf_matrix[24][21] + nb_conf_matrix[25][21] + nb_conf_matrix[26][21] + nb_conf_matrix[27][21] + nb_conf_matrix[28][21]
        #nb_fp_23 = nb_conf_matrix[0][22] + nb_conf_matrix[1][22] + nb_conf_matrix[2][22] + nb_conf_matrix[3][22] + nb_conf_matrix[4][22] + nb_conf_matrix[5][22] + nb_conf_matrix[6][22] + nb_conf_matrix[7][22] + nb_conf_matrix[8][22] + nb_conf_matrix[9][22] + nb_conf_matrix[10][22] + nb_conf_matrix[11][22] + nb_conf_matrix[12][22] + nb_conf_matrix[13][22] + nb_conf_matrix[14][22] + nb_conf_matrix[15][22] + nb_conf_matrix[16][22] + nb_conf_matrix[17][22] + nb_conf_matrix[18][22] + nb_conf_matrix[19][22] + nb_conf_matrix[20][22] + nb_conf_matrix[21][22] + nb_conf_matrix[23][22] + nb_conf_matrix[24][22] + nb_conf_matrix[25][22] + nb_conf_matrix[26][22] + nb_conf_matrix[27][22] + nb_conf_matrix[28][22]
        #nb_fp_24 = nb_conf_matrix[0][23] + nb_conf_matrix[1][23] + nb_conf_matrix[2][23] + nb_conf_matrix[3][23] + nb_conf_matrix[4][23] + nb_conf_matrix[5][23] + nb_conf_matrix[6][23] + nb_conf_matrix[7][23] + nb_conf_matrix[8][23] + nb_conf_matrix[9][23] + nb_conf_matrix[10][23] + nb_conf_matrix[11][23] + nb_conf_matrix[12][23] + nb_conf_matrix[13][23] + nb_conf_matrix[14][23] + nb_conf_matrix[15][23] + nb_conf_matrix[16][23] + nb_conf_matrix[17][23] + nb_conf_matrix[18][23] + nb_conf_matrix[19][23] + nb_conf_matrix[20][23] + nb_conf_matrix[21][23] + nb_conf_matrix[22][23] + nb_conf_matrix[24][23] + nb_conf_matrix[25][23] + nb_conf_matrix[26][23] + nb_conf_matrix[27][23] + nb_conf_matrix[28][23]
        #nb_fp_25 = nb_conf_matrix[0][24] + nb_conf_matrix[1][24] + nb_conf_matrix[2][24] + nb_conf_matrix[3][24] + nb_conf_matrix[4][24] + nb_conf_matrix[5][24] + nb_conf_matrix[6][24] + nb_conf_matrix[7][24] + nb_conf_matrix[8][24] + nb_conf_matrix[9][24] + nb_conf_matrix[10][24] + nb_conf_matrix[11][24] + nb_conf_matrix[12][24] + nb_conf_matrix[13][24] + nb_conf_matrix[14][24] + nb_conf_matrix[15][24] + nb_conf_matrix[16][24] + nb_conf_matrix[17][24] + nb_conf_matrix[18][24] + nb_conf_matrix[19][24] + nb_conf_matrix[20][24] + nb_conf_matrix[21][24] + nb_conf_matrix[22][24] + nb_conf_matrix[23][24] + nb_conf_matrix[25][24] + nb_conf_matrix[26][24] + nb_conf_matrix[27][24] + nb_conf_matrix[28][24]
        #nb_fp_26 = nb_conf_matrix[0][25] + nb_conf_matrix[1][25] + nb_conf_matrix[2][25] + nb_conf_matrix[3][25] + nb_conf_matrix[4][25] + nb_conf_matrix[5][25] + nb_conf_matrix[6][25] + nb_conf_matrix[7][25] + nb_conf_matrix[8][25] + nb_conf_matrix[9][25] + nb_conf_matrix[10][25] + nb_conf_matrix[11][25] + nb_conf_matrix[12][25] + nb_conf_matrix[13][25] + nb_conf_matrix[14][25] + nb_conf_matrix[15][25] + nb_conf_matrix[16][25] + nb_conf_matrix[17][25] + nb_conf_matrix[18][25] + nb_conf_matrix[19][25] + nb_conf_matrix[20][25] + nb_conf_matrix[21][25] + nb_conf_matrix[22][25] + nb_conf_matrix[23][25] + nb_conf_matrix[24][25] + nb_conf_matrix[26][25] + nb_conf_matrix[27][25] + nb_conf_matrix[28][25]
        #nb_fp_27 = nb_conf_matrix[0][26] + nb_conf_matrix[1][26] + nb_conf_matrix[2][26] + nb_conf_matrix[3][26] + nb_conf_matrix[4][26] + nb_conf_matrix[5][26] + nb_conf_matrix[6][26] + nb_conf_matrix[7][26] + nb_conf_matrix[8][26] + nb_conf_matrix[9][26] + nb_conf_matrix[10][26] + nb_conf_matrix[11][26] + nb_conf_matrix[12][26] + nb_conf_matrix[13][26] + nb_conf_matrix[14][26] + nb_conf_matrix[15][26] + nb_conf_matrix[16][26] + nb_conf_matrix[17][26] + nb_conf_matrix[18][26] + nb_conf_matrix[19][26] + nb_conf_matrix[20][26] + nb_conf_matrix[21][26] + nb_conf_matrix[22][26] + nb_conf_matrix[23][26] + nb_conf_matrix[24][26] + nb_conf_matrix[25][26] + nb_conf_matrix[27][26] + nb_conf_matrix[28][26]
        #nb_fp_28 = nb_conf_matrix[0][27] + nb_conf_matrix[1][27] + nb_conf_matrix[2][27] + nb_conf_matrix[3][27] + nb_conf_matrix[4][27] + nb_conf_matrix[5][27] + nb_conf_matrix[6][27] + nb_conf_matrix[7][27] + nb_conf_matrix[8][27] + nb_conf_matrix[9][27] + nb_conf_matrix[10][27] + nb_conf_matrix[11][27] + nb_conf_matrix[12][27] + nb_conf_matrix[13][27] + nb_conf_matrix[14][27] + nb_conf_matrix[15][27] + nb_conf_matrix[16][27] + nb_conf_matrix[17][27] + nb_conf_matrix[18][27] + nb_conf_matrix[19][27] + nb_conf_matrix[20][27] + nb_conf_matrix[21][27] + nb_conf_matrix[22][27] + nb_conf_matrix[23][27] + nb_conf_matrix[24][27] + nb_conf_matrix[25][27] + nb_conf_matrix[26][27] + nb_conf_matrix[28][27]
        #nb_fp_29 = nb_conf_matrix[0][28] + nb_conf_matrix[1][28] + nb_conf_matrix[2][28] + nb_conf_matrix[3][28] + nb_conf_matrix[4][28] + nb_conf_matrix[5][28] + nb_conf_matrix[6][28] + nb_conf_matrix[7][28] + nb_conf_matrix[8][28] + nb_conf_matrix[9][28] + nb_conf_matrix[10][28] + nb_conf_matrix[11][28] + nb_conf_matrix[12][28] + nb_conf_matrix[13][28] + nb_conf_matrix[14][28] + nb_conf_matrix[15][28] + nb_conf_matrix[16][28] + nb_conf_matrix[17][28] + nb_conf_matrix[18][28] + nb_conf_matrix[19][28] + nb_conf_matrix[20][28] + nb_conf_matrix[21][28] + nb_conf_matrix[22][28] + nb_conf_matrix[23][28] + nb_conf_matrix[24][28] + nb_conf_matrix[25][28] + nb_conf_matrix[26][28] + nb_conf_matrix[27][28]

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
        if nb_tp_6 + nb_fp_6 == 0:
            nb_precision_6 = 0
        else:
            nb_precision_6 = nb_tp_6 / (nb_tp_6 + nb_fp_6)
        if nb_tp_7 + nb_fp_7 == 0:
            nb_precision_7 = 0
        else:
            nb_precision_7 = nb_tp_7 / (nb_tp_7 + nb_fp_7)
        if nb_tp_8 + nb_fp_8 == 0:
            nb_precision_8 = 0
        else:
            nb_precision_8 = nb_tp_8 / (nb_tp_8 + nb_fp_8)
        if nb_tp_9 + nb_fp_9 == 0:
            nb_precision_9 = 0
        else:
            nb_precision_9 = nb_tp_9 / (nb_tp_9 + nb_fp_9)
        if nb_tp_10 + nb_fp_10 == 0:
            nb_precision_10 = 0
        else:
            nb_precision_10 = nb_tp_10 / (nb_tp_10 + nb_fp_10)
        if nb_tp_11 + nb_fp_11 == 0:
            nb_precision_11 = 0
        else:
            nb_precision_11 = nb_tp_11 / (nb_tp_11 + nb_fp_11)
        if nb_tp_12 + nb_fp_12 == 0:
            nb_precision_12 = 0
        else:
            nb_precision_12 = nb_tp_12 / (nb_tp_12 + nb_fp_12)
        if nb_tp_13 + nb_fp_13 == 0:
            nb_precision_13 = 0
        else:
            nb_precision_13 = nb_tp_13 / (nb_tp_13 + nb_fp_13)
        if nb_tp_14 + nb_fp_14 == 0:
            nb_precision_14 = 0
        else:
            nb_precision_14 = nb_tp_14 / (nb_tp_14 + nb_fp_14)
        if nb_tp_15 + nb_fp_15 == 0:
            nb_precision_15 = 0
        else:
            nb_precision_15 = nb_tp_15 / (nb_tp_15 + nb_fp_15)
        if nb_tp_16 + nb_fp_16 == 0:
            nb_precision_16 = 0
        else:
            nb_precision_16 = nb_tp_16 / (nb_tp_16 + nb_fp_16)
        if nb_tp_17 + nb_fp_17 == 0:
            nb_precision_17 = 0
        else:
            nb_precision_17 = nb_tp_17 / (nb_tp_17 + nb_fp_17)
        if nb_tp_18 + nb_fp_18 == 0:
            nb_precision_18 = 0
        else:
            nb_precision_18 = nb_tp_18 / (nb_tp_18 + nb_fp_18)
        if nb_tp_19 + nb_fp_19 == 0:
            nb_precision_19 = 0
        else:
            nb_precision_19 = nb_tp_19 / (nb_tp_19 + nb_fp_19)
        if nb_tp_20 + nb_fp_20 == 0:
            nb_precision_20 = 0
        else:
            nb_precision_20 = nb_tp_20 / (nb_tp_20 + nb_fp_20)
        '''
        if nb_tp_21 + nb_fp_21 == 0:
            nb_precision_21 = 0
        else:
            nb_precision_21 = nb_tp_21 / (nb_tp_21 + nb_fp_21)
        if nb_tp_22 + nb_fp_22 == 0:
            nb_precision_22 = 0
        else:
            nb_precision_22 = nb_tp_22 / (nb_tp_22 + nb_fp_22)
        if nb_tp_23 + nb_fp_23 == 0:
            nb_precision_23 = 0
        else:
            nb_precision_23 = nb_tp_23 / (nb_tp_23 + nb_fp_23)
        if nb_tp_24 + nb_fp_24 == 0:
            nb_precision_24 = 0
        else:
            nb_precision_24 = nb_tp_24 / (nb_tp_24 + nb_fp_24)
        if nb_tp_25 + nb_fp_25 == 0:
            nb_precision_25 = 0
        else:
            nb_precision_25 = nb_tp_25 / (nb_tp_25 + nb_fp_25)
        if nb_tp_26 + nb_fp_26 == 0:
            nb_precision_26 = 0
        else:
            nb_precision_26 = nb_tp_26 / (nb_tp_26 + nb_fp_26)
        if nb_tp_27 + nb_fp_27 == 0:
            nb_precision_27 = 0
        else:
            nb_precision_27 = nb_tp_27 / (nb_tp_27 + nb_fp_27)
        if nb_tp_28 + nb_fp_28 == 0:
            nb_precision_28 = 0
        else:
            nb_precision_28 = nb_tp_28 / (nb_tp_28 + nb_fp_28)
        if nb_tp_29 + nb_fp_29 == 0:
            nb_precision_29 = 0
        else:
            nb_precision_29 = nb_tp_29 / (nb_tp_29 + nb_fp_29)
        '''
        nb_precision_avg = (nb_precision_1 + nb_precision_2 + nb_precision_3 + nb_precision_4 + nb_precision_5 + nb_precision_6 + nb_precision_7 + nb_precision_8 + nb_precision_9 + nb_precision_10 + nb_precision_11 + nb_precision_12 + nb_precision_13 + nb_precision_14 + nb_precision_15 + nb_precision_16 + nb_precision_17 + nb_precision_18 + nb_precision_19 + nb_precision_20) / 20
        return nb_precision_avg


    def get_recall_pen_1(nb_conf_matrix):
        nb_tp_1 = nb_conf_matrix[0][0]
        nb_tp_2 = nb_conf_matrix[1][1]
        nb_tp_3 = nb_conf_matrix[2][2]
        nb_tp_4 = nb_conf_matrix[3][3]
        nb_tp_5 = nb_conf_matrix[4][4]
        nb_tp_6 = nb_conf_matrix[5][5]
        nb_tp_7 = nb_conf_matrix[6][6]
        nb_tp_8 = nb_conf_matrix[7][7]
        nb_tp_9 = nb_conf_matrix[8][8]
        nb_tp_10 = nb_conf_matrix[9][9]
        nb_tp_11 = nb_conf_matrix[10][10]
        nb_tp_12 = nb_conf_matrix[11][11]
        nb_tp_13 = nb_conf_matrix[12][12]
        nb_tp_14 = nb_conf_matrix[13][13]
        nb_tp_15 = nb_conf_matrix[14][14]
        nb_tp_16 = nb_conf_matrix[15][15]
        nb_tp_17 = nb_conf_matrix[16][16]
        nb_tp_18 = nb_conf_matrix[17][17]
        nb_tp_19 = nb_conf_matrix[18][18]
        nb_tp_20 = nb_conf_matrix[19][19]
        #nb_tp_21 = nb_conf_matrix[20][20]
        #nb_tp_22 = nb_conf_matrix[21][21]
        #nb_tp_23 = nb_conf_matrix[22][22]
        #nb_tp_24 = nb_conf_matrix[23][23]
        #nb_tp_25 = nb_conf_matrix[24][24]
        #nb_tp_26 = nb_conf_matrix[25][25]
        #nb_tp_27 = nb_conf_matrix[26][26]
        #nb_tp_28 = nb_conf_matrix[27][27]
        #nb_tp_29 = nb_conf_matrix[28][28]

        nb_fn_1 = nb_conf_matrix[0][1] + nb_conf_matrix[0][2] + nb_conf_matrix[0][3] + nb_conf_matrix[0][4] + nb_conf_matrix[0][5] + nb_conf_matrix[0][6] + nb_conf_matrix[0][7] + nb_conf_matrix[0][8] + nb_conf_matrix[0][9] + nb_conf_matrix[0][10] + nb_conf_matrix[0][11] + nb_conf_matrix[0][12] + nb_conf_matrix[0][13] + nb_conf_matrix[0][14] + nb_conf_matrix[0][15] + nb_conf_matrix[0][16] + nb_conf_matrix[0][17] + nb_conf_matrix[0][18] + nb_conf_matrix[0][19]
        nb_fn_2 = nb_conf_matrix[1][0] + nb_conf_matrix[1][2] + nb_conf_matrix[1][3] + nb_conf_matrix[1][4] + nb_conf_matrix[1][5] + nb_conf_matrix[1][6] + nb_conf_matrix[1][7] + nb_conf_matrix[1][8] + nb_conf_matrix[1][9] + nb_conf_matrix[1][10] + nb_conf_matrix[1][11] + nb_conf_matrix[1][12] + nb_conf_matrix[1][13] + nb_conf_matrix[1][14] + nb_conf_matrix[1][15] + nb_conf_matrix[1][16] + nb_conf_matrix[1][17] + nb_conf_matrix[1][18] + nb_conf_matrix[1][19]
        nb_fn_3 = nb_conf_matrix[2][0] + nb_conf_matrix[2][1] + nb_conf_matrix[2][3] + nb_conf_matrix[2][4] + nb_conf_matrix[2][5] + nb_conf_matrix[2][6] + nb_conf_matrix[2][7] + nb_conf_matrix[2][8] + nb_conf_matrix[2][9] + nb_conf_matrix[2][10] + nb_conf_matrix[2][11] + nb_conf_matrix[2][12] + nb_conf_matrix[2][13] + nb_conf_matrix[2][14] + nb_conf_matrix[2][15] + nb_conf_matrix[2][16] + nb_conf_matrix[2][17] + nb_conf_matrix[2][18] + nb_conf_matrix[2][19]
        nb_fn_4 = nb_conf_matrix[3][0] + nb_conf_matrix[3][1] + nb_conf_matrix[3][2] + nb_conf_matrix[3][4] + nb_conf_matrix[3][5] + nb_conf_matrix[3][6] + nb_conf_matrix[3][7] + nb_conf_matrix[3][8] + nb_conf_matrix[3][9] + nb_conf_matrix[3][10] + nb_conf_matrix[3][11] + nb_conf_matrix[3][12] + nb_conf_matrix[3][13] + nb_conf_matrix[3][14] + nb_conf_matrix[3][15] + nb_conf_matrix[3][16] + nb_conf_matrix[3][17] + nb_conf_matrix[3][18] + nb_conf_matrix[3][19]
        nb_fn_5 = nb_conf_matrix[4][0] + nb_conf_matrix[4][1] + nb_conf_matrix[4][2] + nb_conf_matrix[4][3] + nb_conf_matrix[4][5] + nb_conf_matrix[4][6] + nb_conf_matrix[4][7] + nb_conf_matrix[4][8] + nb_conf_matrix[4][9] + nb_conf_matrix[4][10] + nb_conf_matrix[4][11] + nb_conf_matrix[4][12] + nb_conf_matrix[4][13] + nb_conf_matrix[4][14] + nb_conf_matrix[4][15] + nb_conf_matrix[4][16] + nb_conf_matrix[4][17] + nb_conf_matrix[4][18] + nb_conf_matrix[4][19]
        nb_fn_6 = nb_conf_matrix[5][0] + nb_conf_matrix[5][1] + nb_conf_matrix[5][2] + nb_conf_matrix[5][3] + nb_conf_matrix[5][4] + nb_conf_matrix[5][6] + nb_conf_matrix[5][7] + nb_conf_matrix[5][8] + nb_conf_matrix[5][9] + nb_conf_matrix[5][10] + nb_conf_matrix[5][11] + nb_conf_matrix[5][12] + nb_conf_matrix[5][13] + nb_conf_matrix[5][14] + nb_conf_matrix[5][15] + nb_conf_matrix[5][16] + nb_conf_matrix[5][17] + nb_conf_matrix[5][18] + nb_conf_matrix[5][19]
        nb_fn_7 = nb_conf_matrix[6][0] + nb_conf_matrix[6][1] + nb_conf_matrix[6][2] + nb_conf_matrix[6][3] + nb_conf_matrix[6][4] + nb_conf_matrix[6][5] + nb_conf_matrix[6][7] + nb_conf_matrix[6][8] + nb_conf_matrix[6][9] + nb_conf_matrix[6][10] + nb_conf_matrix[6][11] + nb_conf_matrix[6][12] + nb_conf_matrix[6][13] + nb_conf_matrix[6][14] + nb_conf_matrix[6][15] + nb_conf_matrix[6][16] + nb_conf_matrix[6][17] + nb_conf_matrix[6][18] + nb_conf_matrix[6][19]
        nb_fn_8 = nb_conf_matrix[7][0] + nb_conf_matrix[7][1] + nb_conf_matrix[7][2] + nb_conf_matrix[7][3] + nb_conf_matrix[7][4] + nb_conf_matrix[7][5] + nb_conf_matrix[7][6] + nb_conf_matrix[7][8] + nb_conf_matrix[7][9] + nb_conf_matrix[7][10] + nb_conf_matrix[7][11] + nb_conf_matrix[7][12] + nb_conf_matrix[7][13] + nb_conf_matrix[7][14] + nb_conf_matrix[7][15] + nb_conf_matrix[7][16] + nb_conf_matrix[7][17] + nb_conf_matrix[7][18] + nb_conf_matrix[7][19]
        nb_fn_9 = nb_conf_matrix[8][0] + nb_conf_matrix[8][1] + nb_conf_matrix[8][2] + nb_conf_matrix[8][3] + nb_conf_matrix[8][4] + nb_conf_matrix[8][5] + nb_conf_matrix[8][6] + nb_conf_matrix[8][7] + nb_conf_matrix[8][9] + nb_conf_matrix[8][10] + nb_conf_matrix[8][11] + nb_conf_matrix[8][12] + nb_conf_matrix[8][13] + nb_conf_matrix[8][14] + nb_conf_matrix[8][15] + nb_conf_matrix[8][16] + nb_conf_matrix[8][17] + nb_conf_matrix[8][18] + nb_conf_matrix[8][19]
        nb_fn_10 = nb_conf_matrix[9][0] + nb_conf_matrix[9][1] + nb_conf_matrix[9][2] + nb_conf_matrix[9][3] + nb_conf_matrix[9][4] + nb_conf_matrix[9][5] + nb_conf_matrix[9][6] + nb_conf_matrix[9][7] + nb_conf_matrix[9][8] + nb_conf_matrix[9][10] + nb_conf_matrix[9][11] + nb_conf_matrix[9][12] + nb_conf_matrix[9][13] + nb_conf_matrix[9][14] + nb_conf_matrix[9][15] + nb_conf_matrix[9][16] + nb_conf_matrix[9][17] + nb_conf_matrix[9][18] + nb_conf_matrix[9][19]
        nb_fn_11 = nb_conf_matrix[10][0] + nb_conf_matrix[10][1] + nb_conf_matrix[10][2] + nb_conf_matrix[10][3] + nb_conf_matrix[10][4] + nb_conf_matrix[10][5] + nb_conf_matrix[10][6] + nb_conf_matrix[10][7] + nb_conf_matrix[10][8] + nb_conf_matrix[10][9] + nb_conf_matrix[10][11] + nb_conf_matrix[10][12] + nb_conf_matrix[10][13] + nb_conf_matrix[10][14] + nb_conf_matrix[10][15] + nb_conf_matrix[10][16] + nb_conf_matrix[10][17] + nb_conf_matrix[10][18] + nb_conf_matrix[10][19]
        nb_fn_12 = nb_conf_matrix[11][0] + nb_conf_matrix[11][1] + nb_conf_matrix[11][2] + nb_conf_matrix[11][3] + nb_conf_matrix[11][4] + nb_conf_matrix[11][5] + nb_conf_matrix[11][6] + nb_conf_matrix[11][7] + nb_conf_matrix[11][8] + nb_conf_matrix[11][9] + nb_conf_matrix[11][10] + nb_conf_matrix[11][12] + nb_conf_matrix[11][13] + nb_conf_matrix[11][14] + nb_conf_matrix[11][15] + nb_conf_matrix[11][16] + nb_conf_matrix[11][17] + nb_conf_matrix[11][18] + nb_conf_matrix[11][19]
        nb_fn_13 = nb_conf_matrix[12][0] + nb_conf_matrix[12][1] + nb_conf_matrix[12][2] + nb_conf_matrix[12][3] + nb_conf_matrix[12][4] + nb_conf_matrix[12][5] + nb_conf_matrix[12][6] + nb_conf_matrix[12][7] + nb_conf_matrix[12][8] + nb_conf_matrix[12][9] + nb_conf_matrix[12][10] + nb_conf_matrix[12][11] + nb_conf_matrix[12][13] + nb_conf_matrix[12][14] + nb_conf_matrix[12][15] + nb_conf_matrix[12][16] + nb_conf_matrix[12][17] + nb_conf_matrix[12][18] + nb_conf_matrix[12][19]
        nb_fn_14 = nb_conf_matrix[13][0] + nb_conf_matrix[13][1] + nb_conf_matrix[13][2] + nb_conf_matrix[13][3] + nb_conf_matrix[13][4] + nb_conf_matrix[13][5] + nb_conf_matrix[13][6] + nb_conf_matrix[13][7] + nb_conf_matrix[13][8] + nb_conf_matrix[13][9] + nb_conf_matrix[13][10] + nb_conf_matrix[13][11] + nb_conf_matrix[13][12] + nb_conf_matrix[13][14] + nb_conf_matrix[13][15] + nb_conf_matrix[13][16] + nb_conf_matrix[13][17] + nb_conf_matrix[13][18] + nb_conf_matrix[13][19]
        nb_fn_15 = nb_conf_matrix[14][0] + nb_conf_matrix[14][1] + nb_conf_matrix[14][2] + nb_conf_matrix[14][3] + nb_conf_matrix[14][4] + nb_conf_matrix[14][5] + nb_conf_matrix[14][6] + nb_conf_matrix[14][7] + nb_conf_matrix[14][8] + nb_conf_matrix[14][9] + nb_conf_matrix[14][10] + nb_conf_matrix[14][11] + nb_conf_matrix[14][12] + nb_conf_matrix[14][13] + nb_conf_matrix[14][15] + nb_conf_matrix[14][16] + nb_conf_matrix[14][17] + nb_conf_matrix[14][18] + nb_conf_matrix[14][19]
        nb_fn_16 = nb_conf_matrix[15][0] + nb_conf_matrix[15][1] + nb_conf_matrix[15][2] + nb_conf_matrix[15][3] + nb_conf_matrix[15][4] + nb_conf_matrix[15][5] + nb_conf_matrix[15][6] + nb_conf_matrix[15][7] + nb_conf_matrix[15][8] + nb_conf_matrix[15][9] + nb_conf_matrix[15][10] + nb_conf_matrix[15][11] + nb_conf_matrix[15][12] + nb_conf_matrix[15][13] + nb_conf_matrix[15][14] + nb_conf_matrix[15][16] + nb_conf_matrix[15][17] + nb_conf_matrix[15][18] + nb_conf_matrix[15][19]
        nb_fn_17 = nb_conf_matrix[16][0] + nb_conf_matrix[16][1] + nb_conf_matrix[16][2] + nb_conf_matrix[16][3] + nb_conf_matrix[16][4] + nb_conf_matrix[16][5] + nb_conf_matrix[16][6] + nb_conf_matrix[16][7] + nb_conf_matrix[16][8] + nb_conf_matrix[16][9] + nb_conf_matrix[16][10] + nb_conf_matrix[16][11] + nb_conf_matrix[16][12] + nb_conf_matrix[16][13] + nb_conf_matrix[16][14] + nb_conf_matrix[16][15] + nb_conf_matrix[16][17] + nb_conf_matrix[16][18] + nb_conf_matrix[16][19]
        nb_fn_18 = nb_conf_matrix[17][0] + nb_conf_matrix[17][1] + nb_conf_matrix[17][2] + nb_conf_matrix[17][3] + nb_conf_matrix[17][4] + nb_conf_matrix[17][5] + nb_conf_matrix[17][6] + nb_conf_matrix[17][7] + nb_conf_matrix[17][8] + nb_conf_matrix[17][9] + nb_conf_matrix[17][10] + nb_conf_matrix[17][11] + nb_conf_matrix[17][12] + nb_conf_matrix[17][13] + nb_conf_matrix[17][14] + nb_conf_matrix[17][15] + nb_conf_matrix[17][16] + nb_conf_matrix[17][18] + nb_conf_matrix[17][19]
        nb_fn_19 = nb_conf_matrix[18][0] + nb_conf_matrix[18][1] + nb_conf_matrix[18][2] + nb_conf_matrix[18][3] + nb_conf_matrix[18][4] + nb_conf_matrix[18][5] + nb_conf_matrix[18][6] + nb_conf_matrix[18][7] + nb_conf_matrix[18][8] + nb_conf_matrix[18][9] + nb_conf_matrix[18][10] + nb_conf_matrix[18][11] + nb_conf_matrix[18][12] + nb_conf_matrix[18][13] + nb_conf_matrix[18][14] + nb_conf_matrix[18][15] + nb_conf_matrix[18][16] + nb_conf_matrix[18][17] + nb_conf_matrix[18][19]
        nb_fn_20 = nb_conf_matrix[19][0] + nb_conf_matrix[19][1] + nb_conf_matrix[19][2] + nb_conf_matrix[19][3] + nb_conf_matrix[19][4] + nb_conf_matrix[19][5] + nb_conf_matrix[19][6] + nb_conf_matrix[19][7] + nb_conf_matrix[19][8] + nb_conf_matrix[19][9] + nb_conf_matrix[19][10] + nb_conf_matrix[19][11] + nb_conf_matrix[19][12] + nb_conf_matrix[19][13] + nb_conf_matrix[19][14] + nb_conf_matrix[19][15] + nb_conf_matrix[19][16] + nb_conf_matrix[19][17] + nb_conf_matrix[19][18]
        #nb_fn_21 = nb_conf_matrix[20][0] + nb_conf_matrix[20][1] + nb_conf_matrix[20][2] + nb_conf_matrix[20][3] + nb_conf_matrix[20][4] + nb_conf_matrix[20][5] + nb_conf_matrix[20][6] + nb_conf_matrix[20][7] + nb_conf_matrix[20][8] + nb_conf_matrix[20][9] + nb_conf_matrix[20][10] + nb_conf_matrix[20][11] + nb_conf_matrix[20][12] + nb_conf_matrix[20][13] + nb_conf_matrix[20][14] + nb_conf_matrix[20][15] + nb_conf_matrix[20][16] + nb_conf_matrix[20][17] + nb_conf_matrix[20][18] + nb_conf_matrix[20][19] + nb_conf_matrix[20][21] + nb_conf_matrix[20][22] + nb_conf_matrix[20][23] + nb_conf_matrix[20][24] + nb_conf_matrix[20][25] + nb_conf_matrix[20][26] + nb_conf_matrix[20][27] + nb_conf_matrix[20][28]
        #nb_fn_22 = nb_conf_matrix[21][0] + nb_conf_matrix[21][1] + nb_conf_matrix[21][2] + nb_conf_matrix[21][3] + nb_conf_matrix[21][4] + nb_conf_matrix[21][5] + nb_conf_matrix[21][6] + nb_conf_matrix[21][7] + nb_conf_matrix[21][8] + nb_conf_matrix[21][9] + nb_conf_matrix[21][10] + nb_conf_matrix[21][11] + nb_conf_matrix[21][12] + nb_conf_matrix[21][13] + nb_conf_matrix[21][14] + nb_conf_matrix[21][15] + nb_conf_matrix[21][16] + nb_conf_matrix[21][17] + nb_conf_matrix[21][18] + nb_conf_matrix[21][19] + nb_conf_matrix[21][20] + nb_conf_matrix[21][22] + nb_conf_matrix[21][23] + nb_conf_matrix[21][24] + nb_conf_matrix[21][25] + nb_conf_matrix[21][26] + nb_conf_matrix[21][27] + nb_conf_matrix[21][28]
        #nb_fn_23 = nb_conf_matrix[22][0] + nb_conf_matrix[22][1] + nb_conf_matrix[22][2] + nb_conf_matrix[22][3] + nb_conf_matrix[22][4] + nb_conf_matrix[22][5] + nb_conf_matrix[22][6] + nb_conf_matrix[22][7] + nb_conf_matrix[22][8] + nb_conf_matrix[22][9] + nb_conf_matrix[22][10] + nb_conf_matrix[22][11] + nb_conf_matrix[22][12] + nb_conf_matrix[22][13] + nb_conf_matrix[22][14] + nb_conf_matrix[22][15] + nb_conf_matrix[22][16] + nb_conf_matrix[22][17] + nb_conf_matrix[22][18] + nb_conf_matrix[22][19] + nb_conf_matrix[22][20] + nb_conf_matrix[22][21] + nb_conf_matrix[22][23] + nb_conf_matrix[22][24] + nb_conf_matrix[22][25] + nb_conf_matrix[22][26] + nb_conf_matrix[22][27] + nb_conf_matrix[22][28]
        #nb_fn_24 = nb_conf_matrix[23][0] + nb_conf_matrix[23][1] + nb_conf_matrix[23][2] + nb_conf_matrix[23][3] + nb_conf_matrix[23][4] + nb_conf_matrix[23][5] + nb_conf_matrix[23][6] + nb_conf_matrix[23][7] + nb_conf_matrix[23][8] + nb_conf_matrix[23][9] + nb_conf_matrix[23][10] + nb_conf_matrix[23][11] + nb_conf_matrix[23][12] + nb_conf_matrix[23][13] + nb_conf_matrix[23][14] + nb_conf_matrix[23][15] + nb_conf_matrix[23][16] + nb_conf_matrix[23][17] + nb_conf_matrix[23][18] + nb_conf_matrix[23][19] + nb_conf_matrix[23][20] + nb_conf_matrix[23][21] + nb_conf_matrix[23][22] + nb_conf_matrix[23][24] + nb_conf_matrix[23][25] + nb_conf_matrix[23][26] + nb_conf_matrix[23][27] + nb_conf_matrix[23][28]
        #nb_fn_25 = nb_conf_matrix[24][0] + nb_conf_matrix[24][1] + nb_conf_matrix[24][2] + nb_conf_matrix[24][3] + nb_conf_matrix[24][4] + nb_conf_matrix[24][5] + nb_conf_matrix[24][6] + nb_conf_matrix[24][7] + nb_conf_matrix[24][8] + nb_conf_matrix[24][9] + nb_conf_matrix[24][10] + nb_conf_matrix[24][11] + nb_conf_matrix[24][12] + nb_conf_matrix[24][13] + nb_conf_matrix[24][14] + nb_conf_matrix[24][15] + nb_conf_matrix[24][16] + nb_conf_matrix[24][17] + nb_conf_matrix[24][18] + nb_conf_matrix[24][19] + nb_conf_matrix[24][20] + nb_conf_matrix[24][21] + nb_conf_matrix[24][22] + nb_conf_matrix[24][23] + nb_conf_matrix[24][25] + nb_conf_matrix[24][26] + nb_conf_matrix[24][27] + nb_conf_matrix[24][28]
        #nb_fn_26 = nb_conf_matrix[25][0] + nb_conf_matrix[25][1] + nb_conf_matrix[25][2] + nb_conf_matrix[25][3] + nb_conf_matrix[25][4] + nb_conf_matrix[25][5] + nb_conf_matrix[25][6] + nb_conf_matrix[25][7] + nb_conf_matrix[25][8] + nb_conf_matrix[25][9] + nb_conf_matrix[25][10] + nb_conf_matrix[25][11] + nb_conf_matrix[25][12] + nb_conf_matrix[25][13] + nb_conf_matrix[25][14] + nb_conf_matrix[25][15] + nb_conf_matrix[25][16] + nb_conf_matrix[25][17] + nb_conf_matrix[25][18] + nb_conf_matrix[25][19] + nb_conf_matrix[25][20] + nb_conf_matrix[25][21] + nb_conf_matrix[25][22] + nb_conf_matrix[25][23] + nb_conf_matrix[25][24] + nb_conf_matrix[25][26] + nb_conf_matrix[25][27] + nb_conf_matrix[25][28]
        #nb_fn_27 = nb_conf_matrix[26][0] + nb_conf_matrix[26][1] + nb_conf_matrix[26][2] + nb_conf_matrix[26][3] + nb_conf_matrix[26][4] + nb_conf_matrix[26][5] + nb_conf_matrix[26][6] + nb_conf_matrix[26][7] + nb_conf_matrix[26][8] + nb_conf_matrix[26][9] + nb_conf_matrix[26][10] + nb_conf_matrix[26][11] + nb_conf_matrix[26][12] + nb_conf_matrix[26][13] + nb_conf_matrix[26][14] + nb_conf_matrix[26][15] + nb_conf_matrix[26][16] + nb_conf_matrix[26][17] + nb_conf_matrix[26][18] + nb_conf_matrix[26][19] + nb_conf_matrix[26][20] + nb_conf_matrix[26][21] + nb_conf_matrix[26][22] + nb_conf_matrix[26][23] + nb_conf_matrix[26][24] + nb_conf_matrix[26][25] + nb_conf_matrix[26][27] + nb_conf_matrix[26][28]
        #nb_fn_28 = nb_conf_matrix[27][0] + nb_conf_matrix[27][1] + nb_conf_matrix[27][2] + nb_conf_matrix[27][3] + nb_conf_matrix[27][4] + nb_conf_matrix[27][5] + nb_conf_matrix[27][6] + nb_conf_matrix[27][7] + nb_conf_matrix[27][8] + nb_conf_matrix[27][9] + nb_conf_matrix[27][10] + nb_conf_matrix[27][11] + nb_conf_matrix[27][12] + nb_conf_matrix[27][13] + nb_conf_matrix[27][14] + nb_conf_matrix[27][15] + nb_conf_matrix[27][16] + nb_conf_matrix[27][17] + nb_conf_matrix[27][18] + nb_conf_matrix[27][19] + nb_conf_matrix[27][20] + nb_conf_matrix[27][21] + nb_conf_matrix[27][22] + nb_conf_matrix[27][23] + nb_conf_matrix[27][24] + nb_conf_matrix[27][25] + nb_conf_matrix[27][26] + nb_conf_matrix[27][28]
        #nb_fn_29 = nb_conf_matrix[28][0] + nb_conf_matrix[28][1] + nb_conf_matrix[28][2] + nb_conf_matrix[28][3] + nb_conf_matrix[28][4] + nb_conf_matrix[28][5] + nb_conf_matrix[28][6] + nb_conf_matrix[28][7] + nb_conf_matrix[28][8] + nb_conf_matrix[28][9] + nb_conf_matrix[28][10] + nb_conf_matrix[28][11] + nb_conf_matrix[28][12] + nb_conf_matrix[28][13] + nb_conf_matrix[28][14] + nb_conf_matrix[28][15] + nb_conf_matrix[28][16] + nb_conf_matrix[28][17] + nb_conf_matrix[28][18] + nb_conf_matrix[28][19] + nb_conf_matrix[28][20] + nb_conf_matrix[28][21] + nb_conf_matrix[28][22] + nb_conf_matrix[28][23] + nb_conf_matrix[28][24] + nb_conf_matrix[28][25] + nb_conf_matrix[28][26] + nb_conf_matrix[28][27]

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
            nb_recall_5 = nb_tp_5 / (nb_tp_5 + nb_fn_5)
        if nb_tp_6 + nb_fn_6 == 0:
            nb_recall_6 = 0
        else:
            nb_recall_6 = nb_tp_6 / (nb_tp_6 + nb_fn_6)
        if nb_tp_7 + nb_fn_7 == 0:
            nb_recall_7 = 0
        else:
            nb_recall_7 = nb_tp_7 / (nb_tp_7 + nb_fn_7)
        if nb_tp_8 + nb_fn_8 == 0:
            nb_recall_8 = 0
        else:
            nb_recall_8 = nb_tp_8 / (nb_tp_8 + nb_fn_8)
        if nb_tp_9 + nb_fn_9 == 0:
            nb_recall_9 = 0
        else:
            nb_recall_9 = nb_tp_9 / (nb_tp_9 + nb_fn_9)
        if nb_tp_10 + nb_fn_10 == 0:
            nb_recall_10 = 0
        else:
            nb_recall_10 = nb_tp_10 / (nb_tp_10 + nb_fn_10)
        if nb_tp_11 + nb_fn_11 == 0:
            nb_recall_11 = 0
        else:
            nb_recall_11 = nb_tp_11 / (nb_tp_11 + nb_fn_11)
        if nb_tp_12 + nb_fn_12 == 0:
            nb_recall_12 = 0
        else:
            nb_recall_12 = nb_tp_12 / (nb_tp_12 + nb_fn_12)
        if nb_tp_13 + nb_fn_13 == 0:
            nb_recall_13 = 0
        else:
            nb_recall_13 = nb_tp_13 / (nb_tp_13 + nb_fn_13)
        if nb_tp_14 + nb_fn_14 == 0:
            nb_recall_14 = 0
        else:
            nb_recall_14 = nb_tp_14 / (nb_tp_14 + nb_fn_14)
        if nb_tp_15 + nb_fn_15 == 0:
            nb_recall_15 = 0
        else:
            nb_recall_15 = nb_tp_15 / (nb_tp_15 + nb_fn_15)
        if nb_tp_16 + nb_fn_16 == 0:
            nb_recall_16 = 0
        else:
            nb_recall_16 = nb_tp_16 / (nb_tp_16 + nb_fn_16)
        if nb_tp_17 + nb_fn_17 == 0:
            nb_recall_17 = 0
        else:
            nb_recall_17 = nb_tp_17 / (nb_tp_17 + nb_fn_17)
        if nb_tp_18 + nb_fn_18 == 0:
            nb_recall_18 = 0
        else:
            nb_recall_18 = nb_tp_18 / (nb_tp_18 + nb_fn_18)
        if nb_tp_19 + nb_fn_19 == 0:
            nb_recall_19 = 0
        else:
            nb_recall_19 = nb_tp_19 / (nb_tp_19 + nb_fn_19)
        if nb_tp_20 + nb_fn_20 == 0:
            nb_recall_20 = 0
        else:
            nb_recall_20 = nb_tp_20 / (nb_tp_20 + nb_fn_20)
        '''
        if nb_tp_21 + nb_fn_21 == 0:
            nb_recall_21 = 0
        else:
            nb_recall_21 = nb_tp_21 / (nb_tp_21 + nb_fn_21)
        if nb_tp_22 + nb_fn_22 == 0:
            nb_recall_22 = 0
        else:
            nb_recall_22 = nb_tp_22 / (nb_tp_22 + nb_fn_22)
        if nb_tp_23 + nb_fn_23 == 0:
            nb_recall_23 = 0
        else:
            nb_recall_23 = nb_tp_23 / (nb_tp_23 + nb_fn_23)
        if nb_tp_24 + nb_fn_24 == 0:
            nb_recall_24 = 0
        else:
            nb_recall_24 = nb_tp_24 / (nb_tp_24 + nb_fn_24)
        if nb_tp_25 + nb_fn_25 == 0:
            nb_recall_25 = 0
        else:
            nb_recall_25 = nb_tp_25 / (nb_tp_25 + nb_fn_25)
        if nb_tp_26 + nb_fn_26 == 0:
            nb_recall_26 = 0
        else:
            nb_recall_26 = nb_tp_26 / (nb_tp_26 + nb_fn_26)
        if nb_tp_27 + nb_fn_27 == 0:
            nb_recall_27 = 0
        else:
            nb_recall_27 = nb_tp_27 / (nb_tp_27 + nb_fn_27)
        if nb_tp_28 + nb_fn_28 == 0:
            nb_recall_28 = 0
        else:
            nb_recall_28 = nb_tp_28 / (nb_tp_28 + nb_fn_28)
        if nb_tp_29 + nb_fn_29 == 0:
            nb_recall_29 = 0
        else:
            nb_recall_29 = nb_tp_29 / (nb_tp_29 + nb_fn_29)
        '''
        nb_recall_avg_pen_1 = (
                                 nb_recall_1 + nb_recall_2 + nb_recall_3 + nb_recall_4 + nb_recall_5 + nb_recall_6 + nb_recall_7 + nb_recall_8 + nb_recall_9 + nb_recall_10 + nb_recall_11 + nb_recall_12 + nb_recall_13 + nb_recall_14 + nb_recall_15 + nb_recall_16 + nb_recall_17 + nb_recall_18 + nb_recall_19 + nb_recall_20) / (20+1-1)
        return nb_recall_avg_pen_1

    def get_recall_pen_5(nb_conf_matrix):
        nb_tp_1 = nb_conf_matrix[0][0]
        nb_tp_2 = nb_conf_matrix[1][1]
        nb_tp_3 = nb_conf_matrix[2][2]
        nb_tp_4 = nb_conf_matrix[3][3]
        nb_tp_5 = nb_conf_matrix[4][4]
        nb_tp_6 = nb_conf_matrix[5][5]
        nb_tp_7 = nb_conf_matrix[6][6]
        nb_tp_8 = nb_conf_matrix[7][7]
        nb_tp_9 = nb_conf_matrix[8][8]
        nb_tp_10 = nb_conf_matrix[9][9]
        nb_tp_11 = nb_conf_matrix[10][10]
        nb_tp_12 = nb_conf_matrix[11][11]
        nb_tp_13 = nb_conf_matrix[12][12]
        nb_tp_14 = nb_conf_matrix[13][13]
        nb_tp_15 = nb_conf_matrix[14][14]
        nb_tp_16 = nb_conf_matrix[15][15]
        nb_tp_17 = nb_conf_matrix[16][16]
        nb_tp_18 = nb_conf_matrix[17][17]
        nb_tp_19 = nb_conf_matrix[18][18]
        nb_tp_20 = nb_conf_matrix[19][19]
        #nb_tp_21 = nb_conf_matrix[20][20]
        #nb_tp_22 = nb_conf_matrix[21][21]
        #nb_tp_23 = nb_conf_matrix[22][22]
        #nb_tp_24 = nb_conf_matrix[23][23]
        #nb_tp_25 = nb_conf_matrix[24][24]
        #nb_tp_26 = nb_conf_matrix[25][25]
        #nb_tp_27 = nb_conf_matrix[26][26]
        #nb_tp_28 = nb_conf_matrix[27][27]
        #nb_tp_29 = nb_conf_matrix[28][28]

        nb_fn_1 = nb_conf_matrix[0][1] + nb_conf_matrix[0][2] + nb_conf_matrix[0][3] + nb_conf_matrix[0][4] + nb_conf_matrix[0][5] + nb_conf_matrix[0][6] + nb_conf_matrix[0][7] + nb_conf_matrix[0][8] + nb_conf_matrix[0][9] + nb_conf_matrix[0][10] + nb_conf_matrix[0][11] + nb_conf_matrix[0][12] + nb_conf_matrix[0][13] + nb_conf_matrix[0][14] + nb_conf_matrix[0][15] + nb_conf_matrix[0][16] + nb_conf_matrix[0][17] + nb_conf_matrix[0][18] + nb_conf_matrix[0][19]
        nb_fn_2 = nb_conf_matrix[1][0] + nb_conf_matrix[1][2] + nb_conf_matrix[1][3] + nb_conf_matrix[1][4] + nb_conf_matrix[1][5] + nb_conf_matrix[1][6] + nb_conf_matrix[1][7] + nb_conf_matrix[1][8] + nb_conf_matrix[1][9] + nb_conf_matrix[1][10] + nb_conf_matrix[1][11] + nb_conf_matrix[1][12] + nb_conf_matrix[1][13] + nb_conf_matrix[1][14] + nb_conf_matrix[1][15] + nb_conf_matrix[1][16] + nb_conf_matrix[1][17] + nb_conf_matrix[1][18] + nb_conf_matrix[1][19]
        nb_fn_3 = nb_conf_matrix[2][0] + nb_conf_matrix[2][1] + nb_conf_matrix[2][3] + nb_conf_matrix[2][4] + nb_conf_matrix[2][5] + nb_conf_matrix[2][6] + nb_conf_matrix[2][7] + nb_conf_matrix[2][8] + nb_conf_matrix[2][9] + nb_conf_matrix[2][10] + nb_conf_matrix[2][11] + nb_conf_matrix[2][12] + nb_conf_matrix[2][13] + nb_conf_matrix[2][14] + nb_conf_matrix[2][15] + nb_conf_matrix[2][16] + nb_conf_matrix[2][17] + nb_conf_matrix[2][18] + nb_conf_matrix[2][19]
        nb_fn_4 = nb_conf_matrix[3][0] + nb_conf_matrix[3][1] + nb_conf_matrix[3][2] + nb_conf_matrix[3][4] + nb_conf_matrix[3][5] + nb_conf_matrix[3][6] + nb_conf_matrix[3][7] + nb_conf_matrix[3][8] + nb_conf_matrix[3][9] + nb_conf_matrix[3][10] + nb_conf_matrix[3][11] + nb_conf_matrix[3][12] + nb_conf_matrix[3][13] + nb_conf_matrix[3][14] + nb_conf_matrix[3][15] + nb_conf_matrix[3][16] + nb_conf_matrix[3][17] + nb_conf_matrix[3][18] + nb_conf_matrix[3][19]
        nb_fn_5 = nb_conf_matrix[4][0] + nb_conf_matrix[4][1] + nb_conf_matrix[4][2] + nb_conf_matrix[4][3] + nb_conf_matrix[4][5] + nb_conf_matrix[4][6] + nb_conf_matrix[4][7] + nb_conf_matrix[4][8] + nb_conf_matrix[4][9] + nb_conf_matrix[4][10] + nb_conf_matrix[4][11] + nb_conf_matrix[4][12] + nb_conf_matrix[4][13] + nb_conf_matrix[4][14] + nb_conf_matrix[4][15] + nb_conf_matrix[4][16] + nb_conf_matrix[4][17] + nb_conf_matrix[4][18] + nb_conf_matrix[4][19]
        nb_fn_6 = nb_conf_matrix[5][0] + nb_conf_matrix[5][1] + nb_conf_matrix[5][2] + nb_conf_matrix[5][3] + nb_conf_matrix[5][4] + nb_conf_matrix[5][6] + nb_conf_matrix[5][7] + nb_conf_matrix[5][8] + nb_conf_matrix[5][9] + nb_conf_matrix[5][10] + nb_conf_matrix[5][11] + nb_conf_matrix[5][12] + nb_conf_matrix[5][13] + nb_conf_matrix[5][14] + nb_conf_matrix[5][15] + nb_conf_matrix[5][16] + nb_conf_matrix[5][17] + nb_conf_matrix[5][18] + nb_conf_matrix[5][19]
        nb_fn_7 = nb_conf_matrix[6][0] + nb_conf_matrix[6][1] + nb_conf_matrix[6][2] + nb_conf_matrix[6][3] + nb_conf_matrix[6][4] + nb_conf_matrix[6][5] + nb_conf_matrix[6][7] + nb_conf_matrix[6][8] + nb_conf_matrix[6][9] + nb_conf_matrix[6][10] + nb_conf_matrix[6][11] + nb_conf_matrix[6][12] + nb_conf_matrix[6][13] + nb_conf_matrix[6][14] + nb_conf_matrix[6][15] + nb_conf_matrix[6][16] + nb_conf_matrix[6][17] + nb_conf_matrix[6][18] + nb_conf_matrix[6][19]
        nb_fn_8 = nb_conf_matrix[7][0] + nb_conf_matrix[7][1] + nb_conf_matrix[7][2] + nb_conf_matrix[7][3] + nb_conf_matrix[7][4] + nb_conf_matrix[7][5] + nb_conf_matrix[7][6] + nb_conf_matrix[7][8] + nb_conf_matrix[7][9] + nb_conf_matrix[7][10] + nb_conf_matrix[7][11] + nb_conf_matrix[7][12] + nb_conf_matrix[7][13] + nb_conf_matrix[7][14] + nb_conf_matrix[7][15] + nb_conf_matrix[7][16] + nb_conf_matrix[7][17] + nb_conf_matrix[7][18] + nb_conf_matrix[7][19]
        nb_fn_9 = nb_conf_matrix[8][0] + nb_conf_matrix[8][1] + nb_conf_matrix[8][2] + nb_conf_matrix[8][3] + nb_conf_matrix[8][4] + nb_conf_matrix[8][5] + nb_conf_matrix[8][6] + nb_conf_matrix[8][7] + nb_conf_matrix[8][9] + nb_conf_matrix[8][10] + nb_conf_matrix[8][11] + nb_conf_matrix[8][12] + nb_conf_matrix[8][13] + nb_conf_matrix[8][14] + nb_conf_matrix[8][15] + nb_conf_matrix[8][16] + nb_conf_matrix[8][17] + nb_conf_matrix[8][18] + nb_conf_matrix[8][19]
        nb_fn_10 = nb_conf_matrix[9][0] + nb_conf_matrix[9][1] + nb_conf_matrix[9][2] + nb_conf_matrix[9][3] + nb_conf_matrix[9][4] + nb_conf_matrix[9][5] + nb_conf_matrix[9][6] + nb_conf_matrix[9][7] + nb_conf_matrix[9][8] + nb_conf_matrix[9][10] + nb_conf_matrix[9][11] + nb_conf_matrix[9][12] + nb_conf_matrix[9][13] + nb_conf_matrix[9][14] + nb_conf_matrix[9][15] + nb_conf_matrix[9][16] + nb_conf_matrix[9][17] + nb_conf_matrix[9][18] + nb_conf_matrix[9][19]
        nb_fn_11 = nb_conf_matrix[10][0] + nb_conf_matrix[10][1] + nb_conf_matrix[10][2] + nb_conf_matrix[10][3] + nb_conf_matrix[10][4] + nb_conf_matrix[10][5] + nb_conf_matrix[10][6] + nb_conf_matrix[10][7] + nb_conf_matrix[10][8] + nb_conf_matrix[10][9] + nb_conf_matrix[10][11] + nb_conf_matrix[10][12] + nb_conf_matrix[10][13] + nb_conf_matrix[10][14] + nb_conf_matrix[10][15] + nb_conf_matrix[10][16] + nb_conf_matrix[10][17] + nb_conf_matrix[10][18] + nb_conf_matrix[10][19]
        nb_fn_12 = nb_conf_matrix[11][0] + nb_conf_matrix[11][1] + nb_conf_matrix[11][2] + nb_conf_matrix[11][3] + nb_conf_matrix[11][4] + nb_conf_matrix[11][5] + nb_conf_matrix[11][6] + nb_conf_matrix[11][7] + nb_conf_matrix[11][8] + nb_conf_matrix[11][9] + nb_conf_matrix[11][10] + nb_conf_matrix[11][12] + nb_conf_matrix[11][13] + nb_conf_matrix[11][14] + nb_conf_matrix[11][15] + nb_conf_matrix[11][16] + nb_conf_matrix[11][17] + nb_conf_matrix[11][18] + nb_conf_matrix[11][19]
        nb_fn_13 = nb_conf_matrix[12][0] + nb_conf_matrix[12][1] + nb_conf_matrix[12][2] + nb_conf_matrix[12][3] + nb_conf_matrix[12][4] + nb_conf_matrix[12][5] + nb_conf_matrix[12][6] + nb_conf_matrix[12][7] + nb_conf_matrix[12][8] + nb_conf_matrix[12][9] + nb_conf_matrix[12][10] + nb_conf_matrix[12][11] + nb_conf_matrix[12][13] + nb_conf_matrix[12][14] + nb_conf_matrix[12][15] + nb_conf_matrix[12][16] + nb_conf_matrix[12][17] + nb_conf_matrix[12][18] + nb_conf_matrix[12][19]
        nb_fn_14 = nb_conf_matrix[13][0] + nb_conf_matrix[13][1] + nb_conf_matrix[13][2] + nb_conf_matrix[13][3] + nb_conf_matrix[13][4] + nb_conf_matrix[13][5] + nb_conf_matrix[13][6] + nb_conf_matrix[13][7] + nb_conf_matrix[13][8] + nb_conf_matrix[13][9] + nb_conf_matrix[13][10] + nb_conf_matrix[13][11] + nb_conf_matrix[13][12] + nb_conf_matrix[13][14] + nb_conf_matrix[13][15] + nb_conf_matrix[13][16] + nb_conf_matrix[13][17] + nb_conf_matrix[13][18] + nb_conf_matrix[13][19]
        nb_fn_15 = nb_conf_matrix[14][0] + nb_conf_matrix[14][1] + nb_conf_matrix[14][2] + nb_conf_matrix[14][3] + nb_conf_matrix[14][4] + nb_conf_matrix[14][5] + nb_conf_matrix[14][6] + nb_conf_matrix[14][7] + nb_conf_matrix[14][8] + nb_conf_matrix[14][9] + nb_conf_matrix[14][10] + nb_conf_matrix[14][11] + nb_conf_matrix[14][12] + nb_conf_matrix[14][13] + nb_conf_matrix[14][15] + nb_conf_matrix[14][16] + nb_conf_matrix[14][17] + nb_conf_matrix[14][18] + nb_conf_matrix[14][19]
        nb_fn_16 = nb_conf_matrix[15][0] + nb_conf_matrix[15][1] + nb_conf_matrix[15][2] + nb_conf_matrix[15][3] + nb_conf_matrix[15][4] + nb_conf_matrix[15][5] + nb_conf_matrix[15][6] + nb_conf_matrix[15][7] + nb_conf_matrix[15][8] + nb_conf_matrix[15][9] + nb_conf_matrix[15][10] + nb_conf_matrix[15][11] + nb_conf_matrix[15][12] + nb_conf_matrix[15][13] + nb_conf_matrix[15][14] + nb_conf_matrix[15][16] + nb_conf_matrix[15][17] + nb_conf_matrix[15][18] + nb_conf_matrix[15][19]
        nb_fn_17 = nb_conf_matrix[16][0] + nb_conf_matrix[16][1] + nb_conf_matrix[16][2] + nb_conf_matrix[16][3] + nb_conf_matrix[16][4] + nb_conf_matrix[16][5] + nb_conf_matrix[16][6] + nb_conf_matrix[16][7] + nb_conf_matrix[16][8] + nb_conf_matrix[16][9] + nb_conf_matrix[16][10] + nb_conf_matrix[16][11] + nb_conf_matrix[16][12] + nb_conf_matrix[16][13] + nb_conf_matrix[16][14] + nb_conf_matrix[16][15] + nb_conf_matrix[16][17] + nb_conf_matrix[16][18] + nb_conf_matrix[16][19]
        nb_fn_18 = nb_conf_matrix[17][0] + nb_conf_matrix[17][1] + nb_conf_matrix[17][2] + nb_conf_matrix[17][3] + nb_conf_matrix[17][4] + nb_conf_matrix[17][5] + nb_conf_matrix[17][6] + nb_conf_matrix[17][7] + nb_conf_matrix[17][8] + nb_conf_matrix[17][9] + nb_conf_matrix[17][10] + nb_conf_matrix[17][11] + nb_conf_matrix[17][12] + nb_conf_matrix[17][13] + nb_conf_matrix[17][14] + nb_conf_matrix[17][15] + nb_conf_matrix[17][16] + nb_conf_matrix[17][18] + nb_conf_matrix[17][19]
        nb_fn_19 = nb_conf_matrix[18][0] + nb_conf_matrix[18][1] + nb_conf_matrix[18][2] + nb_conf_matrix[18][3] + nb_conf_matrix[18][4] + nb_conf_matrix[18][5] + nb_conf_matrix[18][6] + nb_conf_matrix[18][7] + nb_conf_matrix[18][8] + nb_conf_matrix[18][9] + nb_conf_matrix[18][10] + nb_conf_matrix[18][11] + nb_conf_matrix[18][12] + nb_conf_matrix[18][13] + nb_conf_matrix[18][14] + nb_conf_matrix[18][15] + nb_conf_matrix[18][16] + nb_conf_matrix[18][17] + nb_conf_matrix[18][19]
        nb_fn_20 = nb_conf_matrix[19][0] + nb_conf_matrix[19][1] + nb_conf_matrix[19][2] + nb_conf_matrix[19][3] + nb_conf_matrix[19][4] + nb_conf_matrix[19][5] + nb_conf_matrix[19][6] + nb_conf_matrix[19][7] + nb_conf_matrix[19][8] + nb_conf_matrix[19][9] + nb_conf_matrix[19][10] + nb_conf_matrix[19][11] + nb_conf_matrix[19][12] + nb_conf_matrix[19][13] + nb_conf_matrix[19][14] + nb_conf_matrix[19][15] + nb_conf_matrix[19][16] + nb_conf_matrix[19][17] + nb_conf_matrix[19][18]
        #nb_fn_21 = nb_conf_matrix[20][0] + nb_conf_matrix[20][1] + nb_conf_matrix[20][2] + nb_conf_matrix[20][3] + nb_conf_matrix[20][4] + nb_conf_matrix[20][5] + nb_conf_matrix[20][6] + nb_conf_matrix[20][7] + nb_conf_matrix[20][8] + nb_conf_matrix[20][9] + nb_conf_matrix[20][10] + nb_conf_matrix[20][11] + nb_conf_matrix[20][12] + nb_conf_matrix[20][13] + nb_conf_matrix[20][14] + nb_conf_matrix[20][15] + nb_conf_matrix[20][16] + nb_conf_matrix[20][17] + nb_conf_matrix[20][18] + nb_conf_matrix[20][19] + nb_conf_matrix[20][21] + nb_conf_matrix[20][22] + nb_conf_matrix[20][23] + nb_conf_matrix[20][24] + nb_conf_matrix[20][25] + nb_conf_matrix[20][26] + nb_conf_matrix[20][27] + nb_conf_matrix[20][28]
        #nb_fn_22 = nb_conf_matrix[21][0] + nb_conf_matrix[21][1] + nb_conf_matrix[21][2] + nb_conf_matrix[21][3] + nb_conf_matrix[21][4] + nb_conf_matrix[21][5] + nb_conf_matrix[21][6] + nb_conf_matrix[21][7] + nb_conf_matrix[21][8] + nb_conf_matrix[21][9] + nb_conf_matrix[21][10] + nb_conf_matrix[21][11] + nb_conf_matrix[21][12] + nb_conf_matrix[21][13] + nb_conf_matrix[21][14] + nb_conf_matrix[21][15] + nb_conf_matrix[21][16] + nb_conf_matrix[21][17] + nb_conf_matrix[21][18] + nb_conf_matrix[21][19] + nb_conf_matrix[21][20] + nb_conf_matrix[21][22] + nb_conf_matrix[21][23] + nb_conf_matrix[21][24] + nb_conf_matrix[21][25] + nb_conf_matrix[21][26] + nb_conf_matrix[21][27] + nb_conf_matrix[21][28]
        #nb_fn_23 = nb_conf_matrix[22][0] + nb_conf_matrix[22][1] + nb_conf_matrix[22][2] + nb_conf_matrix[22][3] + nb_conf_matrix[22][4] + nb_conf_matrix[22][5] + nb_conf_matrix[22][6] + nb_conf_matrix[22][7] + nb_conf_matrix[22][8] + nb_conf_matrix[22][9] + nb_conf_matrix[22][10] + nb_conf_matrix[22][11] + nb_conf_matrix[22][12] + nb_conf_matrix[22][13] + nb_conf_matrix[22][14] + nb_conf_matrix[22][15] + nb_conf_matrix[22][16] + nb_conf_matrix[22][17] + nb_conf_matrix[22][18] + nb_conf_matrix[22][19] + nb_conf_matrix[22][20] + nb_conf_matrix[22][21] + nb_conf_matrix[22][23] + nb_conf_matrix[22][24] + nb_conf_matrix[22][25] + nb_conf_matrix[22][26] + nb_conf_matrix[22][27] + nb_conf_matrix[22][28]
        #nb_fn_24 = nb_conf_matrix[23][0] + nb_conf_matrix[23][1] + nb_conf_matrix[23][2] + nb_conf_matrix[23][3] + nb_conf_matrix[23][4] + nb_conf_matrix[23][5] + nb_conf_matrix[23][6] + nb_conf_matrix[23][7] + nb_conf_matrix[23][8] + nb_conf_matrix[23][9] + nb_conf_matrix[23][10] + nb_conf_matrix[23][11] + nb_conf_matrix[23][12] + nb_conf_matrix[23][13] + nb_conf_matrix[23][14] + nb_conf_matrix[23][15] + nb_conf_matrix[23][16] + nb_conf_matrix[23][17] + nb_conf_matrix[23][18] + nb_conf_matrix[23][19] + nb_conf_matrix[23][20] + nb_conf_matrix[23][21] + nb_conf_matrix[23][22] + nb_conf_matrix[23][24] + nb_conf_matrix[23][25] + nb_conf_matrix[23][26] + nb_conf_matrix[23][27] + nb_conf_matrix[23][28]
        #nb_fn_25 = nb_conf_matrix[24][0] + nb_conf_matrix[24][1] + nb_conf_matrix[24][2] + nb_conf_matrix[24][3] + nb_conf_matrix[24][4] + nb_conf_matrix[24][5] + nb_conf_matrix[24][6] + nb_conf_matrix[24][7] + nb_conf_matrix[24][8] + nb_conf_matrix[24][9] + nb_conf_matrix[24][10] + nb_conf_matrix[24][11] + nb_conf_matrix[24][12] + nb_conf_matrix[24][13] + nb_conf_matrix[24][14] + nb_conf_matrix[24][15] + nb_conf_matrix[24][16] + nb_conf_matrix[24][17] + nb_conf_matrix[24][18] + nb_conf_matrix[24][19] + nb_conf_matrix[24][20] + nb_conf_matrix[24][21] + nb_conf_matrix[24][22] + nb_conf_matrix[24][23] + nb_conf_matrix[24][25] + nb_conf_matrix[24][26] + nb_conf_matrix[24][27] + nb_conf_matrix[24][28]
        #nb_fn_26 = nb_conf_matrix[25][0] + nb_conf_matrix[25][1] + nb_conf_matrix[25][2] + nb_conf_matrix[25][3] + nb_conf_matrix[25][4] + nb_conf_matrix[25][5] + nb_conf_matrix[25][6] + nb_conf_matrix[25][7] + nb_conf_matrix[25][8] + nb_conf_matrix[25][9] + nb_conf_matrix[25][10] + nb_conf_matrix[25][11] + nb_conf_matrix[25][12] + nb_conf_matrix[25][13] + nb_conf_matrix[25][14] + nb_conf_matrix[25][15] + nb_conf_matrix[25][16] + nb_conf_matrix[25][17] + nb_conf_matrix[25][18] + nb_conf_matrix[25][19] + nb_conf_matrix[25][20] + nb_conf_matrix[25][21] + nb_conf_matrix[25][22] + nb_conf_matrix[25][23] + nb_conf_matrix[25][24] + nb_conf_matrix[25][26] + nb_conf_matrix[25][27] + nb_conf_matrix[25][28]
        #nb_fn_27 = nb_conf_matrix[26][0] + nb_conf_matrix[26][1] + nb_conf_matrix[26][2] + nb_conf_matrix[26][3] + nb_conf_matrix[26][4] + nb_conf_matrix[26][5] + nb_conf_matrix[26][6] + nb_conf_matrix[26][7] + nb_conf_matrix[26][8] + nb_conf_matrix[26][9] + nb_conf_matrix[26][10] + nb_conf_matrix[26][11] + nb_conf_matrix[26][12] + nb_conf_matrix[26][13] + nb_conf_matrix[26][14] + nb_conf_matrix[26][15] + nb_conf_matrix[26][16] + nb_conf_matrix[26][17] + nb_conf_matrix[26][18] + nb_conf_matrix[26][19] + nb_conf_matrix[26][20] + nb_conf_matrix[26][21] + nb_conf_matrix[26][22] + nb_conf_matrix[26][23] + nb_conf_matrix[26][24] + nb_conf_matrix[26][25] + nb_conf_matrix[26][27] + nb_conf_matrix[26][28]
        #nb_fn_28 = nb_conf_matrix[27][0] + nb_conf_matrix[27][1] + nb_conf_matrix[27][2] + nb_conf_matrix[27][3] + nb_conf_matrix[27][4] + nb_conf_matrix[27][5] + nb_conf_matrix[27][6] + nb_conf_matrix[27][7] + nb_conf_matrix[27][8] + nb_conf_matrix[27][9] + nb_conf_matrix[27][10] + nb_conf_matrix[27][11] + nb_conf_matrix[27][12] + nb_conf_matrix[27][13] + nb_conf_matrix[27][14] + nb_conf_matrix[27][15] + nb_conf_matrix[27][16] + nb_conf_matrix[27][17] + nb_conf_matrix[27][18] + nb_conf_matrix[27][19] + nb_conf_matrix[27][20] + nb_conf_matrix[27][21] + nb_conf_matrix[27][22] + nb_conf_matrix[27][23] + nb_conf_matrix[27][24] + nb_conf_matrix[27][25] + nb_conf_matrix[27][26] + nb_conf_matrix[27][28]
        #nb_fn_29 = nb_conf_matrix[28][0] + nb_conf_matrix[28][1] + nb_conf_matrix[28][2] + nb_conf_matrix[28][3] + nb_conf_matrix[28][4] + nb_conf_matrix[28][5] + nb_conf_matrix[28][6] + nb_conf_matrix[28][7] + nb_conf_matrix[28][8] + nb_conf_matrix[28][9] + nb_conf_matrix[28][10] + nb_conf_matrix[28][11] + nb_conf_matrix[28][12] + nb_conf_matrix[28][13] + nb_conf_matrix[28][14] + nb_conf_matrix[28][15] + nb_conf_matrix[28][16] + nb_conf_matrix[28][17] + nb_conf_matrix[28][18] + nb_conf_matrix[28][19] + nb_conf_matrix[28][20] + nb_conf_matrix[28][21] + nb_conf_matrix[28][22] + nb_conf_matrix[28][23] + nb_conf_matrix[28][24] + nb_conf_matrix[28][25] + nb_conf_matrix[28][26] + nb_conf_matrix[28][27]

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
            nb_recall_5 = nb_tp_5 / (nb_tp_5 + nb_fn_5)
        if nb_tp_6 + nb_fn_6 == 0:
            nb_recall_6 = 0
        else:
            nb_recall_6 = nb_tp_6 / (nb_tp_6 + nb_fn_6)
        if nb_tp_7 + nb_fn_7 == 0:
            nb_recall_7 = 0
        else:
            nb_recall_7 = nb_tp_7 / (nb_tp_7 + nb_fn_7)
        if nb_tp_8 + nb_fn_8 == 0:
            nb_recall_8 = 0
        else:
            nb_recall_8 = nb_tp_8 / (nb_tp_8 + nb_fn_8)
        if nb_tp_9 + nb_fn_9 == 0:
            nb_recall_9 = 0
        else:
            nb_recall_9 = nb_tp_9 / (nb_tp_9 + nb_fn_9)
        if nb_tp_10 + nb_fn_10 == 0:
            nb_recall_10 = 0
        else:
            nb_recall_10 = nb_tp_10 / (nb_tp_10 + nb_fn_10)
        if nb_tp_11 + nb_fn_11 == 0:
            nb_recall_11 = 0
        else:
            nb_recall_11 = nb_tp_11 / (nb_tp_11 + nb_fn_11)
        if nb_tp_12 + nb_fn_12 == 0:
            nb_recall_12 = 0
        else:
            nb_recall_12 = nb_tp_12 / (nb_tp_12 + nb_fn_12)
        if nb_tp_13 + nb_fn_13 == 0:
            nb_recall_13 = 0
        else:
            nb_recall_13 = nb_tp_13 / (nb_tp_13 + nb_fn_13)
        if nb_tp_14 + nb_fn_14 == 0:
            nb_recall_14 = 0
        else:
            nb_recall_14 = nb_tp_14 / (nb_tp_14 + nb_fn_14)
        if nb_tp_15 + nb_fn_15 == 0:
            nb_recall_15 = 0
        else:
            nb_recall_15 = nb_tp_15 / (nb_tp_15 + nb_fn_15)
        if nb_tp_16 + nb_fn_16 == 0:
            nb_recall_16 = 0
        else:
            nb_recall_16 = nb_tp_16 / (nb_tp_16 + nb_fn_16)
        if nb_tp_17 + nb_fn_17 == 0:
            nb_recall_17 = 0
        else:
            nb_recall_17 = nb_tp_17 / (nb_tp_17 + nb_fn_17)
        if nb_tp_18 + nb_fn_18 == 0:
            nb_recall_18 = 0
        else:
            nb_recall_18 = nb_tp_18 / (nb_tp_18 + nb_fn_18)
        if nb_tp_19 + nb_fn_19 == 0:
            nb_recall_19 = 0
        else:
            nb_recall_19 = nb_tp_19 / (nb_tp_19 + nb_fn_19)
        if nb_tp_20 + nb_fn_20 == 0:
            nb_recall_20 = 0
        else:
            nb_recall_20 = nb_tp_20 / (nb_tp_20 + nb_fn_20)
        '''
        if nb_tp_21 + nb_fn_21 == 0:
            nb_recall_21 = 0
        else:
            nb_recall_21 = nb_tp_21 / (nb_tp_21 + nb_fn_21)
        if nb_tp_22 + nb_fn_22 == 0:
            nb_recall_22 = 0
        else:
            nb_recall_22 = nb_tp_22 / (nb_tp_22 + nb_fn_22)
        if nb_tp_23 + nb_fn_23 == 0:
            nb_recall_23 = 0
        else:
            nb_recall_23 = nb_tp_23 / (nb_tp_23 + nb_fn_23)
        if nb_tp_24 + nb_fn_24 == 0:
            nb_recall_24 = 0
        else:
            nb_recall_24 = nb_tp_24 / (nb_tp_24 + nb_fn_24)
        if nb_tp_25 + nb_fn_25 == 0:
            nb_recall_25 = 0
        else:
            nb_recall_25 = nb_tp_25 / (nb_tp_25 + nb_fn_25)
        if nb_tp_26 + nb_fn_26 == 0:
            nb_recall_26 = 0
        else:
            nb_recall_26 = nb_tp_26 / (nb_tp_26 + nb_fn_26)
        if nb_tp_27 + nb_fn_27 == 0:
            nb_recall_27 = 0
        else:
            nb_recall_27 = nb_tp_27 / (nb_tp_27 + nb_fn_27)
        if nb_tp_28 + nb_fn_28 == 0:
            nb_recall_28 = 0
        else:
            nb_recall_28 = nb_tp_28 / (nb_tp_28 + nb_fn_28)
        if nb_tp_29 + nb_fn_29 == 0:
            nb_recall_29 = 0
        else:
            nb_recall_29 = nb_tp_29 / (nb_tp_29 + nb_fn_29)
        '''
        nb_recall_avg_pen_5 = (
                                 nb_recall_1 + nb_recall_2 + nb_recall_3 + nb_recall_4 + nb_recall_5 + nb_recall_6 + nb_recall_7 + nb_recall_8 + nb_recall_9 + nb_recall_10 + nb_recall_11 + nb_recall_12 + nb_recall_13 + nb_recall_14 + nb_recall_15 + nb_recall_16 + nb_recall_17 + nb_recall_18 + nb_recall_19 + (5*nb_recall_20)) / (20+5-1)
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
    nb_ovr_accuracy = (nb_conf_matrix[0][0] + nb_conf_matrix[1][1] + nb_conf_matrix[2][2] + nb_conf_matrix[3][3] + nb_conf_matrix[4][4] + nb_conf_matrix[5][5] + nb_conf_matrix[6][6] + nb_conf_matrix[7][7] + nb_conf_matrix[8][8] + nb_conf_matrix[9][9] + nb_conf_matrix[10][10] + nb_conf_matrix[11][11] + nb_conf_matrix[12][12] + nb_conf_matrix[13][13] + nb_conf_matrix[14][14] + nb_conf_matrix[15][15] + nb_conf_matrix[16][16] + nb_conf_matrix[17][17] + nb_conf_matrix[18][18] + nb_conf_matrix[19][19]) / (
                sum(nb_conf_matrix[0]) + sum(nb_conf_matrix[1]) + sum(nb_conf_matrix[2]) + sum(nb_conf_matrix[3]) + sum(nb_conf_matrix[4]) + sum(nb_conf_matrix[5]) + sum(nb_conf_matrix[6]) + sum(nb_conf_matrix[7]) + sum(nb_conf_matrix[8]) + sum(nb_conf_matrix[9]) + sum(nb_conf_matrix[10]) + sum(nb_conf_matrix[11]) + sum(nb_conf_matrix[12]) + sum(nb_conf_matrix[13]) + sum(nb_conf_matrix[14]) + sum(nb_conf_matrix[15]) + sum(nb_conf_matrix[16]) + sum(nb_conf_matrix[17]) + sum(nb_conf_matrix[18]) + sum(nb_conf_matrix[19]))
    print("nb_f1 score of pen 1 is:")
    print(nb_f1_score_pen_1)
    print("nb_f1 score of pen 5 is:")
    print(nb_f1_score_pen_5)
    print("nb_overall accuracy is:")
    print(nb_ovr_accuracy)
    nb_conf_matrix = pd.DataFrame(nb_conf_matrix)
    nb_conf_matrix.to_csv('conf_matrix_' + imb_technique + '_nb_production_' + str(nsplits) + 'foldcv_' + str(repeat+1) + '.csv', header=False, index=False)  # First repetition
    #nb_conf_matrix.to_csv('conf_matrix_' + imb_technique + '_penalty_' + str(penalty) + '_nb_production_' + str(nsplits) + 'foldcv_' + str(repeat+6) + '.csv', header=False, index=False)  # First repetition
    nb_f1_score_pen_1_kfoldcv[repeat] = nb_f1_score_pen_1
    nb_f1_score_pen_5_kfoldcv[repeat] = nb_f1_score_pen_5
    nb_ovr_accuracy_kfoldcv[repeat] = nb_ovr_accuracy



    for i in range(0, len(y_test)):
        #rf_DM_index = 0
        rf_FI_index = 0
        rf_FG_index = 0
        #rf_GR_index = 0
        #rf_GR12_index = 0
        rf_GR27_index = 0
        rf_LM_index = 0
        rf_LMM_index = 0
        #rf_MM14_index = 0
        #rf_MM16_index = 0
        rf_PC_index = 0
        rf_RG12_index = 0
        #rf_RG19_index = 0
        rf_RG2_index = 0
        rf_RG3_index = 0
        rf_RGM_index = 0
        rf_RGQC_index = 0
        #rf_TMSA10_index = 0
        rf_T8_index = 0
        #rf_T9_index = 0
        rf_TM10_index = 0
        rf_TM4_index = 0
        rf_TM5_index = 0
        rf_TM6_index = 0
        rf_TM8_index = 0
        rf_TM9_index = 0
        rf_TMQC_index = 0
        rf_TQC_index = 0
        #rf_WC13_index = 0

        """
        if rf_pred_class_DM[i] == "Deburring - Manual":
            if rf_pred_prob_DM[i][0] >= 0.5:
                rf_DM_index = 0
            else:
                rf_DM_index = 1
        elif rf_pred_class_DM[i] == "Others":
            if rf_pred_prob_DM[i][0] < 0.5:
                rf_DM_index = 0
            else:
                rf_DM_index = 1
        """
        if rf_pred_class_FI[i] == "Final Inspection Q.C.":
            if rf_pred_prob_FI[i][0] >= 0.5:
                rf_FI_index = 0
            else:
                rf_FI_index = 1
        elif rf_pred_class_FI[i] == "Others":
            if rf_pred_prob_FI[i][0] < 0.5:
                rf_FI_index = 0
            else:
                rf_FI_index = 1
        if rf_pred_class_FG[i] == "Flat Grinding - Machine 11":
            if rf_pred_prob_FG[i][0] >= 0.5:
                rf_FG_index = 0
            else:
                rf_FG_index = 1
        elif rf_pred_class_FG[i] == "Others":
            if rf_pred_prob_FG[i][0] < 0.5:
                rf_FG_index = 0
            else:
                rf_FG_index = 1
        """
        if rf_pred_class_GR[i] == "Grinding Rework":
            if rf_pred_prob_GR[i][0] >= 0.5:
                rf_GR_index = 0
            else:
                rf_GR_index = 1
        elif rf_pred_class_GR[i] == "Others":
            if rf_pred_prob_GR[i][0] < 0.5:
                rf_GR_index = 0
            else:
                rf_GR_index = 1
        """
        """
        if rf_pred_class_GR12[i] == "Grinding Rework - Machine 12":
            if rf_pred_prob_GR12[i][0] >= 0.5:
                rf_GR12_index = 0
            else:
                rf_GR12_index = 1
        elif rf_pred_class_GR12[i] == "Others":
            if rf_pred_prob_GR12[i][0] < 0.5:
                rf_GR12_index = 0
            else:
                rf_GR12_index = 1
        """
        if rf_pred_class_GR27[i] == "Grinding Rework - Machine 27":
            if rf_pred_prob_GR27[i][0] >= 0.5:
                rf_GR27_index = 0
            else:
                rf_GR27_index = 1
        elif rf_pred_class_GR27[i] == "Others":
            if rf_pred_prob_GR27[i][0] < 0.5:
                rf_GR27_index = 0
            else:
                rf_GR27_index = 1
        if rf_pred_class_LM[i] == "Lapping - Machine 1":
            if rf_pred_prob_LM[i][0] >= 0.5:
                rf_LM_index = 0
            else:
                rf_LM_index = 1
        elif rf_pred_class_LM[i] == "Others":
            if rf_pred_prob_LM[i][0] < 0.5:
                rf_LM_index = 0
            else:
                rf_LM_index = 1
        if rf_pred_class_LMM[i] == "Laser Marking - Machine 7":
            if rf_pred_prob_LMM[i][0] >= 0.5:
                rf_LMM_index = 0
            else:
                rf_LMM_index = 1
        elif rf_pred_class_LMM[i] == "Others":
            if rf_pred_prob_LMM[i][0] < 0.5:
                rf_LMM_index = 0
            else:
                rf_LMM_index = 1
        """
        if rf_pred_class_MM14[i] == "Milling - Machine 14":
            if rf_pred_prob_MM14[i][0] >= 0.5:
                rf_MM14_index = 0
            else:
                rf_MM14_index = 1
        elif rf_pred_class_MM14[i] == "Others":
            if rf_pred_prob_MM14[i][0] < 0.5:
                rf_MM14_index = 0
            else:
                rf_MM14_index = 1
        """
        """
        if rf_pred_class_MM16[i] == "Milling - Machine 16":
            if rf_pred_prob_MM16[i][0] >= 0.5:
                rf_MM16_index = 0
            else:
                rf_MM16_index = 1
        elif rf_pred_class_MM16[i] == "Others":
            if rf_pred_prob_MM16[i][0] < 0.5:
                rf_MM16_index = 0
            else:
                rf_MM16_index = 1
        """
        if rf_pred_class_PC[i] == "Packing":
            if rf_pred_prob_PC[i][0] >= 0.5:
                rf_PC_index = 0
            else:
                rf_PC_index = 1
        elif rf_pred_class_PC[i] == "Others":
            if rf_pred_prob_PC[i][0] < 0.5:
                rf_PC_index = 0
            else:
                rf_PC_index = 1
        if rf_pred_class_RG12[i] == "Round Grinding - Machine 12":
            if rf_pred_prob_RG12[i][0] >= 0.5:
                rf_RG12_index = 0
            else:
                rf_RG12_index = 1
        elif rf_pred_class_RG12[i] == "Others":
            if rf_pred_prob_RG12[i][0] < 0.5:
                rf_RG12_index = 0
            else:
                rf_RG12_index = 1
        """
        if rf_pred_class_RG19[i] == "Round Grinding - Machine 19":
            if rf_pred_prob_RG19[i][0] >= 0.5:
                rf_RG19_index = 0
            else:
                rf_RG19_index = 1
        elif rf_pred_class_RG19[i] == "Others":
            if rf_pred_prob_RG19[i][0] < 0.5:
                rf_RG19_index = 0
            else:
                rf_RG19_index = 1
        """
        if rf_pred_class_RG2[i] == "Round Grinding - Machine 2":
            if rf_pred_prob_RG2[i][0] >= 0.5:
                rf_RG2_index = 0
            else:
                rf_RG2_index = 1
        elif rf_pred_class_RG2[i] == "Others":
            if rf_pred_prob_RG2[i][0] < 0.5:
                rf_RG2_index = 0
            else:
                rf_RG2_index = 1
        if rf_pred_class_RG3[i] == "Round Grinding - Machine 3":
            if rf_pred_prob_RG3[i][0] >= 0.5:
                rf_RG3_index = 0
            else:
                rf_RG3_index = 1
        elif rf_pred_class_RG3[i] == "Others":
            if rf_pred_prob_RG3[i][0] < 0.5:
                rf_RG3_index = 0
            else:
                rf_RG3_index = 1
        if rf_pred_class_RGM[i] == "Round Grinding - Manual":
            if rf_pred_prob_RGM[i][0] >= 0.5:
                rf_RGM_index = 0
            else:
                rf_RGM_index = 1
        elif rf_pred_class_RGM[i] == "Others":
            if rf_pred_prob_RGM[i][0] < 0.5:
                rf_RGM_index = 0
            else:
                rf_RGM_index = 1
        if rf_pred_class_RGQC[i] == "Round Grinding - Q.C.":
            if rf_pred_prob_RGQC[i][0] >= 0.5:
                rf_RGQC_index = 0
            else:
                rf_RGQC_index = 1
        elif rf_pred_class_RGQC[i] == "Others":
            if rf_pred_prob_RGQC[i][0] < 0.5:
                rf_RGQC_index = 0
            else:
                rf_RGQC_index = 1
        """
        if rf_pred_class_TMSA10[i] == "Turn & Mill. & Screw Assem - Machine 10":
            if rf_pred_prob_TMSA10[i][0] >= 0.5:
                rf_TMSA10_index = 0
            else:
                rf_TMSA10_index = 1
        elif rf_pred_class_TMSA10[i] == "Others":
            if rf_pred_prob_TMSA10[i][0] < 0.5:
                rf_TMSA10_index = 0
            else:
                rf_TMSA10_index = 1
        """
        if rf_pred_class_T8[i] == "Turning - Machine 8":
            if rf_pred_prob_T8[i][0] >= 0.5:
                rf_T8_index = 0
            else:
                rf_T8_index = 1
        elif rf_pred_class_T8[i] == "Others":
            if rf_pred_prob_T8[i][0] < 0.5:
                rf_T8_index = 0
            else:
                rf_T8_index = 1
        """
        if rf_pred_class_T9[i] == "Turning - Machine 9":
            if rf_pred_prob_T9[i][0] >= 0.5:
                rf_T9_index = 0
            else:
                rf_T9_index = 1
        elif rf_pred_class_T9[i] == "Others":
            if rf_pred_prob_T9[i][0] < 0.5:
                rf_T9_index = 0
            else:
                rf_T9_index = 1
        """
        if rf_pred_class_TM10[i] == "Turning & Milling - Machine 10":
            if rf_pred_prob_TM10[i][0] >= 0.5:
                rf_TM10_index = 0
            else:
                rf_TM10_index = 1
        elif rf_pred_class_TM10[i] == "Others":
            if rf_pred_prob_TM10[i][0] < 0.5:
                rf_TM10_index = 0
            else:
                rf_TM10_index = 1
        if rf_pred_class_TM4[i] == "Turning & Milling - Machine 4":
            if rf_pred_prob_TM4[i][0] >= 0.5:
                rf_TM4_index = 0
            else:
                rf_TM4_index = 1
        elif rf_pred_class_TM4[i] == "Others":
            if rf_pred_prob_TM4[i][0] < 0.5:
                rf_TM4_index = 0
            else:
                rf_TM4_index = 1
        if rf_pred_class_TM5[i] == "Turning & Milling - Machine 5":
            if rf_pred_prob_TM5[i][0] >= 0.5:
                rf_TM5_index = 0
            else:
                rf_TM5_index = 1
        elif rf_pred_class_TM5[i] == "Others":
            if rf_pred_prob_TM5[i][0] < 0.5:
                rf_TM5_index = 0
            else:
                rf_TM5_index = 1
        if rf_pred_class_TM6[i] == "Turning & Milling - Machine 6":
            if rf_pred_prob_TM6[i][0] >= 0.5:
                rf_TM6_index = 0
            else:
                rf_TM6_index = 1
        elif rf_pred_class_TM6[i] == "Others":
            if rf_pred_prob_TM6[i][0] < 0.5:
                rf_TM6_index = 0
            else:
                rf_TM6_index = 1
        if rf_pred_class_TM8[i] == "Turning & Milling - Machine 8":
            if rf_pred_prob_TM8[i][0] >= 0.5:
                rf_TM8_index = 0
            else:
                rf_TM8_index = 1
        elif rf_pred_class_TM8[i] == "Others":
            if rf_pred_prob_TM8[i][0] < 0.5:
                rf_TM8_index = 0
            else:
                rf_TM8_index = 1
        if rf_pred_class_TM9[i] == "Turning & Milling - Machine 9":
            if rf_pred_prob_TM9[i][0] >= 0.5:
                rf_TM9_index = 0
            else:
                rf_TM9_index = 1
        elif rf_pred_class_TM9[i] == "Others":
            if rf_pred_prob_TM9[i][0] < 0.5:
                rf_TM9_index = 0
            else:
                rf_TM9_index = 1
        if rf_pred_class_TMQC[i] == "Turning & Milling Q.C.":
            if rf_pred_prob_TMQC[i][0] >= 0.5:
                rf_TMQC_index = 0
            else:
                rf_TMQC_index = 1
        elif rf_pred_class_TMQC[i] == "Others":
            if rf_pred_prob_TMQC[i][0] < 0.5:
                rf_TMQC_index = 0
            else:
                rf_TMQC_index = 1
        if rf_pred_class_TQC[i] == "Turning Q.C.":
            if rf_pred_prob_TQC[i][0] >= 0.5:
                rf_TQC_index = 0
            else:
                rf_TQC_index = 1
        elif rf_pred_class_TQC[i] == "Others":
            if rf_pred_prob_TQC[i][0] < 0.5:
                rf_TQC_index = 0
            else:
                rf_TQC_index = 1
        """
        if rf_pred_class_WC13[i] == "Wire Cut - Machine 13":
            if rf_pred_prob_WC13[i][0] >= 0.5:
                rf_WC13_index = 0
            else:
                rf_WC13_index = 1
        elif rf_pred_class_WC13[i] == "Others":
            if rf_pred_prob_WC13[i][0] < 0.5:
                rf_WC13_index = 0
            else:
                rf_WC13_index = 1
        """
        #if rf_pred_prob_DM[i][rf_DM_index] == max(rf_pred_prob_FI[i][rf_FI_index], rf_pred_prob_FG[i][rf_FG_index], rf_pred_prob_GR27[i][rf_GR27_index], rf_pred_prob_LM[i][rf_LM_index], rf_pred_prob_LMM[i][rf_LMM_index], rf_pred_prob_PC[i][rf_PC_index], rf_pred_prob_RG12[i][rf_RG12_index], rf_pred_prob_RG2[i][rf_RG2_index], rf_pred_prob_RG3[i][rf_RG3_index], rf_pred_prob_RGM[i][rf_RGM_index], rf_pred_prob_RGQC[i][rf_RGQC_index], rf_pred_prob_T8[i][rf_T8_index], rf_pred_prob_TM10[i][rf_TM10_index], rf_pred_prob_TM4[i][rf_TM4_index], rf_pred_prob_TM5[i][rf_TM5_index], rf_pred_prob_TM6[i][rf_TM6_index], rf_pred_prob_TM8[i][rf_TM8_index], rf_pred_prob_TM9[i][rf_TM9_index], rf_pred_prob_TMQC[i][rf_TMQC_index], rf_pred_prob_TQC[i][rf_TQC_index]):
        #    rf_prediction.loc[i] = "Deburring - Manual"
        if rf_pred_prob_FI[i][rf_FI_index] == max(rf_pred_prob_FI[i][rf_FI_index], rf_pred_prob_FG[i][rf_FG_index], rf_pred_prob_GR27[i][rf_GR27_index], rf_pred_prob_LM[i][rf_LM_index], rf_pred_prob_LMM[i][rf_LMM_index], rf_pred_prob_PC[i][rf_PC_index], rf_pred_prob_RG12[i][rf_RG12_index], rf_pred_prob_RG2[i][rf_RG2_index], rf_pred_prob_RG3[i][rf_RG3_index], rf_pred_prob_RGM[i][rf_RGM_index], rf_pred_prob_RGQC[i][rf_RGQC_index], rf_pred_prob_T8[i][rf_T8_index], rf_pred_prob_TM10[i][rf_TM10_index], rf_pred_prob_TM4[i][rf_TM4_index], rf_pred_prob_TM5[i][rf_TM5_index], rf_pred_prob_TM6[i][rf_TM6_index], rf_pred_prob_TM8[i][rf_TM8_index], rf_pred_prob_TM9[i][rf_TM9_index], rf_pred_prob_TMQC[i][rf_TMQC_index], rf_pred_prob_TQC[i][rf_TQC_index]):
            rf_prediction.loc[i] = "Final Inspection Q.C."
        elif rf_pred_prob_FG[i][rf_FG_index] == max(rf_pred_prob_FI[i][rf_FI_index], rf_pred_prob_FG[i][rf_FG_index], rf_pred_prob_GR27[i][rf_GR27_index], rf_pred_prob_LM[i][rf_LM_index], rf_pred_prob_LMM[i][rf_LMM_index], rf_pred_prob_PC[i][rf_PC_index], rf_pred_prob_RG12[i][rf_RG12_index], rf_pred_prob_RG2[i][rf_RG2_index], rf_pred_prob_RG3[i][rf_RG3_index], rf_pred_prob_RGM[i][rf_RGM_index], rf_pred_prob_RGQC[i][rf_RGQC_index], rf_pred_prob_T8[i][rf_T8_index], rf_pred_prob_TM10[i][rf_TM10_index], rf_pred_prob_TM4[i][rf_TM4_index], rf_pred_prob_TM5[i][rf_TM5_index], rf_pred_prob_TM6[i][rf_TM6_index], rf_pred_prob_TM8[i][rf_TM8_index], rf_pred_prob_TM9[i][rf_TM9_index], rf_pred_prob_TMQC[i][rf_TMQC_index], rf_pred_prob_TQC[i][rf_TQC_index]):
            rf_prediction.loc[i] = "Flat Grinding - Machine 11"
        #elif rf_pred_prob_GR[i][rf_GR_index] == max(rf_pred_prob_FI[i][rf_FI_index], rf_pred_prob_FG[i][rf_FG_index], rf_pred_prob_GR27[i][rf_GR27_index], rf_pred_prob_LM[i][rf_LM_index], rf_pred_prob_LMM[i][rf_LMM_index], rf_pred_prob_PC[i][rf_PC_index], rf_pred_prob_RG12[i][rf_RG12_index], rf_pred_prob_RG2[i][rf_RG2_index], rf_pred_prob_RG3[i][rf_RG3_index], rf_pred_prob_RGM[i][rf_RGM_index], rf_pred_prob_RGQC[i][rf_RGQC_index], rf_pred_prob_T8[i][rf_T8_index], rf_pred_prob_TM10[i][rf_TM10_index], rf_pred_prob_TM4[i][rf_TM4_index], rf_pred_prob_TM5[i][rf_TM5_index], rf_pred_prob_TM6[i][rf_TM6_index], rf_pred_prob_TM8[i][rf_TM8_index], rf_pred_prob_TM9[i][rf_TM9_index], rf_pred_prob_TMQC[i][rf_TMQC_index], rf_pred_prob_TQC[i][rf_TQC_index]):
        #    rf_prediction.loc[i] = "Grinding Rework"
        #elif rf_pred_prob_GR12[i][rf_GR12_index] == max(rf_pred_prob_FI[i][rf_FI_index], rf_pred_prob_FG[i][rf_FG_index], rf_pred_prob_GR27[i][rf_GR27_index], rf_pred_prob_LM[i][rf_LM_index], rf_pred_prob_LMM[i][rf_LMM_index], rf_pred_prob_PC[i][rf_PC_index], rf_pred_prob_RG12[i][rf_RG12_index], rf_pred_prob_RG2[i][rf_RG2_index], rf_pred_prob_RG3[i][rf_RG3_index], rf_pred_prob_RGM[i][rf_RGM_index], rf_pred_prob_RGQC[i][rf_RGQC_index], rf_pred_prob_T8[i][rf_T8_index], rf_pred_prob_TM10[i][rf_TM10_index], rf_pred_prob_TM4[i][rf_TM4_index], rf_pred_prob_TM5[i][rf_TM5_index], rf_pred_prob_TM6[i][rf_TM6_index], rf_pred_prob_TM8[i][rf_TM8_index], rf_pred_prob_TM9[i][rf_TM9_index], rf_pred_prob_TMQC[i][rf_TMQC_index], rf_pred_prob_TQC[i][rf_TQC_index]):
        #    rf_prediction.loc[i] = "Grinding Rework - Machine 12"
        elif rf_pred_prob_GR27[i][rf_GR27_index] == max(rf_pred_prob_FI[i][rf_FI_index], rf_pred_prob_FG[i][rf_FG_index], rf_pred_prob_GR27[i][rf_GR27_index], rf_pred_prob_LM[i][rf_LM_index], rf_pred_prob_LMM[i][rf_LMM_index], rf_pred_prob_PC[i][rf_PC_index], rf_pred_prob_RG12[i][rf_RG12_index], rf_pred_prob_RG2[i][rf_RG2_index], rf_pred_prob_RG3[i][rf_RG3_index], rf_pred_prob_RGM[i][rf_RGM_index], rf_pred_prob_RGQC[i][rf_RGQC_index], rf_pred_prob_T8[i][rf_T8_index], rf_pred_prob_TM10[i][rf_TM10_index], rf_pred_prob_TM4[i][rf_TM4_index], rf_pred_prob_TM5[i][rf_TM5_index], rf_pred_prob_TM6[i][rf_TM6_index], rf_pred_prob_TM8[i][rf_TM8_index], rf_pred_prob_TM9[i][rf_TM9_index], rf_pred_prob_TMQC[i][rf_TMQC_index], rf_pred_prob_TQC[i][rf_TQC_index]):
            rf_prediction.loc[i] = "Grinding Rework - Machine 27"
        elif rf_pred_prob_LM[i][rf_LM_index] == max(rf_pred_prob_FI[i][rf_FI_index], rf_pred_prob_FG[i][rf_FG_index], rf_pred_prob_GR27[i][rf_GR27_index], rf_pred_prob_LM[i][rf_LM_index], rf_pred_prob_LMM[i][rf_LMM_index], rf_pred_prob_PC[i][rf_PC_index], rf_pred_prob_RG12[i][rf_RG12_index], rf_pred_prob_RG2[i][rf_RG2_index], rf_pred_prob_RG3[i][rf_RG3_index], rf_pred_prob_RGM[i][rf_RGM_index], rf_pred_prob_RGQC[i][rf_RGQC_index], rf_pred_prob_T8[i][rf_T8_index], rf_pred_prob_TM10[i][rf_TM10_index], rf_pred_prob_TM4[i][rf_TM4_index], rf_pred_prob_TM5[i][rf_TM5_index], rf_pred_prob_TM6[i][rf_TM6_index], rf_pred_prob_TM8[i][rf_TM8_index], rf_pred_prob_TM9[i][rf_TM9_index], rf_pred_prob_TMQC[i][rf_TMQC_index], rf_pred_prob_TQC[i][rf_TQC_index]):
            rf_prediction.loc[i] = "Lapping - Machine 1"
        elif rf_pred_prob_LMM[i][rf_LMM_index] == max(rf_pred_prob_FI[i][rf_FI_index], rf_pred_prob_FG[i][rf_FG_index], rf_pred_prob_GR27[i][rf_GR27_index], rf_pred_prob_LM[i][rf_LM_index], rf_pred_prob_LMM[i][rf_LMM_index], rf_pred_prob_PC[i][rf_PC_index], rf_pred_prob_RG12[i][rf_RG12_index], rf_pred_prob_RG2[i][rf_RG2_index], rf_pred_prob_RG3[i][rf_RG3_index], rf_pred_prob_RGM[i][rf_RGM_index], rf_pred_prob_RGQC[i][rf_RGQC_index], rf_pred_prob_T8[i][rf_T8_index], rf_pred_prob_TM10[i][rf_TM10_index], rf_pred_prob_TM4[i][rf_TM4_index], rf_pred_prob_TM5[i][rf_TM5_index], rf_pred_prob_TM6[i][rf_TM6_index], rf_pred_prob_TM8[i][rf_TM8_index], rf_pred_prob_TM9[i][rf_TM9_index], rf_pred_prob_TMQC[i][rf_TMQC_index], rf_pred_prob_TQC[i][rf_TQC_index]):
            rf_prediction.loc[i] = "Laser Marking - Machine 7"
        #elif rf_pred_prob_MM14[i][rf_MM14_index] == max(rf_pred_prob_FI[i][rf_FI_index], rf_pred_prob_FG[i][rf_FG_index], rf_pred_prob_GR27[i][rf_GR27_index], rf_pred_prob_LM[i][rf_LM_index], rf_pred_prob_LMM[i][rf_LMM_index], rf_pred_prob_PC[i][rf_PC_index], rf_pred_prob_RG12[i][rf_RG12_index], rf_pred_prob_RG2[i][rf_RG2_index], rf_pred_prob_RG3[i][rf_RG3_index], rf_pred_prob_RGM[i][rf_RGM_index], rf_pred_prob_RGQC[i][rf_RGQC_index], rf_pred_prob_T8[i][rf_T8_index], rf_pred_prob_TM10[i][rf_TM10_index], rf_pred_prob_TM4[i][rf_TM4_index], rf_pred_prob_TM5[i][rf_TM5_index], rf_pred_prob_TM6[i][rf_TM6_index], rf_pred_prob_TM8[i][rf_TM8_index], rf_pred_prob_TM9[i][rf_TM9_index], rf_pred_prob_TMQC[i][rf_TMQC_index], rf_pred_prob_TQC[i][rf_TQC_index]):
        #    rf_prediction.loc[i] = "Milling - Machine 14"
        #elif rf_pred_prob_MM16[i][rf_MM16_index] == max(rf_pred_prob_FI[i][rf_FI_index], rf_pred_prob_FG[i][rf_FG_index], rf_pred_prob_GR27[i][rf_GR27_index], rf_pred_prob_LM[i][rf_LM_index], rf_pred_prob_LMM[i][rf_LMM_index], rf_pred_prob_PC[i][rf_PC_index], rf_pred_prob_RG12[i][rf_RG12_index], rf_pred_prob_RG2[i][rf_RG2_index], rf_pred_prob_RG3[i][rf_RG3_index], rf_pred_prob_RGM[i][rf_RGM_index], rf_pred_prob_RGQC[i][rf_RGQC_index], rf_pred_prob_T8[i][rf_T8_index], rf_pred_prob_TM10[i][rf_TM10_index], rf_pred_prob_TM4[i][rf_TM4_index], rf_pred_prob_TM5[i][rf_TM5_index], rf_pred_prob_TM6[i][rf_TM6_index], rf_pred_prob_TM8[i][rf_TM8_index], rf_pred_prob_TM9[i][rf_TM9_index], rf_pred_prob_TMQC[i][rf_TMQC_index], rf_pred_prob_TQC[i][rf_TQC_index]):
        #    rf_prediction.loc[i] = "Milling - Machine 16"
        elif rf_pred_prob_PC[i][rf_PC_index] == max(rf_pred_prob_FI[i][rf_FI_index], rf_pred_prob_FG[i][rf_FG_index], rf_pred_prob_GR27[i][rf_GR27_index], rf_pred_prob_LM[i][rf_LM_index], rf_pred_prob_LMM[i][rf_LMM_index], rf_pred_prob_PC[i][rf_PC_index], rf_pred_prob_RG12[i][rf_RG12_index], rf_pred_prob_RG2[i][rf_RG2_index], rf_pred_prob_RG3[i][rf_RG3_index], rf_pred_prob_RGM[i][rf_RGM_index], rf_pred_prob_RGQC[i][rf_RGQC_index], rf_pred_prob_T8[i][rf_T8_index], rf_pred_prob_TM10[i][rf_TM10_index], rf_pred_prob_TM4[i][rf_TM4_index], rf_pred_prob_TM5[i][rf_TM5_index], rf_pred_prob_TM6[i][rf_TM6_index], rf_pred_prob_TM8[i][rf_TM8_index], rf_pred_prob_TM9[i][rf_TM9_index], rf_pred_prob_TMQC[i][rf_TMQC_index], rf_pred_prob_TQC[i][rf_TQC_index]):
            rf_prediction.loc[i] = "Packing"
        elif rf_pred_prob_RG12[i][rf_RG12_index] == max(rf_pred_prob_FI[i][rf_FI_index], rf_pred_prob_FG[i][rf_FG_index], rf_pred_prob_GR27[i][rf_GR27_index], rf_pred_prob_LM[i][rf_LM_index], rf_pred_prob_LMM[i][rf_LMM_index], rf_pred_prob_PC[i][rf_PC_index], rf_pred_prob_RG12[i][rf_RG12_index], rf_pred_prob_RG2[i][rf_RG2_index], rf_pred_prob_RG3[i][rf_RG3_index], rf_pred_prob_RGM[i][rf_RGM_index], rf_pred_prob_RGQC[i][rf_RGQC_index], rf_pred_prob_T8[i][rf_T8_index], rf_pred_prob_TM10[i][rf_TM10_index], rf_pred_prob_TM4[i][rf_TM4_index], rf_pred_prob_TM5[i][rf_TM5_index], rf_pred_prob_TM6[i][rf_TM6_index], rf_pred_prob_TM8[i][rf_TM8_index], rf_pred_prob_TM9[i][rf_TM9_index], rf_pred_prob_TMQC[i][rf_TMQC_index], rf_pred_prob_TQC[i][rf_TQC_index]):
            rf_prediction.loc[i] = "Round Grinding - Machine 12"
        #elif rf_pred_prob_RG19[i][rf_RG19_index] == max(rf_pred_prob_FI[i][rf_FI_index], rf_pred_prob_FG[i][rf_FG_index], rf_pred_prob_GR27[i][rf_GR27_index], rf_pred_prob_LM[i][rf_LM_index], rf_pred_prob_LMM[i][rf_LMM_index], rf_pred_prob_PC[i][rf_PC_index], rf_pred_prob_RG12[i][rf_RG12_index], rf_pred_prob_RG2[i][rf_RG2_index], rf_pred_prob_RG3[i][rf_RG3_index], rf_pred_prob_RGM[i][rf_RGM_index], rf_pred_prob_RGQC[i][rf_RGQC_index], rf_pred_prob_T8[i][rf_T8_index], rf_pred_prob_TM10[i][rf_TM10_index], rf_pred_prob_TM4[i][rf_TM4_index], rf_pred_prob_TM5[i][rf_TM5_index], rf_pred_prob_TM6[i][rf_TM6_index], rf_pred_prob_TM8[i][rf_TM8_index], rf_pred_prob_TM9[i][rf_TM9_index], rf_pred_prob_TMQC[i][rf_TMQC_index], rf_pred_prob_TQC[i][rf_TQC_index]):
        #    rf_prediction.loc[i] = "Round Grinding - Machine 19"
        elif rf_pred_prob_RG2[i][rf_RG2_index] == max(rf_pred_prob_FI[i][rf_FI_index], rf_pred_prob_FG[i][rf_FG_index], rf_pred_prob_GR27[i][rf_GR27_index], rf_pred_prob_LM[i][rf_LM_index], rf_pred_prob_LMM[i][rf_LMM_index], rf_pred_prob_PC[i][rf_PC_index], rf_pred_prob_RG12[i][rf_RG12_index], rf_pred_prob_RG2[i][rf_RG2_index], rf_pred_prob_RG3[i][rf_RG3_index], rf_pred_prob_RGM[i][rf_RGM_index], rf_pred_prob_RGQC[i][rf_RGQC_index], rf_pred_prob_T8[i][rf_T8_index], rf_pred_prob_TM10[i][rf_TM10_index], rf_pred_prob_TM4[i][rf_TM4_index], rf_pred_prob_TM5[i][rf_TM5_index], rf_pred_prob_TM6[i][rf_TM6_index], rf_pred_prob_TM8[i][rf_TM8_index], rf_pred_prob_TM9[i][rf_TM9_index], rf_pred_prob_TMQC[i][rf_TMQC_index], rf_pred_prob_TQC[i][rf_TQC_index]):
            rf_prediction.loc[i] = "Round Grinding - Machine 2"
        elif rf_pred_prob_RG3[i][rf_RG3_index] == max(rf_pred_prob_FI[i][rf_FI_index], rf_pred_prob_FG[i][rf_FG_index], rf_pred_prob_GR27[i][rf_GR27_index], rf_pred_prob_LM[i][rf_LM_index], rf_pred_prob_LMM[i][rf_LMM_index], rf_pred_prob_PC[i][rf_PC_index], rf_pred_prob_RG12[i][rf_RG12_index], rf_pred_prob_RG2[i][rf_RG2_index], rf_pred_prob_RG3[i][rf_RG3_index], rf_pred_prob_RGM[i][rf_RGM_index], rf_pred_prob_RGQC[i][rf_RGQC_index], rf_pred_prob_T8[i][rf_T8_index], rf_pred_prob_TM10[i][rf_TM10_index], rf_pred_prob_TM4[i][rf_TM4_index], rf_pred_prob_TM5[i][rf_TM5_index], rf_pred_prob_TM6[i][rf_TM6_index], rf_pred_prob_TM8[i][rf_TM8_index], rf_pred_prob_TM9[i][rf_TM9_index], rf_pred_prob_TMQC[i][rf_TMQC_index], rf_pred_prob_TQC[i][rf_TQC_index]):
            rf_prediction.loc[i] = "Round Grinding - Machine 3"
        elif rf_pred_prob_RGM[i][rf_RGM_index] == max(rf_pred_prob_FI[i][rf_FI_index], rf_pred_prob_FG[i][rf_FG_index], rf_pred_prob_GR27[i][rf_GR27_index], rf_pred_prob_LM[i][rf_LM_index], rf_pred_prob_LMM[i][rf_LMM_index], rf_pred_prob_PC[i][rf_PC_index], rf_pred_prob_RG12[i][rf_RG12_index], rf_pred_prob_RG2[i][rf_RG2_index], rf_pred_prob_RG3[i][rf_RG3_index], rf_pred_prob_RGM[i][rf_RGM_index], rf_pred_prob_RGQC[i][rf_RGQC_index], rf_pred_prob_T8[i][rf_T8_index], rf_pred_prob_TM10[i][rf_TM10_index], rf_pred_prob_TM4[i][rf_TM4_index], rf_pred_prob_TM5[i][rf_TM5_index], rf_pred_prob_TM6[i][rf_TM6_index], rf_pred_prob_TM8[i][rf_TM8_index], rf_pred_prob_TM9[i][rf_TM9_index], rf_pred_prob_TMQC[i][rf_TMQC_index], rf_pred_prob_TQC[i][rf_TQC_index]):
            rf_prediction.loc[i] = "Round Grinding - Manual"
        elif rf_pred_prob_RGQC[i][rf_RGQC_index] == max(rf_pred_prob_FI[i][rf_FI_index], rf_pred_prob_FG[i][rf_FG_index], rf_pred_prob_GR27[i][rf_GR27_index], rf_pred_prob_LM[i][rf_LM_index], rf_pred_prob_LMM[i][rf_LMM_index], rf_pred_prob_PC[i][rf_PC_index], rf_pred_prob_RG12[i][rf_RG12_index], rf_pred_prob_RG2[i][rf_RG2_index], rf_pred_prob_RG3[i][rf_RG3_index], rf_pred_prob_RGM[i][rf_RGM_index], rf_pred_prob_RGQC[i][rf_RGQC_index], rf_pred_prob_T8[i][rf_T8_index], rf_pred_prob_TM10[i][rf_TM10_index], rf_pred_prob_TM4[i][rf_TM4_index], rf_pred_prob_TM5[i][rf_TM5_index], rf_pred_prob_TM6[i][rf_TM6_index], rf_pred_prob_TM8[i][rf_TM8_index], rf_pred_prob_TM9[i][rf_TM9_index], rf_pred_prob_TMQC[i][rf_TMQC_index], rf_pred_prob_TQC[i][rf_TQC_index]):
            rf_prediction.loc[i] = "Round Grinding - Q.C."
        #elif rf_pred_prob_TMSA10[i][rf_TMSA10_index] == max(rf_pred_prob_FI[i][rf_FI_index], rf_pred_prob_FG[i][rf_FG_index], rf_pred_prob_GR27[i][rf_GR27_index], rf_pred_prob_LM[i][rf_LM_index], rf_pred_prob_LMM[i][rf_LMM_index], rf_pred_prob_PC[i][rf_PC_index], rf_pred_prob_RG12[i][rf_RG12_index], rf_pred_prob_RG2[i][rf_RG2_index], rf_pred_prob_RG3[i][rf_RG3_index], rf_pred_prob_RGM[i][rf_RGM_index], rf_pred_prob_RGQC[i][rf_RGQC_index], rf_pred_prob_T8[i][rf_T8_index], rf_pred_prob_TM10[i][rf_TM10_index], rf_pred_prob_TM4[i][rf_TM4_index], rf_pred_prob_TM5[i][rf_TM5_index], rf_pred_prob_TM6[i][rf_TM6_index], rf_pred_prob_TM8[i][rf_TM8_index], rf_pred_prob_TM9[i][rf_TM9_index], rf_pred_prob_TMQC[i][rf_TMQC_index], rf_pred_prob_TQC[i][rf_TQC_index]):
        #    rf_prediction.loc[i] = "Turn & Mill. & Screw Assem - Machine 10"
        elif rf_pred_prob_T8[i][rf_T8_index] == max(rf_pred_prob_FI[i][rf_FI_index], rf_pred_prob_FG[i][rf_FG_index], rf_pred_prob_GR27[i][rf_GR27_index], rf_pred_prob_LM[i][rf_LM_index], rf_pred_prob_LMM[i][rf_LMM_index], rf_pred_prob_PC[i][rf_PC_index], rf_pred_prob_RG12[i][rf_RG12_index], rf_pred_prob_RG2[i][rf_RG2_index], rf_pred_prob_RG3[i][rf_RG3_index], rf_pred_prob_RGM[i][rf_RGM_index], rf_pred_prob_RGQC[i][rf_RGQC_index], rf_pred_prob_T8[i][rf_T8_index], rf_pred_prob_TM10[i][rf_TM10_index], rf_pred_prob_TM4[i][rf_TM4_index], rf_pred_prob_TM5[i][rf_TM5_index], rf_pred_prob_TM6[i][rf_TM6_index], rf_pred_prob_TM8[i][rf_TM8_index], rf_pred_prob_TM9[i][rf_TM9_index], rf_pred_prob_TMQC[i][rf_TMQC_index], rf_pred_prob_TQC[i][rf_TQC_index]):
            rf_prediction.loc[i] = "Turning - Machine 8"
        #elif rf_pred_prob_T9[i][rf_T9_index] == max(rf_pred_prob_FI[i][rf_FI_index], rf_pred_prob_FG[i][rf_FG_index], rf_pred_prob_GR27[i][rf_GR27_index], rf_pred_prob_LM[i][rf_LM_index], rf_pred_prob_LMM[i][rf_LMM_index], rf_pred_prob_PC[i][rf_PC_index], rf_pred_prob_RG12[i][rf_RG12_index], rf_pred_prob_RG2[i][rf_RG2_index], rf_pred_prob_RG3[i][rf_RG3_index], rf_pred_prob_RGM[i][rf_RGM_index], rf_pred_prob_RGQC[i][rf_RGQC_index], rf_pred_prob_T8[i][rf_T8_index], rf_pred_prob_TM10[i][rf_TM10_index], rf_pred_prob_TM4[i][rf_TM4_index], rf_pred_prob_TM5[i][rf_TM5_index], rf_pred_prob_TM6[i][rf_TM6_index], rf_pred_prob_TM8[i][rf_TM8_index], rf_pred_prob_TM9[i][rf_TM9_index], rf_pred_prob_TMQC[i][rf_TMQC_index], rf_pred_prob_TQC[i][rf_TQC_index]):
        #    rf_prediction.loc[i] = "Turning - Machine 9"
        elif rf_pred_prob_TM10[i][rf_TM10_index] == max(rf_pred_prob_FI[i][rf_FI_index], rf_pred_prob_FG[i][rf_FG_index], rf_pred_prob_GR27[i][rf_GR27_index], rf_pred_prob_LM[i][rf_LM_index], rf_pred_prob_LMM[i][rf_LMM_index], rf_pred_prob_PC[i][rf_PC_index], rf_pred_prob_RG12[i][rf_RG12_index], rf_pred_prob_RG2[i][rf_RG2_index], rf_pred_prob_RG3[i][rf_RG3_index], rf_pred_prob_RGM[i][rf_RGM_index], rf_pred_prob_RGQC[i][rf_RGQC_index], rf_pred_prob_T8[i][rf_T8_index], rf_pred_prob_TM10[i][rf_TM10_index], rf_pred_prob_TM4[i][rf_TM4_index], rf_pred_prob_TM5[i][rf_TM5_index], rf_pred_prob_TM6[i][rf_TM6_index], rf_pred_prob_TM8[i][rf_TM8_index], rf_pred_prob_TM9[i][rf_TM9_index], rf_pred_prob_TMQC[i][rf_TMQC_index], rf_pred_prob_TQC[i][rf_TQC_index]):
            rf_prediction.loc[i] = "Turning & Milling - Machine 10"
        elif rf_pred_prob_TM4[i][rf_TM4_index] == max(rf_pred_prob_FI[i][rf_FI_index], rf_pred_prob_FG[i][rf_FG_index], rf_pred_prob_GR27[i][rf_GR27_index], rf_pred_prob_LM[i][rf_LM_index], rf_pred_prob_LMM[i][rf_LMM_index], rf_pred_prob_PC[i][rf_PC_index], rf_pred_prob_RG12[i][rf_RG12_index], rf_pred_prob_RG2[i][rf_RG2_index], rf_pred_prob_RG3[i][rf_RG3_index], rf_pred_prob_RGM[i][rf_RGM_index], rf_pred_prob_RGQC[i][rf_RGQC_index], rf_pred_prob_T8[i][rf_T8_index], rf_pred_prob_TM10[i][rf_TM10_index], rf_pred_prob_TM4[i][rf_TM4_index], rf_pred_prob_TM5[i][rf_TM5_index], rf_pred_prob_TM6[i][rf_TM6_index], rf_pred_prob_TM8[i][rf_TM8_index], rf_pred_prob_TM9[i][rf_TM9_index], rf_pred_prob_TMQC[i][rf_TMQC_index], rf_pred_prob_TQC[i][rf_TQC_index]):
            rf_prediction.loc[i] = "Turning & Milling - Machine 4"
        elif rf_pred_prob_TM5[i][rf_TM5_index] == max(rf_pred_prob_FI[i][rf_FI_index], rf_pred_prob_FG[i][rf_FG_index], rf_pred_prob_GR27[i][rf_GR27_index], rf_pred_prob_LM[i][rf_LM_index], rf_pred_prob_LMM[i][rf_LMM_index], rf_pred_prob_PC[i][rf_PC_index], rf_pred_prob_RG12[i][rf_RG12_index], rf_pred_prob_RG2[i][rf_RG2_index], rf_pred_prob_RG3[i][rf_RG3_index], rf_pred_prob_RGM[i][rf_RGM_index], rf_pred_prob_RGQC[i][rf_RGQC_index], rf_pred_prob_T8[i][rf_T8_index], rf_pred_prob_TM10[i][rf_TM10_index], rf_pred_prob_TM4[i][rf_TM4_index], rf_pred_prob_TM5[i][rf_TM5_index], rf_pred_prob_TM6[i][rf_TM6_index], rf_pred_prob_TM8[i][rf_TM8_index], rf_pred_prob_TM9[i][rf_TM9_index], rf_pred_prob_TMQC[i][rf_TMQC_index], rf_pred_prob_TQC[i][rf_TQC_index]):
            rf_prediction.loc[i] = "Turning & Milling - Machine 5"
        elif rf_pred_prob_TM6[i][rf_TM6_index] == max(rf_pred_prob_FI[i][rf_FI_index], rf_pred_prob_FG[i][rf_FG_index], rf_pred_prob_GR27[i][rf_GR27_index], rf_pred_prob_LM[i][rf_LM_index], rf_pred_prob_LMM[i][rf_LMM_index], rf_pred_prob_PC[i][rf_PC_index], rf_pred_prob_RG12[i][rf_RG12_index], rf_pred_prob_RG2[i][rf_RG2_index], rf_pred_prob_RG3[i][rf_RG3_index], rf_pred_prob_RGM[i][rf_RGM_index], rf_pred_prob_RGQC[i][rf_RGQC_index], rf_pred_prob_T8[i][rf_T8_index], rf_pred_prob_TM10[i][rf_TM10_index], rf_pred_prob_TM4[i][rf_TM4_index], rf_pred_prob_TM5[i][rf_TM5_index], rf_pred_prob_TM6[i][rf_TM6_index], rf_pred_prob_TM8[i][rf_TM8_index], rf_pred_prob_TM9[i][rf_TM9_index], rf_pred_prob_TMQC[i][rf_TMQC_index], rf_pred_prob_TQC[i][rf_TQC_index]):
            rf_prediction.loc[i] = "Turning & Milling - Machine 6"
        elif rf_pred_prob_TM8[i][rf_TM8_index] == max(rf_pred_prob_FI[i][rf_FI_index], rf_pred_prob_FG[i][rf_FG_index], rf_pred_prob_GR27[i][rf_GR27_index], rf_pred_prob_LM[i][rf_LM_index], rf_pred_prob_LMM[i][rf_LMM_index], rf_pred_prob_PC[i][rf_PC_index], rf_pred_prob_RG12[i][rf_RG12_index], rf_pred_prob_RG2[i][rf_RG2_index], rf_pred_prob_RG3[i][rf_RG3_index], rf_pred_prob_RGM[i][rf_RGM_index], rf_pred_prob_RGQC[i][rf_RGQC_index], rf_pred_prob_T8[i][rf_T8_index], rf_pred_prob_TM10[i][rf_TM10_index], rf_pred_prob_TM4[i][rf_TM4_index], rf_pred_prob_TM5[i][rf_TM5_index], rf_pred_prob_TM6[i][rf_TM6_index], rf_pred_prob_TM8[i][rf_TM8_index], rf_pred_prob_TM9[i][rf_TM9_index], rf_pred_prob_TMQC[i][rf_TMQC_index], rf_pred_prob_TQC[i][rf_TQC_index]):
            rf_prediction.loc[i] = "Turning & Milling - Machine 8"
        elif rf_pred_prob_TM9[i][rf_TM9_index] == max(rf_pred_prob_FI[i][rf_FI_index], rf_pred_prob_FG[i][rf_FG_index], rf_pred_prob_GR27[i][rf_GR27_index], rf_pred_prob_LM[i][rf_LM_index], rf_pred_prob_LMM[i][rf_LMM_index], rf_pred_prob_PC[i][rf_PC_index], rf_pred_prob_RG12[i][rf_RG12_index], rf_pred_prob_RG2[i][rf_RG2_index], rf_pred_prob_RG3[i][rf_RG3_index], rf_pred_prob_RGM[i][rf_RGM_index], rf_pred_prob_RGQC[i][rf_RGQC_index], rf_pred_prob_T8[i][rf_T8_index], rf_pred_prob_TM10[i][rf_TM10_index], rf_pred_prob_TM4[i][rf_TM4_index], rf_pred_prob_TM5[i][rf_TM5_index], rf_pred_prob_TM6[i][rf_TM6_index], rf_pred_prob_TM8[i][rf_TM8_index], rf_pred_prob_TM9[i][rf_TM9_index], rf_pred_prob_TMQC[i][rf_TMQC_index], rf_pred_prob_TQC[i][rf_TQC_index]):
            rf_prediction.loc[i] = "Turning & Milling - Machine 9"
        elif rf_pred_prob_TMQC[i][rf_TMQC_index] == max(rf_pred_prob_FI[i][rf_FI_index], rf_pred_prob_FG[i][rf_FG_index], rf_pred_prob_GR27[i][rf_GR27_index], rf_pred_prob_LM[i][rf_LM_index], rf_pred_prob_LMM[i][rf_LMM_index], rf_pred_prob_PC[i][rf_PC_index], rf_pred_prob_RG12[i][rf_RG12_index], rf_pred_prob_RG2[i][rf_RG2_index], rf_pred_prob_RG3[i][rf_RG3_index], rf_pred_prob_RGM[i][rf_RGM_index], rf_pred_prob_RGQC[i][rf_RGQC_index], rf_pred_prob_T8[i][rf_T8_index], rf_pred_prob_TM10[i][rf_TM10_index], rf_pred_prob_TM4[i][rf_TM4_index], rf_pred_prob_TM5[i][rf_TM5_index], rf_pred_prob_TM6[i][rf_TM6_index], rf_pred_prob_TM8[i][rf_TM8_index], rf_pred_prob_TM9[i][rf_TM9_index], rf_pred_prob_TMQC[i][rf_TMQC_index], rf_pred_prob_TQC[i][rf_TQC_index]):
            rf_prediction.loc[i] = "Turning & Milling Q.C."
        elif rf_pred_prob_TQC[i][rf_TQC_index] == max(rf_pred_prob_FI[i][rf_FI_index], rf_pred_prob_FG[i][rf_FG_index], rf_pred_prob_GR27[i][rf_GR27_index], rf_pred_prob_LM[i][rf_LM_index], rf_pred_prob_LMM[i][rf_LMM_index], rf_pred_prob_PC[i][rf_PC_index], rf_pred_prob_RG12[i][rf_RG12_index], rf_pred_prob_RG2[i][rf_RG2_index], rf_pred_prob_RG3[i][rf_RG3_index], rf_pred_prob_RGM[i][rf_RGM_index], rf_pred_prob_RGQC[i][rf_RGQC_index], rf_pred_prob_T8[i][rf_T8_index], rf_pred_prob_TM10[i][rf_TM10_index], rf_pred_prob_TM4[i][rf_TM4_index], rf_pred_prob_TM5[i][rf_TM5_index], rf_pred_prob_TM6[i][rf_TM6_index], rf_pred_prob_TM8[i][rf_TM8_index], rf_pred_prob_TM9[i][rf_TM9_index], rf_pred_prob_TMQC[i][rf_TMQC_index], rf_pred_prob_TQC[i][rf_TQC_index]):
            rf_prediction.loc[i] = "Turning Q.C."
        #elif rf_pred_prob_WC13[i][rf_WC13_index] == max(rf_pred_prob_FI[i][rf_FI_index], rf_pred_prob_FG[i][rf_FG_index], rf_pred_prob_GR27[i][rf_GR27_index], rf_pred_prob_LM[i][rf_LM_index], rf_pred_prob_LMM[i][rf_LMM_index], rf_pred_prob_PC[i][rf_PC_index], rf_pred_prob_RG12[i][rf_RG12_index], rf_pred_prob_RG2[i][rf_RG2_index], rf_pred_prob_RG3[i][rf_RG3_index], rf_pred_prob_RGM[i][rf_RGM_index], rf_pred_prob_RGQC[i][rf_RGQC_index], rf_pred_prob_T8[i][rf_T8_index], rf_pred_prob_TM10[i][rf_TM10_index], rf_pred_prob_TM4[i][rf_TM4_index], rf_pred_prob_TM5[i][rf_TM5_index], rf_pred_prob_TM6[i][rf_TM6_index], rf_pred_prob_TM8[i][rf_TM8_index], rf_pred_prob_TM9[i][rf_TM9_index], rf_pred_prob_TMQC[i][rf_TMQC_index], rf_pred_prob_TQC[i][rf_TQC_index]):
        #    rf_prediction.loc[i] = "Wire Cut - Machine 13"


    def get_precision(rf_conf_matrix):
        rf_tp_1 = rf_conf_matrix[0][0]
        rf_tp_2 = rf_conf_matrix[1][1]
        rf_tp_3 = rf_conf_matrix[2][2]
        rf_tp_4 = rf_conf_matrix[3][3]
        rf_tp_5 = rf_conf_matrix[4][4]
        rf_tp_6 = rf_conf_matrix[5][5]
        rf_tp_7 = rf_conf_matrix[6][6]
        rf_tp_8 = rf_conf_matrix[7][7]
        rf_tp_9 = rf_conf_matrix[8][8]
        rf_tp_10 = rf_conf_matrix[9][9]
        rf_tp_11 = rf_conf_matrix[10][10]
        rf_tp_12 = rf_conf_matrix[11][11]
        rf_tp_13 = rf_conf_matrix[12][12]
        rf_tp_14 = rf_conf_matrix[13][13]
        rf_tp_15 = rf_conf_matrix[14][14]
        rf_tp_16 = rf_conf_matrix[15][15]
        rf_tp_17 = rf_conf_matrix[16][16]
        rf_tp_18 = rf_conf_matrix[17][17]
        rf_tp_19 = rf_conf_matrix[18][18]
        rf_tp_20 = rf_conf_matrix[19][19]
        #rf_tp_21 = rf_conf_matrix[20][20]
        #rf_tp_22 = rf_conf_matrix[21][21]
        #rf_tp_23 = rf_conf_matrix[22][22]
        #rf_tp_24 = rf_conf_matrix[23][23]
        #rf_tp_25 = rf_conf_matrix[24][24]
        #rf_tp_26 = rf_conf_matrix[25][25]
        #rf_tp_27 = rf_conf_matrix[26][26]
        #rf_tp_28 = rf_conf_matrix[27][27]
        #rf_tp_29 = rf_conf_matrix[28][28]

        rf_fp_1 = rf_conf_matrix[1][0] + rf_conf_matrix[2][0] + rf_conf_matrix[3][0] + rf_conf_matrix[4][0] + rf_conf_matrix[5][0] + rf_conf_matrix[6][0] + rf_conf_matrix[7][0] + rf_conf_matrix[8][0] + rf_conf_matrix[9][0] + rf_conf_matrix[10][0] + rf_conf_matrix[11][0] + rf_conf_matrix[12][0] + rf_conf_matrix[13][0] + rf_conf_matrix[14][0] + rf_conf_matrix[15][0] + rf_conf_matrix[16][0] + rf_conf_matrix[17][0] + rf_conf_matrix[18][0] + rf_conf_matrix[19][0]
        rf_fp_2 = rf_conf_matrix[0][1] + rf_conf_matrix[2][1] + rf_conf_matrix[3][1] + rf_conf_matrix[4][1] + rf_conf_matrix[5][1] + rf_conf_matrix[6][1] + rf_conf_matrix[7][1] + rf_conf_matrix[8][1] + rf_conf_matrix[9][1] + rf_conf_matrix[10][1] + rf_conf_matrix[11][1] + rf_conf_matrix[12][1] + rf_conf_matrix[13][1] + rf_conf_matrix[14][1] + rf_conf_matrix[15][1] + rf_conf_matrix[16][1] + rf_conf_matrix[17][1] + rf_conf_matrix[18][1] + rf_conf_matrix[19][1]
        rf_fp_3 = rf_conf_matrix[0][2] + rf_conf_matrix[1][2] + rf_conf_matrix[3][2] + rf_conf_matrix[4][2] + rf_conf_matrix[5][2] + rf_conf_matrix[6][2] + rf_conf_matrix[7][2] + rf_conf_matrix[8][2] + rf_conf_matrix[9][2] + rf_conf_matrix[10][2] + rf_conf_matrix[11][2] + rf_conf_matrix[12][2] + rf_conf_matrix[13][2] + rf_conf_matrix[14][2] + rf_conf_matrix[15][2] + rf_conf_matrix[16][2] + rf_conf_matrix[17][2] + rf_conf_matrix[18][2] + rf_conf_matrix[19][2]
        rf_fp_4 = rf_conf_matrix[0][3] + rf_conf_matrix[1][3] + rf_conf_matrix[2][3] + rf_conf_matrix[4][3] + rf_conf_matrix[5][3] + rf_conf_matrix[6][3] + rf_conf_matrix[7][3] + rf_conf_matrix[8][3] + rf_conf_matrix[9][3] + rf_conf_matrix[10][3] + rf_conf_matrix[11][3] + rf_conf_matrix[12][3] + rf_conf_matrix[13][3] + rf_conf_matrix[14][3] + rf_conf_matrix[15][3] + rf_conf_matrix[16][3] + rf_conf_matrix[17][3] + rf_conf_matrix[18][3] + rf_conf_matrix[19][3]
        rf_fp_5 = rf_conf_matrix[0][4] + rf_conf_matrix[1][4] + rf_conf_matrix[2][4] + rf_conf_matrix[3][4] + rf_conf_matrix[5][4] + rf_conf_matrix[6][4] + rf_conf_matrix[7][4] + rf_conf_matrix[8][4] + rf_conf_matrix[9][4] + rf_conf_matrix[10][4] + rf_conf_matrix[11][4] + rf_conf_matrix[12][4] + rf_conf_matrix[13][4] + rf_conf_matrix[14][4] + rf_conf_matrix[15][4] + rf_conf_matrix[16][4] + rf_conf_matrix[17][4] + rf_conf_matrix[18][4] + rf_conf_matrix[19][4]
        rf_fp_6 = rf_conf_matrix[0][5] + rf_conf_matrix[1][5] + rf_conf_matrix[2][5] + rf_conf_matrix[3][5] + rf_conf_matrix[4][5] + rf_conf_matrix[6][5] + rf_conf_matrix[7][5] + rf_conf_matrix[8][5] + rf_conf_matrix[9][5] + rf_conf_matrix[10][5] + rf_conf_matrix[11][5] + rf_conf_matrix[12][5] + rf_conf_matrix[13][5] + rf_conf_matrix[14][5] + rf_conf_matrix[15][5] + rf_conf_matrix[16][5] + rf_conf_matrix[17][5] + rf_conf_matrix[18][5] + rf_conf_matrix[19][5]
        rf_fp_7 = rf_conf_matrix[0][6] + rf_conf_matrix[1][6] + rf_conf_matrix[2][6] + rf_conf_matrix[3][6] + rf_conf_matrix[4][6] + rf_conf_matrix[5][6] + rf_conf_matrix[7][6] + rf_conf_matrix[8][6] + rf_conf_matrix[9][6] + rf_conf_matrix[10][6] + rf_conf_matrix[11][6] + rf_conf_matrix[12][6] + rf_conf_matrix[13][6] + rf_conf_matrix[14][6] + rf_conf_matrix[15][6] + rf_conf_matrix[16][6] + rf_conf_matrix[17][6] + rf_conf_matrix[18][6] + rf_conf_matrix[19][6]
        rf_fp_8 = rf_conf_matrix[0][7] + rf_conf_matrix[1][7] + rf_conf_matrix[2][7] + rf_conf_matrix[3][7] + rf_conf_matrix[4][7] + rf_conf_matrix[5][7] + rf_conf_matrix[6][7] + rf_conf_matrix[8][7] + rf_conf_matrix[9][7] + rf_conf_matrix[10][7] + rf_conf_matrix[11][7] + rf_conf_matrix[12][7] + rf_conf_matrix[13][7] + rf_conf_matrix[14][7] + rf_conf_matrix[15][7] + rf_conf_matrix[16][7] + rf_conf_matrix[17][7] + rf_conf_matrix[18][7] + rf_conf_matrix[19][7]
        rf_fp_9 = rf_conf_matrix[0][8] + rf_conf_matrix[1][8] + rf_conf_matrix[2][8] + rf_conf_matrix[3][8] + rf_conf_matrix[4][8] + rf_conf_matrix[5][8] + rf_conf_matrix[6][8] + rf_conf_matrix[7][8] + rf_conf_matrix[9][8] + rf_conf_matrix[10][8] + rf_conf_matrix[11][8] + rf_conf_matrix[12][8] + rf_conf_matrix[13][8] + rf_conf_matrix[14][8] + rf_conf_matrix[15][8] + rf_conf_matrix[16][8] + rf_conf_matrix[17][8] + rf_conf_matrix[18][8] + rf_conf_matrix[19][8]
        rf_fp_10 = rf_conf_matrix[0][9] + rf_conf_matrix[1][9] + rf_conf_matrix[2][9] + rf_conf_matrix[3][9] + rf_conf_matrix[4][9] + rf_conf_matrix[5][9] + rf_conf_matrix[6][9] + rf_conf_matrix[7][9] + rf_conf_matrix[8][9] + rf_conf_matrix[10][9] + rf_conf_matrix[11][9] + rf_conf_matrix[12][9] + rf_conf_matrix[13][9] + rf_conf_matrix[14][9] + rf_conf_matrix[15][9] + rf_conf_matrix[16][9] + rf_conf_matrix[17][9] + rf_conf_matrix[18][9] + rf_conf_matrix[19][9]
        rf_fp_11 = rf_conf_matrix[0][10] + rf_conf_matrix[1][10] + rf_conf_matrix[2][10] + rf_conf_matrix[3][10] + rf_conf_matrix[4][10] + rf_conf_matrix[5][10] + rf_conf_matrix[6][10] + rf_conf_matrix[7][10] + rf_conf_matrix[8][10] + rf_conf_matrix[9][10] + rf_conf_matrix[11][10] + rf_conf_matrix[12][10] + rf_conf_matrix[13][10] + rf_conf_matrix[14][10] + rf_conf_matrix[15][10] + rf_conf_matrix[16][10] + rf_conf_matrix[17][10] + rf_conf_matrix[18][10] + rf_conf_matrix[19][10]
        rf_fp_12 = rf_conf_matrix[0][11] + rf_conf_matrix[1][11] + rf_conf_matrix[2][11] + rf_conf_matrix[3][11] + rf_conf_matrix[4][11] + rf_conf_matrix[5][11] + rf_conf_matrix[6][11] + rf_conf_matrix[7][11] + rf_conf_matrix[8][11] + rf_conf_matrix[9][11] + rf_conf_matrix[10][11] + rf_conf_matrix[12][11] + rf_conf_matrix[13][11] + rf_conf_matrix[14][11] + rf_conf_matrix[15][11] + rf_conf_matrix[16][11] + rf_conf_matrix[17][11] + rf_conf_matrix[18][11] + rf_conf_matrix[19][11]
        rf_fp_13 = rf_conf_matrix[0][12] + rf_conf_matrix[1][12] + rf_conf_matrix[2][12] + rf_conf_matrix[3][12] + rf_conf_matrix[4][12] + rf_conf_matrix[5][12] + rf_conf_matrix[6][12] + rf_conf_matrix[7][12] + rf_conf_matrix[8][12] + rf_conf_matrix[9][12] + rf_conf_matrix[10][12] + rf_conf_matrix[11][12] + rf_conf_matrix[13][12] + rf_conf_matrix[14][12] + rf_conf_matrix[15][12] + rf_conf_matrix[16][12] + rf_conf_matrix[17][12] + rf_conf_matrix[18][12] + rf_conf_matrix[19][12]
        rf_fp_14 = rf_conf_matrix[0][13] + rf_conf_matrix[1][13] + rf_conf_matrix[2][13] + rf_conf_matrix[3][13] + rf_conf_matrix[4][13] + rf_conf_matrix[5][13] + rf_conf_matrix[6][13] + rf_conf_matrix[7][13] + rf_conf_matrix[8][13] + rf_conf_matrix[9][13] + rf_conf_matrix[10][13] + rf_conf_matrix[11][13] + rf_conf_matrix[12][13] + rf_conf_matrix[14][13] + rf_conf_matrix[15][13] + rf_conf_matrix[16][13] + rf_conf_matrix[17][13] + rf_conf_matrix[18][13] + rf_conf_matrix[19][13]
        rf_fp_15 = rf_conf_matrix[0][14] + rf_conf_matrix[1][14] + rf_conf_matrix[2][14] + rf_conf_matrix[3][14] + rf_conf_matrix[4][14] + rf_conf_matrix[5][14] + rf_conf_matrix[6][14] + rf_conf_matrix[7][14] + rf_conf_matrix[8][14] + rf_conf_matrix[9][14] + rf_conf_matrix[10][14] + rf_conf_matrix[11][14] + rf_conf_matrix[12][14] + rf_conf_matrix[13][14] + rf_conf_matrix[15][14] + rf_conf_matrix[16][14] + rf_conf_matrix[17][14] + rf_conf_matrix[18][14] + rf_conf_matrix[19][14]
        rf_fp_16 = rf_conf_matrix[0][15] + rf_conf_matrix[1][15] + rf_conf_matrix[2][15] + rf_conf_matrix[3][15] + rf_conf_matrix[4][15] + rf_conf_matrix[5][15] + rf_conf_matrix[6][15] + rf_conf_matrix[7][15] + rf_conf_matrix[8][15] + rf_conf_matrix[9][15] + rf_conf_matrix[10][15] + rf_conf_matrix[11][15] + rf_conf_matrix[12][15] + rf_conf_matrix[13][15] + rf_conf_matrix[14][15] + rf_conf_matrix[16][15] + rf_conf_matrix[17][15] + rf_conf_matrix[18][15] + rf_conf_matrix[19][15]
        rf_fp_17 = rf_conf_matrix[0][16] + rf_conf_matrix[1][16] + rf_conf_matrix[2][16] + rf_conf_matrix[3][16] + rf_conf_matrix[4][16] + rf_conf_matrix[5][16] + rf_conf_matrix[6][16] + rf_conf_matrix[7][16] + rf_conf_matrix[8][16] + rf_conf_matrix[9][16] + rf_conf_matrix[10][16] + rf_conf_matrix[11][16] + rf_conf_matrix[12][16] + rf_conf_matrix[13][16] + rf_conf_matrix[14][16] + rf_conf_matrix[15][16] + rf_conf_matrix[17][16] + rf_conf_matrix[18][16] + rf_conf_matrix[19][16]
        rf_fp_18 = rf_conf_matrix[0][17] + rf_conf_matrix[1][17] + rf_conf_matrix[2][17] + rf_conf_matrix[3][17] + rf_conf_matrix[4][17] + rf_conf_matrix[5][17] + rf_conf_matrix[6][17] + rf_conf_matrix[7][17] + rf_conf_matrix[8][17] + rf_conf_matrix[9][17] + rf_conf_matrix[10][17] + rf_conf_matrix[11][17] + rf_conf_matrix[12][17] + rf_conf_matrix[13][17] + rf_conf_matrix[14][17] + rf_conf_matrix[15][17] + rf_conf_matrix[16][17] + rf_conf_matrix[18][17] + rf_conf_matrix[19][17]
        rf_fp_19 = rf_conf_matrix[0][18] + rf_conf_matrix[1][18] + rf_conf_matrix[2][18] + rf_conf_matrix[3][18] + rf_conf_matrix[4][18] + rf_conf_matrix[5][18] + rf_conf_matrix[6][18] + rf_conf_matrix[7][18] + rf_conf_matrix[8][18] + rf_conf_matrix[9][18] + rf_conf_matrix[10][18] + rf_conf_matrix[11][18] + rf_conf_matrix[12][18] + rf_conf_matrix[13][18] + rf_conf_matrix[14][18] + rf_conf_matrix[15][18] + rf_conf_matrix[16][18] + rf_conf_matrix[17][18] + rf_conf_matrix[19][18]
        rf_fp_20 = rf_conf_matrix[0][19] + rf_conf_matrix[1][19] + rf_conf_matrix[2][19] + rf_conf_matrix[3][19] + rf_conf_matrix[4][19] + rf_conf_matrix[5][19] + rf_conf_matrix[6][19] + rf_conf_matrix[7][19] + rf_conf_matrix[8][19] + rf_conf_matrix[9][19] + rf_conf_matrix[10][19] + rf_conf_matrix[11][19] + rf_conf_matrix[12][19] + rf_conf_matrix[13][19] + rf_conf_matrix[14][19] + rf_conf_matrix[15][19] + rf_conf_matrix[16][19] + rf_conf_matrix[17][19] + rf_conf_matrix[18][19]
        #rf_fp_21 = rf_conf_matrix[0][20] + rf_conf_matrix[1][20] + rf_conf_matrix[2][20] + rf_conf_matrix[3][20] + rf_conf_matrix[4][20] + rf_conf_matrix[5][20] + rf_conf_matrix[6][20] + rf_conf_matrix[7][20] + rf_conf_matrix[8][20] + rf_conf_matrix[9][20] + rf_conf_matrix[10][20] + rf_conf_matrix[11][20] + rf_conf_matrix[12][20] + rf_conf_matrix[13][20] + rf_conf_matrix[14][20] + rf_conf_matrix[15][20] + rf_conf_matrix[16][20] + rf_conf_matrix[17][20] + rf_conf_matrix[18][20] + rf_conf_matrix[19][20] + rf_conf_matrix[21][20] + rf_conf_matrix[22][20] + rf_conf_matrix[23][20] + rf_conf_matrix[24][20] + rf_conf_matrix[25][20] + rf_conf_matrix[26][20] + rf_conf_matrix[27][20] + rf_conf_matrix[28][20]
        #rf_fp_22 = rf_conf_matrix[0][21] + rf_conf_matrix[1][21] + rf_conf_matrix[2][21] + rf_conf_matrix[3][21] + rf_conf_matrix[4][21] + rf_conf_matrix[5][21] + rf_conf_matrix[6][21] + rf_conf_matrix[7][21] + rf_conf_matrix[8][21] + rf_conf_matrix[9][21] + rf_conf_matrix[10][21] + rf_conf_matrix[11][21] + rf_conf_matrix[12][21] + rf_conf_matrix[13][21] + rf_conf_matrix[14][21] + rf_conf_matrix[15][21] + rf_conf_matrix[16][21] + rf_conf_matrix[17][21] + rf_conf_matrix[18][21] + rf_conf_matrix[19][21] + rf_conf_matrix[20][21] + rf_conf_matrix[22][21] + rf_conf_matrix[23][21] + rf_conf_matrix[24][21] + rf_conf_matrix[25][21] + rf_conf_matrix[26][21] + rf_conf_matrix[27][21] + rf_conf_matrix[28][21]
        #rf_fp_23 = rf_conf_matrix[0][22] + rf_conf_matrix[1][22] + rf_conf_matrix[2][22] + rf_conf_matrix[3][22] + rf_conf_matrix[4][22] + rf_conf_matrix[5][22] + rf_conf_matrix[6][22] + rf_conf_matrix[7][22] + rf_conf_matrix[8][22] + rf_conf_matrix[9][22] + rf_conf_matrix[10][22] + rf_conf_matrix[11][22] + rf_conf_matrix[12][22] + rf_conf_matrix[13][22] + rf_conf_matrix[14][22] + rf_conf_matrix[15][22] + rf_conf_matrix[16][22] + rf_conf_matrix[17][22] + rf_conf_matrix[18][22] + rf_conf_matrix[19][22] + rf_conf_matrix[20][22] + rf_conf_matrix[21][22] + rf_conf_matrix[23][22] + rf_conf_matrix[24][22] + rf_conf_matrix[25][22] + rf_conf_matrix[26][22] + rf_conf_matrix[27][22] + rf_conf_matrix[28][22]
        #rf_fp_24 = rf_conf_matrix[0][23] + rf_conf_matrix[1][23] + rf_conf_matrix[2][23] + rf_conf_matrix[3][23] + rf_conf_matrix[4][23] + rf_conf_matrix[5][23] + rf_conf_matrix[6][23] + rf_conf_matrix[7][23] + rf_conf_matrix[8][23] + rf_conf_matrix[9][23] + rf_conf_matrix[10][23] + rf_conf_matrix[11][23] + rf_conf_matrix[12][23] + rf_conf_matrix[13][23] + rf_conf_matrix[14][23] + rf_conf_matrix[15][23] + rf_conf_matrix[16][23] + rf_conf_matrix[17][23] + rf_conf_matrix[18][23] + rf_conf_matrix[19][23] + rf_conf_matrix[20][23] + rf_conf_matrix[21][23] + rf_conf_matrix[22][23] + rf_conf_matrix[24][23] + rf_conf_matrix[25][23] + rf_conf_matrix[26][23] + rf_conf_matrix[27][23] + rf_conf_matrix[28][23]
        #rf_fp_25 = rf_conf_matrix[0][24] + rf_conf_matrix[1][24] + rf_conf_matrix[2][24] + rf_conf_matrix[3][24] + rf_conf_matrix[4][24] + rf_conf_matrix[5][24] + rf_conf_matrix[6][24] + rf_conf_matrix[7][24] + rf_conf_matrix[8][24] + rf_conf_matrix[9][24] + rf_conf_matrix[10][24] + rf_conf_matrix[11][24] + rf_conf_matrix[12][24] + rf_conf_matrix[13][24] + rf_conf_matrix[14][24] + rf_conf_matrix[15][24] + rf_conf_matrix[16][24] + rf_conf_matrix[17][24] + rf_conf_matrix[18][24] + rf_conf_matrix[19][24] + rf_conf_matrix[20][24] + rf_conf_matrix[21][24] + rf_conf_matrix[22][24] + rf_conf_matrix[23][24] + rf_conf_matrix[25][24] + rf_conf_matrix[26][24] + rf_conf_matrix[27][24] + rf_conf_matrix[28][24]
        #rf_fp_26 = rf_conf_matrix[0][25] + rf_conf_matrix[1][25] + rf_conf_matrix[2][25] + rf_conf_matrix[3][25] + rf_conf_matrix[4][25] + rf_conf_matrix[5][25] + rf_conf_matrix[6][25] + rf_conf_matrix[7][25] + rf_conf_matrix[8][25] + rf_conf_matrix[9][25] + rf_conf_matrix[10][25] + rf_conf_matrix[11][25] + rf_conf_matrix[12][25] + rf_conf_matrix[13][25] + rf_conf_matrix[14][25] + rf_conf_matrix[15][25] + rf_conf_matrix[16][25] + rf_conf_matrix[17][25] + rf_conf_matrix[18][25] + rf_conf_matrix[19][25] + rf_conf_matrix[20][25] + rf_conf_matrix[21][25] + rf_conf_matrix[22][25] + rf_conf_matrix[23][25] + rf_conf_matrix[24][25] + rf_conf_matrix[26][25] + rf_conf_matrix[27][25] + rf_conf_matrix[28][25]
        #rf_fp_27 = rf_conf_matrix[0][26] + rf_conf_matrix[1][26] + rf_conf_matrix[2][26] + rf_conf_matrix[3][26] + rf_conf_matrix[4][26] + rf_conf_matrix[5][26] + rf_conf_matrix[6][26] + rf_conf_matrix[7][26] + rf_conf_matrix[8][26] + rf_conf_matrix[9][26] + rf_conf_matrix[10][26] + rf_conf_matrix[11][26] + rf_conf_matrix[12][26] + rf_conf_matrix[13][26] + rf_conf_matrix[14][26] + rf_conf_matrix[15][26] + rf_conf_matrix[16][26] + rf_conf_matrix[17][26] + rf_conf_matrix[18][26] + rf_conf_matrix[19][26] + rf_conf_matrix[20][26] + rf_conf_matrix[21][26] + rf_conf_matrix[22][26] + rf_conf_matrix[23][26] + rf_conf_matrix[24][26] + rf_conf_matrix[25][26] + rf_conf_matrix[27][26] + rf_conf_matrix[28][26]
        #rf_fp_28 = rf_conf_matrix[0][27] + rf_conf_matrix[1][27] + rf_conf_matrix[2][27] + rf_conf_matrix[3][27] + rf_conf_matrix[4][27] + rf_conf_matrix[5][27] + rf_conf_matrix[6][27] + rf_conf_matrix[7][27] + rf_conf_matrix[8][27] + rf_conf_matrix[9][27] + rf_conf_matrix[10][27] + rf_conf_matrix[11][27] + rf_conf_matrix[12][27] + rf_conf_matrix[13][27] + rf_conf_matrix[14][27] + rf_conf_matrix[15][27] + rf_conf_matrix[16][27] + rf_conf_matrix[17][27] + rf_conf_matrix[18][27] + rf_conf_matrix[19][27] + rf_conf_matrix[20][27] + rf_conf_matrix[21][27] + rf_conf_matrix[22][27] + rf_conf_matrix[23][27] + rf_conf_matrix[24][27] + rf_conf_matrix[25][27] + rf_conf_matrix[26][27] + rf_conf_matrix[28][27]
        #rf_fp_29 = rf_conf_matrix[0][28] + rf_conf_matrix[1][28] + rf_conf_matrix[2][28] + rf_conf_matrix[3][28] + rf_conf_matrix[4][28] + rf_conf_matrix[5][28] + rf_conf_matrix[6][28] + rf_conf_matrix[7][28] + rf_conf_matrix[8][28] + rf_conf_matrix[9][28] + rf_conf_matrix[10][28] + rf_conf_matrix[11][28] + rf_conf_matrix[12][28] + rf_conf_matrix[13][28] + rf_conf_matrix[14][28] + rf_conf_matrix[15][28] + rf_conf_matrix[16][28] + rf_conf_matrix[17][28] + rf_conf_matrix[18][28] + rf_conf_matrix[19][28] + rf_conf_matrix[20][28] + rf_conf_matrix[21][28] + rf_conf_matrix[22][28] + rf_conf_matrix[23][28] + rf_conf_matrix[24][28] + rf_conf_matrix[25][28] + rf_conf_matrix[26][28] + rf_conf_matrix[27][28]

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
        if rf_tp_6 + rf_fp_6 == 0:
            rf_precision_6 = 0
        else:
            rf_precision_6 = rf_tp_6 / (rf_tp_6 + rf_fp_6)
        if rf_tp_7 + rf_fp_7 == 0:
            rf_precision_7 = 0
        else:
            rf_precision_7 = rf_tp_7 / (rf_tp_7 + rf_fp_7)
        if rf_tp_8 + rf_fp_8 == 0:
            rf_precision_8 = 0
        else:
            rf_precision_8 = rf_tp_8 / (rf_tp_8 + rf_fp_8)
        if rf_tp_9 + rf_fp_9 == 0:
            rf_precision_9 = 0
        else:
            rf_precision_9 = rf_tp_9 / (rf_tp_9 + rf_fp_9)
        if rf_tp_10 + rf_fp_10 == 0:
            rf_precision_10 = 0
        else:
            rf_precision_10 = rf_tp_10 / (rf_tp_10 + rf_fp_10)
        if rf_tp_11 + rf_fp_11 == 0:
            rf_precision_11 = 0
        else:
            rf_precision_11 = rf_tp_11 / (rf_tp_11 + rf_fp_11)
        if rf_tp_12 + rf_fp_12 == 0:
            rf_precision_12 = 0
        else:
            rf_precision_12 = rf_tp_12 / (rf_tp_12 + rf_fp_12)
        if rf_tp_13 + rf_fp_13 == 0:
            rf_precision_13 = 0
        else:
            rf_precision_13 = rf_tp_13 / (rf_tp_13 + rf_fp_13)
        if rf_tp_14 + rf_fp_14 == 0:
            rf_precision_14 = 0
        else:
            rf_precision_14 = rf_tp_14 / (rf_tp_14 + rf_fp_14)
        if rf_tp_15 + rf_fp_15 == 0:
            rf_precision_15 = 0
        else:
            rf_precision_15 = rf_tp_15 / (rf_tp_15 + rf_fp_15)
        if rf_tp_16 + rf_fp_16 == 0:
            rf_precision_16 = 0
        else:
            rf_precision_16 = rf_tp_16 / (rf_tp_16 + rf_fp_16)
        if rf_tp_17 + rf_fp_17 == 0:
            rf_precision_17 = 0
        else:
            rf_precision_17 = rf_tp_17 / (rf_tp_17 + rf_fp_17)
        if rf_tp_18 + rf_fp_18 == 0:
            rf_precision_18 = 0
        else:
            rf_precision_18 = rf_tp_18 / (rf_tp_18 + rf_fp_18)
        if rf_tp_19 + rf_fp_19 == 0:
            rf_precision_19 = 0
        else:
            rf_precision_19 = rf_tp_19 / (rf_tp_19 + rf_fp_19)
        if rf_tp_20 + rf_fp_20 == 0:
            rf_precision_20 = 0
        else:
            rf_precision_20 = rf_tp_20 / (rf_tp_20 + rf_fp_20)
        '''
        if rf_tp_21 + rf_fp_21 == 0:
            rf_precision_21 = 0
        else:
            rf_precision_21 = rf_tp_21 / (rf_tp_21 + rf_fp_21)
        if rf_tp_22 + rf_fp_22 == 0:
            rf_precision_22 = 0
        else:
            rf_precision_22 = rf_tp_22 / (rf_tp_22 + rf_fp_22)
        if rf_tp_23 + rf_fp_23 == 0:
            rf_precision_23 = 0
        else:
            rf_precision_23 = rf_tp_23 / (rf_tp_23 + rf_fp_23)
        if rf_tp_24 + rf_fp_24 == 0:
            rf_precision_24 = 0
        else:
            rf_precision_24 = rf_tp_24 / (rf_tp_24 + rf_fp_24)
        if rf_tp_25 + rf_fp_25 == 0:
            rf_precision_25 = 0
        else:
            rf_precision_25 = rf_tp_25 / (rf_tp_25 + rf_fp_25)
        if rf_tp_26 + rf_fp_26 == 0:
            rf_precision_26 = 0
        else:
            rf_precision_26 = rf_tp_26 / (rf_tp_26 + rf_fp_26)
        if rf_tp_27 + rf_fp_27 == 0:
            rf_precision_27 = 0
        else:
            rf_precision_27 = rf_tp_27 / (rf_tp_27 + rf_fp_27)
        if rf_tp_28 + rf_fp_28 == 0:
            rf_precision_28 = 0
        else:
            rf_precision_28 = rf_tp_28 / (rf_tp_28 + rf_fp_28)
        if rf_tp_29 + rf_fp_29 == 0:
            rf_precision_29 = 0
        else:
            rf_precision_29 = rf_tp_29 / (rf_tp_29 + rf_fp_29)
        '''
        rf_precision_avg = (rf_precision_1 + rf_precision_2 + rf_precision_3 + rf_precision_4 + rf_precision_5 + rf_precision_6 + rf_precision_7 + rf_precision_8 + rf_precision_9 + rf_precision_10 + rf_precision_11 + rf_precision_12 + rf_precision_13 + rf_precision_14 + rf_precision_15 + rf_precision_16 + rf_precision_17 + rf_precision_18 + rf_precision_19 + rf_precision_20) / 20
        return rf_precision_avg


    def get_recall_pen_1(rf_conf_matrix):
        rf_tp_1 = rf_conf_matrix[0][0]
        rf_tp_2 = rf_conf_matrix[1][1]
        rf_tp_3 = rf_conf_matrix[2][2]
        rf_tp_4 = rf_conf_matrix[3][3]
        rf_tp_5 = rf_conf_matrix[4][4]
        rf_tp_6 = rf_conf_matrix[5][5]
        rf_tp_7 = rf_conf_matrix[6][6]
        rf_tp_8 = rf_conf_matrix[7][7]
        rf_tp_9 = rf_conf_matrix[8][8]
        rf_tp_10 = rf_conf_matrix[9][9]
        rf_tp_11 = rf_conf_matrix[10][10]
        rf_tp_12 = rf_conf_matrix[11][11]
        rf_tp_13 = rf_conf_matrix[12][12]
        rf_tp_14 = rf_conf_matrix[13][13]
        rf_tp_15 = rf_conf_matrix[14][14]
        rf_tp_16 = rf_conf_matrix[15][15]
        rf_tp_17 = rf_conf_matrix[16][16]
        rf_tp_18 = rf_conf_matrix[17][17]
        rf_tp_19 = rf_conf_matrix[18][18]
        rf_tp_20 = rf_conf_matrix[19][19]
        #rf_tp_21 = rf_conf_matrix[20][20]
        #rf_tp_22 = rf_conf_matrix[21][21]
        #rf_tp_23 = rf_conf_matrix[22][22]
        #rf_tp_24 = rf_conf_matrix[23][23]
        #rf_tp_25 = rf_conf_matrix[24][24]
        #rf_tp_26 = rf_conf_matrix[25][25]
        #rf_tp_27 = rf_conf_matrix[26][26]
        #rf_tp_28 = rf_conf_matrix[27][27]
        #rf_tp_29 = rf_conf_matrix[28][28]

        rf_fn_1 = rf_conf_matrix[0][1] + rf_conf_matrix[0][2] + rf_conf_matrix[0][3] + rf_conf_matrix[0][4] + rf_conf_matrix[0][5] + rf_conf_matrix[0][6] + rf_conf_matrix[0][7] + rf_conf_matrix[0][8] + rf_conf_matrix[0][9] + rf_conf_matrix[0][10] + rf_conf_matrix[0][11] + rf_conf_matrix[0][12] + rf_conf_matrix[0][13] + rf_conf_matrix[0][14] + rf_conf_matrix[0][15] + rf_conf_matrix[0][16] + rf_conf_matrix[0][17] + rf_conf_matrix[0][18] + rf_conf_matrix[0][19]
        rf_fn_2 = rf_conf_matrix[1][0] + rf_conf_matrix[1][2] + rf_conf_matrix[1][3] + rf_conf_matrix[1][4] + rf_conf_matrix[1][5] + rf_conf_matrix[1][6] + rf_conf_matrix[1][7] + rf_conf_matrix[1][8] + rf_conf_matrix[1][9] + rf_conf_matrix[1][10] + rf_conf_matrix[1][11] + rf_conf_matrix[1][12] + rf_conf_matrix[1][13] + rf_conf_matrix[1][14] + rf_conf_matrix[1][15] + rf_conf_matrix[1][16] + rf_conf_matrix[1][17] + rf_conf_matrix[1][18] + rf_conf_matrix[1][19]
        rf_fn_3 = rf_conf_matrix[2][0] + rf_conf_matrix[2][1] + rf_conf_matrix[2][3] + rf_conf_matrix[2][4] + rf_conf_matrix[2][5] + rf_conf_matrix[2][6] + rf_conf_matrix[2][7] + rf_conf_matrix[2][8] + rf_conf_matrix[2][9] + rf_conf_matrix[2][10] + rf_conf_matrix[2][11] + rf_conf_matrix[2][12] + rf_conf_matrix[2][13] + rf_conf_matrix[2][14] + rf_conf_matrix[2][15] + rf_conf_matrix[2][16] + rf_conf_matrix[2][17] + rf_conf_matrix[2][18] + rf_conf_matrix[2][19]
        rf_fn_4 = rf_conf_matrix[3][0] + rf_conf_matrix[3][1] + rf_conf_matrix[3][2] + rf_conf_matrix[3][4] + rf_conf_matrix[3][5] + rf_conf_matrix[3][6] + rf_conf_matrix[3][7] + rf_conf_matrix[3][8] + rf_conf_matrix[3][9] + rf_conf_matrix[3][10] + rf_conf_matrix[3][11] + rf_conf_matrix[3][12] + rf_conf_matrix[3][13] + rf_conf_matrix[3][14] + rf_conf_matrix[3][15] + rf_conf_matrix[3][16] + rf_conf_matrix[3][17] + rf_conf_matrix[3][18] + rf_conf_matrix[3][19]
        rf_fn_5 = rf_conf_matrix[4][0] + rf_conf_matrix[4][1] + rf_conf_matrix[4][2] + rf_conf_matrix[4][3] + rf_conf_matrix[4][5] + rf_conf_matrix[4][6] + rf_conf_matrix[4][7] + rf_conf_matrix[4][8] + rf_conf_matrix[4][9] + rf_conf_matrix[4][10] + rf_conf_matrix[4][11] + rf_conf_matrix[4][12] + rf_conf_matrix[4][13] + rf_conf_matrix[4][14] + rf_conf_matrix[4][15] + rf_conf_matrix[4][16] + rf_conf_matrix[4][17] + rf_conf_matrix[4][18] + rf_conf_matrix[4][19]
        rf_fn_6 = rf_conf_matrix[5][0] + rf_conf_matrix[5][1] + rf_conf_matrix[5][2] + rf_conf_matrix[5][3] + rf_conf_matrix[5][4] + rf_conf_matrix[5][6] + rf_conf_matrix[5][7] + rf_conf_matrix[5][8] + rf_conf_matrix[5][9] + rf_conf_matrix[5][10] + rf_conf_matrix[5][11] + rf_conf_matrix[5][12] + rf_conf_matrix[5][13] + rf_conf_matrix[5][14] + rf_conf_matrix[5][15] + rf_conf_matrix[5][16] + rf_conf_matrix[5][17] + rf_conf_matrix[5][18] + rf_conf_matrix[5][19]
        rf_fn_7 = rf_conf_matrix[6][0] + rf_conf_matrix[6][1] + rf_conf_matrix[6][2] + rf_conf_matrix[6][3] + rf_conf_matrix[6][4] + rf_conf_matrix[6][5] + rf_conf_matrix[6][7] + rf_conf_matrix[6][8] + rf_conf_matrix[6][9] + rf_conf_matrix[6][10] + rf_conf_matrix[6][11] + rf_conf_matrix[6][12] + rf_conf_matrix[6][13] + rf_conf_matrix[6][14] + rf_conf_matrix[6][15] + rf_conf_matrix[6][16] + rf_conf_matrix[6][17] + rf_conf_matrix[6][18] + rf_conf_matrix[6][19]
        rf_fn_8 = rf_conf_matrix[7][0] + rf_conf_matrix[7][1] + rf_conf_matrix[7][2] + rf_conf_matrix[7][3] + rf_conf_matrix[7][4] + rf_conf_matrix[7][5] + rf_conf_matrix[7][6] + rf_conf_matrix[7][8] + rf_conf_matrix[7][9] + rf_conf_matrix[7][10] + rf_conf_matrix[7][11] + rf_conf_matrix[7][12] + rf_conf_matrix[7][13] + rf_conf_matrix[7][14] + rf_conf_matrix[7][15] + rf_conf_matrix[7][16] + rf_conf_matrix[7][17] + rf_conf_matrix[7][18] + rf_conf_matrix[7][19]
        rf_fn_9 = rf_conf_matrix[8][0] + rf_conf_matrix[8][1] + rf_conf_matrix[8][2] + rf_conf_matrix[8][3] + rf_conf_matrix[8][4] + rf_conf_matrix[8][5] + rf_conf_matrix[8][6] + rf_conf_matrix[8][7] + rf_conf_matrix[8][9] + rf_conf_matrix[8][10] + rf_conf_matrix[8][11] + rf_conf_matrix[8][12] + rf_conf_matrix[8][13] + rf_conf_matrix[8][14] + rf_conf_matrix[8][15] + rf_conf_matrix[8][16] + rf_conf_matrix[8][17] + rf_conf_matrix[8][18] + rf_conf_matrix[8][19]
        rf_fn_10 = rf_conf_matrix[9][0] + rf_conf_matrix[9][1] + rf_conf_matrix[9][2] + rf_conf_matrix[9][3] + rf_conf_matrix[9][4] + rf_conf_matrix[9][5] + rf_conf_matrix[9][6] + rf_conf_matrix[9][7] + rf_conf_matrix[9][8] + rf_conf_matrix[9][10] + rf_conf_matrix[9][11] + rf_conf_matrix[9][12] + rf_conf_matrix[9][13] + rf_conf_matrix[9][14] + rf_conf_matrix[9][15] + rf_conf_matrix[9][16] + rf_conf_matrix[9][17] + rf_conf_matrix[9][18] + rf_conf_matrix[9][19]
        rf_fn_11 = rf_conf_matrix[10][0] + rf_conf_matrix[10][1] + rf_conf_matrix[10][2] + rf_conf_matrix[10][3] + rf_conf_matrix[10][4] + rf_conf_matrix[10][5] + rf_conf_matrix[10][6] + rf_conf_matrix[10][7] + rf_conf_matrix[10][8] + rf_conf_matrix[10][9] + rf_conf_matrix[10][11] + rf_conf_matrix[10][12] + rf_conf_matrix[10][13] + rf_conf_matrix[10][14] + rf_conf_matrix[10][15] + rf_conf_matrix[10][16] + rf_conf_matrix[10][17] + rf_conf_matrix[10][18] + rf_conf_matrix[10][19]
        rf_fn_12 = rf_conf_matrix[11][0] + rf_conf_matrix[11][1] + rf_conf_matrix[11][2] + rf_conf_matrix[11][3] + rf_conf_matrix[11][4] + rf_conf_matrix[11][5] + rf_conf_matrix[11][6] + rf_conf_matrix[11][7] + rf_conf_matrix[11][8] + rf_conf_matrix[11][9] + rf_conf_matrix[11][10] + rf_conf_matrix[11][12] + rf_conf_matrix[11][13] + rf_conf_matrix[11][14] + rf_conf_matrix[11][15] + rf_conf_matrix[11][16] + rf_conf_matrix[11][17] + rf_conf_matrix[11][18] + rf_conf_matrix[11][19]
        rf_fn_13 = rf_conf_matrix[12][0] + rf_conf_matrix[12][1] + rf_conf_matrix[12][2] + rf_conf_matrix[12][3] + rf_conf_matrix[12][4] + rf_conf_matrix[12][5] + rf_conf_matrix[12][6] + rf_conf_matrix[12][7] + rf_conf_matrix[12][8] + rf_conf_matrix[12][9] + rf_conf_matrix[12][10] + rf_conf_matrix[12][11] + rf_conf_matrix[12][13] + rf_conf_matrix[12][14] + rf_conf_matrix[12][15] + rf_conf_matrix[12][16] + rf_conf_matrix[12][17] + rf_conf_matrix[12][18] + rf_conf_matrix[12][19]
        rf_fn_14 = rf_conf_matrix[13][0] + rf_conf_matrix[13][1] + rf_conf_matrix[13][2] + rf_conf_matrix[13][3] + rf_conf_matrix[13][4] + rf_conf_matrix[13][5] + rf_conf_matrix[13][6] + rf_conf_matrix[13][7] + rf_conf_matrix[13][8] + rf_conf_matrix[13][9] + rf_conf_matrix[13][10] + rf_conf_matrix[13][11] + rf_conf_matrix[13][12] + rf_conf_matrix[13][14] + rf_conf_matrix[13][15] + rf_conf_matrix[13][16] + rf_conf_matrix[13][17] + rf_conf_matrix[13][18] + rf_conf_matrix[13][19]
        rf_fn_15 = rf_conf_matrix[14][0] + rf_conf_matrix[14][1] + rf_conf_matrix[14][2] + rf_conf_matrix[14][3] + rf_conf_matrix[14][4] + rf_conf_matrix[14][5] + rf_conf_matrix[14][6] + rf_conf_matrix[14][7] + rf_conf_matrix[14][8] + rf_conf_matrix[14][9] + rf_conf_matrix[14][10] + rf_conf_matrix[14][11] + rf_conf_matrix[14][12] + rf_conf_matrix[14][13] + rf_conf_matrix[14][15] + rf_conf_matrix[14][16] + rf_conf_matrix[14][17] + rf_conf_matrix[14][18] + rf_conf_matrix[14][19]
        rf_fn_16 = rf_conf_matrix[15][0] + rf_conf_matrix[15][1] + rf_conf_matrix[15][2] + rf_conf_matrix[15][3] + rf_conf_matrix[15][4] + rf_conf_matrix[15][5] + rf_conf_matrix[15][6] + rf_conf_matrix[15][7] + rf_conf_matrix[15][8] + rf_conf_matrix[15][9] + rf_conf_matrix[15][10] + rf_conf_matrix[15][11] + rf_conf_matrix[15][12] + rf_conf_matrix[15][13] + rf_conf_matrix[15][14] + rf_conf_matrix[15][16] + rf_conf_matrix[15][17] + rf_conf_matrix[15][18] + rf_conf_matrix[15][19]
        rf_fn_17 = rf_conf_matrix[16][0] + rf_conf_matrix[16][1] + rf_conf_matrix[16][2] + rf_conf_matrix[16][3] + rf_conf_matrix[16][4] + rf_conf_matrix[16][5] + rf_conf_matrix[16][6] + rf_conf_matrix[16][7] + rf_conf_matrix[16][8] + rf_conf_matrix[16][9] + rf_conf_matrix[16][10] + rf_conf_matrix[16][11] + rf_conf_matrix[16][12] + rf_conf_matrix[16][13] + rf_conf_matrix[16][14] + rf_conf_matrix[16][15] + rf_conf_matrix[16][17] + rf_conf_matrix[16][18] + rf_conf_matrix[16][19]
        rf_fn_18 = rf_conf_matrix[17][0] + rf_conf_matrix[17][1] + rf_conf_matrix[17][2] + rf_conf_matrix[17][3] + rf_conf_matrix[17][4] + rf_conf_matrix[17][5] + rf_conf_matrix[17][6] + rf_conf_matrix[17][7] + rf_conf_matrix[17][8] + rf_conf_matrix[17][9] + rf_conf_matrix[17][10] + rf_conf_matrix[17][11] + rf_conf_matrix[17][12] + rf_conf_matrix[17][13] + rf_conf_matrix[17][14] + rf_conf_matrix[17][15] + rf_conf_matrix[17][16] + rf_conf_matrix[17][18] + rf_conf_matrix[17][19]
        rf_fn_19 = rf_conf_matrix[18][0] + rf_conf_matrix[18][1] + rf_conf_matrix[18][2] + rf_conf_matrix[18][3] + rf_conf_matrix[18][4] + rf_conf_matrix[18][5] + rf_conf_matrix[18][6] + rf_conf_matrix[18][7] + rf_conf_matrix[18][8] + rf_conf_matrix[18][9] + rf_conf_matrix[18][10] + rf_conf_matrix[18][11] + rf_conf_matrix[18][12] + rf_conf_matrix[18][13] + rf_conf_matrix[18][14] + rf_conf_matrix[18][15] + rf_conf_matrix[18][16] + rf_conf_matrix[18][17] + rf_conf_matrix[18][19]
        rf_fn_20 = rf_conf_matrix[19][0] + rf_conf_matrix[19][1] + rf_conf_matrix[19][2] + rf_conf_matrix[19][3] + rf_conf_matrix[19][4] + rf_conf_matrix[19][5] + rf_conf_matrix[19][6] + rf_conf_matrix[19][7] + rf_conf_matrix[19][8] + rf_conf_matrix[19][9] + rf_conf_matrix[19][10] + rf_conf_matrix[19][11] + rf_conf_matrix[19][12] + rf_conf_matrix[19][13] + rf_conf_matrix[19][14] + rf_conf_matrix[19][15] + rf_conf_matrix[19][16] + rf_conf_matrix[19][17] + rf_conf_matrix[19][18]
        #rf_fn_21 = rf_conf_matrix[20][0] + rf_conf_matrix[20][1] + rf_conf_matrix[20][2] + rf_conf_matrix[20][3] + rf_conf_matrix[20][4] + rf_conf_matrix[20][5] + rf_conf_matrix[20][6] + rf_conf_matrix[20][7] + rf_conf_matrix[20][8] + rf_conf_matrix[20][9] + rf_conf_matrix[20][10] + rf_conf_matrix[20][11] + rf_conf_matrix[20][12] + rf_conf_matrix[20][13] + rf_conf_matrix[20][14] + rf_conf_matrix[20][15] + rf_conf_matrix[20][16] + rf_conf_matrix[20][17] + rf_conf_matrix[20][18] + rf_conf_matrix[20][19] + rf_conf_matrix[20][21] + rf_conf_matrix[20][22] + rf_conf_matrix[20][23] + rf_conf_matrix[20][24] + rf_conf_matrix[20][25] + rf_conf_matrix[20][26] + rf_conf_matrix[20][27] + rf_conf_matrix[20][28]
        #rf_fn_22 = rf_conf_matrix[21][0] + rf_conf_matrix[21][1] + rf_conf_matrix[21][2] + rf_conf_matrix[21][3] + rf_conf_matrix[21][4] + rf_conf_matrix[21][5] + rf_conf_matrix[21][6] + rf_conf_matrix[21][7] + rf_conf_matrix[21][8] + rf_conf_matrix[21][9] + rf_conf_matrix[21][10] + rf_conf_matrix[21][11] + rf_conf_matrix[21][12] + rf_conf_matrix[21][13] + rf_conf_matrix[21][14] + rf_conf_matrix[21][15] + rf_conf_matrix[21][16] + rf_conf_matrix[21][17] + rf_conf_matrix[21][18] + rf_conf_matrix[21][19] + rf_conf_matrix[21][20] + rf_conf_matrix[21][22] + rf_conf_matrix[21][23] + rf_conf_matrix[21][24] + rf_conf_matrix[21][25] + rf_conf_matrix[21][26] + rf_conf_matrix[21][27] + rf_conf_matrix[21][28]
        #rf_fn_23 = rf_conf_matrix[22][0] + rf_conf_matrix[22][1] + rf_conf_matrix[22][2] + rf_conf_matrix[22][3] + rf_conf_matrix[22][4] + rf_conf_matrix[22][5] + rf_conf_matrix[22][6] + rf_conf_matrix[22][7] + rf_conf_matrix[22][8] + rf_conf_matrix[22][9] + rf_conf_matrix[22][10] + rf_conf_matrix[22][11] + rf_conf_matrix[22][12] + rf_conf_matrix[22][13] + rf_conf_matrix[22][14] + rf_conf_matrix[22][15] + rf_conf_matrix[22][16] + rf_conf_matrix[22][17] + rf_conf_matrix[22][18] + rf_conf_matrix[22][19] + rf_conf_matrix[22][20] + rf_conf_matrix[22][21] + rf_conf_matrix[22][23] + rf_conf_matrix[22][24] + rf_conf_matrix[22][25] + rf_conf_matrix[22][26] + rf_conf_matrix[22][27] + rf_conf_matrix[22][28]
        #rf_fn_24 = rf_conf_matrix[23][0] + rf_conf_matrix[23][1] + rf_conf_matrix[23][2] + rf_conf_matrix[23][3] + rf_conf_matrix[23][4] + rf_conf_matrix[23][5] + rf_conf_matrix[23][6] + rf_conf_matrix[23][7] + rf_conf_matrix[23][8] + rf_conf_matrix[23][9] + rf_conf_matrix[23][10] + rf_conf_matrix[23][11] + rf_conf_matrix[23][12] + rf_conf_matrix[23][13] + rf_conf_matrix[23][14] + rf_conf_matrix[23][15] + rf_conf_matrix[23][16] + rf_conf_matrix[23][17] + rf_conf_matrix[23][18] + rf_conf_matrix[23][19] + rf_conf_matrix[23][20] + rf_conf_matrix[23][21] + rf_conf_matrix[23][22] + rf_conf_matrix[23][24] + rf_conf_matrix[23][25] + rf_conf_matrix[23][26] + rf_conf_matrix[23][27] + rf_conf_matrix[23][28]
        #rf_fn_25 = rf_conf_matrix[24][0] + rf_conf_matrix[24][1] + rf_conf_matrix[24][2] + rf_conf_matrix[24][3] + rf_conf_matrix[24][4] + rf_conf_matrix[24][5] + rf_conf_matrix[24][6] + rf_conf_matrix[24][7] + rf_conf_matrix[24][8] + rf_conf_matrix[24][9] + rf_conf_matrix[24][10] + rf_conf_matrix[24][11] + rf_conf_matrix[24][12] + rf_conf_matrix[24][13] + rf_conf_matrix[24][14] + rf_conf_matrix[24][15] + rf_conf_matrix[24][16] + rf_conf_matrix[24][17] + rf_conf_matrix[24][18] + rf_conf_matrix[24][19] + rf_conf_matrix[24][20] + rf_conf_matrix[24][21] + rf_conf_matrix[24][22] + rf_conf_matrix[24][23] + rf_conf_matrix[24][25] + rf_conf_matrix[24][26] + rf_conf_matrix[24][27] + rf_conf_matrix[24][28]
        #rf_fn_26 = rf_conf_matrix[25][0] + rf_conf_matrix[25][1] + rf_conf_matrix[25][2] + rf_conf_matrix[25][3] + rf_conf_matrix[25][4] + rf_conf_matrix[25][5] + rf_conf_matrix[25][6] + rf_conf_matrix[25][7] + rf_conf_matrix[25][8] + rf_conf_matrix[25][9] + rf_conf_matrix[25][10] + rf_conf_matrix[25][11] + rf_conf_matrix[25][12] + rf_conf_matrix[25][13] + rf_conf_matrix[25][14] + rf_conf_matrix[25][15] + rf_conf_matrix[25][16] + rf_conf_matrix[25][17] + rf_conf_matrix[25][18] + rf_conf_matrix[25][19] + rf_conf_matrix[25][20] + rf_conf_matrix[25][21] + rf_conf_matrix[25][22] + rf_conf_matrix[25][23] + rf_conf_matrix[25][24] + rf_conf_matrix[25][26] + rf_conf_matrix[25][27] + rf_conf_matrix[25][28]
        #rf_fn_27 = rf_conf_matrix[26][0] + rf_conf_matrix[26][1] + rf_conf_matrix[26][2] + rf_conf_matrix[26][3] + rf_conf_matrix[26][4] + rf_conf_matrix[26][5] + rf_conf_matrix[26][6] + rf_conf_matrix[26][7] + rf_conf_matrix[26][8] + rf_conf_matrix[26][9] + rf_conf_matrix[26][10] + rf_conf_matrix[26][11] + rf_conf_matrix[26][12] + rf_conf_matrix[26][13] + rf_conf_matrix[26][14] + rf_conf_matrix[26][15] + rf_conf_matrix[26][16] + rf_conf_matrix[26][17] + rf_conf_matrix[26][18] + rf_conf_matrix[26][19] + rf_conf_matrix[26][20] + rf_conf_matrix[26][21] + rf_conf_matrix[26][22] + rf_conf_matrix[26][23] + rf_conf_matrix[26][24] + rf_conf_matrix[26][25] + rf_conf_matrix[26][27] + rf_conf_matrix[26][28]
        #rf_fn_28 = rf_conf_matrix[27][0] + rf_conf_matrix[27][1] + rf_conf_matrix[27][2] + rf_conf_matrix[27][3] + rf_conf_matrix[27][4] + rf_conf_matrix[27][5] + rf_conf_matrix[27][6] + rf_conf_matrix[27][7] + rf_conf_matrix[27][8] + rf_conf_matrix[27][9] + rf_conf_matrix[27][10] + rf_conf_matrix[27][11] + rf_conf_matrix[27][12] + rf_conf_matrix[27][13] + rf_conf_matrix[27][14] + rf_conf_matrix[27][15] + rf_conf_matrix[27][16] + rf_conf_matrix[27][17] + rf_conf_matrix[27][18] + rf_conf_matrix[27][19] + rf_conf_matrix[27][20] + rf_conf_matrix[27][21] + rf_conf_matrix[27][22] + rf_conf_matrix[27][23] + rf_conf_matrix[27][24] + rf_conf_matrix[27][25] + rf_conf_matrix[27][26] + rf_conf_matrix[27][28]
        #rf_fn_29 = rf_conf_matrix[28][0] + rf_conf_matrix[28][1] + rf_conf_matrix[28][2] + rf_conf_matrix[28][3] + rf_conf_matrix[28][4] + rf_conf_matrix[28][5] + rf_conf_matrix[28][6] + rf_conf_matrix[28][7] + rf_conf_matrix[28][8] + rf_conf_matrix[28][9] + rf_conf_matrix[28][10] + rf_conf_matrix[28][11] + rf_conf_matrix[28][12] + rf_conf_matrix[28][13] + rf_conf_matrix[28][14] + rf_conf_matrix[28][15] + rf_conf_matrix[28][16] + rf_conf_matrix[28][17] + rf_conf_matrix[28][18] + rf_conf_matrix[28][19] + rf_conf_matrix[28][20] + rf_conf_matrix[28][21] + rf_conf_matrix[28][22] + rf_conf_matrix[28][23] + rf_conf_matrix[28][24] + rf_conf_matrix[28][25] + rf_conf_matrix[28][26] + rf_conf_matrix[28][27]

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
            rf_recall_5 = rf_tp_5 / (rf_tp_5 + rf_fn_5)
        if rf_tp_6 + rf_fn_6 == 0:
            rf_recall_6 = 0
        else:
            rf_recall_6 = rf_tp_6 / (rf_tp_6 + rf_fn_6)
        if rf_tp_7 + rf_fn_7 == 0:
            rf_recall_7 = 0
        else:
            rf_recall_7 = rf_tp_7 / (rf_tp_7 + rf_fn_7)
        if rf_tp_8 + rf_fn_8 == 0:
            rf_recall_8 = 0
        else:
            rf_recall_8 = rf_tp_8 / (rf_tp_8 + rf_fn_8)
        if rf_tp_9 + rf_fn_9 == 0:
            rf_recall_9 = 0
        else:
            rf_recall_9 = rf_tp_9 / (rf_tp_9 + rf_fn_9)
        if rf_tp_10 + rf_fn_10 == 0:
            rf_recall_10 = 0
        else:
            rf_recall_10 = rf_tp_10 / (rf_tp_10 + rf_fn_10)
        if rf_tp_11 + rf_fn_11 == 0:
            rf_recall_11 = 0
        else:
            rf_recall_11 = rf_tp_11 / (rf_tp_11 + rf_fn_11)
        if rf_tp_12 + rf_fn_12 == 0:
            rf_recall_12 = 0
        else:
            rf_recall_12 = rf_tp_12 / (rf_tp_12 + rf_fn_12)
        if rf_tp_13 + rf_fn_13 == 0:
            rf_recall_13 = 0
        else:
            rf_recall_13 = rf_tp_13 / (rf_tp_13 + rf_fn_13)
        if rf_tp_14 + rf_fn_14 == 0:
            rf_recall_14 = 0
        else:
            rf_recall_14 = rf_tp_14 / (rf_tp_14 + rf_fn_14)
        if rf_tp_15 + rf_fn_15 == 0:
            rf_recall_15 = 0
        else:
            rf_recall_15 = rf_tp_15 / (rf_tp_15 + rf_fn_15)
        if rf_tp_16 + rf_fn_16 == 0:
            rf_recall_16 = 0
        else:
            rf_recall_16 = rf_tp_16 / (rf_tp_16 + rf_fn_16)
        if rf_tp_17 + rf_fn_17 == 0:
            rf_recall_17 = 0
        else:
            rf_recall_17 = rf_tp_17 / (rf_tp_17 + rf_fn_17)
        if rf_tp_18 + rf_fn_18 == 0:
            rf_recall_18 = 0
        else:
            rf_recall_18 = rf_tp_18 / (rf_tp_18 + rf_fn_18)
        if rf_tp_19 + rf_fn_19 == 0:
            rf_recall_19 = 0
        else:
            rf_recall_19 = rf_tp_19 / (rf_tp_19 + rf_fn_19)
        if rf_tp_20 + rf_fn_20 == 0:
            rf_recall_20 = 0
        else:
            rf_recall_20 = rf_tp_20 / (rf_tp_20 + rf_fn_20)
        '''
        if rf_tp_21 + rf_fn_21 == 0:
            rf_recall_21 = 0
        else:
            rf_recall_21 = rf_tp_21 / (rf_tp_21 + rf_fn_21)
        if rf_tp_22 + rf_fn_22 == 0:
            rf_recall_22 = 0
        else:
            rf_recall_22 = rf_tp_22 / (rf_tp_22 + rf_fn_22)
        if rf_tp_23 + rf_fn_23 == 0:
            rf_recall_23 = 0
        else:
            rf_recall_23 = rf_tp_23 / (rf_tp_23 + rf_fn_23)
        if rf_tp_24 + rf_fn_24 == 0:
            rf_recall_24 = 0
        else:
            rf_recall_24 = rf_tp_24 / (rf_tp_24 + rf_fn_24)
        if rf_tp_25 + rf_fn_25 == 0:
            rf_recall_25 = 0
        else:
            rf_recall_25 = rf_tp_25 / (rf_tp_25 + rf_fn_25)
        if rf_tp_26 + rf_fn_26 == 0:
            rf_recall_26 = 0
        else:
            rf_recall_26 = rf_tp_26 / (rf_tp_26 + rf_fn_26)
        if rf_tp_27 + rf_fn_27 == 0:
            rf_recall_27 = 0
        else:
            rf_recall_27 = rf_tp_27 / (rf_tp_27 + rf_fn_27)
        if rf_tp_28 + rf_fn_28 == 0:
            rf_recall_28 = 0
        else:
            rf_recall_28 = rf_tp_28 / (rf_tp_28 + rf_fn_28)
        if rf_tp_29 + rf_fn_29 == 0:
            rf_recall_29 = 0
        else:
            rf_recall_29 = rf_tp_29 / (rf_tp_29 + rf_fn_29)
        '''
        rf_recall_avg_pen_1 = (
                                 rf_recall_1 + rf_recall_2 + rf_recall_3 + rf_recall_4 + rf_recall_5 + rf_recall_6 + rf_recall_7 + rf_recall_8 + rf_recall_9 + rf_recall_10 + rf_recall_11 + rf_recall_12 + rf_recall_13 + rf_recall_14 + rf_recall_15 + rf_recall_16 + rf_recall_17 + rf_recall_18 + rf_recall_19 + rf_recall_20) / (20+1-1)
        return rf_recall_avg_pen_1

    def get_recall_pen_5(rf_conf_matrix):
        rf_tp_1 = rf_conf_matrix[0][0]
        rf_tp_2 = rf_conf_matrix[1][1]
        rf_tp_3 = rf_conf_matrix[2][2]
        rf_tp_4 = rf_conf_matrix[3][3]
        rf_tp_5 = rf_conf_matrix[4][4]
        rf_tp_6 = rf_conf_matrix[5][5]
        rf_tp_7 = rf_conf_matrix[6][6]
        rf_tp_8 = rf_conf_matrix[7][7]
        rf_tp_9 = rf_conf_matrix[8][8]
        rf_tp_10 = rf_conf_matrix[9][9]
        rf_tp_11 = rf_conf_matrix[10][10]
        rf_tp_12 = rf_conf_matrix[11][11]
        rf_tp_13 = rf_conf_matrix[12][12]
        rf_tp_14 = rf_conf_matrix[13][13]
        rf_tp_15 = rf_conf_matrix[14][14]
        rf_tp_16 = rf_conf_matrix[15][15]
        rf_tp_17 = rf_conf_matrix[16][16]
        rf_tp_18 = rf_conf_matrix[17][17]
        rf_tp_19 = rf_conf_matrix[18][18]
        rf_tp_20 = rf_conf_matrix[19][19]
        #rf_tp_21 = rf_conf_matrix[20][20]
        #rf_tp_22 = rf_conf_matrix[21][21]
        #rf_tp_23 = rf_conf_matrix[22][22]
        #rf_tp_24 = rf_conf_matrix[23][23]
        #rf_tp_25 = rf_conf_matrix[24][24]
        #rf_tp_26 = rf_conf_matrix[25][25]
        #rf_tp_27 = rf_conf_matrix[26][26]
        #rf_tp_28 = rf_conf_matrix[27][27]
        #rf_tp_29 = rf_conf_matrix[28][28]

        rf_fn_1 = rf_conf_matrix[0][1] + rf_conf_matrix[0][2] + rf_conf_matrix[0][3] + rf_conf_matrix[0][4] + rf_conf_matrix[0][5] + rf_conf_matrix[0][6] + rf_conf_matrix[0][7] + rf_conf_matrix[0][8] + rf_conf_matrix[0][9] + rf_conf_matrix[0][10] + rf_conf_matrix[0][11] + rf_conf_matrix[0][12] + rf_conf_matrix[0][13] + rf_conf_matrix[0][14] + rf_conf_matrix[0][15] + rf_conf_matrix[0][16] + rf_conf_matrix[0][17] + rf_conf_matrix[0][18] + rf_conf_matrix[0][19]
        rf_fn_2 = rf_conf_matrix[1][0] + rf_conf_matrix[1][2] + rf_conf_matrix[1][3] + rf_conf_matrix[1][4] + rf_conf_matrix[1][5] + rf_conf_matrix[1][6] + rf_conf_matrix[1][7] + rf_conf_matrix[1][8] + rf_conf_matrix[1][9] + rf_conf_matrix[1][10] + rf_conf_matrix[1][11] + rf_conf_matrix[1][12] + rf_conf_matrix[1][13] + rf_conf_matrix[1][14] + rf_conf_matrix[1][15] + rf_conf_matrix[1][16] + rf_conf_matrix[1][17] + rf_conf_matrix[1][18] + rf_conf_matrix[1][19]
        rf_fn_3 = rf_conf_matrix[2][0] + rf_conf_matrix[2][1] + rf_conf_matrix[2][3] + rf_conf_matrix[2][4] + rf_conf_matrix[2][5] + rf_conf_matrix[2][6] + rf_conf_matrix[2][7] + rf_conf_matrix[2][8] + rf_conf_matrix[2][9] + rf_conf_matrix[2][10] + rf_conf_matrix[2][11] + rf_conf_matrix[2][12] + rf_conf_matrix[2][13] + rf_conf_matrix[2][14] + rf_conf_matrix[2][15] + rf_conf_matrix[2][16] + rf_conf_matrix[2][17] + rf_conf_matrix[2][18] + rf_conf_matrix[2][19]
        rf_fn_4 = rf_conf_matrix[3][0] + rf_conf_matrix[3][1] + rf_conf_matrix[3][2] + rf_conf_matrix[3][4] + rf_conf_matrix[3][5] + rf_conf_matrix[3][6] + rf_conf_matrix[3][7] + rf_conf_matrix[3][8] + rf_conf_matrix[3][9] + rf_conf_matrix[3][10] + rf_conf_matrix[3][11] + rf_conf_matrix[3][12] + rf_conf_matrix[3][13] + rf_conf_matrix[3][14] + rf_conf_matrix[3][15] + rf_conf_matrix[3][16] + rf_conf_matrix[3][17] + rf_conf_matrix[3][18] + rf_conf_matrix[3][19]
        rf_fn_5 = rf_conf_matrix[4][0] + rf_conf_matrix[4][1] + rf_conf_matrix[4][2] + rf_conf_matrix[4][3] + rf_conf_matrix[4][5] + rf_conf_matrix[4][6] + rf_conf_matrix[4][7] + rf_conf_matrix[4][8] + rf_conf_matrix[4][9] + rf_conf_matrix[4][10] + rf_conf_matrix[4][11] + rf_conf_matrix[4][12] + rf_conf_matrix[4][13] + rf_conf_matrix[4][14] + rf_conf_matrix[4][15] + rf_conf_matrix[4][16] + rf_conf_matrix[4][17] + rf_conf_matrix[4][18] + rf_conf_matrix[4][19]
        rf_fn_6 = rf_conf_matrix[5][0] + rf_conf_matrix[5][1] + rf_conf_matrix[5][2] + rf_conf_matrix[5][3] + rf_conf_matrix[5][4] + rf_conf_matrix[5][6] + rf_conf_matrix[5][7] + rf_conf_matrix[5][8] + rf_conf_matrix[5][9] + rf_conf_matrix[5][10] + rf_conf_matrix[5][11] + rf_conf_matrix[5][12] + rf_conf_matrix[5][13] + rf_conf_matrix[5][14] + rf_conf_matrix[5][15] + rf_conf_matrix[5][16] + rf_conf_matrix[5][17] + rf_conf_matrix[5][18] + rf_conf_matrix[5][19]
        rf_fn_7 = rf_conf_matrix[6][0] + rf_conf_matrix[6][1] + rf_conf_matrix[6][2] + rf_conf_matrix[6][3] + rf_conf_matrix[6][4] + rf_conf_matrix[6][5] + rf_conf_matrix[6][7] + rf_conf_matrix[6][8] + rf_conf_matrix[6][9] + rf_conf_matrix[6][10] + rf_conf_matrix[6][11] + rf_conf_matrix[6][12] + rf_conf_matrix[6][13] + rf_conf_matrix[6][14] + rf_conf_matrix[6][15] + rf_conf_matrix[6][16] + rf_conf_matrix[6][17] + rf_conf_matrix[6][18] + rf_conf_matrix[6][19]
        rf_fn_8 = rf_conf_matrix[7][0] + rf_conf_matrix[7][1] + rf_conf_matrix[7][2] + rf_conf_matrix[7][3] + rf_conf_matrix[7][4] + rf_conf_matrix[7][5] + rf_conf_matrix[7][6] + rf_conf_matrix[7][8] + rf_conf_matrix[7][9] + rf_conf_matrix[7][10] + rf_conf_matrix[7][11] + rf_conf_matrix[7][12] + rf_conf_matrix[7][13] + rf_conf_matrix[7][14] + rf_conf_matrix[7][15] + rf_conf_matrix[7][16] + rf_conf_matrix[7][17] + rf_conf_matrix[7][18] + rf_conf_matrix[7][19]
        rf_fn_9 = rf_conf_matrix[8][0] + rf_conf_matrix[8][1] + rf_conf_matrix[8][2] + rf_conf_matrix[8][3] + rf_conf_matrix[8][4] + rf_conf_matrix[8][5] + rf_conf_matrix[8][6] + rf_conf_matrix[8][7] + rf_conf_matrix[8][9] + rf_conf_matrix[8][10] + rf_conf_matrix[8][11] + rf_conf_matrix[8][12] + rf_conf_matrix[8][13] + rf_conf_matrix[8][14] + rf_conf_matrix[8][15] + rf_conf_matrix[8][16] + rf_conf_matrix[8][17] + rf_conf_matrix[8][18] + rf_conf_matrix[8][19]
        rf_fn_10 = rf_conf_matrix[9][0] + rf_conf_matrix[9][1] + rf_conf_matrix[9][2] + rf_conf_matrix[9][3] + rf_conf_matrix[9][4] + rf_conf_matrix[9][5] + rf_conf_matrix[9][6] + rf_conf_matrix[9][7] + rf_conf_matrix[9][8] + rf_conf_matrix[9][10] + rf_conf_matrix[9][11] + rf_conf_matrix[9][12] + rf_conf_matrix[9][13] + rf_conf_matrix[9][14] + rf_conf_matrix[9][15] + rf_conf_matrix[9][16] + rf_conf_matrix[9][17] + rf_conf_matrix[9][18] + rf_conf_matrix[9][19]
        rf_fn_11 = rf_conf_matrix[10][0] + rf_conf_matrix[10][1] + rf_conf_matrix[10][2] + rf_conf_matrix[10][3] + rf_conf_matrix[10][4] + rf_conf_matrix[10][5] + rf_conf_matrix[10][6] + rf_conf_matrix[10][7] + rf_conf_matrix[10][8] + rf_conf_matrix[10][9] + rf_conf_matrix[10][11] + rf_conf_matrix[10][12] + rf_conf_matrix[10][13] + rf_conf_matrix[10][14] + rf_conf_matrix[10][15] + rf_conf_matrix[10][16] + rf_conf_matrix[10][17] + rf_conf_matrix[10][18] + rf_conf_matrix[10][19]
        rf_fn_12 = rf_conf_matrix[11][0] + rf_conf_matrix[11][1] + rf_conf_matrix[11][2] + rf_conf_matrix[11][3] + rf_conf_matrix[11][4] + rf_conf_matrix[11][5] + rf_conf_matrix[11][6] + rf_conf_matrix[11][7] + rf_conf_matrix[11][8] + rf_conf_matrix[11][9] + rf_conf_matrix[11][10] + rf_conf_matrix[11][12] + rf_conf_matrix[11][13] + rf_conf_matrix[11][14] + rf_conf_matrix[11][15] + rf_conf_matrix[11][16] + rf_conf_matrix[11][17] + rf_conf_matrix[11][18] + rf_conf_matrix[11][19]
        rf_fn_13 = rf_conf_matrix[12][0] + rf_conf_matrix[12][1] + rf_conf_matrix[12][2] + rf_conf_matrix[12][3] + rf_conf_matrix[12][4] + rf_conf_matrix[12][5] + rf_conf_matrix[12][6] + rf_conf_matrix[12][7] + rf_conf_matrix[12][8] + rf_conf_matrix[12][9] + rf_conf_matrix[12][10] + rf_conf_matrix[12][11] + rf_conf_matrix[12][13] + rf_conf_matrix[12][14] + rf_conf_matrix[12][15] + rf_conf_matrix[12][16] + rf_conf_matrix[12][17] + rf_conf_matrix[12][18] + rf_conf_matrix[12][19]
        rf_fn_14 = rf_conf_matrix[13][0] + rf_conf_matrix[13][1] + rf_conf_matrix[13][2] + rf_conf_matrix[13][3] + rf_conf_matrix[13][4] + rf_conf_matrix[13][5] + rf_conf_matrix[13][6] + rf_conf_matrix[13][7] + rf_conf_matrix[13][8] + rf_conf_matrix[13][9] + rf_conf_matrix[13][10] + rf_conf_matrix[13][11] + rf_conf_matrix[13][12] + rf_conf_matrix[13][14] + rf_conf_matrix[13][15] + rf_conf_matrix[13][16] + rf_conf_matrix[13][17] + rf_conf_matrix[13][18] + rf_conf_matrix[13][19]
        rf_fn_15 = rf_conf_matrix[14][0] + rf_conf_matrix[14][1] + rf_conf_matrix[14][2] + rf_conf_matrix[14][3] + rf_conf_matrix[14][4] + rf_conf_matrix[14][5] + rf_conf_matrix[14][6] + rf_conf_matrix[14][7] + rf_conf_matrix[14][8] + rf_conf_matrix[14][9] + rf_conf_matrix[14][10] + rf_conf_matrix[14][11] + rf_conf_matrix[14][12] + rf_conf_matrix[14][13] + rf_conf_matrix[14][15] + rf_conf_matrix[14][16] + rf_conf_matrix[14][17] + rf_conf_matrix[14][18] + rf_conf_matrix[14][19]
        rf_fn_16 = rf_conf_matrix[15][0] + rf_conf_matrix[15][1] + rf_conf_matrix[15][2] + rf_conf_matrix[15][3] + rf_conf_matrix[15][4] + rf_conf_matrix[15][5] + rf_conf_matrix[15][6] + rf_conf_matrix[15][7] + rf_conf_matrix[15][8] + rf_conf_matrix[15][9] + rf_conf_matrix[15][10] + rf_conf_matrix[15][11] + rf_conf_matrix[15][12] + rf_conf_matrix[15][13] + rf_conf_matrix[15][14] + rf_conf_matrix[15][16] + rf_conf_matrix[15][17] + rf_conf_matrix[15][18] + rf_conf_matrix[15][19]
        rf_fn_17 = rf_conf_matrix[16][0] + rf_conf_matrix[16][1] + rf_conf_matrix[16][2] + rf_conf_matrix[16][3] + rf_conf_matrix[16][4] + rf_conf_matrix[16][5] + rf_conf_matrix[16][6] + rf_conf_matrix[16][7] + rf_conf_matrix[16][8] + rf_conf_matrix[16][9] + rf_conf_matrix[16][10] + rf_conf_matrix[16][11] + rf_conf_matrix[16][12] + rf_conf_matrix[16][13] + rf_conf_matrix[16][14] + rf_conf_matrix[16][15] + rf_conf_matrix[16][17] + rf_conf_matrix[16][18] + rf_conf_matrix[16][19]
        rf_fn_18 = rf_conf_matrix[17][0] + rf_conf_matrix[17][1] + rf_conf_matrix[17][2] + rf_conf_matrix[17][3] + rf_conf_matrix[17][4] + rf_conf_matrix[17][5] + rf_conf_matrix[17][6] + rf_conf_matrix[17][7] + rf_conf_matrix[17][8] + rf_conf_matrix[17][9] + rf_conf_matrix[17][10] + rf_conf_matrix[17][11] + rf_conf_matrix[17][12] + rf_conf_matrix[17][13] + rf_conf_matrix[17][14] + rf_conf_matrix[17][15] + rf_conf_matrix[17][16] + rf_conf_matrix[17][18] + rf_conf_matrix[17][19]
        rf_fn_19 = rf_conf_matrix[18][0] + rf_conf_matrix[18][1] + rf_conf_matrix[18][2] + rf_conf_matrix[18][3] + rf_conf_matrix[18][4] + rf_conf_matrix[18][5] + rf_conf_matrix[18][6] + rf_conf_matrix[18][7] + rf_conf_matrix[18][8] + rf_conf_matrix[18][9] + rf_conf_matrix[18][10] + rf_conf_matrix[18][11] + rf_conf_matrix[18][12] + rf_conf_matrix[18][13] + rf_conf_matrix[18][14] + rf_conf_matrix[18][15] + rf_conf_matrix[18][16] + rf_conf_matrix[18][17] + rf_conf_matrix[18][19]
        rf_fn_20 = rf_conf_matrix[19][0] + rf_conf_matrix[19][1] + rf_conf_matrix[19][2] + rf_conf_matrix[19][3] + rf_conf_matrix[19][4] + rf_conf_matrix[19][5] + rf_conf_matrix[19][6] + rf_conf_matrix[19][7] + rf_conf_matrix[19][8] + rf_conf_matrix[19][9] + rf_conf_matrix[19][10] + rf_conf_matrix[19][11] + rf_conf_matrix[19][12] + rf_conf_matrix[19][13] + rf_conf_matrix[19][14] + rf_conf_matrix[19][15] + rf_conf_matrix[19][16] + rf_conf_matrix[19][17] + rf_conf_matrix[19][18]
        #rf_fn_21 = rf_conf_matrix[20][0] + rf_conf_matrix[20][1] + rf_conf_matrix[20][2] + rf_conf_matrix[20][3] + rf_conf_matrix[20][4] + rf_conf_matrix[20][5] + rf_conf_matrix[20][6] + rf_conf_matrix[20][7] + rf_conf_matrix[20][8] + rf_conf_matrix[20][9] + rf_conf_matrix[20][10] + rf_conf_matrix[20][11] + rf_conf_matrix[20][12] + rf_conf_matrix[20][13] + rf_conf_matrix[20][14] + rf_conf_matrix[20][15] + rf_conf_matrix[20][16] + rf_conf_matrix[20][17] + rf_conf_matrix[20][18] + rf_conf_matrix[20][19] + rf_conf_matrix[20][21] + rf_conf_matrix[20][22] + rf_conf_matrix[20][23] + rf_conf_matrix[20][24] + rf_conf_matrix[20][25] + rf_conf_matrix[20][26] + rf_conf_matrix[20][27] + rf_conf_matrix[20][28]
        #rf_fn_22 = rf_conf_matrix[21][0] + rf_conf_matrix[21][1] + rf_conf_matrix[21][2] + rf_conf_matrix[21][3] + rf_conf_matrix[21][4] + rf_conf_matrix[21][5] + rf_conf_matrix[21][6] + rf_conf_matrix[21][7] + rf_conf_matrix[21][8] + rf_conf_matrix[21][9] + rf_conf_matrix[21][10] + rf_conf_matrix[21][11] + rf_conf_matrix[21][12] + rf_conf_matrix[21][13] + rf_conf_matrix[21][14] + rf_conf_matrix[21][15] + rf_conf_matrix[21][16] + rf_conf_matrix[21][17] + rf_conf_matrix[21][18] + rf_conf_matrix[21][19] + rf_conf_matrix[21][20] + rf_conf_matrix[21][22] + rf_conf_matrix[21][23] + rf_conf_matrix[21][24] + rf_conf_matrix[21][25] + rf_conf_matrix[21][26] + rf_conf_matrix[21][27] + rf_conf_matrix[21][28]
        #rf_fn_23 = rf_conf_matrix[22][0] + rf_conf_matrix[22][1] + rf_conf_matrix[22][2] + rf_conf_matrix[22][3] + rf_conf_matrix[22][4] + rf_conf_matrix[22][5] + rf_conf_matrix[22][6] + rf_conf_matrix[22][7] + rf_conf_matrix[22][8] + rf_conf_matrix[22][9] + rf_conf_matrix[22][10] + rf_conf_matrix[22][11] + rf_conf_matrix[22][12] + rf_conf_matrix[22][13] + rf_conf_matrix[22][14] + rf_conf_matrix[22][15] + rf_conf_matrix[22][16] + rf_conf_matrix[22][17] + rf_conf_matrix[22][18] + rf_conf_matrix[22][19] + rf_conf_matrix[22][20] + rf_conf_matrix[22][21] + rf_conf_matrix[22][23] + rf_conf_matrix[22][24] + rf_conf_matrix[22][25] + rf_conf_matrix[22][26] + rf_conf_matrix[22][27] + rf_conf_matrix[22][28]
        #rf_fn_24 = rf_conf_matrix[23][0] + rf_conf_matrix[23][1] + rf_conf_matrix[23][2] + rf_conf_matrix[23][3] + rf_conf_matrix[23][4] + rf_conf_matrix[23][5] + rf_conf_matrix[23][6] + rf_conf_matrix[23][7] + rf_conf_matrix[23][8] + rf_conf_matrix[23][9] + rf_conf_matrix[23][10] + rf_conf_matrix[23][11] + rf_conf_matrix[23][12] + rf_conf_matrix[23][13] + rf_conf_matrix[23][14] + rf_conf_matrix[23][15] + rf_conf_matrix[23][16] + rf_conf_matrix[23][17] + rf_conf_matrix[23][18] + rf_conf_matrix[23][19] + rf_conf_matrix[23][20] + rf_conf_matrix[23][21] + rf_conf_matrix[23][22] + rf_conf_matrix[23][24] + rf_conf_matrix[23][25] + rf_conf_matrix[23][26] + rf_conf_matrix[23][27] + rf_conf_matrix[23][28]
        #rf_fn_25 = rf_conf_matrix[24][0] + rf_conf_matrix[24][1] + rf_conf_matrix[24][2] + rf_conf_matrix[24][3] + rf_conf_matrix[24][4] + rf_conf_matrix[24][5] + rf_conf_matrix[24][6] + rf_conf_matrix[24][7] + rf_conf_matrix[24][8] + rf_conf_matrix[24][9] + rf_conf_matrix[24][10] + rf_conf_matrix[24][11] + rf_conf_matrix[24][12] + rf_conf_matrix[24][13] + rf_conf_matrix[24][14] + rf_conf_matrix[24][15] + rf_conf_matrix[24][16] + rf_conf_matrix[24][17] + rf_conf_matrix[24][18] + rf_conf_matrix[24][19] + rf_conf_matrix[24][20] + rf_conf_matrix[24][21] + rf_conf_matrix[24][22] + rf_conf_matrix[24][23] + rf_conf_matrix[24][25] + rf_conf_matrix[24][26] + rf_conf_matrix[24][27] + rf_conf_matrix[24][28]
        #rf_fn_26 = rf_conf_matrix[25][0] + rf_conf_matrix[25][1] + rf_conf_matrix[25][2] + rf_conf_matrix[25][3] + rf_conf_matrix[25][4] + rf_conf_matrix[25][5] + rf_conf_matrix[25][6] + rf_conf_matrix[25][7] + rf_conf_matrix[25][8] + rf_conf_matrix[25][9] + rf_conf_matrix[25][10] + rf_conf_matrix[25][11] + rf_conf_matrix[25][12] + rf_conf_matrix[25][13] + rf_conf_matrix[25][14] + rf_conf_matrix[25][15] + rf_conf_matrix[25][16] + rf_conf_matrix[25][17] + rf_conf_matrix[25][18] + rf_conf_matrix[25][19] + rf_conf_matrix[25][20] + rf_conf_matrix[25][21] + rf_conf_matrix[25][22] + rf_conf_matrix[25][23] + rf_conf_matrix[25][24] + rf_conf_matrix[25][26] + rf_conf_matrix[25][27] + rf_conf_matrix[25][28]
        #rf_fn_27 = rf_conf_matrix[26][0] + rf_conf_matrix[26][1] + rf_conf_matrix[26][2] + rf_conf_matrix[26][3] + rf_conf_matrix[26][4] + rf_conf_matrix[26][5] + rf_conf_matrix[26][6] + rf_conf_matrix[26][7] + rf_conf_matrix[26][8] + rf_conf_matrix[26][9] + rf_conf_matrix[26][10] + rf_conf_matrix[26][11] + rf_conf_matrix[26][12] + rf_conf_matrix[26][13] + rf_conf_matrix[26][14] + rf_conf_matrix[26][15] + rf_conf_matrix[26][16] + rf_conf_matrix[26][17] + rf_conf_matrix[26][18] + rf_conf_matrix[26][19] + rf_conf_matrix[26][20] + rf_conf_matrix[26][21] + rf_conf_matrix[26][22] + rf_conf_matrix[26][23] + rf_conf_matrix[26][24] + rf_conf_matrix[26][25] + rf_conf_matrix[26][27] + rf_conf_matrix[26][28]
        #rf_fn_28 = rf_conf_matrix[27][0] + rf_conf_matrix[27][1] + rf_conf_matrix[27][2] + rf_conf_matrix[27][3] + rf_conf_matrix[27][4] + rf_conf_matrix[27][5] + rf_conf_matrix[27][6] + rf_conf_matrix[27][7] + rf_conf_matrix[27][8] + rf_conf_matrix[27][9] + rf_conf_matrix[27][10] + rf_conf_matrix[27][11] + rf_conf_matrix[27][12] + rf_conf_matrix[27][13] + rf_conf_matrix[27][14] + rf_conf_matrix[27][15] + rf_conf_matrix[27][16] + rf_conf_matrix[27][17] + rf_conf_matrix[27][18] + rf_conf_matrix[27][19] + rf_conf_matrix[27][20] + rf_conf_matrix[27][21] + rf_conf_matrix[27][22] + rf_conf_matrix[27][23] + rf_conf_matrix[27][24] + rf_conf_matrix[27][25] + rf_conf_matrix[27][26] + rf_conf_matrix[27][28]
        #rf_fn_29 = rf_conf_matrix[28][0] + rf_conf_matrix[28][1] + rf_conf_matrix[28][2] + rf_conf_matrix[28][3] + rf_conf_matrix[28][4] + rf_conf_matrix[28][5] + rf_conf_matrix[28][6] + rf_conf_matrix[28][7] + rf_conf_matrix[28][8] + rf_conf_matrix[28][9] + rf_conf_matrix[28][10] + rf_conf_matrix[28][11] + rf_conf_matrix[28][12] + rf_conf_matrix[28][13] + rf_conf_matrix[28][14] + rf_conf_matrix[28][15] + rf_conf_matrix[28][16] + rf_conf_matrix[28][17] + rf_conf_matrix[28][18] + rf_conf_matrix[28][19] + rf_conf_matrix[28][20] + rf_conf_matrix[28][21] + rf_conf_matrix[28][22] + rf_conf_matrix[28][23] + rf_conf_matrix[28][24] + rf_conf_matrix[28][25] + rf_conf_matrix[28][26] + rf_conf_matrix[28][27]

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
            rf_recall_5 = rf_tp_5 / (rf_tp_5 + rf_fn_5)
        if rf_tp_6 + rf_fn_6 == 0:
            rf_recall_6 = 0
        else:
            rf_recall_6 = rf_tp_6 / (rf_tp_6 + rf_fn_6)
        if rf_tp_7 + rf_fn_7 == 0:
            rf_recall_7 = 0
        else:
            rf_recall_7 = rf_tp_7 / (rf_tp_7 + rf_fn_7)
        if rf_tp_8 + rf_fn_8 == 0:
            rf_recall_8 = 0
        else:
            rf_recall_8 = rf_tp_8 / (rf_tp_8 + rf_fn_8)
        if rf_tp_9 + rf_fn_9 == 0:
            rf_recall_9 = 0
        else:
            rf_recall_9 = rf_tp_9 / (rf_tp_9 + rf_fn_9)
        if rf_tp_10 + rf_fn_10 == 0:
            rf_recall_10 = 0
        else:
            rf_recall_10 = rf_tp_10 / (rf_tp_10 + rf_fn_10)
        if rf_tp_11 + rf_fn_11 == 0:
            rf_recall_11 = 0
        else:
            rf_recall_11 = rf_tp_11 / (rf_tp_11 + rf_fn_11)
        if rf_tp_12 + rf_fn_12 == 0:
            rf_recall_12 = 0
        else:
            rf_recall_12 = rf_tp_12 / (rf_tp_12 + rf_fn_12)
        if rf_tp_13 + rf_fn_13 == 0:
            rf_recall_13 = 0
        else:
            rf_recall_13 = rf_tp_13 / (rf_tp_13 + rf_fn_13)
        if rf_tp_14 + rf_fn_14 == 0:
            rf_recall_14 = 0
        else:
            rf_recall_14 = rf_tp_14 / (rf_tp_14 + rf_fn_14)
        if rf_tp_15 + rf_fn_15 == 0:
            rf_recall_15 = 0
        else:
            rf_recall_15 = rf_tp_15 / (rf_tp_15 + rf_fn_15)
        if rf_tp_16 + rf_fn_16 == 0:
            rf_recall_16 = 0
        else:
            rf_recall_16 = rf_tp_16 / (rf_tp_16 + rf_fn_16)
        if rf_tp_17 + rf_fn_17 == 0:
            rf_recall_17 = 0
        else:
            rf_recall_17 = rf_tp_17 / (rf_tp_17 + rf_fn_17)
        if rf_tp_18 + rf_fn_18 == 0:
            rf_recall_18 = 0
        else:
            rf_recall_18 = rf_tp_18 / (rf_tp_18 + rf_fn_18)
        if rf_tp_19 + rf_fn_19 == 0:
            rf_recall_19 = 0
        else:
            rf_recall_19 = rf_tp_19 / (rf_tp_19 + rf_fn_19)
        if rf_tp_20 + rf_fn_20 == 0:
            rf_recall_20 = 0
        else:
            rf_recall_20 = rf_tp_20 / (rf_tp_20 + rf_fn_20)
        '''
        if rf_tp_21 + rf_fn_21 == 0:
            rf_recall_21 = 0
        else:
            rf_recall_21 = rf_tp_21 / (rf_tp_21 + rf_fn_21)
        if rf_tp_22 + rf_fn_22 == 0:
            rf_recall_22 = 0
        else:
            rf_recall_22 = rf_tp_22 / (rf_tp_22 + rf_fn_22)
        if rf_tp_23 + rf_fn_23 == 0:
            rf_recall_23 = 0
        else:
            rf_recall_23 = rf_tp_23 / (rf_tp_23 + rf_fn_23)
        if rf_tp_24 + rf_fn_24 == 0:
            rf_recall_24 = 0
        else:
            rf_recall_24 = rf_tp_24 / (rf_tp_24 + rf_fn_24)
        if rf_tp_25 + rf_fn_25 == 0:
            rf_recall_25 = 0
        else:
            rf_recall_25 = rf_tp_25 / (rf_tp_25 + rf_fn_25)
        if rf_tp_26 + rf_fn_26 == 0:
            rf_recall_26 = 0
        else:
            rf_recall_26 = rf_tp_26 / (rf_tp_26 + rf_fn_26)
        if rf_tp_27 + rf_fn_27 == 0:
            rf_recall_27 = 0
        else:
            rf_recall_27 = rf_tp_27 / (rf_tp_27 + rf_fn_27)
        if rf_tp_28 + rf_fn_28 == 0:
            rf_recall_28 = 0
        else:
            rf_recall_28 = rf_tp_28 / (rf_tp_28 + rf_fn_28)
        if rf_tp_29 + rf_fn_29 == 0:
            rf_recall_29 = 0
        else:
            rf_recall_29 = rf_tp_29 / (rf_tp_29 + rf_fn_29)
        '''
        rf_recall_avg_pen_5 = (
                                 rf_recall_1 + rf_recall_2 + rf_recall_3 + rf_recall_4 + rf_recall_5 + rf_recall_6 + rf_recall_7 + rf_recall_8 + rf_recall_9 + rf_recall_10 + rf_recall_11 + rf_recall_12 + rf_recall_13 + rf_recall_14 + rf_recall_15 + rf_recall_16 + rf_recall_17 + rf_recall_18 + rf_recall_19 + (5*rf_recall_20)) / (20+5-1)
        return rf_recall_avg_pen_5

    from sklearn.metrics import classification_report, confusion_matrix

    rf_conf_matrix = confusion_matrix(y_test, rf_prediction, labels = np.unique(data['ACT_4']))


    print("rf_confusion matrix:")
    print(rf_conf_matrix)
    rf_precision = get_precision(rf_conf_matrix)
    rf_recall_pen_1 = get_recall_pen_1(rf_conf_matrix)
    rf_recall_pen_5 = get_recall_pen_5(rf_conf_matrix)
    rf_f1_score_pen_1 = 2 * (rf_precision * rf_recall_pen_1) / (rf_precision + rf_recall_pen_1)
    rf_f1_score_pen_5 = 2 * (rf_precision * rf_recall_pen_5) / (rf_precision + rf_recall_pen_5)
    rf_ovr_accuracy = (rf_conf_matrix[0][0] + rf_conf_matrix[1][1] + rf_conf_matrix[2][2] + rf_conf_matrix[3][3] + rf_conf_matrix[4][4] + rf_conf_matrix[5][5] + rf_conf_matrix[6][6] + rf_conf_matrix[7][7] + rf_conf_matrix[8][8] + rf_conf_matrix[9][9] + rf_conf_matrix[10][10] + rf_conf_matrix[11][11] + rf_conf_matrix[12][12] + rf_conf_matrix[13][13] + rf_conf_matrix[14][14] + rf_conf_matrix[15][15] + rf_conf_matrix[16][16] + rf_conf_matrix[17][17] + rf_conf_matrix[18][18] + rf_conf_matrix[19][19]) / (
                sum(rf_conf_matrix[0]) + sum(rf_conf_matrix[1]) + sum(rf_conf_matrix[2]) + sum(rf_conf_matrix[3]) + sum(rf_conf_matrix[4]) + sum(rf_conf_matrix[5]) + sum(rf_conf_matrix[6]) + sum(rf_conf_matrix[7]) + sum(rf_conf_matrix[8]) + sum(rf_conf_matrix[9]) + sum(rf_conf_matrix[10]) + sum(rf_conf_matrix[11]) + sum(rf_conf_matrix[12]) + sum(rf_conf_matrix[13]) + sum(rf_conf_matrix[14]) + sum(rf_conf_matrix[15]) + sum(rf_conf_matrix[16]) + sum(rf_conf_matrix[17]) + sum(rf_conf_matrix[18]) + sum(rf_conf_matrix[19]))
    print("rf_f1 score of pen 1 is:")
    print(rf_f1_score_pen_1)
    print("rf_f1 score of pen 5 is:")
    print(rf_f1_score_pen_5)
    print("rf_overall accuracy is:")
    print(rf_ovr_accuracy)
    rf_conf_matrix = pd.DataFrame(rf_conf_matrix)
    rf_conf_matrix.to_csv('conf_matrix_' + imb_technique + '_rf_production_' + str(nsplits) + 'foldcv_' + str(repeat+1) + '.csv', header=False, index=False)  # First repetition
    #rf_conf_matrix.to_csv('conf_matrix_' + imb_technique + '_penalty_' + str(penalty) + '_rf_production_' + str(nsplits) + 'foldcv_' + str(repeat+6) + '.csv', header=False, index=False)  # First repetition
    rf_f1_score_pen_1_kfoldcv[repeat] = rf_f1_score_pen_1
    rf_f1_score_pen_5_kfoldcv[repeat] = rf_f1_score_pen_5
    rf_ovr_accuracy_kfoldcv[repeat] = rf_ovr_accuracy



    for i in range(0, len(y_test)):
        #svm_DM_index = 0
        svm_FI_index = 0
        svm_FG_index = 0
        #svm_GR_index = 0
        #svm_GR12_index = 0
        svm_GR27_index = 0
        svm_LM_index = 0
        svm_LMM_index = 0
        #svm_MM14_index = 0
        #svm_MM16_index = 0
        svm_PC_index = 0
        svm_RG12_index = 0
        #svm_RG19_index = 0
        svm_RG2_index = 0
        svm_RG3_index = 0
        svm_RGM_index = 0
        svm_RGQC_index = 0
        #svm_TMSA10_index = 0
        svm_T8_index = 0
        #svm_T9_index = 0
        svm_TM10_index = 0
        svm_TM4_index = 0
        svm_TM5_index = 0
        svm_TM6_index = 0
        svm_TM8_index = 0
        svm_TM9_index = 0
        svm_TMQC_index = 0
        svm_TQC_index = 0
        #svm_WC13_index = 0

        """
        if svm_pred_class_DM[i] == "Deburring - Manual":
            if svm_pred_prob_DM[i][0] >= 0.5:
                svm_DM_index = 0
            else:
                svm_DM_index = 1
        elif svm_pred_class_DM[i] == "Others":
            if svm_pred_prob_DM[i][0] < 0.5:
                svm_DM_index = 0
            else:
                svm_DM_index = 1
        """
        if svm_pred_class_FI[i] == "Final Inspection Q.C.":
            if svm_pred_prob_FI[i][0] >= 0.5:
                svm_FI_index = 0
            else:
                svm_FI_index = 1
        elif svm_pred_class_FI[i] == "Others":
            if svm_pred_prob_FI[i][0] < 0.5:
                svm_FI_index = 0
            else:
                svm_FI_index = 1
        if svm_pred_class_FG[i] == "Flat Grinding - Machine 11":
            if svm_pred_prob_FG[i][0] >= 0.5:
                svm_FG_index = 0
            else:
                svm_FG_index = 1
        elif svm_pred_class_FG[i] == "Others":
            if svm_pred_prob_FG[i][0] < 0.5:
                svm_FG_index = 0
            else:
                svm_FG_index = 1
        """
        if svm_pred_class_GR[i] == "Grinding Rework":
            if svm_pred_prob_GR[i][0] >= 0.5:
                svm_GR_index = 0
            else:
                svm_GR_index = 1
        elif svm_pred_class_GR[i] == "Others":
            if svm_pred_prob_GR[i][0] < 0.5:
                svm_GR_index = 0
            else:
                svm_GR_index = 1
        """
        """
        if svm_pred_class_GR12[i] == "Grinding Rework - Machine 12":
            if svm_pred_prob_GR12[i][0] >= 0.5:
                svm_GR12_index = 0
            else:
                svm_GR12_index = 1
        elif svm_pred_class_GR12[i] == "Others":
            if svm_pred_prob_GR12[i][0] < 0.5:
                svm_GR12_index = 0
            else:
                svm_GR12_index = 1
        """
        if svm_pred_class_GR27[i] == "Grinding Rework - Machine 27":
            if svm_pred_prob_GR27[i][0] >= 0.5:
                svm_GR27_index = 0
            else:
                svm_GR27_index = 1
        elif svm_pred_class_GR27[i] == "Others":
            if svm_pred_prob_GR27[i][0] < 0.5:
                svm_GR27_index = 0
            else:
                svm_GR27_index = 1
        if svm_pred_class_LM[i] == "Lapping - Machine 1":
            if svm_pred_prob_LM[i][0] >= 0.5:
                svm_LM_index = 0
            else:
                svm_LM_index = 1
        elif svm_pred_class_LM[i] == "Others":
            if svm_pred_prob_LM[i][0] < 0.5:
                svm_LM_index = 0
            else:
                svm_LM_index = 1
        if svm_pred_class_LMM[i] == "Laser Marking - Machine 7":
            if svm_pred_prob_LMM[i][0] >= 0.5:
                svm_LMM_index = 0
            else:
                svm_LMM_index = 1
        elif svm_pred_class_LMM[i] == "Others":
            if svm_pred_prob_LMM[i][0] < 0.5:
                svm_LMM_index = 0
            else:
                svm_LMM_index = 1
        """
        if svm_pred_class_MM14[i] == "Milling - Machine 14":
            if svm_pred_prob_MM14[i][0] >= 0.5:
                svm_MM14_index = 0
            else:
                svm_MM14_index = 1
        elif svm_pred_class_MM14[i] == "Others":
            if svm_pred_prob_MM14[i][0] < 0.5:
                svm_MM14_index = 0
            else:
                svm_MM14_index = 1
        """
        """
        if svm_pred_class_MM16[i] == "Milling - Machine 16":
            if svm_pred_prob_MM16[i][0] >= 0.5:
                svm_MM16_index = 0
            else:
                svm_MM16_index = 1
        elif svm_pred_class_MM16[i] == "Others":
            if svm_pred_prob_MM16[i][0] < 0.5:
                svm_MM16_index = 0
            else:
                svm_MM16_index = 1
        """
        if svm_pred_class_PC[i] == "Packing":
            if svm_pred_prob_PC[i][0] >= 0.5:
                svm_PC_index = 0
            else:
                svm_PC_index = 1
        elif svm_pred_class_PC[i] == "Others":
            if svm_pred_prob_PC[i][0] < 0.5:
                svm_PC_index = 0
            else:
                svm_PC_index = 1
        if svm_pred_class_RG12[i] == "Round Grinding - Machine 12":
            if svm_pred_prob_RG12[i][0] >= 0.5:
                svm_RG12_index = 0
            else:
                svm_RG12_index = 1
        elif svm_pred_class_RG12[i] == "Others":
            if svm_pred_prob_RG12[i][0] < 0.5:
                svm_RG12_index = 0
            else:
                svm_RG12_index = 1
        """
        if svm_pred_class_RG19[i] == "Round Grinding - Machine 19":
            if svm_pred_prob_RG19[i][0] >= 0.5:
                svm_RG19_index = 0
            else:
                svm_RG19_index = 1
        elif svm_pred_class_RG19[i] == "Others":
            if svm_pred_prob_RG19[i][0] < 0.5:
                svm_RG19_index = 0
            else:
                svm_RG19_index = 1
        """
        if svm_pred_class_RG2[i] == "Round Grinding - Machine 2":
            if svm_pred_prob_RG2[i][0] >= 0.5:
                svm_RG2_index = 0
            else:
                svm_RG2_index = 1
        elif svm_pred_class_RG2[i] == "Others":
            if svm_pred_prob_RG2[i][0] < 0.5:
                svm_RG2_index = 0
            else:
                svm_RG2_index = 1
        if svm_pred_class_RG3[i] == "Round Grinding - Machine 3":
            if svm_pred_prob_RG3[i][0] >= 0.5:
                svm_RG3_index = 0
            else:
                svm_RG3_index = 1
        elif svm_pred_class_RG3[i] == "Others":
            if svm_pred_prob_RG3[i][0] < 0.5:
                svm_RG3_index = 0
            else:
                svm_RG3_index = 1
        if svm_pred_class_RGM[i] == "Round Grinding - Manual":
            if svm_pred_prob_RGM[i][0] >= 0.5:
                svm_RGM_index = 0
            else:
                svm_RGM_index = 1
        elif svm_pred_class_RGM[i] == "Others":
            if svm_pred_prob_RGM[i][0] < 0.5:
                svm_RGM_index = 0
            else:
                svm_RGM_index = 1
        if svm_pred_class_RGQC[i] == "Round Grinding - Q.C.":
            if svm_pred_prob_RGQC[i][0] >= 0.5:
                svm_RGQC_index = 0
            else:
                svm_RGQC_index = 1
        elif svm_pred_class_RGQC[i] == "Others":
            if svm_pred_prob_RGQC[i][0] < 0.5:
                svm_RGQC_index = 0
            else:
                svm_RGQC_index = 1
        """
        if svm_pred_class_TMSA10[i] == "Turn & Mill. & Screw Assem - Machine 10":
            if svm_pred_prob_TMSA10[i][0] >= 0.5:
                svm_TMSA10_index = 0
            else:
                svm_TMSA10_index = 1
        elif svm_pred_class_TMSA10[i] == "Others":
            if svm_pred_prob_TMSA10[i][0] < 0.5:
                svm_TMSA10_index = 0
            else:
                svm_TMSA10_index = 1
        """
        if svm_pred_class_T8[i] == "Turning - Machine 8":
            if svm_pred_prob_T8[i][0] >= 0.5:
                svm_T8_index = 0
            else:
                svm_T8_index = 1
        elif svm_pred_class_T8[i] == "Others":
            if svm_pred_prob_T8[i][0] < 0.5:
                svm_T8_index = 0
            else:
                svm_T8_index = 1
        """
        if svm_pred_class_T9[i] == "Turning - Machine 9":
            if svm_pred_prob_T9[i][0] >= 0.5:
                svm_T9_index = 0
            else:
                svm_T9_index = 1
        elif svm_pred_class_T9[i] == "Others":
            if svm_pred_prob_T9[i][0] < 0.5:
                svm_T9_index = 0
            else:
                svm_T9_index = 1
        """
        if svm_pred_class_TM10[i] == "Turning & Milling - Machine 10":
            if svm_pred_prob_TM10[i][0] >= 0.5:
                svm_TM10_index = 0
            else:
                svm_TM10_index = 1
        elif svm_pred_class_TM10[i] == "Others":
            if svm_pred_prob_TM10[i][0] < 0.5:
                svm_TM10_index = 0
            else:
                svm_TM10_index = 1
        if svm_pred_class_TM4[i] == "Turning & Milling - Machine 4":
            if svm_pred_prob_TM4[i][0] >= 0.5:
                svm_TM4_index = 0
            else:
                svm_TM4_index = 1
        elif svm_pred_class_TM4[i] == "Others":
            if svm_pred_prob_TM4[i][0] < 0.5:
                svm_TM4_index = 0
            else:
                svm_TM4_index = 1
        if svm_pred_class_TM5[i] == "Turning & Milling - Machine 5":
            if svm_pred_prob_TM5[i][0] >= 0.5:
                svm_TM5_index = 0
            else:
                svm_TM5_index = 1
        elif svm_pred_class_TM5[i] == "Others":
            if svm_pred_prob_TM5[i][0] < 0.5:
                svm_TM5_index = 0
            else:
                svm_TM5_index = 1
        if svm_pred_class_TM6[i] == "Turning & Milling - Machine 6":
            if svm_pred_prob_TM6[i][0] >= 0.5:
                svm_TM6_index = 0
            else:
                svm_TM6_index = 1
        elif svm_pred_class_TM6[i] == "Others":
            if svm_pred_prob_TM6[i][0] < 0.5:
                svm_TM6_index = 0
            else:
                svm_TM6_index = 1
        if svm_pred_class_TM8[i] == "Turning & Milling - Machine 8":
            if svm_pred_prob_TM8[i][0] >= 0.5:
                svm_TM8_index = 0
            else:
                svm_TM8_index = 1
        elif svm_pred_class_TM8[i] == "Others":
            if svm_pred_prob_TM8[i][0] < 0.5:
                svm_TM8_index = 0
            else:
                svm_TM8_index = 1
        if svm_pred_class_TM9[i] == "Turning & Milling - Machine 9":
            if svm_pred_prob_TM9[i][0] >= 0.5:
                svm_TM9_index = 0
            else:
                svm_TM9_index = 1
        elif svm_pred_class_TM9[i] == "Others":
            if svm_pred_prob_TM9[i][0] < 0.5:
                svm_TM9_index = 0
            else:
                svm_TM9_index = 1
        if svm_pred_class_TMQC[i] == "Turning & Milling Q.C.":
            if svm_pred_prob_TMQC[i][0] >= 0.5:
                svm_TMQC_index = 0
            else:
                svm_TMQC_index = 1
        elif svm_pred_class_TMQC[i] == "Others":
            if svm_pred_prob_TMQC[i][0] < 0.5:
                svm_TMQC_index = 0
            else:
                svm_TMQC_index = 1
        if svm_pred_class_TQC[i] == "Turning Q.C.":
            if svm_pred_prob_TQC[i][0] >= 0.5:
                svm_TQC_index = 0
            else:
                svm_TQC_index = 1
        elif svm_pred_class_TQC[i] == "Others":
            if svm_pred_prob_TQC[i][0] < 0.5:
                svm_TQC_index = 0
            else:
                svm_TQC_index = 1
        """
        if svm_pred_class_WC13[i] == "Wire Cut - Machine 13":
            if svm_pred_prob_WC13[i][0] >= 0.5:
                svm_WC13_index = 0
            else:
                svm_WC13_index = 1
        elif svm_pred_class_WC13[i] == "Others":
            if svm_pred_prob_WC13[i][0] < 0.5:
                svm_WC13_index = 0
            else:
                svm_WC13_index = 1
        """
        #if svm_pred_prob_DM[i][svm_DM_index] == max(svm_pred_prob_FI[i][svm_FI_index], svm_pred_prob_FG[i][svm_FG_index], svm_pred_prob_GR27[i][svm_GR27_index], svm_pred_prob_LM[i][svm_LM_index], svm_pred_prob_LMM[i][svm_LMM_index], svm_pred_prob_PC[i][svm_PC_index], svm_pred_prob_RG12[i][svm_RG12_index], svm_pred_prob_RG2[i][svm_RG2_index], svm_pred_prob_RG3[i][svm_RG3_index], svm_pred_prob_RGM[i][svm_RGM_index], svm_pred_prob_RGQC[i][svm_RGQC_index], svm_pred_prob_T8[i][svm_T8_index], svm_pred_prob_TM10[i][svm_TM10_index], svm_pred_prob_TM4[i][svm_TM4_index], svm_pred_prob_TM5[i][svm_TM5_index], svm_pred_prob_TM6[i][svm_TM6_index], svm_pred_prob_TM8[i][svm_TM8_index], svm_pred_prob_TM9[i][svm_TM9_index], svm_pred_prob_TMQC[i][svm_TMQC_index], svm_pred_prob_TQC[i][svm_TQC_index]):
        #    svm_prediction.loc[i] = "Deburring - Manual"
        if svm_pred_prob_FI[i][svm_FI_index] == max(svm_pred_prob_FI[i][svm_FI_index], svm_pred_prob_FG[i][svm_FG_index], svm_pred_prob_GR27[i][svm_GR27_index], svm_pred_prob_LM[i][svm_LM_index], svm_pred_prob_LMM[i][svm_LMM_index], svm_pred_prob_PC[i][svm_PC_index], svm_pred_prob_RG12[i][svm_RG12_index], svm_pred_prob_RG2[i][svm_RG2_index], svm_pred_prob_RG3[i][svm_RG3_index], svm_pred_prob_RGM[i][svm_RGM_index], svm_pred_prob_RGQC[i][svm_RGQC_index], svm_pred_prob_T8[i][svm_T8_index], svm_pred_prob_TM10[i][svm_TM10_index], svm_pred_prob_TM4[i][svm_TM4_index], svm_pred_prob_TM5[i][svm_TM5_index], svm_pred_prob_TM6[i][svm_TM6_index], svm_pred_prob_TM8[i][svm_TM8_index], svm_pred_prob_TM9[i][svm_TM9_index], svm_pred_prob_TMQC[i][svm_TMQC_index], svm_pred_prob_TQC[i][svm_TQC_index]):
            svm_prediction.loc[i] = "Final Inspection Q.C."
        elif svm_pred_prob_FG[i][svm_FG_index] == max(svm_pred_prob_FI[i][svm_FI_index], svm_pred_prob_FG[i][svm_FG_index], svm_pred_prob_GR27[i][svm_GR27_index], svm_pred_prob_LM[i][svm_LM_index], svm_pred_prob_LMM[i][svm_LMM_index], svm_pred_prob_PC[i][svm_PC_index], svm_pred_prob_RG12[i][svm_RG12_index], svm_pred_prob_RG2[i][svm_RG2_index], svm_pred_prob_RG3[i][svm_RG3_index], svm_pred_prob_RGM[i][svm_RGM_index], svm_pred_prob_RGQC[i][svm_RGQC_index], svm_pred_prob_T8[i][svm_T8_index], svm_pred_prob_TM10[i][svm_TM10_index], svm_pred_prob_TM4[i][svm_TM4_index], svm_pred_prob_TM5[i][svm_TM5_index], svm_pred_prob_TM6[i][svm_TM6_index], svm_pred_prob_TM8[i][svm_TM8_index], svm_pred_prob_TM9[i][svm_TM9_index], svm_pred_prob_TMQC[i][svm_TMQC_index], svm_pred_prob_TQC[i][svm_TQC_index]):
            svm_prediction.loc[i] = "Flat Grinding - Machine 11"
        #elif svm_pred_prob_GR[i][svm_GR_index] == max(svm_pred_prob_FI[i][svm_FI_index], svm_pred_prob_FG[i][svm_FG_index], svm_pred_prob_GR27[i][svm_GR27_index], svm_pred_prob_LM[i][svm_LM_index], svm_pred_prob_LMM[i][svm_LMM_index], svm_pred_prob_PC[i][svm_PC_index], svm_pred_prob_RG12[i][svm_RG12_index], svm_pred_prob_RG2[i][svm_RG2_index], svm_pred_prob_RG3[i][svm_RG3_index], svm_pred_prob_RGM[i][svm_RGM_index], svm_pred_prob_RGQC[i][svm_RGQC_index], svm_pred_prob_T8[i][svm_T8_index], svm_pred_prob_TM10[i][svm_TM10_index], svm_pred_prob_TM4[i][svm_TM4_index], svm_pred_prob_TM5[i][svm_TM5_index], svm_pred_prob_TM6[i][svm_TM6_index], svm_pred_prob_TM8[i][svm_TM8_index], svm_pred_prob_TM9[i][svm_TM9_index], svm_pred_prob_TMQC[i][svm_TMQC_index], svm_pred_prob_TQC[i][svm_TQC_index]):
        #    svm_prediction.loc[i] = "Grinding Rework"
        #elif svm_pred_prob_GR12[i][svm_GR12_index] == max(svm_pred_prob_FI[i][svm_FI_index], svm_pred_prob_FG[i][svm_FG_index], svm_pred_prob_GR27[i][svm_GR27_index], svm_pred_prob_LM[i][svm_LM_index], svm_pred_prob_LMM[i][svm_LMM_index], svm_pred_prob_PC[i][svm_PC_index], svm_pred_prob_RG12[i][svm_RG12_index], svm_pred_prob_RG2[i][svm_RG2_index], svm_pred_prob_RG3[i][svm_RG3_index], svm_pred_prob_RGM[i][svm_RGM_index], svm_pred_prob_RGQC[i][svm_RGQC_index], svm_pred_prob_T8[i][svm_T8_index], svm_pred_prob_TM10[i][svm_TM10_index], svm_pred_prob_TM4[i][svm_TM4_index], svm_pred_prob_TM5[i][svm_TM5_index], svm_pred_prob_TM6[i][svm_TM6_index], svm_pred_prob_TM8[i][svm_TM8_index], svm_pred_prob_TM9[i][svm_TM9_index], svm_pred_prob_TMQC[i][svm_TMQC_index], svm_pred_prob_TQC[i][svm_TQC_index]):
        #    svm_prediction.loc[i] = "Grinding Rework - Machine 12"
        elif svm_pred_prob_GR27[i][svm_GR27_index] == max(svm_pred_prob_FI[i][svm_FI_index], svm_pred_prob_FG[i][svm_FG_index], svm_pred_prob_GR27[i][svm_GR27_index], svm_pred_prob_LM[i][svm_LM_index], svm_pred_prob_LMM[i][svm_LMM_index], svm_pred_prob_PC[i][svm_PC_index], svm_pred_prob_RG12[i][svm_RG12_index], svm_pred_prob_RG2[i][svm_RG2_index], svm_pred_prob_RG3[i][svm_RG3_index], svm_pred_prob_RGM[i][svm_RGM_index], svm_pred_prob_RGQC[i][svm_RGQC_index], svm_pred_prob_T8[i][svm_T8_index], svm_pred_prob_TM10[i][svm_TM10_index], svm_pred_prob_TM4[i][svm_TM4_index], svm_pred_prob_TM5[i][svm_TM5_index], svm_pred_prob_TM6[i][svm_TM6_index], svm_pred_prob_TM8[i][svm_TM8_index], svm_pred_prob_TM9[i][svm_TM9_index], svm_pred_prob_TMQC[i][svm_TMQC_index], svm_pred_prob_TQC[i][svm_TQC_index]):
            svm_prediction.loc[i] = "Grinding Rework - Machine 27"
        elif svm_pred_prob_LM[i][svm_LM_index] == max(svm_pred_prob_FI[i][svm_FI_index], svm_pred_prob_FG[i][svm_FG_index], svm_pred_prob_GR27[i][svm_GR27_index], svm_pred_prob_LM[i][svm_LM_index], svm_pred_prob_LMM[i][svm_LMM_index], svm_pred_prob_PC[i][svm_PC_index], svm_pred_prob_RG12[i][svm_RG12_index], svm_pred_prob_RG2[i][svm_RG2_index], svm_pred_prob_RG3[i][svm_RG3_index], svm_pred_prob_RGM[i][svm_RGM_index], svm_pred_prob_RGQC[i][svm_RGQC_index], svm_pred_prob_T8[i][svm_T8_index], svm_pred_prob_TM10[i][svm_TM10_index], svm_pred_prob_TM4[i][svm_TM4_index], svm_pred_prob_TM5[i][svm_TM5_index], svm_pred_prob_TM6[i][svm_TM6_index], svm_pred_prob_TM8[i][svm_TM8_index], svm_pred_prob_TM9[i][svm_TM9_index], svm_pred_prob_TMQC[i][svm_TMQC_index], svm_pred_prob_TQC[i][svm_TQC_index]):
            svm_prediction.loc[i] = "Lapping - Machine 1"
        elif svm_pred_prob_LMM[i][svm_LMM_index] == max(svm_pred_prob_FI[i][svm_FI_index], svm_pred_prob_FG[i][svm_FG_index], svm_pred_prob_GR27[i][svm_GR27_index], svm_pred_prob_LM[i][svm_LM_index], svm_pred_prob_LMM[i][svm_LMM_index], svm_pred_prob_PC[i][svm_PC_index], svm_pred_prob_RG12[i][svm_RG12_index], svm_pred_prob_RG2[i][svm_RG2_index], svm_pred_prob_RG3[i][svm_RG3_index], svm_pred_prob_RGM[i][svm_RGM_index], svm_pred_prob_RGQC[i][svm_RGQC_index], svm_pred_prob_T8[i][svm_T8_index], svm_pred_prob_TM10[i][svm_TM10_index], svm_pred_prob_TM4[i][svm_TM4_index], svm_pred_prob_TM5[i][svm_TM5_index], svm_pred_prob_TM6[i][svm_TM6_index], svm_pred_prob_TM8[i][svm_TM8_index], svm_pred_prob_TM9[i][svm_TM9_index], svm_pred_prob_TMQC[i][svm_TMQC_index], svm_pred_prob_TQC[i][svm_TQC_index]):
            svm_prediction.loc[i] = "Laser Marking - Machine 7"
        #elif svm_pred_prob_MM14[i][svm_MM14_index] == max(svm_pred_prob_FI[i][svm_FI_index], svm_pred_prob_FG[i][svm_FG_index], svm_pred_prob_GR27[i][svm_GR27_index], svm_pred_prob_LM[i][svm_LM_index], svm_pred_prob_LMM[i][svm_LMM_index], svm_pred_prob_PC[i][svm_PC_index], svm_pred_prob_RG12[i][svm_RG12_index], svm_pred_prob_RG2[i][svm_RG2_index], svm_pred_prob_RG3[i][svm_RG3_index], svm_pred_prob_RGM[i][svm_RGM_index], svm_pred_prob_RGQC[i][svm_RGQC_index], svm_pred_prob_T8[i][svm_T8_index], svm_pred_prob_TM10[i][svm_TM10_index], svm_pred_prob_TM4[i][svm_TM4_index], svm_pred_prob_TM5[i][svm_TM5_index], svm_pred_prob_TM6[i][svm_TM6_index], svm_pred_prob_TM8[i][svm_TM8_index], svm_pred_prob_TM9[i][svm_TM9_index], svm_pred_prob_TMQC[i][svm_TMQC_index], svm_pred_prob_TQC[i][svm_TQC_index]):
        #    svm_prediction.loc[i] = "Milling - Machine 14"
        #elif svm_pred_prob_MM16[i][svm_MM16_index] == max(svm_pred_prob_FI[i][svm_FI_index], svm_pred_prob_FG[i][svm_FG_index], svm_pred_prob_GR27[i][svm_GR27_index], svm_pred_prob_LM[i][svm_LM_index], svm_pred_prob_LMM[i][svm_LMM_index], svm_pred_prob_PC[i][svm_PC_index], svm_pred_prob_RG12[i][svm_RG12_index], svm_pred_prob_RG2[i][svm_RG2_index], svm_pred_prob_RG3[i][svm_RG3_index], svm_pred_prob_RGM[i][svm_RGM_index], svm_pred_prob_RGQC[i][svm_RGQC_index], svm_pred_prob_T8[i][svm_T8_index], svm_pred_prob_TM10[i][svm_TM10_index], svm_pred_prob_TM4[i][svm_TM4_index], svm_pred_prob_TM5[i][svm_TM5_index], svm_pred_prob_TM6[i][svm_TM6_index], svm_pred_prob_TM8[i][svm_TM8_index], svm_pred_prob_TM9[i][svm_TM9_index], svm_pred_prob_TMQC[i][svm_TMQC_index], svm_pred_prob_TQC[i][svm_TQC_index]):
        #    svm_prediction.loc[i] = "Milling - Machine 16"
        elif svm_pred_prob_PC[i][svm_PC_index] == max(svm_pred_prob_FI[i][svm_FI_index], svm_pred_prob_FG[i][svm_FG_index], svm_pred_prob_GR27[i][svm_GR27_index], svm_pred_prob_LM[i][svm_LM_index], svm_pred_prob_LMM[i][svm_LMM_index], svm_pred_prob_PC[i][svm_PC_index], svm_pred_prob_RG12[i][svm_RG12_index], svm_pred_prob_RG2[i][svm_RG2_index], svm_pred_prob_RG3[i][svm_RG3_index], svm_pred_prob_RGM[i][svm_RGM_index], svm_pred_prob_RGQC[i][svm_RGQC_index], svm_pred_prob_T8[i][svm_T8_index], svm_pred_prob_TM10[i][svm_TM10_index], svm_pred_prob_TM4[i][svm_TM4_index], svm_pred_prob_TM5[i][svm_TM5_index], svm_pred_prob_TM6[i][svm_TM6_index], svm_pred_prob_TM8[i][svm_TM8_index], svm_pred_prob_TM9[i][svm_TM9_index], svm_pred_prob_TMQC[i][svm_TMQC_index], svm_pred_prob_TQC[i][svm_TQC_index]):
            svm_prediction.loc[i] = "Packing"
        elif svm_pred_prob_RG12[i][svm_RG12_index] == max(svm_pred_prob_FI[i][svm_FI_index], svm_pred_prob_FG[i][svm_FG_index], svm_pred_prob_GR27[i][svm_GR27_index], svm_pred_prob_LM[i][svm_LM_index], svm_pred_prob_LMM[i][svm_LMM_index], svm_pred_prob_PC[i][svm_PC_index], svm_pred_prob_RG12[i][svm_RG12_index], svm_pred_prob_RG2[i][svm_RG2_index], svm_pred_prob_RG3[i][svm_RG3_index], svm_pred_prob_RGM[i][svm_RGM_index], svm_pred_prob_RGQC[i][svm_RGQC_index], svm_pred_prob_T8[i][svm_T8_index], svm_pred_prob_TM10[i][svm_TM10_index], svm_pred_prob_TM4[i][svm_TM4_index], svm_pred_prob_TM5[i][svm_TM5_index], svm_pred_prob_TM6[i][svm_TM6_index], svm_pred_prob_TM8[i][svm_TM8_index], svm_pred_prob_TM9[i][svm_TM9_index], svm_pred_prob_TMQC[i][svm_TMQC_index], svm_pred_prob_TQC[i][svm_TQC_index]):
            svm_prediction.loc[i] = "Round Grinding - Machine 12"
        #elif svm_pred_prob_RG19[i][svm_RG19_index] == max(svm_pred_prob_FI[i][svm_FI_index], svm_pred_prob_FG[i][svm_FG_index], svm_pred_prob_GR27[i][svm_GR27_index], svm_pred_prob_LM[i][svm_LM_index], svm_pred_prob_LMM[i][svm_LMM_index], svm_pred_prob_PC[i][svm_PC_index], svm_pred_prob_RG12[i][svm_RG12_index], svm_pred_prob_RG2[i][svm_RG2_index], svm_pred_prob_RG3[i][svm_RG3_index], svm_pred_prob_RGM[i][svm_RGM_index], svm_pred_prob_RGQC[i][svm_RGQC_index], svm_pred_prob_T8[i][svm_T8_index], svm_pred_prob_TM10[i][svm_TM10_index], svm_pred_prob_TM4[i][svm_TM4_index], svm_pred_prob_TM5[i][svm_TM5_index], svm_pred_prob_TM6[i][svm_TM6_index], svm_pred_prob_TM8[i][svm_TM8_index], svm_pred_prob_TM9[i][svm_TM9_index], svm_pred_prob_TMQC[i][svm_TMQC_index], svm_pred_prob_TQC[i][svm_TQC_index]):
        #    svm_prediction.loc[i] = "Round Grinding - Machine 19"
        elif svm_pred_prob_RG2[i][svm_RG2_index] == max(svm_pred_prob_FI[i][svm_FI_index], svm_pred_prob_FG[i][svm_FG_index], svm_pred_prob_GR27[i][svm_GR27_index], svm_pred_prob_LM[i][svm_LM_index], svm_pred_prob_LMM[i][svm_LMM_index], svm_pred_prob_PC[i][svm_PC_index], svm_pred_prob_RG12[i][svm_RG12_index], svm_pred_prob_RG2[i][svm_RG2_index], svm_pred_prob_RG3[i][svm_RG3_index], svm_pred_prob_RGM[i][svm_RGM_index], svm_pred_prob_RGQC[i][svm_RGQC_index], svm_pred_prob_T8[i][svm_T8_index], svm_pred_prob_TM10[i][svm_TM10_index], svm_pred_prob_TM4[i][svm_TM4_index], svm_pred_prob_TM5[i][svm_TM5_index], svm_pred_prob_TM6[i][svm_TM6_index], svm_pred_prob_TM8[i][svm_TM8_index], svm_pred_prob_TM9[i][svm_TM9_index], svm_pred_prob_TMQC[i][svm_TMQC_index], svm_pred_prob_TQC[i][svm_TQC_index]):
            svm_prediction.loc[i] = "Round Grinding - Machine 2"
        elif svm_pred_prob_RG3[i][svm_RG3_index] == max(svm_pred_prob_FI[i][svm_FI_index], svm_pred_prob_FG[i][svm_FG_index], svm_pred_prob_GR27[i][svm_GR27_index], svm_pred_prob_LM[i][svm_LM_index], svm_pred_prob_LMM[i][svm_LMM_index], svm_pred_prob_PC[i][svm_PC_index], svm_pred_prob_RG12[i][svm_RG12_index], svm_pred_prob_RG2[i][svm_RG2_index], svm_pred_prob_RG3[i][svm_RG3_index], svm_pred_prob_RGM[i][svm_RGM_index], svm_pred_prob_RGQC[i][svm_RGQC_index], svm_pred_prob_T8[i][svm_T8_index], svm_pred_prob_TM10[i][svm_TM10_index], svm_pred_prob_TM4[i][svm_TM4_index], svm_pred_prob_TM5[i][svm_TM5_index], svm_pred_prob_TM6[i][svm_TM6_index], svm_pred_prob_TM8[i][svm_TM8_index], svm_pred_prob_TM9[i][svm_TM9_index], svm_pred_prob_TMQC[i][svm_TMQC_index], svm_pred_prob_TQC[i][svm_TQC_index]):
            svm_prediction.loc[i] = "Round Grinding - Machine 3"
        elif svm_pred_prob_RGM[i][svm_RGM_index] == max(svm_pred_prob_FI[i][svm_FI_index], svm_pred_prob_FG[i][svm_FG_index], svm_pred_prob_GR27[i][svm_GR27_index], svm_pred_prob_LM[i][svm_LM_index], svm_pred_prob_LMM[i][svm_LMM_index], svm_pred_prob_PC[i][svm_PC_index], svm_pred_prob_RG12[i][svm_RG12_index], svm_pred_prob_RG2[i][svm_RG2_index], svm_pred_prob_RG3[i][svm_RG3_index], svm_pred_prob_RGM[i][svm_RGM_index], svm_pred_prob_RGQC[i][svm_RGQC_index], svm_pred_prob_T8[i][svm_T8_index], svm_pred_prob_TM10[i][svm_TM10_index], svm_pred_prob_TM4[i][svm_TM4_index], svm_pred_prob_TM5[i][svm_TM5_index], svm_pred_prob_TM6[i][svm_TM6_index], svm_pred_prob_TM8[i][svm_TM8_index], svm_pred_prob_TM9[i][svm_TM9_index], svm_pred_prob_TMQC[i][svm_TMQC_index], svm_pred_prob_TQC[i][svm_TQC_index]):
            svm_prediction.loc[i] = "Round Grinding - Manual"
        elif svm_pred_prob_RGQC[i][svm_RGQC_index] == max(svm_pred_prob_FI[i][svm_FI_index], svm_pred_prob_FG[i][svm_FG_index], svm_pred_prob_GR27[i][svm_GR27_index], svm_pred_prob_LM[i][svm_LM_index], svm_pred_prob_LMM[i][svm_LMM_index], svm_pred_prob_PC[i][svm_PC_index], svm_pred_prob_RG12[i][svm_RG12_index], svm_pred_prob_RG2[i][svm_RG2_index], svm_pred_prob_RG3[i][svm_RG3_index], svm_pred_prob_RGM[i][svm_RGM_index], svm_pred_prob_RGQC[i][svm_RGQC_index], svm_pred_prob_T8[i][svm_T8_index], svm_pred_prob_TM10[i][svm_TM10_index], svm_pred_prob_TM4[i][svm_TM4_index], svm_pred_prob_TM5[i][svm_TM5_index], svm_pred_prob_TM6[i][svm_TM6_index], svm_pred_prob_TM8[i][svm_TM8_index], svm_pred_prob_TM9[i][svm_TM9_index], svm_pred_prob_TMQC[i][svm_TMQC_index], svm_pred_prob_TQC[i][svm_TQC_index]):
            svm_prediction.loc[i] = "Round Grinding - Q.C."
        #elif svm_pred_prob_TMSA10[i][svm_TMSA10_index] == max(svm_pred_prob_FI[i][svm_FI_index], svm_pred_prob_FG[i][svm_FG_index], svm_pred_prob_GR27[i][svm_GR27_index], svm_pred_prob_LM[i][svm_LM_index], svm_pred_prob_LMM[i][svm_LMM_index], svm_pred_prob_PC[i][svm_PC_index], svm_pred_prob_RG12[i][svm_RG12_index], svm_pred_prob_RG2[i][svm_RG2_index], svm_pred_prob_RG3[i][svm_RG3_index], svm_pred_prob_RGM[i][svm_RGM_index], svm_pred_prob_RGQC[i][svm_RGQC_index], svm_pred_prob_T8[i][svm_T8_index], svm_pred_prob_TM10[i][svm_TM10_index], svm_pred_prob_TM4[i][svm_TM4_index], svm_pred_prob_TM5[i][svm_TM5_index], svm_pred_prob_TM6[i][svm_TM6_index], svm_pred_prob_TM8[i][svm_TM8_index], svm_pred_prob_TM9[i][svm_TM9_index], svm_pred_prob_TMQC[i][svm_TMQC_index], svm_pred_prob_TQC[i][svm_TQC_index]):
        #    svm_prediction.loc[i] = "Turn & Mill. & Screw Assem - Machine 10"
        elif svm_pred_prob_T8[i][svm_T8_index] == max(svm_pred_prob_FI[i][svm_FI_index], svm_pred_prob_FG[i][svm_FG_index], svm_pred_prob_GR27[i][svm_GR27_index], svm_pred_prob_LM[i][svm_LM_index], svm_pred_prob_LMM[i][svm_LMM_index], svm_pred_prob_PC[i][svm_PC_index], svm_pred_prob_RG12[i][svm_RG12_index], svm_pred_prob_RG2[i][svm_RG2_index], svm_pred_prob_RG3[i][svm_RG3_index], svm_pred_prob_RGM[i][svm_RGM_index], svm_pred_prob_RGQC[i][svm_RGQC_index], svm_pred_prob_T8[i][svm_T8_index], svm_pred_prob_TM10[i][svm_TM10_index], svm_pred_prob_TM4[i][svm_TM4_index], svm_pred_prob_TM5[i][svm_TM5_index], svm_pred_prob_TM6[i][svm_TM6_index], svm_pred_prob_TM8[i][svm_TM8_index], svm_pred_prob_TM9[i][svm_TM9_index], svm_pred_prob_TMQC[i][svm_TMQC_index], svm_pred_prob_TQC[i][svm_TQC_index]):
            svm_prediction.loc[i] = "Turning - Machine 8"
        #elif svm_pred_prob_T9[i][svm_T9_index] == max(svm_pred_prob_FI[i][svm_FI_index], svm_pred_prob_FG[i][svm_FG_index], svm_pred_prob_GR27[i][svm_GR27_index], svm_pred_prob_LM[i][svm_LM_index], svm_pred_prob_LMM[i][svm_LMM_index], svm_pred_prob_PC[i][svm_PC_index], svm_pred_prob_RG12[i][svm_RG12_index], svm_pred_prob_RG2[i][svm_RG2_index], svm_pred_prob_RG3[i][svm_RG3_index], svm_pred_prob_RGM[i][svm_RGM_index], svm_pred_prob_RGQC[i][svm_RGQC_index], svm_pred_prob_T8[i][svm_T8_index], svm_pred_prob_TM10[i][svm_TM10_index], svm_pred_prob_TM4[i][svm_TM4_index], svm_pred_prob_TM5[i][svm_TM5_index], svm_pred_prob_TM6[i][svm_TM6_index], svm_pred_prob_TM8[i][svm_TM8_index], svm_pred_prob_TM9[i][svm_TM9_index], svm_pred_prob_TMQC[i][svm_TMQC_index], svm_pred_prob_TQC[i][svm_TQC_index]):
        #    svm_prediction.loc[i] = "Turning - Machine 9"
        elif svm_pred_prob_TM10[i][svm_TM10_index] == max(svm_pred_prob_FI[i][svm_FI_index], svm_pred_prob_FG[i][svm_FG_index], svm_pred_prob_GR27[i][svm_GR27_index], svm_pred_prob_LM[i][svm_LM_index], svm_pred_prob_LMM[i][svm_LMM_index], svm_pred_prob_PC[i][svm_PC_index], svm_pred_prob_RG12[i][svm_RG12_index], svm_pred_prob_RG2[i][svm_RG2_index], svm_pred_prob_RG3[i][svm_RG3_index], svm_pred_prob_RGM[i][svm_RGM_index], svm_pred_prob_RGQC[i][svm_RGQC_index], svm_pred_prob_T8[i][svm_T8_index], svm_pred_prob_TM10[i][svm_TM10_index], svm_pred_prob_TM4[i][svm_TM4_index], svm_pred_prob_TM5[i][svm_TM5_index], svm_pred_prob_TM6[i][svm_TM6_index], svm_pred_prob_TM8[i][svm_TM8_index], svm_pred_prob_TM9[i][svm_TM9_index], svm_pred_prob_TMQC[i][svm_TMQC_index], svm_pred_prob_TQC[i][svm_TQC_index]):
            svm_prediction.loc[i] = "Turning & Milling - Machine 10"
        elif svm_pred_prob_TM4[i][svm_TM4_index] == max(svm_pred_prob_FI[i][svm_FI_index], svm_pred_prob_FG[i][svm_FG_index], svm_pred_prob_GR27[i][svm_GR27_index], svm_pred_prob_LM[i][svm_LM_index], svm_pred_prob_LMM[i][svm_LMM_index], svm_pred_prob_PC[i][svm_PC_index], svm_pred_prob_RG12[i][svm_RG12_index], svm_pred_prob_RG2[i][svm_RG2_index], svm_pred_prob_RG3[i][svm_RG3_index], svm_pred_prob_RGM[i][svm_RGM_index], svm_pred_prob_RGQC[i][svm_RGQC_index], svm_pred_prob_T8[i][svm_T8_index], svm_pred_prob_TM10[i][svm_TM10_index], svm_pred_prob_TM4[i][svm_TM4_index], svm_pred_prob_TM5[i][svm_TM5_index], svm_pred_prob_TM6[i][svm_TM6_index], svm_pred_prob_TM8[i][svm_TM8_index], svm_pred_prob_TM9[i][svm_TM9_index], svm_pred_prob_TMQC[i][svm_TMQC_index], svm_pred_prob_TQC[i][svm_TQC_index]):
            svm_prediction.loc[i] = "Turning & Milling - Machine 4"
        elif svm_pred_prob_TM5[i][svm_TM5_index] == max(svm_pred_prob_FI[i][svm_FI_index], svm_pred_prob_FG[i][svm_FG_index], svm_pred_prob_GR27[i][svm_GR27_index], svm_pred_prob_LM[i][svm_LM_index], svm_pred_prob_LMM[i][svm_LMM_index], svm_pred_prob_PC[i][svm_PC_index], svm_pred_prob_RG12[i][svm_RG12_index], svm_pred_prob_RG2[i][svm_RG2_index], svm_pred_prob_RG3[i][svm_RG3_index], svm_pred_prob_RGM[i][svm_RGM_index], svm_pred_prob_RGQC[i][svm_RGQC_index], svm_pred_prob_T8[i][svm_T8_index], svm_pred_prob_TM10[i][svm_TM10_index], svm_pred_prob_TM4[i][svm_TM4_index], svm_pred_prob_TM5[i][svm_TM5_index], svm_pred_prob_TM6[i][svm_TM6_index], svm_pred_prob_TM8[i][svm_TM8_index], svm_pred_prob_TM9[i][svm_TM9_index], svm_pred_prob_TMQC[i][svm_TMQC_index], svm_pred_prob_TQC[i][svm_TQC_index]):
            svm_prediction.loc[i] = "Turning & Milling - Machine 5"
        elif svm_pred_prob_TM6[i][svm_TM6_index] == max(svm_pred_prob_FI[i][svm_FI_index], svm_pred_prob_FG[i][svm_FG_index], svm_pred_prob_GR27[i][svm_GR27_index], svm_pred_prob_LM[i][svm_LM_index], svm_pred_prob_LMM[i][svm_LMM_index], svm_pred_prob_PC[i][svm_PC_index], svm_pred_prob_RG12[i][svm_RG12_index], svm_pred_prob_RG2[i][svm_RG2_index], svm_pred_prob_RG3[i][svm_RG3_index], svm_pred_prob_RGM[i][svm_RGM_index], svm_pred_prob_RGQC[i][svm_RGQC_index], svm_pred_prob_T8[i][svm_T8_index], svm_pred_prob_TM10[i][svm_TM10_index], svm_pred_prob_TM4[i][svm_TM4_index], svm_pred_prob_TM5[i][svm_TM5_index], svm_pred_prob_TM6[i][svm_TM6_index], svm_pred_prob_TM8[i][svm_TM8_index], svm_pred_prob_TM9[i][svm_TM9_index], svm_pred_prob_TMQC[i][svm_TMQC_index], svm_pred_prob_TQC[i][svm_TQC_index]):
            svm_prediction.loc[i] = "Turning & Milling - Machine 6"
        elif svm_pred_prob_TM8[i][svm_TM8_index] == max(svm_pred_prob_FI[i][svm_FI_index], svm_pred_prob_FG[i][svm_FG_index], svm_pred_prob_GR27[i][svm_GR27_index], svm_pred_prob_LM[i][svm_LM_index], svm_pred_prob_LMM[i][svm_LMM_index], svm_pred_prob_PC[i][svm_PC_index], svm_pred_prob_RG12[i][svm_RG12_index], svm_pred_prob_RG2[i][svm_RG2_index], svm_pred_prob_RG3[i][svm_RG3_index], svm_pred_prob_RGM[i][svm_RGM_index], svm_pred_prob_RGQC[i][svm_RGQC_index], svm_pred_prob_T8[i][svm_T8_index], svm_pred_prob_TM10[i][svm_TM10_index], svm_pred_prob_TM4[i][svm_TM4_index], svm_pred_prob_TM5[i][svm_TM5_index], svm_pred_prob_TM6[i][svm_TM6_index], svm_pred_prob_TM8[i][svm_TM8_index], svm_pred_prob_TM9[i][svm_TM9_index], svm_pred_prob_TMQC[i][svm_TMQC_index], svm_pred_prob_TQC[i][svm_TQC_index]):
            svm_prediction.loc[i] = "Turning & Milling - Machine 8"
        elif svm_pred_prob_TM9[i][svm_TM9_index] == max(svm_pred_prob_FI[i][svm_FI_index], svm_pred_prob_FG[i][svm_FG_index], svm_pred_prob_GR27[i][svm_GR27_index], svm_pred_prob_LM[i][svm_LM_index], svm_pred_prob_LMM[i][svm_LMM_index], svm_pred_prob_PC[i][svm_PC_index], svm_pred_prob_RG12[i][svm_RG12_index], svm_pred_prob_RG2[i][svm_RG2_index], svm_pred_prob_RG3[i][svm_RG3_index], svm_pred_prob_RGM[i][svm_RGM_index], svm_pred_prob_RGQC[i][svm_RGQC_index], svm_pred_prob_T8[i][svm_T8_index], svm_pred_prob_TM10[i][svm_TM10_index], svm_pred_prob_TM4[i][svm_TM4_index], svm_pred_prob_TM5[i][svm_TM5_index], svm_pred_prob_TM6[i][svm_TM6_index], svm_pred_prob_TM8[i][svm_TM8_index], svm_pred_prob_TM9[i][svm_TM9_index], svm_pred_prob_TMQC[i][svm_TMQC_index], svm_pred_prob_TQC[i][svm_TQC_index]):
            svm_prediction.loc[i] = "Turning & Milling - Machine 9"
        elif svm_pred_prob_TMQC[i][svm_TMQC_index] == max(svm_pred_prob_FI[i][svm_FI_index], svm_pred_prob_FG[i][svm_FG_index], svm_pred_prob_GR27[i][svm_GR27_index], svm_pred_prob_LM[i][svm_LM_index], svm_pred_prob_LMM[i][svm_LMM_index], svm_pred_prob_PC[i][svm_PC_index], svm_pred_prob_RG12[i][svm_RG12_index], svm_pred_prob_RG2[i][svm_RG2_index], svm_pred_prob_RG3[i][svm_RG3_index], svm_pred_prob_RGM[i][svm_RGM_index], svm_pred_prob_RGQC[i][svm_RGQC_index], svm_pred_prob_T8[i][svm_T8_index], svm_pred_prob_TM10[i][svm_TM10_index], svm_pred_prob_TM4[i][svm_TM4_index], svm_pred_prob_TM5[i][svm_TM5_index], svm_pred_prob_TM6[i][svm_TM6_index], svm_pred_prob_TM8[i][svm_TM8_index], svm_pred_prob_TM9[i][svm_TM9_index], svm_pred_prob_TMQC[i][svm_TMQC_index], svm_pred_prob_TQC[i][svm_TQC_index]):
            svm_prediction.loc[i] = "Turning & Milling Q.C."
        elif svm_pred_prob_TQC[i][svm_TQC_index] == max(svm_pred_prob_FI[i][svm_FI_index], svm_pred_prob_FG[i][svm_FG_index], svm_pred_prob_GR27[i][svm_GR27_index], svm_pred_prob_LM[i][svm_LM_index], svm_pred_prob_LMM[i][svm_LMM_index], svm_pred_prob_PC[i][svm_PC_index], svm_pred_prob_RG12[i][svm_RG12_index], svm_pred_prob_RG2[i][svm_RG2_index], svm_pred_prob_RG3[i][svm_RG3_index], svm_pred_prob_RGM[i][svm_RGM_index], svm_pred_prob_RGQC[i][svm_RGQC_index], svm_pred_prob_T8[i][svm_T8_index], svm_pred_prob_TM10[i][svm_TM10_index], svm_pred_prob_TM4[i][svm_TM4_index], svm_pred_prob_TM5[i][svm_TM5_index], svm_pred_prob_TM6[i][svm_TM6_index], svm_pred_prob_TM8[i][svm_TM8_index], svm_pred_prob_TM9[i][svm_TM9_index], svm_pred_prob_TMQC[i][svm_TMQC_index], svm_pred_prob_TQC[i][svm_TQC_index]):
            svm_prediction.loc[i] = "Turning Q.C."
        #elif svm_pred_prob_WC13[i][svm_WC13_index] == max(svm_pred_prob_FI[i][svm_FI_index], svm_pred_prob_FG[i][svm_FG_index], svm_pred_prob_GR27[i][svm_GR27_index], svm_pred_prob_LM[i][svm_LM_index], svm_pred_prob_LMM[i][svm_LMM_index], svm_pred_prob_PC[i][svm_PC_index], svm_pred_prob_RG12[i][svm_RG12_index], svm_pred_prob_RG2[i][svm_RG2_index], svm_pred_prob_RG3[i][svm_RG3_index], svm_pred_prob_RGM[i][svm_RGM_index], svm_pred_prob_RGQC[i][svm_RGQC_index], svm_pred_prob_T8[i][svm_T8_index], svm_pred_prob_TM10[i][svm_TM10_index], svm_pred_prob_TM4[i][svm_TM4_index], svm_pred_prob_TM5[i][svm_TM5_index], svm_pred_prob_TM6[i][svm_TM6_index], svm_pred_prob_TM8[i][svm_TM8_index], svm_pred_prob_TM9[i][svm_TM9_index], svm_pred_prob_TMQC[i][svm_TMQC_index], svm_pred_prob_TQC[i][svm_TQC_index]):
        #    svm_prediction.loc[i] = "Wire Cut - Machine 13"


    def get_precision(svm_conf_matrix):
        svm_tp_1 = svm_conf_matrix[0][0]
        svm_tp_2 = svm_conf_matrix[1][1]
        svm_tp_3 = svm_conf_matrix[2][2]
        svm_tp_4 = svm_conf_matrix[3][3]
        svm_tp_5 = svm_conf_matrix[4][4]
        svm_tp_6 = svm_conf_matrix[5][5]
        svm_tp_7 = svm_conf_matrix[6][6]
        svm_tp_8 = svm_conf_matrix[7][7]
        svm_tp_9 = svm_conf_matrix[8][8]
        svm_tp_10 = svm_conf_matrix[9][9]
        svm_tp_11 = svm_conf_matrix[10][10]
        svm_tp_12 = svm_conf_matrix[11][11]
        svm_tp_13 = svm_conf_matrix[12][12]
        svm_tp_14 = svm_conf_matrix[13][13]
        svm_tp_15 = svm_conf_matrix[14][14]
        svm_tp_16 = svm_conf_matrix[15][15]
        svm_tp_17 = svm_conf_matrix[16][16]
        svm_tp_18 = svm_conf_matrix[17][17]
        svm_tp_19 = svm_conf_matrix[18][18]
        svm_tp_20 = svm_conf_matrix[19][19]
        #svm_tp_21 = svm_conf_matrix[20][20]
        #svm_tp_22 = svm_conf_matrix[21][21]
        #svm_tp_23 = svm_conf_matrix[22][22]
        #svm_tp_24 = svm_conf_matrix[23][23]
        #svm_tp_25 = svm_conf_matrix[24][24]
        #svm_tp_26 = svm_conf_matrix[25][25]
        #svm_tp_27 = svm_conf_matrix[26][26]
        #svm_tp_28 = svm_conf_matrix[27][27]
        #svm_tp_29 = svm_conf_matrix[28][28]

        svm_fp_1 = svm_conf_matrix[1][0] + svm_conf_matrix[2][0] + svm_conf_matrix[3][0] + svm_conf_matrix[4][0] + svm_conf_matrix[5][0] + svm_conf_matrix[6][0] + svm_conf_matrix[7][0] + svm_conf_matrix[8][0] + svm_conf_matrix[9][0] + svm_conf_matrix[10][0] + svm_conf_matrix[11][0] + svm_conf_matrix[12][0] + svm_conf_matrix[13][0] + svm_conf_matrix[14][0] + svm_conf_matrix[15][0] + svm_conf_matrix[16][0] + svm_conf_matrix[17][0] + svm_conf_matrix[18][0] + svm_conf_matrix[19][0]
        svm_fp_2 = svm_conf_matrix[0][1] + svm_conf_matrix[2][1] + svm_conf_matrix[3][1] + svm_conf_matrix[4][1] + svm_conf_matrix[5][1] + svm_conf_matrix[6][1] + svm_conf_matrix[7][1] + svm_conf_matrix[8][1] + svm_conf_matrix[9][1] + svm_conf_matrix[10][1] + svm_conf_matrix[11][1] + svm_conf_matrix[12][1] + svm_conf_matrix[13][1] + svm_conf_matrix[14][1] + svm_conf_matrix[15][1] + svm_conf_matrix[16][1] + svm_conf_matrix[17][1] + svm_conf_matrix[18][1] + svm_conf_matrix[19][1]
        svm_fp_3 = svm_conf_matrix[0][2] + svm_conf_matrix[1][2] + svm_conf_matrix[3][2] + svm_conf_matrix[4][2] + svm_conf_matrix[5][2] + svm_conf_matrix[6][2] + svm_conf_matrix[7][2] + svm_conf_matrix[8][2] + svm_conf_matrix[9][2] + svm_conf_matrix[10][2] + svm_conf_matrix[11][2] + svm_conf_matrix[12][2] + svm_conf_matrix[13][2] + svm_conf_matrix[14][2] + svm_conf_matrix[15][2] + svm_conf_matrix[16][2] + svm_conf_matrix[17][2] + svm_conf_matrix[18][2] + svm_conf_matrix[19][2]
        svm_fp_4 = svm_conf_matrix[0][3] + svm_conf_matrix[1][3] + svm_conf_matrix[2][3] + svm_conf_matrix[4][3] + svm_conf_matrix[5][3] + svm_conf_matrix[6][3] + svm_conf_matrix[7][3] + svm_conf_matrix[8][3] + svm_conf_matrix[9][3] + svm_conf_matrix[10][3] + svm_conf_matrix[11][3] + svm_conf_matrix[12][3] + svm_conf_matrix[13][3] + svm_conf_matrix[14][3] + svm_conf_matrix[15][3] + svm_conf_matrix[16][3] + svm_conf_matrix[17][3] + svm_conf_matrix[18][3] + svm_conf_matrix[19][3]
        svm_fp_5 = svm_conf_matrix[0][4] + svm_conf_matrix[1][4] + svm_conf_matrix[2][4] + svm_conf_matrix[3][4] + svm_conf_matrix[5][4] + svm_conf_matrix[6][4] + svm_conf_matrix[7][4] + svm_conf_matrix[8][4] + svm_conf_matrix[9][4] + svm_conf_matrix[10][4] + svm_conf_matrix[11][4] + svm_conf_matrix[12][4] + svm_conf_matrix[13][4] + svm_conf_matrix[14][4] + svm_conf_matrix[15][4] + svm_conf_matrix[16][4] + svm_conf_matrix[17][4] + svm_conf_matrix[18][4] + svm_conf_matrix[19][4]
        svm_fp_6 = svm_conf_matrix[0][5] + svm_conf_matrix[1][5] + svm_conf_matrix[2][5] + svm_conf_matrix[3][5] + svm_conf_matrix[4][5] + svm_conf_matrix[6][5] + svm_conf_matrix[7][5] + svm_conf_matrix[8][5] + svm_conf_matrix[9][5] + svm_conf_matrix[10][5] + svm_conf_matrix[11][5] + svm_conf_matrix[12][5] + svm_conf_matrix[13][5] + svm_conf_matrix[14][5] + svm_conf_matrix[15][5] + svm_conf_matrix[16][5] + svm_conf_matrix[17][5] + svm_conf_matrix[18][5] + svm_conf_matrix[19][5]
        svm_fp_7 = svm_conf_matrix[0][6] + svm_conf_matrix[1][6] + svm_conf_matrix[2][6] + svm_conf_matrix[3][6] + svm_conf_matrix[4][6] + svm_conf_matrix[5][6] + svm_conf_matrix[7][6] + svm_conf_matrix[8][6] + svm_conf_matrix[9][6] + svm_conf_matrix[10][6] + svm_conf_matrix[11][6] + svm_conf_matrix[12][6] + svm_conf_matrix[13][6] + svm_conf_matrix[14][6] + svm_conf_matrix[15][6] + svm_conf_matrix[16][6] + svm_conf_matrix[17][6] + svm_conf_matrix[18][6] + svm_conf_matrix[19][6]
        svm_fp_8 = svm_conf_matrix[0][7] + svm_conf_matrix[1][7] + svm_conf_matrix[2][7] + svm_conf_matrix[3][7] + svm_conf_matrix[4][7] + svm_conf_matrix[5][7] + svm_conf_matrix[6][7] + svm_conf_matrix[8][7] + svm_conf_matrix[9][7] + svm_conf_matrix[10][7] + svm_conf_matrix[11][7] + svm_conf_matrix[12][7] + svm_conf_matrix[13][7] + svm_conf_matrix[14][7] + svm_conf_matrix[15][7] + svm_conf_matrix[16][7] + svm_conf_matrix[17][7] + svm_conf_matrix[18][7] + svm_conf_matrix[19][7]
        svm_fp_9 = svm_conf_matrix[0][8] + svm_conf_matrix[1][8] + svm_conf_matrix[2][8] + svm_conf_matrix[3][8] + svm_conf_matrix[4][8] + svm_conf_matrix[5][8] + svm_conf_matrix[6][8] + svm_conf_matrix[7][8] + svm_conf_matrix[9][8] + svm_conf_matrix[10][8] + svm_conf_matrix[11][8] + svm_conf_matrix[12][8] + svm_conf_matrix[13][8] + svm_conf_matrix[14][8] + svm_conf_matrix[15][8] + svm_conf_matrix[16][8] + svm_conf_matrix[17][8] + svm_conf_matrix[18][8] + svm_conf_matrix[19][8]
        svm_fp_10 = svm_conf_matrix[0][9] + svm_conf_matrix[1][9] + svm_conf_matrix[2][9] + svm_conf_matrix[3][9] + svm_conf_matrix[4][9] + svm_conf_matrix[5][9] + svm_conf_matrix[6][9] + svm_conf_matrix[7][9] + svm_conf_matrix[8][9] + svm_conf_matrix[10][9] + svm_conf_matrix[11][9] + svm_conf_matrix[12][9] + svm_conf_matrix[13][9] + svm_conf_matrix[14][9] + svm_conf_matrix[15][9] + svm_conf_matrix[16][9] + svm_conf_matrix[17][9] + svm_conf_matrix[18][9] + svm_conf_matrix[19][9]
        svm_fp_11 = svm_conf_matrix[0][10] + svm_conf_matrix[1][10] + svm_conf_matrix[2][10] + svm_conf_matrix[3][10] + svm_conf_matrix[4][10] + svm_conf_matrix[5][10] + svm_conf_matrix[6][10] + svm_conf_matrix[7][10] + svm_conf_matrix[8][10] + svm_conf_matrix[9][10] + svm_conf_matrix[11][10] + svm_conf_matrix[12][10] + svm_conf_matrix[13][10] + svm_conf_matrix[14][10] + svm_conf_matrix[15][10] + svm_conf_matrix[16][10] + svm_conf_matrix[17][10] + svm_conf_matrix[18][10] + svm_conf_matrix[19][10]
        svm_fp_12 = svm_conf_matrix[0][11] + svm_conf_matrix[1][11] + svm_conf_matrix[2][11] + svm_conf_matrix[3][11] + svm_conf_matrix[4][11] + svm_conf_matrix[5][11] + svm_conf_matrix[6][11] + svm_conf_matrix[7][11] + svm_conf_matrix[8][11] + svm_conf_matrix[9][11] + svm_conf_matrix[10][11] + svm_conf_matrix[12][11] + svm_conf_matrix[13][11] + svm_conf_matrix[14][11] + svm_conf_matrix[15][11] + svm_conf_matrix[16][11] + svm_conf_matrix[17][11] + svm_conf_matrix[18][11] + svm_conf_matrix[19][11]
        svm_fp_13 = svm_conf_matrix[0][12] + svm_conf_matrix[1][12] + svm_conf_matrix[2][12] + svm_conf_matrix[3][12] + svm_conf_matrix[4][12] + svm_conf_matrix[5][12] + svm_conf_matrix[6][12] + svm_conf_matrix[7][12] + svm_conf_matrix[8][12] + svm_conf_matrix[9][12] + svm_conf_matrix[10][12] + svm_conf_matrix[11][12] + svm_conf_matrix[13][12] + svm_conf_matrix[14][12] + svm_conf_matrix[15][12] + svm_conf_matrix[16][12] + svm_conf_matrix[17][12] + svm_conf_matrix[18][12] + svm_conf_matrix[19][12]
        svm_fp_14 = svm_conf_matrix[0][13] + svm_conf_matrix[1][13] + svm_conf_matrix[2][13] + svm_conf_matrix[3][13] + svm_conf_matrix[4][13] + svm_conf_matrix[5][13] + svm_conf_matrix[6][13] + svm_conf_matrix[7][13] + svm_conf_matrix[8][13] + svm_conf_matrix[9][13] + svm_conf_matrix[10][13] + svm_conf_matrix[11][13] + svm_conf_matrix[12][13] + svm_conf_matrix[14][13] + svm_conf_matrix[15][13] + svm_conf_matrix[16][13] + svm_conf_matrix[17][13] + svm_conf_matrix[18][13] + svm_conf_matrix[19][13]
        svm_fp_15 = svm_conf_matrix[0][14] + svm_conf_matrix[1][14] + svm_conf_matrix[2][14] + svm_conf_matrix[3][14] + svm_conf_matrix[4][14] + svm_conf_matrix[5][14] + svm_conf_matrix[6][14] + svm_conf_matrix[7][14] + svm_conf_matrix[8][14] + svm_conf_matrix[9][14] + svm_conf_matrix[10][14] + svm_conf_matrix[11][14] + svm_conf_matrix[12][14] + svm_conf_matrix[13][14] + svm_conf_matrix[15][14] + svm_conf_matrix[16][14] + svm_conf_matrix[17][14] + svm_conf_matrix[18][14] + svm_conf_matrix[19][14]
        svm_fp_16 = svm_conf_matrix[0][15] + svm_conf_matrix[1][15] + svm_conf_matrix[2][15] + svm_conf_matrix[3][15] + svm_conf_matrix[4][15] + svm_conf_matrix[5][15] + svm_conf_matrix[6][15] + svm_conf_matrix[7][15] + svm_conf_matrix[8][15] + svm_conf_matrix[9][15] + svm_conf_matrix[10][15] + svm_conf_matrix[11][15] + svm_conf_matrix[12][15] + svm_conf_matrix[13][15] + svm_conf_matrix[14][15] + svm_conf_matrix[16][15] + svm_conf_matrix[17][15] + svm_conf_matrix[18][15] + svm_conf_matrix[19][15]
        svm_fp_17 = svm_conf_matrix[0][16] + svm_conf_matrix[1][16] + svm_conf_matrix[2][16] + svm_conf_matrix[3][16] + svm_conf_matrix[4][16] + svm_conf_matrix[5][16] + svm_conf_matrix[6][16] + svm_conf_matrix[7][16] + svm_conf_matrix[8][16] + svm_conf_matrix[9][16] + svm_conf_matrix[10][16] + svm_conf_matrix[11][16] + svm_conf_matrix[12][16] + svm_conf_matrix[13][16] + svm_conf_matrix[14][16] + svm_conf_matrix[15][16] + svm_conf_matrix[17][16] + svm_conf_matrix[18][16] + svm_conf_matrix[19][16]
        svm_fp_18 = svm_conf_matrix[0][17] + svm_conf_matrix[1][17] + svm_conf_matrix[2][17] + svm_conf_matrix[3][17] + svm_conf_matrix[4][17] + svm_conf_matrix[5][17] + svm_conf_matrix[6][17] + svm_conf_matrix[7][17] + svm_conf_matrix[8][17] + svm_conf_matrix[9][17] + svm_conf_matrix[10][17] + svm_conf_matrix[11][17] + svm_conf_matrix[12][17] + svm_conf_matrix[13][17] + svm_conf_matrix[14][17] + svm_conf_matrix[15][17] + svm_conf_matrix[16][17] + svm_conf_matrix[18][17] + svm_conf_matrix[19][17]
        svm_fp_19 = svm_conf_matrix[0][18] + svm_conf_matrix[1][18] + svm_conf_matrix[2][18] + svm_conf_matrix[3][18] + svm_conf_matrix[4][18] + svm_conf_matrix[5][18] + svm_conf_matrix[6][18] + svm_conf_matrix[7][18] + svm_conf_matrix[8][18] + svm_conf_matrix[9][18] + svm_conf_matrix[10][18] + svm_conf_matrix[11][18] + svm_conf_matrix[12][18] + svm_conf_matrix[13][18] + svm_conf_matrix[14][18] + svm_conf_matrix[15][18] + svm_conf_matrix[16][18] + svm_conf_matrix[17][18] + svm_conf_matrix[19][18]
        svm_fp_20 = svm_conf_matrix[0][19] + svm_conf_matrix[1][19] + svm_conf_matrix[2][19] + svm_conf_matrix[3][19] + svm_conf_matrix[4][19] + svm_conf_matrix[5][19] + svm_conf_matrix[6][19] + svm_conf_matrix[7][19] + svm_conf_matrix[8][19] + svm_conf_matrix[9][19] + svm_conf_matrix[10][19] + svm_conf_matrix[11][19] + svm_conf_matrix[12][19] + svm_conf_matrix[13][19] + svm_conf_matrix[14][19] + svm_conf_matrix[15][19] + svm_conf_matrix[16][19] + svm_conf_matrix[17][19] + svm_conf_matrix[18][19]
        #svm_fp_21 = svm_conf_matrix[0][20] + svm_conf_matrix[1][20] + svm_conf_matrix[2][20] + svm_conf_matrix[3][20] + svm_conf_matrix[4][20] + svm_conf_matrix[5][20] + svm_conf_matrix[6][20] + svm_conf_matrix[7][20] + svm_conf_matrix[8][20] + svm_conf_matrix[9][20] + svm_conf_matrix[10][20] + svm_conf_matrix[11][20] + svm_conf_matrix[12][20] + svm_conf_matrix[13][20] + svm_conf_matrix[14][20] + svm_conf_matrix[15][20] + svm_conf_matrix[16][20] + svm_conf_matrix[17][20] + svm_conf_matrix[18][20] + svm_conf_matrix[19][20] + svm_conf_matrix[21][20] + svm_conf_matrix[22][20] + svm_conf_matrix[23][20] + svm_conf_matrix[24][20] + svm_conf_matrix[25][20] + svm_conf_matrix[26][20] + svm_conf_matrix[27][20] + svm_conf_matrix[28][20]
        #svm_fp_22 = svm_conf_matrix[0][21] + svm_conf_matrix[1][21] + svm_conf_matrix[2][21] + svm_conf_matrix[3][21] + svm_conf_matrix[4][21] + svm_conf_matrix[5][21] + svm_conf_matrix[6][21] + svm_conf_matrix[7][21] + svm_conf_matrix[8][21] + svm_conf_matrix[9][21] + svm_conf_matrix[10][21] + svm_conf_matrix[11][21] + svm_conf_matrix[12][21] + svm_conf_matrix[13][21] + svm_conf_matrix[14][21] + svm_conf_matrix[15][21] + svm_conf_matrix[16][21] + svm_conf_matrix[17][21] + svm_conf_matrix[18][21] + svm_conf_matrix[19][21] + svm_conf_matrix[20][21] + svm_conf_matrix[22][21] + svm_conf_matrix[23][21] + svm_conf_matrix[24][21] + svm_conf_matrix[25][21] + svm_conf_matrix[26][21] + svm_conf_matrix[27][21] + svm_conf_matrix[28][21]
        #svm_fp_23 = svm_conf_matrix[0][22] + svm_conf_matrix[1][22] + svm_conf_matrix[2][22] + svm_conf_matrix[3][22] + svm_conf_matrix[4][22] + svm_conf_matrix[5][22] + svm_conf_matrix[6][22] + svm_conf_matrix[7][22] + svm_conf_matrix[8][22] + svm_conf_matrix[9][22] + svm_conf_matrix[10][22] + svm_conf_matrix[11][22] + svm_conf_matrix[12][22] + svm_conf_matrix[13][22] + svm_conf_matrix[14][22] + svm_conf_matrix[15][22] + svm_conf_matrix[16][22] + svm_conf_matrix[17][22] + svm_conf_matrix[18][22] + svm_conf_matrix[19][22] + svm_conf_matrix[20][22] + svm_conf_matrix[21][22] + svm_conf_matrix[23][22] + svm_conf_matrix[24][22] + svm_conf_matrix[25][22] + svm_conf_matrix[26][22] + svm_conf_matrix[27][22] + svm_conf_matrix[28][22]
        #svm_fp_24 = svm_conf_matrix[0][23] + svm_conf_matrix[1][23] + svm_conf_matrix[2][23] + svm_conf_matrix[3][23] + svm_conf_matrix[4][23] + svm_conf_matrix[5][23] + svm_conf_matrix[6][23] + svm_conf_matrix[7][23] + svm_conf_matrix[8][23] + svm_conf_matrix[9][23] + svm_conf_matrix[10][23] + svm_conf_matrix[11][23] + svm_conf_matrix[12][23] + svm_conf_matrix[13][23] + svm_conf_matrix[14][23] + svm_conf_matrix[15][23] + svm_conf_matrix[16][23] + svm_conf_matrix[17][23] + svm_conf_matrix[18][23] + svm_conf_matrix[19][23] + svm_conf_matrix[20][23] + svm_conf_matrix[21][23] + svm_conf_matrix[22][23] + svm_conf_matrix[24][23] + svm_conf_matrix[25][23] + svm_conf_matrix[26][23] + svm_conf_matrix[27][23] + svm_conf_matrix[28][23]
        #svm_fp_25 = svm_conf_matrix[0][24] + svm_conf_matrix[1][24] + svm_conf_matrix[2][24] + svm_conf_matrix[3][24] + svm_conf_matrix[4][24] + svm_conf_matrix[5][24] + svm_conf_matrix[6][24] + svm_conf_matrix[7][24] + svm_conf_matrix[8][24] + svm_conf_matrix[9][24] + svm_conf_matrix[10][24] + svm_conf_matrix[11][24] + svm_conf_matrix[12][24] + svm_conf_matrix[13][24] + svm_conf_matrix[14][24] + svm_conf_matrix[15][24] + svm_conf_matrix[16][24] + svm_conf_matrix[17][24] + svm_conf_matrix[18][24] + svm_conf_matrix[19][24] + svm_conf_matrix[20][24] + svm_conf_matrix[21][24] + svm_conf_matrix[22][24] + svm_conf_matrix[23][24] + svm_conf_matrix[25][24] + svm_conf_matrix[26][24] + svm_conf_matrix[27][24] + svm_conf_matrix[28][24]
        #svm_fp_26 = svm_conf_matrix[0][25] + svm_conf_matrix[1][25] + svm_conf_matrix[2][25] + svm_conf_matrix[3][25] + svm_conf_matrix[4][25] + svm_conf_matrix[5][25] + svm_conf_matrix[6][25] + svm_conf_matrix[7][25] + svm_conf_matrix[8][25] + svm_conf_matrix[9][25] + svm_conf_matrix[10][25] + svm_conf_matrix[11][25] + svm_conf_matrix[12][25] + svm_conf_matrix[13][25] + svm_conf_matrix[14][25] + svm_conf_matrix[15][25] + svm_conf_matrix[16][25] + svm_conf_matrix[17][25] + svm_conf_matrix[18][25] + svm_conf_matrix[19][25] + svm_conf_matrix[20][25] + svm_conf_matrix[21][25] + svm_conf_matrix[22][25] + svm_conf_matrix[23][25] + svm_conf_matrix[24][25] + svm_conf_matrix[26][25] + svm_conf_matrix[27][25] + svm_conf_matrix[28][25]
        #svm_fp_27 = svm_conf_matrix[0][26] + svm_conf_matrix[1][26] + svm_conf_matrix[2][26] + svm_conf_matrix[3][26] + svm_conf_matrix[4][26] + svm_conf_matrix[5][26] + svm_conf_matrix[6][26] + svm_conf_matrix[7][26] + svm_conf_matrix[8][26] + svm_conf_matrix[9][26] + svm_conf_matrix[10][26] + svm_conf_matrix[11][26] + svm_conf_matrix[12][26] + svm_conf_matrix[13][26] + svm_conf_matrix[14][26] + svm_conf_matrix[15][26] + svm_conf_matrix[16][26] + svm_conf_matrix[17][26] + svm_conf_matrix[18][26] + svm_conf_matrix[19][26] + svm_conf_matrix[20][26] + svm_conf_matrix[21][26] + svm_conf_matrix[22][26] + svm_conf_matrix[23][26] + svm_conf_matrix[24][26] + svm_conf_matrix[25][26] + svm_conf_matrix[27][26] + svm_conf_matrix[28][26]
        #svm_fp_28 = svm_conf_matrix[0][27] + svm_conf_matrix[1][27] + svm_conf_matrix[2][27] + svm_conf_matrix[3][27] + svm_conf_matrix[4][27] + svm_conf_matrix[5][27] + svm_conf_matrix[6][27] + svm_conf_matrix[7][27] + svm_conf_matrix[8][27] + svm_conf_matrix[9][27] + svm_conf_matrix[10][27] + svm_conf_matrix[11][27] + svm_conf_matrix[12][27] + svm_conf_matrix[13][27] + svm_conf_matrix[14][27] + svm_conf_matrix[15][27] + svm_conf_matrix[16][27] + svm_conf_matrix[17][27] + svm_conf_matrix[18][27] + svm_conf_matrix[19][27] + svm_conf_matrix[20][27] + svm_conf_matrix[21][27] + svm_conf_matrix[22][27] + svm_conf_matrix[23][27] + svm_conf_matrix[24][27] + svm_conf_matrix[25][27] + svm_conf_matrix[26][27] + svm_conf_matrix[28][27]
        #svm_fp_29 = svm_conf_matrix[0][28] + svm_conf_matrix[1][28] + svm_conf_matrix[2][28] + svm_conf_matrix[3][28] + svm_conf_matrix[4][28] + svm_conf_matrix[5][28] + svm_conf_matrix[6][28] + svm_conf_matrix[7][28] + svm_conf_matrix[8][28] + svm_conf_matrix[9][28] + svm_conf_matrix[10][28] + svm_conf_matrix[11][28] + svm_conf_matrix[12][28] + svm_conf_matrix[13][28] + svm_conf_matrix[14][28] + svm_conf_matrix[15][28] + svm_conf_matrix[16][28] + svm_conf_matrix[17][28] + svm_conf_matrix[18][28] + svm_conf_matrix[19][28] + svm_conf_matrix[20][28] + svm_conf_matrix[21][28] + svm_conf_matrix[22][28] + svm_conf_matrix[23][28] + svm_conf_matrix[24][28] + svm_conf_matrix[25][28] + svm_conf_matrix[26][28] + svm_conf_matrix[27][28]

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
        if svm_tp_6 + svm_fp_6 == 0:
            svm_precision_6 = 0
        else:
            svm_precision_6 = svm_tp_6 / (svm_tp_6 + svm_fp_6)
        if svm_tp_7 + svm_fp_7 == 0:
            svm_precision_7 = 0
        else:
            svm_precision_7 = svm_tp_7 / (svm_tp_7 + svm_fp_7)
        if svm_tp_8 + svm_fp_8 == 0:
            svm_precision_8 = 0
        else:
            svm_precision_8 = svm_tp_8 / (svm_tp_8 + svm_fp_8)
        if svm_tp_9 + svm_fp_9 == 0:
            svm_precision_9 = 0
        else:
            svm_precision_9 = svm_tp_9 / (svm_tp_9 + svm_fp_9)
        if svm_tp_10 + svm_fp_10 == 0:
            svm_precision_10 = 0
        else:
            svm_precision_10 = svm_tp_10 / (svm_tp_10 + svm_fp_10)
        if svm_tp_11 + svm_fp_11 == 0:
            svm_precision_11 = 0
        else:
            svm_precision_11 = svm_tp_11 / (svm_tp_11 + svm_fp_11)
        if svm_tp_12 + svm_fp_12 == 0:
            svm_precision_12 = 0
        else:
            svm_precision_12 = svm_tp_12 / (svm_tp_12 + svm_fp_12)
        if svm_tp_13 + svm_fp_13 == 0:
            svm_precision_13 = 0
        else:
            svm_precision_13 = svm_tp_13 / (svm_tp_13 + svm_fp_13)
        if svm_tp_14 + svm_fp_14 == 0:
            svm_precision_14 = 0
        else:
            svm_precision_14 = svm_tp_14 / (svm_tp_14 + svm_fp_14)
        if svm_tp_15 + svm_fp_15 == 0:
            svm_precision_15 = 0
        else:
            svm_precision_15 = svm_tp_15 / (svm_tp_15 + svm_fp_15)
        if svm_tp_16 + svm_fp_16 == 0:
            svm_precision_16 = 0
        else:
            svm_precision_16 = svm_tp_16 / (svm_tp_16 + svm_fp_16)
        if svm_tp_17 + svm_fp_17 == 0:
            svm_precision_17 = 0
        else:
            svm_precision_17 = svm_tp_17 / (svm_tp_17 + svm_fp_17)
        if svm_tp_18 + svm_fp_18 == 0:
            svm_precision_18 = 0
        else:
            svm_precision_18 = svm_tp_18 / (svm_tp_18 + svm_fp_18)
        if svm_tp_19 + svm_fp_19 == 0:
            svm_precision_19 = 0
        else:
            svm_precision_19 = svm_tp_19 / (svm_tp_19 + svm_fp_19)
        if svm_tp_20 + svm_fp_20 == 0:
            svm_precision_20 = 0
        else:
            svm_precision_20 = svm_tp_20 / (svm_tp_20 + svm_fp_20)
        '''
        if svm_tp_21 + svm_fp_21 == 0:
            svm_precision_21 = 0
        else:
            svm_precision_21 = svm_tp_21 / (svm_tp_21 + svm_fp_21)
        if svm_tp_22 + svm_fp_22 == 0:
            svm_precision_22 = 0
        else:
            svm_precision_22 = svm_tp_22 / (svm_tp_22 + svm_fp_22)
        if svm_tp_23 + svm_fp_23 == 0:
            svm_precision_23 = 0
        else:
            svm_precision_23 = svm_tp_23 / (svm_tp_23 + svm_fp_23)
        if svm_tp_24 + svm_fp_24 == 0:
            svm_precision_24 = 0
        else:
            svm_precision_24 = svm_tp_24 / (svm_tp_24 + svm_fp_24)
        if svm_tp_25 + svm_fp_25 == 0:
            svm_precision_25 = 0
        else:
            svm_precision_25 = svm_tp_25 / (svm_tp_25 + svm_fp_25)
        if svm_tp_26 + svm_fp_26 == 0:
            svm_precision_26 = 0
        else:
            svm_precision_26 = svm_tp_26 / (svm_tp_26 + svm_fp_26)
        if svm_tp_27 + svm_fp_27 == 0:
            svm_precision_27 = 0
        else:
            svm_precision_27 = svm_tp_27 / (svm_tp_27 + svm_fp_27)
        if svm_tp_28 + svm_fp_28 == 0:
            svm_precision_28 = 0
        else:
            svm_precision_28 = svm_tp_28 / (svm_tp_28 + svm_fp_28)
        if svm_tp_29 + svm_fp_29 == 0:
            svm_precision_29 = 0
        else:
            svm_precision_29 = svm_tp_29 / (svm_tp_29 + svm_fp_29)
        '''
        svm_precision_avg = (svm_precision_1 + svm_precision_2 + svm_precision_3 + svm_precision_4 + svm_precision_5 + svm_precision_6 + svm_precision_7 + svm_precision_8 + svm_precision_9 + svm_precision_10 + svm_precision_11 + svm_precision_12 + svm_precision_13 + svm_precision_14 + svm_precision_15 + svm_precision_16 + svm_precision_17 + svm_precision_18 + svm_precision_19 + svm_precision_20) / 20
        return svm_precision_avg


    def get_recall_pen_1(svm_conf_matrix):
        svm_tp_1 = svm_conf_matrix[0][0]
        svm_tp_2 = svm_conf_matrix[1][1]
        svm_tp_3 = svm_conf_matrix[2][2]
        svm_tp_4 = svm_conf_matrix[3][3]
        svm_tp_5 = svm_conf_matrix[4][4]
        svm_tp_6 = svm_conf_matrix[5][5]
        svm_tp_7 = svm_conf_matrix[6][6]
        svm_tp_8 = svm_conf_matrix[7][7]
        svm_tp_9 = svm_conf_matrix[8][8]
        svm_tp_10 = svm_conf_matrix[9][9]
        svm_tp_11 = svm_conf_matrix[10][10]
        svm_tp_12 = svm_conf_matrix[11][11]
        svm_tp_13 = svm_conf_matrix[12][12]
        svm_tp_14 = svm_conf_matrix[13][13]
        svm_tp_15 = svm_conf_matrix[14][14]
        svm_tp_16 = svm_conf_matrix[15][15]
        svm_tp_17 = svm_conf_matrix[16][16]
        svm_tp_18 = svm_conf_matrix[17][17]
        svm_tp_19 = svm_conf_matrix[18][18]
        svm_tp_20 = svm_conf_matrix[19][19]
        #svm_tp_21 = svm_conf_matrix[20][20]
        #svm_tp_22 = svm_conf_matrix[21][21]
        #svm_tp_23 = svm_conf_matrix[22][22]
        #svm_tp_24 = svm_conf_matrix[23][23]
        #svm_tp_25 = svm_conf_matrix[24][24]
        #svm_tp_26 = svm_conf_matrix[25][25]
        #svm_tp_27 = svm_conf_matrix[26][26]
        #svm_tp_28 = svm_conf_matrix[27][27]
        #svm_tp_29 = svm_conf_matrix[28][28]

        svm_fn_1 = svm_conf_matrix[0][1] + svm_conf_matrix[0][2] + svm_conf_matrix[0][3] + svm_conf_matrix[0][4] + svm_conf_matrix[0][5] + svm_conf_matrix[0][6] + svm_conf_matrix[0][7] + svm_conf_matrix[0][8] + svm_conf_matrix[0][9] + svm_conf_matrix[0][10] + svm_conf_matrix[0][11] + svm_conf_matrix[0][12] + svm_conf_matrix[0][13] + svm_conf_matrix[0][14] + svm_conf_matrix[0][15] + svm_conf_matrix[0][16] + svm_conf_matrix[0][17] + svm_conf_matrix[0][18] + svm_conf_matrix[0][19]
        svm_fn_2 = svm_conf_matrix[1][0] + svm_conf_matrix[1][2] + svm_conf_matrix[1][3] + svm_conf_matrix[1][4] + svm_conf_matrix[1][5] + svm_conf_matrix[1][6] + svm_conf_matrix[1][7] + svm_conf_matrix[1][8] + svm_conf_matrix[1][9] + svm_conf_matrix[1][10] + svm_conf_matrix[1][11] + svm_conf_matrix[1][12] + svm_conf_matrix[1][13] + svm_conf_matrix[1][14] + svm_conf_matrix[1][15] + svm_conf_matrix[1][16] + svm_conf_matrix[1][17] + svm_conf_matrix[1][18] + svm_conf_matrix[1][19]
        svm_fn_3 = svm_conf_matrix[2][0] + svm_conf_matrix[2][1] + svm_conf_matrix[2][3] + svm_conf_matrix[2][4] + svm_conf_matrix[2][5] + svm_conf_matrix[2][6] + svm_conf_matrix[2][7] + svm_conf_matrix[2][8] + svm_conf_matrix[2][9] + svm_conf_matrix[2][10] + svm_conf_matrix[2][11] + svm_conf_matrix[2][12] + svm_conf_matrix[2][13] + svm_conf_matrix[2][14] + svm_conf_matrix[2][15] + svm_conf_matrix[2][16] + svm_conf_matrix[2][17] + svm_conf_matrix[2][18] + svm_conf_matrix[2][19]
        svm_fn_4 = svm_conf_matrix[3][0] + svm_conf_matrix[3][1] + svm_conf_matrix[3][2] + svm_conf_matrix[3][4] + svm_conf_matrix[3][5] + svm_conf_matrix[3][6] + svm_conf_matrix[3][7] + svm_conf_matrix[3][8] + svm_conf_matrix[3][9] + svm_conf_matrix[3][10] + svm_conf_matrix[3][11] + svm_conf_matrix[3][12] + svm_conf_matrix[3][13] + svm_conf_matrix[3][14] + svm_conf_matrix[3][15] + svm_conf_matrix[3][16] + svm_conf_matrix[3][17] + svm_conf_matrix[3][18] + svm_conf_matrix[3][19]
        svm_fn_5 = svm_conf_matrix[4][0] + svm_conf_matrix[4][1] + svm_conf_matrix[4][2] + svm_conf_matrix[4][3] + svm_conf_matrix[4][5] + svm_conf_matrix[4][6] + svm_conf_matrix[4][7] + svm_conf_matrix[4][8] + svm_conf_matrix[4][9] + svm_conf_matrix[4][10] + svm_conf_matrix[4][11] + svm_conf_matrix[4][12] + svm_conf_matrix[4][13] + svm_conf_matrix[4][14] + svm_conf_matrix[4][15] + svm_conf_matrix[4][16] + svm_conf_matrix[4][17] + svm_conf_matrix[4][18] + svm_conf_matrix[4][19]
        svm_fn_6 = svm_conf_matrix[5][0] + svm_conf_matrix[5][1] + svm_conf_matrix[5][2] + svm_conf_matrix[5][3] + svm_conf_matrix[5][4] + svm_conf_matrix[5][6] + svm_conf_matrix[5][7] + svm_conf_matrix[5][8] + svm_conf_matrix[5][9] + svm_conf_matrix[5][10] + svm_conf_matrix[5][11] + svm_conf_matrix[5][12] + svm_conf_matrix[5][13] + svm_conf_matrix[5][14] + svm_conf_matrix[5][15] + svm_conf_matrix[5][16] + svm_conf_matrix[5][17] + svm_conf_matrix[5][18] + svm_conf_matrix[5][19]
        svm_fn_7 = svm_conf_matrix[6][0] + svm_conf_matrix[6][1] + svm_conf_matrix[6][2] + svm_conf_matrix[6][3] + svm_conf_matrix[6][4] + svm_conf_matrix[6][5] + svm_conf_matrix[6][7] + svm_conf_matrix[6][8] + svm_conf_matrix[6][9] + svm_conf_matrix[6][10] + svm_conf_matrix[6][11] + svm_conf_matrix[6][12] + svm_conf_matrix[6][13] + svm_conf_matrix[6][14] + svm_conf_matrix[6][15] + svm_conf_matrix[6][16] + svm_conf_matrix[6][17] + svm_conf_matrix[6][18] + svm_conf_matrix[6][19]
        svm_fn_8 = svm_conf_matrix[7][0] + svm_conf_matrix[7][1] + svm_conf_matrix[7][2] + svm_conf_matrix[7][3] + svm_conf_matrix[7][4] + svm_conf_matrix[7][5] + svm_conf_matrix[7][6] + svm_conf_matrix[7][8] + svm_conf_matrix[7][9] + svm_conf_matrix[7][10] + svm_conf_matrix[7][11] + svm_conf_matrix[7][12] + svm_conf_matrix[7][13] + svm_conf_matrix[7][14] + svm_conf_matrix[7][15] + svm_conf_matrix[7][16] + svm_conf_matrix[7][17] + svm_conf_matrix[7][18] + svm_conf_matrix[7][19]
        svm_fn_9 = svm_conf_matrix[8][0] + svm_conf_matrix[8][1] + svm_conf_matrix[8][2] + svm_conf_matrix[8][3] + svm_conf_matrix[8][4] + svm_conf_matrix[8][5] + svm_conf_matrix[8][6] + svm_conf_matrix[8][7] + svm_conf_matrix[8][9] + svm_conf_matrix[8][10] + svm_conf_matrix[8][11] + svm_conf_matrix[8][12] + svm_conf_matrix[8][13] + svm_conf_matrix[8][14] + svm_conf_matrix[8][15] + svm_conf_matrix[8][16] + svm_conf_matrix[8][17] + svm_conf_matrix[8][18] + svm_conf_matrix[8][19]
        svm_fn_10 = svm_conf_matrix[9][0] + svm_conf_matrix[9][1] + svm_conf_matrix[9][2] + svm_conf_matrix[9][3] + svm_conf_matrix[9][4] + svm_conf_matrix[9][5] + svm_conf_matrix[9][6] + svm_conf_matrix[9][7] + svm_conf_matrix[9][8] + svm_conf_matrix[9][10] + svm_conf_matrix[9][11] + svm_conf_matrix[9][12] + svm_conf_matrix[9][13] + svm_conf_matrix[9][14] + svm_conf_matrix[9][15] + svm_conf_matrix[9][16] + svm_conf_matrix[9][17] + svm_conf_matrix[9][18] + svm_conf_matrix[9][19]
        svm_fn_11 = svm_conf_matrix[10][0] + svm_conf_matrix[10][1] + svm_conf_matrix[10][2] + svm_conf_matrix[10][3] + svm_conf_matrix[10][4] + svm_conf_matrix[10][5] + svm_conf_matrix[10][6] + svm_conf_matrix[10][7] + svm_conf_matrix[10][8] + svm_conf_matrix[10][9] + svm_conf_matrix[10][11] + svm_conf_matrix[10][12] + svm_conf_matrix[10][13] + svm_conf_matrix[10][14] + svm_conf_matrix[10][15] + svm_conf_matrix[10][16] + svm_conf_matrix[10][17] + svm_conf_matrix[10][18] + svm_conf_matrix[10][19]
        svm_fn_12 = svm_conf_matrix[11][0] + svm_conf_matrix[11][1] + svm_conf_matrix[11][2] + svm_conf_matrix[11][3] + svm_conf_matrix[11][4] + svm_conf_matrix[11][5] + svm_conf_matrix[11][6] + svm_conf_matrix[11][7] + svm_conf_matrix[11][8] + svm_conf_matrix[11][9] + svm_conf_matrix[11][10] + svm_conf_matrix[11][12] + svm_conf_matrix[11][13] + svm_conf_matrix[11][14] + svm_conf_matrix[11][15] + svm_conf_matrix[11][16] + svm_conf_matrix[11][17] + svm_conf_matrix[11][18] + svm_conf_matrix[11][19]
        svm_fn_13 = svm_conf_matrix[12][0] + svm_conf_matrix[12][1] + svm_conf_matrix[12][2] + svm_conf_matrix[12][3] + svm_conf_matrix[12][4] + svm_conf_matrix[12][5] + svm_conf_matrix[12][6] + svm_conf_matrix[12][7] + svm_conf_matrix[12][8] + svm_conf_matrix[12][9] + svm_conf_matrix[12][10] + svm_conf_matrix[12][11] + svm_conf_matrix[12][13] + svm_conf_matrix[12][14] + svm_conf_matrix[12][15] + svm_conf_matrix[12][16] + svm_conf_matrix[12][17] + svm_conf_matrix[12][18] + svm_conf_matrix[12][19]
        svm_fn_14 = svm_conf_matrix[13][0] + svm_conf_matrix[13][1] + svm_conf_matrix[13][2] + svm_conf_matrix[13][3] + svm_conf_matrix[13][4] + svm_conf_matrix[13][5] + svm_conf_matrix[13][6] + svm_conf_matrix[13][7] + svm_conf_matrix[13][8] + svm_conf_matrix[13][9] + svm_conf_matrix[13][10] + svm_conf_matrix[13][11] + svm_conf_matrix[13][12] + svm_conf_matrix[13][14] + svm_conf_matrix[13][15] + svm_conf_matrix[13][16] + svm_conf_matrix[13][17] + svm_conf_matrix[13][18] + svm_conf_matrix[13][19]
        svm_fn_15 = svm_conf_matrix[14][0] + svm_conf_matrix[14][1] + svm_conf_matrix[14][2] + svm_conf_matrix[14][3] + svm_conf_matrix[14][4] + svm_conf_matrix[14][5] + svm_conf_matrix[14][6] + svm_conf_matrix[14][7] + svm_conf_matrix[14][8] + svm_conf_matrix[14][9] + svm_conf_matrix[14][10] + svm_conf_matrix[14][11] + svm_conf_matrix[14][12] + svm_conf_matrix[14][13] + svm_conf_matrix[14][15] + svm_conf_matrix[14][16] + svm_conf_matrix[14][17] + svm_conf_matrix[14][18] + svm_conf_matrix[14][19]
        svm_fn_16 = svm_conf_matrix[15][0] + svm_conf_matrix[15][1] + svm_conf_matrix[15][2] + svm_conf_matrix[15][3] + svm_conf_matrix[15][4] + svm_conf_matrix[15][5] + svm_conf_matrix[15][6] + svm_conf_matrix[15][7] + svm_conf_matrix[15][8] + svm_conf_matrix[15][9] + svm_conf_matrix[15][10] + svm_conf_matrix[15][11] + svm_conf_matrix[15][12] + svm_conf_matrix[15][13] + svm_conf_matrix[15][14] + svm_conf_matrix[15][16] + svm_conf_matrix[15][17] + svm_conf_matrix[15][18] + svm_conf_matrix[15][19]
        svm_fn_17 = svm_conf_matrix[16][0] + svm_conf_matrix[16][1] + svm_conf_matrix[16][2] + svm_conf_matrix[16][3] + svm_conf_matrix[16][4] + svm_conf_matrix[16][5] + svm_conf_matrix[16][6] + svm_conf_matrix[16][7] + svm_conf_matrix[16][8] + svm_conf_matrix[16][9] + svm_conf_matrix[16][10] + svm_conf_matrix[16][11] + svm_conf_matrix[16][12] + svm_conf_matrix[16][13] + svm_conf_matrix[16][14] + svm_conf_matrix[16][15] + svm_conf_matrix[16][17] + svm_conf_matrix[16][18] + svm_conf_matrix[16][19]
        svm_fn_18 = svm_conf_matrix[17][0] + svm_conf_matrix[17][1] + svm_conf_matrix[17][2] + svm_conf_matrix[17][3] + svm_conf_matrix[17][4] + svm_conf_matrix[17][5] + svm_conf_matrix[17][6] + svm_conf_matrix[17][7] + svm_conf_matrix[17][8] + svm_conf_matrix[17][9] + svm_conf_matrix[17][10] + svm_conf_matrix[17][11] + svm_conf_matrix[17][12] + svm_conf_matrix[17][13] + svm_conf_matrix[17][14] + svm_conf_matrix[17][15] + svm_conf_matrix[17][16] + svm_conf_matrix[17][18] + svm_conf_matrix[17][19]
        svm_fn_19 = svm_conf_matrix[18][0] + svm_conf_matrix[18][1] + svm_conf_matrix[18][2] + svm_conf_matrix[18][3] + svm_conf_matrix[18][4] + svm_conf_matrix[18][5] + svm_conf_matrix[18][6] + svm_conf_matrix[18][7] + svm_conf_matrix[18][8] + svm_conf_matrix[18][9] + svm_conf_matrix[18][10] + svm_conf_matrix[18][11] + svm_conf_matrix[18][12] + svm_conf_matrix[18][13] + svm_conf_matrix[18][14] + svm_conf_matrix[18][15] + svm_conf_matrix[18][16] + svm_conf_matrix[18][17] + svm_conf_matrix[18][19]
        svm_fn_20 = svm_conf_matrix[19][0] + svm_conf_matrix[19][1] + svm_conf_matrix[19][2] + svm_conf_matrix[19][3] + svm_conf_matrix[19][4] + svm_conf_matrix[19][5] + svm_conf_matrix[19][6] + svm_conf_matrix[19][7] + svm_conf_matrix[19][8] + svm_conf_matrix[19][9] + svm_conf_matrix[19][10] + svm_conf_matrix[19][11] + svm_conf_matrix[19][12] + svm_conf_matrix[19][13] + svm_conf_matrix[19][14] + svm_conf_matrix[19][15] + svm_conf_matrix[19][16] + svm_conf_matrix[19][17] + svm_conf_matrix[19][18]
        #svm_fn_21 = svm_conf_matrix[20][0] + svm_conf_matrix[20][1] + svm_conf_matrix[20][2] + svm_conf_matrix[20][3] + svm_conf_matrix[20][4] + svm_conf_matrix[20][5] + svm_conf_matrix[20][6] + svm_conf_matrix[20][7] + svm_conf_matrix[20][8] + svm_conf_matrix[20][9] + svm_conf_matrix[20][10] + svm_conf_matrix[20][11] + svm_conf_matrix[20][12] + svm_conf_matrix[20][13] + svm_conf_matrix[20][14] + svm_conf_matrix[20][15] + svm_conf_matrix[20][16] + svm_conf_matrix[20][17] + svm_conf_matrix[20][18] + svm_conf_matrix[20][19] + svm_conf_matrix[20][21] + svm_conf_matrix[20][22] + svm_conf_matrix[20][23] + svm_conf_matrix[20][24] + svm_conf_matrix[20][25] + svm_conf_matrix[20][26] + svm_conf_matrix[20][27] + svm_conf_matrix[20][28]
        #svm_fn_22 = svm_conf_matrix[21][0] + svm_conf_matrix[21][1] + svm_conf_matrix[21][2] + svm_conf_matrix[21][3] + svm_conf_matrix[21][4] + svm_conf_matrix[21][5] + svm_conf_matrix[21][6] + svm_conf_matrix[21][7] + svm_conf_matrix[21][8] + svm_conf_matrix[21][9] + svm_conf_matrix[21][10] + svm_conf_matrix[21][11] + svm_conf_matrix[21][12] + svm_conf_matrix[21][13] + svm_conf_matrix[21][14] + svm_conf_matrix[21][15] + svm_conf_matrix[21][16] + svm_conf_matrix[21][17] + svm_conf_matrix[21][18] + svm_conf_matrix[21][19] + svm_conf_matrix[21][20] + svm_conf_matrix[21][22] + svm_conf_matrix[21][23] + svm_conf_matrix[21][24] + svm_conf_matrix[21][25] + svm_conf_matrix[21][26] + svm_conf_matrix[21][27] + svm_conf_matrix[21][28]
        #svm_fn_23 = svm_conf_matrix[22][0] + svm_conf_matrix[22][1] + svm_conf_matrix[22][2] + svm_conf_matrix[22][3] + svm_conf_matrix[22][4] + svm_conf_matrix[22][5] + svm_conf_matrix[22][6] + svm_conf_matrix[22][7] + svm_conf_matrix[22][8] + svm_conf_matrix[22][9] + svm_conf_matrix[22][10] + svm_conf_matrix[22][11] + svm_conf_matrix[22][12] + svm_conf_matrix[22][13] + svm_conf_matrix[22][14] + svm_conf_matrix[22][15] + svm_conf_matrix[22][16] + svm_conf_matrix[22][17] + svm_conf_matrix[22][18] + svm_conf_matrix[22][19] + svm_conf_matrix[22][20] + svm_conf_matrix[22][21] + svm_conf_matrix[22][23] + svm_conf_matrix[22][24] + svm_conf_matrix[22][25] + svm_conf_matrix[22][26] + svm_conf_matrix[22][27] + svm_conf_matrix[22][28]
        #svm_fn_24 = svm_conf_matrix[23][0] + svm_conf_matrix[23][1] + svm_conf_matrix[23][2] + svm_conf_matrix[23][3] + svm_conf_matrix[23][4] + svm_conf_matrix[23][5] + svm_conf_matrix[23][6] + svm_conf_matrix[23][7] + svm_conf_matrix[23][8] + svm_conf_matrix[23][9] + svm_conf_matrix[23][10] + svm_conf_matrix[23][11] + svm_conf_matrix[23][12] + svm_conf_matrix[23][13] + svm_conf_matrix[23][14] + svm_conf_matrix[23][15] + svm_conf_matrix[23][16] + svm_conf_matrix[23][17] + svm_conf_matrix[23][18] + svm_conf_matrix[23][19] + svm_conf_matrix[23][20] + svm_conf_matrix[23][21] + svm_conf_matrix[23][22] + svm_conf_matrix[23][24] + svm_conf_matrix[23][25] + svm_conf_matrix[23][26] + svm_conf_matrix[23][27] + svm_conf_matrix[23][28]
        #svm_fn_25 = svm_conf_matrix[24][0] + svm_conf_matrix[24][1] + svm_conf_matrix[24][2] + svm_conf_matrix[24][3] + svm_conf_matrix[24][4] + svm_conf_matrix[24][5] + svm_conf_matrix[24][6] + svm_conf_matrix[24][7] + svm_conf_matrix[24][8] + svm_conf_matrix[24][9] + svm_conf_matrix[24][10] + svm_conf_matrix[24][11] + svm_conf_matrix[24][12] + svm_conf_matrix[24][13] + svm_conf_matrix[24][14] + svm_conf_matrix[24][15] + svm_conf_matrix[24][16] + svm_conf_matrix[24][17] + svm_conf_matrix[24][18] + svm_conf_matrix[24][19] + svm_conf_matrix[24][20] + svm_conf_matrix[24][21] + svm_conf_matrix[24][22] + svm_conf_matrix[24][23] + svm_conf_matrix[24][25] + svm_conf_matrix[24][26] + svm_conf_matrix[24][27] + svm_conf_matrix[24][28]
        #svm_fn_26 = svm_conf_matrix[25][0] + svm_conf_matrix[25][1] + svm_conf_matrix[25][2] + svm_conf_matrix[25][3] + svm_conf_matrix[25][4] + svm_conf_matrix[25][5] + svm_conf_matrix[25][6] + svm_conf_matrix[25][7] + svm_conf_matrix[25][8] + svm_conf_matrix[25][9] + svm_conf_matrix[25][10] + svm_conf_matrix[25][11] + svm_conf_matrix[25][12] + svm_conf_matrix[25][13] + svm_conf_matrix[25][14] + svm_conf_matrix[25][15] + svm_conf_matrix[25][16] + svm_conf_matrix[25][17] + svm_conf_matrix[25][18] + svm_conf_matrix[25][19] + svm_conf_matrix[25][20] + svm_conf_matrix[25][21] + svm_conf_matrix[25][22] + svm_conf_matrix[25][23] + svm_conf_matrix[25][24] + svm_conf_matrix[25][26] + svm_conf_matrix[25][27] + svm_conf_matrix[25][28]
        #svm_fn_27 = svm_conf_matrix[26][0] + svm_conf_matrix[26][1] + svm_conf_matrix[26][2] + svm_conf_matrix[26][3] + svm_conf_matrix[26][4] + svm_conf_matrix[26][5] + svm_conf_matrix[26][6] + svm_conf_matrix[26][7] + svm_conf_matrix[26][8] + svm_conf_matrix[26][9] + svm_conf_matrix[26][10] + svm_conf_matrix[26][11] + svm_conf_matrix[26][12] + svm_conf_matrix[26][13] + svm_conf_matrix[26][14] + svm_conf_matrix[26][15] + svm_conf_matrix[26][16] + svm_conf_matrix[26][17] + svm_conf_matrix[26][18] + svm_conf_matrix[26][19] + svm_conf_matrix[26][20] + svm_conf_matrix[26][21] + svm_conf_matrix[26][22] + svm_conf_matrix[26][23] + svm_conf_matrix[26][24] + svm_conf_matrix[26][25] + svm_conf_matrix[26][27] + svm_conf_matrix[26][28]
        #svm_fn_28 = svm_conf_matrix[27][0] + svm_conf_matrix[27][1] + svm_conf_matrix[27][2] + svm_conf_matrix[27][3] + svm_conf_matrix[27][4] + svm_conf_matrix[27][5] + svm_conf_matrix[27][6] + svm_conf_matrix[27][7] + svm_conf_matrix[27][8] + svm_conf_matrix[27][9] + svm_conf_matrix[27][10] + svm_conf_matrix[27][11] + svm_conf_matrix[27][12] + svm_conf_matrix[27][13] + svm_conf_matrix[27][14] + svm_conf_matrix[27][15] + svm_conf_matrix[27][16] + svm_conf_matrix[27][17] + svm_conf_matrix[27][18] + svm_conf_matrix[27][19] + svm_conf_matrix[27][20] + svm_conf_matrix[27][21] + svm_conf_matrix[27][22] + svm_conf_matrix[27][23] + svm_conf_matrix[27][24] + svm_conf_matrix[27][25] + svm_conf_matrix[27][26] + svm_conf_matrix[27][28]
        #svm_fn_29 = svm_conf_matrix[28][0] + svm_conf_matrix[28][1] + svm_conf_matrix[28][2] + svm_conf_matrix[28][3] + svm_conf_matrix[28][4] + svm_conf_matrix[28][5] + svm_conf_matrix[28][6] + svm_conf_matrix[28][7] + svm_conf_matrix[28][8] + svm_conf_matrix[28][9] + svm_conf_matrix[28][10] + svm_conf_matrix[28][11] + svm_conf_matrix[28][12] + svm_conf_matrix[28][13] + svm_conf_matrix[28][14] + svm_conf_matrix[28][15] + svm_conf_matrix[28][16] + svm_conf_matrix[28][17] + svm_conf_matrix[28][18] + svm_conf_matrix[28][19] + svm_conf_matrix[28][20] + svm_conf_matrix[28][21] + svm_conf_matrix[28][22] + svm_conf_matrix[28][23] + svm_conf_matrix[28][24] + svm_conf_matrix[28][25] + svm_conf_matrix[28][26] + svm_conf_matrix[28][27]

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
            svm_recall_5 = svm_tp_5 / (svm_tp_5 + svm_fn_5)
        if svm_tp_6 + svm_fn_6 == 0:
            svm_recall_6 = 0
        else:
            svm_recall_6 = svm_tp_6 / (svm_tp_6 + svm_fn_6)
        if svm_tp_7 + svm_fn_7 == 0:
            svm_recall_7 = 0
        else:
            svm_recall_7 = svm_tp_7 / (svm_tp_7 + svm_fn_7)
        if svm_tp_8 + svm_fn_8 == 0:
            svm_recall_8 = 0
        else:
            svm_recall_8 = svm_tp_8 / (svm_tp_8 + svm_fn_8)
        if svm_tp_9 + svm_fn_9 == 0:
            svm_recall_9 = 0
        else:
            svm_recall_9 = svm_tp_9 / (svm_tp_9 + svm_fn_9)
        if svm_tp_10 + svm_fn_10 == 0:
            svm_recall_10 = 0
        else:
            svm_recall_10 = svm_tp_10 / (svm_tp_10 + svm_fn_10)
        if svm_tp_11 + svm_fn_11 == 0:
            svm_recall_11 = 0
        else:
            svm_recall_11 = svm_tp_11 / (svm_tp_11 + svm_fn_11)
        if svm_tp_12 + svm_fn_12 == 0:
            svm_recall_12 = 0
        else:
            svm_recall_12 = svm_tp_12 / (svm_tp_12 + svm_fn_12)
        if svm_tp_13 + svm_fn_13 == 0:
            svm_recall_13 = 0
        else:
            svm_recall_13 = svm_tp_13 / (svm_tp_13 + svm_fn_13)
        if svm_tp_14 + svm_fn_14 == 0:
            svm_recall_14 = 0
        else:
            svm_recall_14 = svm_tp_14 / (svm_tp_14 + svm_fn_14)
        if svm_tp_15 + svm_fn_15 == 0:
            svm_recall_15 = 0
        else:
            svm_recall_15 = svm_tp_15 / (svm_tp_15 + svm_fn_15)
        if svm_tp_16 + svm_fn_16 == 0:
            svm_recall_16 = 0
        else:
            svm_recall_16 = svm_tp_16 / (svm_tp_16 + svm_fn_16)
        if svm_tp_17 + svm_fn_17 == 0:
            svm_recall_17 = 0
        else:
            svm_recall_17 = svm_tp_17 / (svm_tp_17 + svm_fn_17)
        if svm_tp_18 + svm_fn_18 == 0:
            svm_recall_18 = 0
        else:
            svm_recall_18 = svm_tp_18 / (svm_tp_18 + svm_fn_18)
        if svm_tp_19 + svm_fn_19 == 0:
            svm_recall_19 = 0
        else:
            svm_recall_19 = svm_tp_19 / (svm_tp_19 + svm_fn_19)
        if svm_tp_20 + svm_fn_20 == 0:
            svm_recall_20 = 0
        else:
            svm_recall_20 = svm_tp_20 / (svm_tp_20 + svm_fn_20)
        '''
        if svm_tp_21 + svm_fn_21 == 0:
            svm_recall_21 = 0
        else:
            svm_recall_21 = svm_tp_21 / (svm_tp_21 + svm_fn_21)
        if svm_tp_22 + svm_fn_22 == 0:
            svm_recall_22 = 0
        else:
            svm_recall_22 = svm_tp_22 / (svm_tp_22 + svm_fn_22)
        if svm_tp_23 + svm_fn_23 == 0:
            svm_recall_23 = 0
        else:
            svm_recall_23 = svm_tp_23 / (svm_tp_23 + svm_fn_23)
        if svm_tp_24 + svm_fn_24 == 0:
            svm_recall_24 = 0
        else:
            svm_recall_24 = svm_tp_24 / (svm_tp_24 + svm_fn_24)
        if svm_tp_25 + svm_fn_25 == 0:
            svm_recall_25 = 0
        else:
            svm_recall_25 = svm_tp_25 / (svm_tp_25 + svm_fn_25)
        if svm_tp_26 + svm_fn_26 == 0:
            svm_recall_26 = 0
        else:
            svm_recall_26 = svm_tp_26 / (svm_tp_26 + svm_fn_26)
        if svm_tp_27 + svm_fn_27 == 0:
            svm_recall_27 = 0
        else:
            svm_recall_27 = svm_tp_27 / (svm_tp_27 + svm_fn_27)
        if svm_tp_28 + svm_fn_28 == 0:
            svm_recall_28 = 0
        else:
            svm_recall_28 = svm_tp_28 / (svm_tp_28 + svm_fn_28)
        if svm_tp_29 + svm_fn_29 == 0:
            svm_recall_29 = 0
        else:
            svm_recall_29 = svm_tp_29 / (svm_tp_29 + svm_fn_29)
        '''
        svm_recall_avg_pen_1 = (
                                 svm_recall_1 + svm_recall_2 + svm_recall_3 + svm_recall_4 + svm_recall_5 + svm_recall_6 + svm_recall_7 + svm_recall_8 + svm_recall_9 + svm_recall_10 + svm_recall_11 + svm_recall_12 + svm_recall_13 + svm_recall_14 + svm_recall_15 + svm_recall_16 + svm_recall_17 + svm_recall_18 + svm_recall_19 + svm_recall_20) / (20+1-1)
        return svm_recall_avg_pen_1

    def get_recall_pen_5(svm_conf_matrix):
        svm_tp_1 = svm_conf_matrix[0][0]
        svm_tp_2 = svm_conf_matrix[1][1]
        svm_tp_3 = svm_conf_matrix[2][2]
        svm_tp_4 = svm_conf_matrix[3][3]
        svm_tp_5 = svm_conf_matrix[4][4]
        svm_tp_6 = svm_conf_matrix[5][5]
        svm_tp_7 = svm_conf_matrix[6][6]
        svm_tp_8 = svm_conf_matrix[7][7]
        svm_tp_9 = svm_conf_matrix[8][8]
        svm_tp_10 = svm_conf_matrix[9][9]
        svm_tp_11 = svm_conf_matrix[10][10]
        svm_tp_12 = svm_conf_matrix[11][11]
        svm_tp_13 = svm_conf_matrix[12][12]
        svm_tp_14 = svm_conf_matrix[13][13]
        svm_tp_15 = svm_conf_matrix[14][14]
        svm_tp_16 = svm_conf_matrix[15][15]
        svm_tp_17 = svm_conf_matrix[16][16]
        svm_tp_18 = svm_conf_matrix[17][17]
        svm_tp_19 = svm_conf_matrix[18][18]
        svm_tp_20 = svm_conf_matrix[19][19]
        #svm_tp_21 = svm_conf_matrix[20][20]
        #svm_tp_22 = svm_conf_matrix[21][21]
        #svm_tp_23 = svm_conf_matrix[22][22]
        #svm_tp_24 = svm_conf_matrix[23][23]
        #svm_tp_25 = svm_conf_matrix[24][24]
        #svm_tp_26 = svm_conf_matrix[25][25]
        #svm_tp_27 = svm_conf_matrix[26][26]
        #svm_tp_28 = svm_conf_matrix[27][27]
        #svm_tp_29 = svm_conf_matrix[28][28]

        svm_fn_1 = svm_conf_matrix[0][1] + svm_conf_matrix[0][2] + svm_conf_matrix[0][3] + svm_conf_matrix[0][4] + svm_conf_matrix[0][5] + svm_conf_matrix[0][6] + svm_conf_matrix[0][7] + svm_conf_matrix[0][8] + svm_conf_matrix[0][9] + svm_conf_matrix[0][10] + svm_conf_matrix[0][11] + svm_conf_matrix[0][12] + svm_conf_matrix[0][13] + svm_conf_matrix[0][14] + svm_conf_matrix[0][15] + svm_conf_matrix[0][16] + svm_conf_matrix[0][17] + svm_conf_matrix[0][18] + svm_conf_matrix[0][19]
        svm_fn_2 = svm_conf_matrix[1][0] + svm_conf_matrix[1][2] + svm_conf_matrix[1][3] + svm_conf_matrix[1][4] + svm_conf_matrix[1][5] + svm_conf_matrix[1][6] + svm_conf_matrix[1][7] + svm_conf_matrix[1][8] + svm_conf_matrix[1][9] + svm_conf_matrix[1][10] + svm_conf_matrix[1][11] + svm_conf_matrix[1][12] + svm_conf_matrix[1][13] + svm_conf_matrix[1][14] + svm_conf_matrix[1][15] + svm_conf_matrix[1][16] + svm_conf_matrix[1][17] + svm_conf_matrix[1][18] + svm_conf_matrix[1][19]
        svm_fn_3 = svm_conf_matrix[2][0] + svm_conf_matrix[2][1] + svm_conf_matrix[2][3] + svm_conf_matrix[2][4] + svm_conf_matrix[2][5] + svm_conf_matrix[2][6] + svm_conf_matrix[2][7] + svm_conf_matrix[2][8] + svm_conf_matrix[2][9] + svm_conf_matrix[2][10] + svm_conf_matrix[2][11] + svm_conf_matrix[2][12] + svm_conf_matrix[2][13] + svm_conf_matrix[2][14] + svm_conf_matrix[2][15] + svm_conf_matrix[2][16] + svm_conf_matrix[2][17] + svm_conf_matrix[2][18] + svm_conf_matrix[2][19]
        svm_fn_4 = svm_conf_matrix[3][0] + svm_conf_matrix[3][1] + svm_conf_matrix[3][2] + svm_conf_matrix[3][4] + svm_conf_matrix[3][5] + svm_conf_matrix[3][6] + svm_conf_matrix[3][7] + svm_conf_matrix[3][8] + svm_conf_matrix[3][9] + svm_conf_matrix[3][10] + svm_conf_matrix[3][11] + svm_conf_matrix[3][12] + svm_conf_matrix[3][13] + svm_conf_matrix[3][14] + svm_conf_matrix[3][15] + svm_conf_matrix[3][16] + svm_conf_matrix[3][17] + svm_conf_matrix[3][18] + svm_conf_matrix[3][19]
        svm_fn_5 = svm_conf_matrix[4][0] + svm_conf_matrix[4][1] + svm_conf_matrix[4][2] + svm_conf_matrix[4][3] + svm_conf_matrix[4][5] + svm_conf_matrix[4][6] + svm_conf_matrix[4][7] + svm_conf_matrix[4][8] + svm_conf_matrix[4][9] + svm_conf_matrix[4][10] + svm_conf_matrix[4][11] + svm_conf_matrix[4][12] + svm_conf_matrix[4][13] + svm_conf_matrix[4][14] + svm_conf_matrix[4][15] + svm_conf_matrix[4][16] + svm_conf_matrix[4][17] + svm_conf_matrix[4][18] + svm_conf_matrix[4][19]
        svm_fn_6 = svm_conf_matrix[5][0] + svm_conf_matrix[5][1] + svm_conf_matrix[5][2] + svm_conf_matrix[5][3] + svm_conf_matrix[5][4] + svm_conf_matrix[5][6] + svm_conf_matrix[5][7] + svm_conf_matrix[5][8] + svm_conf_matrix[5][9] + svm_conf_matrix[5][10] + svm_conf_matrix[5][11] + svm_conf_matrix[5][12] + svm_conf_matrix[5][13] + svm_conf_matrix[5][14] + svm_conf_matrix[5][15] + svm_conf_matrix[5][16] + svm_conf_matrix[5][17] + svm_conf_matrix[5][18] + svm_conf_matrix[5][19]
        svm_fn_7 = svm_conf_matrix[6][0] + svm_conf_matrix[6][1] + svm_conf_matrix[6][2] + svm_conf_matrix[6][3] + svm_conf_matrix[6][4] + svm_conf_matrix[6][5] + svm_conf_matrix[6][7] + svm_conf_matrix[6][8] + svm_conf_matrix[6][9] + svm_conf_matrix[6][10] + svm_conf_matrix[6][11] + svm_conf_matrix[6][12] + svm_conf_matrix[6][13] + svm_conf_matrix[6][14] + svm_conf_matrix[6][15] + svm_conf_matrix[6][16] + svm_conf_matrix[6][17] + svm_conf_matrix[6][18] + svm_conf_matrix[6][19]
        svm_fn_8 = svm_conf_matrix[7][0] + svm_conf_matrix[7][1] + svm_conf_matrix[7][2] + svm_conf_matrix[7][3] + svm_conf_matrix[7][4] + svm_conf_matrix[7][5] + svm_conf_matrix[7][6] + svm_conf_matrix[7][8] + svm_conf_matrix[7][9] + svm_conf_matrix[7][10] + svm_conf_matrix[7][11] + svm_conf_matrix[7][12] + svm_conf_matrix[7][13] + svm_conf_matrix[7][14] + svm_conf_matrix[7][15] + svm_conf_matrix[7][16] + svm_conf_matrix[7][17] + svm_conf_matrix[7][18] + svm_conf_matrix[7][19]
        svm_fn_9 = svm_conf_matrix[8][0] + svm_conf_matrix[8][1] + svm_conf_matrix[8][2] + svm_conf_matrix[8][3] + svm_conf_matrix[8][4] + svm_conf_matrix[8][5] + svm_conf_matrix[8][6] + svm_conf_matrix[8][7] + svm_conf_matrix[8][9] + svm_conf_matrix[8][10] + svm_conf_matrix[8][11] + svm_conf_matrix[8][12] + svm_conf_matrix[8][13] + svm_conf_matrix[8][14] + svm_conf_matrix[8][15] + svm_conf_matrix[8][16] + svm_conf_matrix[8][17] + svm_conf_matrix[8][18] + svm_conf_matrix[8][19]
        svm_fn_10 = svm_conf_matrix[9][0] + svm_conf_matrix[9][1] + svm_conf_matrix[9][2] + svm_conf_matrix[9][3] + svm_conf_matrix[9][4] + svm_conf_matrix[9][5] + svm_conf_matrix[9][6] + svm_conf_matrix[9][7] + svm_conf_matrix[9][8] + svm_conf_matrix[9][10] + svm_conf_matrix[9][11] + svm_conf_matrix[9][12] + svm_conf_matrix[9][13] + svm_conf_matrix[9][14] + svm_conf_matrix[9][15] + svm_conf_matrix[9][16] + svm_conf_matrix[9][17] + svm_conf_matrix[9][18] + svm_conf_matrix[9][19]
        svm_fn_11 = svm_conf_matrix[10][0] + svm_conf_matrix[10][1] + svm_conf_matrix[10][2] + svm_conf_matrix[10][3] + svm_conf_matrix[10][4] + svm_conf_matrix[10][5] + svm_conf_matrix[10][6] + svm_conf_matrix[10][7] + svm_conf_matrix[10][8] + svm_conf_matrix[10][9] + svm_conf_matrix[10][11] + svm_conf_matrix[10][12] + svm_conf_matrix[10][13] + svm_conf_matrix[10][14] + svm_conf_matrix[10][15] + svm_conf_matrix[10][16] + svm_conf_matrix[10][17] + svm_conf_matrix[10][18] + svm_conf_matrix[10][19]
        svm_fn_12 = svm_conf_matrix[11][0] + svm_conf_matrix[11][1] + svm_conf_matrix[11][2] + svm_conf_matrix[11][3] + svm_conf_matrix[11][4] + svm_conf_matrix[11][5] + svm_conf_matrix[11][6] + svm_conf_matrix[11][7] + svm_conf_matrix[11][8] + svm_conf_matrix[11][9] + svm_conf_matrix[11][10] + svm_conf_matrix[11][12] + svm_conf_matrix[11][13] + svm_conf_matrix[11][14] + svm_conf_matrix[11][15] + svm_conf_matrix[11][16] + svm_conf_matrix[11][17] + svm_conf_matrix[11][18] + svm_conf_matrix[11][19]
        svm_fn_13 = svm_conf_matrix[12][0] + svm_conf_matrix[12][1] + svm_conf_matrix[12][2] + svm_conf_matrix[12][3] + svm_conf_matrix[12][4] + svm_conf_matrix[12][5] + svm_conf_matrix[12][6] + svm_conf_matrix[12][7] + svm_conf_matrix[12][8] + svm_conf_matrix[12][9] + svm_conf_matrix[12][10] + svm_conf_matrix[12][11] + svm_conf_matrix[12][13] + svm_conf_matrix[12][14] + svm_conf_matrix[12][15] + svm_conf_matrix[12][16] + svm_conf_matrix[12][17] + svm_conf_matrix[12][18] + svm_conf_matrix[12][19]
        svm_fn_14 = svm_conf_matrix[13][0] + svm_conf_matrix[13][1] + svm_conf_matrix[13][2] + svm_conf_matrix[13][3] + svm_conf_matrix[13][4] + svm_conf_matrix[13][5] + svm_conf_matrix[13][6] + svm_conf_matrix[13][7] + svm_conf_matrix[13][8] + svm_conf_matrix[13][9] + svm_conf_matrix[13][10] + svm_conf_matrix[13][11] + svm_conf_matrix[13][12] + svm_conf_matrix[13][14] + svm_conf_matrix[13][15] + svm_conf_matrix[13][16] + svm_conf_matrix[13][17] + svm_conf_matrix[13][18] + svm_conf_matrix[13][19]
        svm_fn_15 = svm_conf_matrix[14][0] + svm_conf_matrix[14][1] + svm_conf_matrix[14][2] + svm_conf_matrix[14][3] + svm_conf_matrix[14][4] + svm_conf_matrix[14][5] + svm_conf_matrix[14][6] + svm_conf_matrix[14][7] + svm_conf_matrix[14][8] + svm_conf_matrix[14][9] + svm_conf_matrix[14][10] + svm_conf_matrix[14][11] + svm_conf_matrix[14][12] + svm_conf_matrix[14][13] + svm_conf_matrix[14][15] + svm_conf_matrix[14][16] + svm_conf_matrix[14][17] + svm_conf_matrix[14][18] + svm_conf_matrix[14][19]
        svm_fn_16 = svm_conf_matrix[15][0] + svm_conf_matrix[15][1] + svm_conf_matrix[15][2] + svm_conf_matrix[15][3] + svm_conf_matrix[15][4] + svm_conf_matrix[15][5] + svm_conf_matrix[15][6] + svm_conf_matrix[15][7] + svm_conf_matrix[15][8] + svm_conf_matrix[15][9] + svm_conf_matrix[15][10] + svm_conf_matrix[15][11] + svm_conf_matrix[15][12] + svm_conf_matrix[15][13] + svm_conf_matrix[15][14] + svm_conf_matrix[15][16] + svm_conf_matrix[15][17] + svm_conf_matrix[15][18] + svm_conf_matrix[15][19]
        svm_fn_17 = svm_conf_matrix[16][0] + svm_conf_matrix[16][1] + svm_conf_matrix[16][2] + svm_conf_matrix[16][3] + svm_conf_matrix[16][4] + svm_conf_matrix[16][5] + svm_conf_matrix[16][6] + svm_conf_matrix[16][7] + svm_conf_matrix[16][8] + svm_conf_matrix[16][9] + svm_conf_matrix[16][10] + svm_conf_matrix[16][11] + svm_conf_matrix[16][12] + svm_conf_matrix[16][13] + svm_conf_matrix[16][14] + svm_conf_matrix[16][15] + svm_conf_matrix[16][17] + svm_conf_matrix[16][18] + svm_conf_matrix[16][19]
        svm_fn_18 = svm_conf_matrix[17][0] + svm_conf_matrix[17][1] + svm_conf_matrix[17][2] + svm_conf_matrix[17][3] + svm_conf_matrix[17][4] + svm_conf_matrix[17][5] + svm_conf_matrix[17][6] + svm_conf_matrix[17][7] + svm_conf_matrix[17][8] + svm_conf_matrix[17][9] + svm_conf_matrix[17][10] + svm_conf_matrix[17][11] + svm_conf_matrix[17][12] + svm_conf_matrix[17][13] + svm_conf_matrix[17][14] + svm_conf_matrix[17][15] + svm_conf_matrix[17][16] + svm_conf_matrix[17][18] + svm_conf_matrix[17][19]
        svm_fn_19 = svm_conf_matrix[18][0] + svm_conf_matrix[18][1] + svm_conf_matrix[18][2] + svm_conf_matrix[18][3] + svm_conf_matrix[18][4] + svm_conf_matrix[18][5] + svm_conf_matrix[18][6] + svm_conf_matrix[18][7] + svm_conf_matrix[18][8] + svm_conf_matrix[18][9] + svm_conf_matrix[18][10] + svm_conf_matrix[18][11] + svm_conf_matrix[18][12] + svm_conf_matrix[18][13] + svm_conf_matrix[18][14] + svm_conf_matrix[18][15] + svm_conf_matrix[18][16] + svm_conf_matrix[18][17] + svm_conf_matrix[18][19]
        svm_fn_20 = svm_conf_matrix[19][0] + svm_conf_matrix[19][1] + svm_conf_matrix[19][2] + svm_conf_matrix[19][3] + svm_conf_matrix[19][4] + svm_conf_matrix[19][5] + svm_conf_matrix[19][6] + svm_conf_matrix[19][7] + svm_conf_matrix[19][8] + svm_conf_matrix[19][9] + svm_conf_matrix[19][10] + svm_conf_matrix[19][11] + svm_conf_matrix[19][12] + svm_conf_matrix[19][13] + svm_conf_matrix[19][14] + svm_conf_matrix[19][15] + svm_conf_matrix[19][16] + svm_conf_matrix[19][17] + svm_conf_matrix[19][18]
        #svm_fn_21 = svm_conf_matrix[20][0] + svm_conf_matrix[20][1] + svm_conf_matrix[20][2] + svm_conf_matrix[20][3] + svm_conf_matrix[20][4] + svm_conf_matrix[20][5] + svm_conf_matrix[20][6] + svm_conf_matrix[20][7] + svm_conf_matrix[20][8] + svm_conf_matrix[20][9] + svm_conf_matrix[20][10] + svm_conf_matrix[20][11] + svm_conf_matrix[20][12] + svm_conf_matrix[20][13] + svm_conf_matrix[20][14] + svm_conf_matrix[20][15] + svm_conf_matrix[20][16] + svm_conf_matrix[20][17] + svm_conf_matrix[20][18] + svm_conf_matrix[20][19] + svm_conf_matrix[20][21] + svm_conf_matrix[20][22] + svm_conf_matrix[20][23] + svm_conf_matrix[20][24] + svm_conf_matrix[20][25] + svm_conf_matrix[20][26] + svm_conf_matrix[20][27] + svm_conf_matrix[20][28]
        #svm_fn_22 = svm_conf_matrix[21][0] + svm_conf_matrix[21][1] + svm_conf_matrix[21][2] + svm_conf_matrix[21][3] + svm_conf_matrix[21][4] + svm_conf_matrix[21][5] + svm_conf_matrix[21][6] + svm_conf_matrix[21][7] + svm_conf_matrix[21][8] + svm_conf_matrix[21][9] + svm_conf_matrix[21][10] + svm_conf_matrix[21][11] + svm_conf_matrix[21][12] + svm_conf_matrix[21][13] + svm_conf_matrix[21][14] + svm_conf_matrix[21][15] + svm_conf_matrix[21][16] + svm_conf_matrix[21][17] + svm_conf_matrix[21][18] + svm_conf_matrix[21][19] + svm_conf_matrix[21][20] + svm_conf_matrix[21][22] + svm_conf_matrix[21][23] + svm_conf_matrix[21][24] + svm_conf_matrix[21][25] + svm_conf_matrix[21][26] + svm_conf_matrix[21][27] + svm_conf_matrix[21][28]
        #svm_fn_23 = svm_conf_matrix[22][0] + svm_conf_matrix[22][1] + svm_conf_matrix[22][2] + svm_conf_matrix[22][3] + svm_conf_matrix[22][4] + svm_conf_matrix[22][5] + svm_conf_matrix[22][6] + svm_conf_matrix[22][7] + svm_conf_matrix[22][8] + svm_conf_matrix[22][9] + svm_conf_matrix[22][10] + svm_conf_matrix[22][11] + svm_conf_matrix[22][12] + svm_conf_matrix[22][13] + svm_conf_matrix[22][14] + svm_conf_matrix[22][15] + svm_conf_matrix[22][16] + svm_conf_matrix[22][17] + svm_conf_matrix[22][18] + svm_conf_matrix[22][19] + svm_conf_matrix[22][20] + svm_conf_matrix[22][21] + svm_conf_matrix[22][23] + svm_conf_matrix[22][24] + svm_conf_matrix[22][25] + svm_conf_matrix[22][26] + svm_conf_matrix[22][27] + svm_conf_matrix[22][28]
        #svm_fn_24 = svm_conf_matrix[23][0] + svm_conf_matrix[23][1] + svm_conf_matrix[23][2] + svm_conf_matrix[23][3] + svm_conf_matrix[23][4] + svm_conf_matrix[23][5] + svm_conf_matrix[23][6] + svm_conf_matrix[23][7] + svm_conf_matrix[23][8] + svm_conf_matrix[23][9] + svm_conf_matrix[23][10] + svm_conf_matrix[23][11] + svm_conf_matrix[23][12] + svm_conf_matrix[23][13] + svm_conf_matrix[23][14] + svm_conf_matrix[23][15] + svm_conf_matrix[23][16] + svm_conf_matrix[23][17] + svm_conf_matrix[23][18] + svm_conf_matrix[23][19] + svm_conf_matrix[23][20] + svm_conf_matrix[23][21] + svm_conf_matrix[23][22] + svm_conf_matrix[23][24] + svm_conf_matrix[23][25] + svm_conf_matrix[23][26] + svm_conf_matrix[23][27] + svm_conf_matrix[23][28]
        #svm_fn_25 = svm_conf_matrix[24][0] + svm_conf_matrix[24][1] + svm_conf_matrix[24][2] + svm_conf_matrix[24][3] + svm_conf_matrix[24][4] + svm_conf_matrix[24][5] + svm_conf_matrix[24][6] + svm_conf_matrix[24][7] + svm_conf_matrix[24][8] + svm_conf_matrix[24][9] + svm_conf_matrix[24][10] + svm_conf_matrix[24][11] + svm_conf_matrix[24][12] + svm_conf_matrix[24][13] + svm_conf_matrix[24][14] + svm_conf_matrix[24][15] + svm_conf_matrix[24][16] + svm_conf_matrix[24][17] + svm_conf_matrix[24][18] + svm_conf_matrix[24][19] + svm_conf_matrix[24][20] + svm_conf_matrix[24][21] + svm_conf_matrix[24][22] + svm_conf_matrix[24][23] + svm_conf_matrix[24][25] + svm_conf_matrix[24][26] + svm_conf_matrix[24][27] + svm_conf_matrix[24][28]
        #svm_fn_26 = svm_conf_matrix[25][0] + svm_conf_matrix[25][1] + svm_conf_matrix[25][2] + svm_conf_matrix[25][3] + svm_conf_matrix[25][4] + svm_conf_matrix[25][5] + svm_conf_matrix[25][6] + svm_conf_matrix[25][7] + svm_conf_matrix[25][8] + svm_conf_matrix[25][9] + svm_conf_matrix[25][10] + svm_conf_matrix[25][11] + svm_conf_matrix[25][12] + svm_conf_matrix[25][13] + svm_conf_matrix[25][14] + svm_conf_matrix[25][15] + svm_conf_matrix[25][16] + svm_conf_matrix[25][17] + svm_conf_matrix[25][18] + svm_conf_matrix[25][19] + svm_conf_matrix[25][20] + svm_conf_matrix[25][21] + svm_conf_matrix[25][22] + svm_conf_matrix[25][23] + svm_conf_matrix[25][24] + svm_conf_matrix[25][26] + svm_conf_matrix[25][27] + svm_conf_matrix[25][28]
        #svm_fn_27 = svm_conf_matrix[26][0] + svm_conf_matrix[26][1] + svm_conf_matrix[26][2] + svm_conf_matrix[26][3] + svm_conf_matrix[26][4] + svm_conf_matrix[26][5] + svm_conf_matrix[26][6] + svm_conf_matrix[26][7] + svm_conf_matrix[26][8] + svm_conf_matrix[26][9] + svm_conf_matrix[26][10] + svm_conf_matrix[26][11] + svm_conf_matrix[26][12] + svm_conf_matrix[26][13] + svm_conf_matrix[26][14] + svm_conf_matrix[26][15] + svm_conf_matrix[26][16] + svm_conf_matrix[26][17] + svm_conf_matrix[26][18] + svm_conf_matrix[26][19] + svm_conf_matrix[26][20] + svm_conf_matrix[26][21] + svm_conf_matrix[26][22] + svm_conf_matrix[26][23] + svm_conf_matrix[26][24] + svm_conf_matrix[26][25] + svm_conf_matrix[26][27] + svm_conf_matrix[26][28]
        #svm_fn_28 = svm_conf_matrix[27][0] + svm_conf_matrix[27][1] + svm_conf_matrix[27][2] + svm_conf_matrix[27][3] + svm_conf_matrix[27][4] + svm_conf_matrix[27][5] + svm_conf_matrix[27][6] + svm_conf_matrix[27][7] + svm_conf_matrix[27][8] + svm_conf_matrix[27][9] + svm_conf_matrix[27][10] + svm_conf_matrix[27][11] + svm_conf_matrix[27][12] + svm_conf_matrix[27][13] + svm_conf_matrix[27][14] + svm_conf_matrix[27][15] + svm_conf_matrix[27][16] + svm_conf_matrix[27][17] + svm_conf_matrix[27][18] + svm_conf_matrix[27][19] + svm_conf_matrix[27][20] + svm_conf_matrix[27][21] + svm_conf_matrix[27][22] + svm_conf_matrix[27][23] + svm_conf_matrix[27][24] + svm_conf_matrix[27][25] + svm_conf_matrix[27][26] + svm_conf_matrix[27][28]
        #svm_fn_29 = svm_conf_matrix[28][0] + svm_conf_matrix[28][1] + svm_conf_matrix[28][2] + svm_conf_matrix[28][3] + svm_conf_matrix[28][4] + svm_conf_matrix[28][5] + svm_conf_matrix[28][6] + svm_conf_matrix[28][7] + svm_conf_matrix[28][8] + svm_conf_matrix[28][9] + svm_conf_matrix[28][10] + svm_conf_matrix[28][11] + svm_conf_matrix[28][12] + svm_conf_matrix[28][13] + svm_conf_matrix[28][14] + svm_conf_matrix[28][15] + svm_conf_matrix[28][16] + svm_conf_matrix[28][17] + svm_conf_matrix[28][18] + svm_conf_matrix[28][19] + svm_conf_matrix[28][20] + svm_conf_matrix[28][21] + svm_conf_matrix[28][22] + svm_conf_matrix[28][23] + svm_conf_matrix[28][24] + svm_conf_matrix[28][25] + svm_conf_matrix[28][26] + svm_conf_matrix[28][27]

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
            svm_recall_5 = svm_tp_5 / (svm_tp_5 + svm_fn_5)
        if svm_tp_6 + svm_fn_6 == 0:
            svm_recall_6 = 0
        else:
            svm_recall_6 = svm_tp_6 / (svm_tp_6 + svm_fn_6)
        if svm_tp_7 + svm_fn_7 == 0:
            svm_recall_7 = 0
        else:
            svm_recall_7 = svm_tp_7 / (svm_tp_7 + svm_fn_7)
        if svm_tp_8 + svm_fn_8 == 0:
            svm_recall_8 = 0
        else:
            svm_recall_8 = svm_tp_8 / (svm_tp_8 + svm_fn_8)
        if svm_tp_9 + svm_fn_9 == 0:
            svm_recall_9 = 0
        else:
            svm_recall_9 = svm_tp_9 / (svm_tp_9 + svm_fn_9)
        if svm_tp_10 + svm_fn_10 == 0:
            svm_recall_10 = 0
        else:
            svm_recall_10 = svm_tp_10 / (svm_tp_10 + svm_fn_10)
        if svm_tp_11 + svm_fn_11 == 0:
            svm_recall_11 = 0
        else:
            svm_recall_11 = svm_tp_11 / (svm_tp_11 + svm_fn_11)
        if svm_tp_12 + svm_fn_12 == 0:
            svm_recall_12 = 0
        else:
            svm_recall_12 = svm_tp_12 / (svm_tp_12 + svm_fn_12)
        if svm_tp_13 + svm_fn_13 == 0:
            svm_recall_13 = 0
        else:
            svm_recall_13 = svm_tp_13 / (svm_tp_13 + svm_fn_13)
        if svm_tp_14 + svm_fn_14 == 0:
            svm_recall_14 = 0
        else:
            svm_recall_14 = svm_tp_14 / (svm_tp_14 + svm_fn_14)
        if svm_tp_15 + svm_fn_15 == 0:
            svm_recall_15 = 0
        else:
            svm_recall_15 = svm_tp_15 / (svm_tp_15 + svm_fn_15)
        if svm_tp_16 + svm_fn_16 == 0:
            svm_recall_16 = 0
        else:
            svm_recall_16 = svm_tp_16 / (svm_tp_16 + svm_fn_16)
        if svm_tp_17 + svm_fn_17 == 0:
            svm_recall_17 = 0
        else:
            svm_recall_17 = svm_tp_17 / (svm_tp_17 + svm_fn_17)
        if svm_tp_18 + svm_fn_18 == 0:
            svm_recall_18 = 0
        else:
            svm_recall_18 = svm_tp_18 / (svm_tp_18 + svm_fn_18)
        if svm_tp_19 + svm_fn_19 == 0:
            svm_recall_19 = 0
        else:
            svm_recall_19 = svm_tp_19 / (svm_tp_19 + svm_fn_19)
        if svm_tp_20 + svm_fn_20 == 0:
            svm_recall_20 = 0
        else:
            svm_recall_20 = svm_tp_20 / (svm_tp_20 + svm_fn_20)
        '''
        if svm_tp_21 + svm_fn_21 == 0:
            svm_recall_21 = 0
        else:
            svm_recall_21 = svm_tp_21 / (svm_tp_21 + svm_fn_21)
        if svm_tp_22 + svm_fn_22 == 0:
            svm_recall_22 = 0
        else:
            svm_recall_22 = svm_tp_22 / (svm_tp_22 + svm_fn_22)
        if svm_tp_23 + svm_fn_23 == 0:
            svm_recall_23 = 0
        else:
            svm_recall_23 = svm_tp_23 / (svm_tp_23 + svm_fn_23)
        if svm_tp_24 + svm_fn_24 == 0:
            svm_recall_24 = 0
        else:
            svm_recall_24 = svm_tp_24 / (svm_tp_24 + svm_fn_24)
        if svm_tp_25 + svm_fn_25 == 0:
            svm_recall_25 = 0
        else:
            svm_recall_25 = svm_tp_25 / (svm_tp_25 + svm_fn_25)
        if svm_tp_26 + svm_fn_26 == 0:
            svm_recall_26 = 0
        else:
            svm_recall_26 = svm_tp_26 / (svm_tp_26 + svm_fn_26)
        if svm_tp_27 + svm_fn_27 == 0:
            svm_recall_27 = 0
        else:
            svm_recall_27 = svm_tp_27 / (svm_tp_27 + svm_fn_27)
        if svm_tp_28 + svm_fn_28 == 0:
            svm_recall_28 = 0
        else:
            svm_recall_28 = svm_tp_28 / (svm_tp_28 + svm_fn_28)
        if svm_tp_29 + svm_fn_29 == 0:
            svm_recall_29 = 0
        else:
            svm_recall_29 = svm_tp_29 / (svm_tp_29 + svm_fn_29)
        '''
        svm_recall_avg_pen_5 = (
                                 svm_recall_1 + svm_recall_2 + svm_recall_3 + svm_recall_4 + svm_recall_5 + svm_recall_6 + svm_recall_7 + svm_recall_8 + svm_recall_9 + svm_recall_10 + svm_recall_11 + svm_recall_12 + svm_recall_13 + svm_recall_14 + svm_recall_15 + svm_recall_16 + svm_recall_17 + svm_recall_18 + svm_recall_19 + (5*svm_recall_20)) / (20+5-1)
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
    svm_ovr_accuracy = (svm_conf_matrix[0][0] + svm_conf_matrix[1][1] + svm_conf_matrix[2][2] + svm_conf_matrix[3][3] + svm_conf_matrix[4][4] + svm_conf_matrix[5][5] + svm_conf_matrix[6][6] + svm_conf_matrix[7][7] + svm_conf_matrix[8][8] + svm_conf_matrix[9][9] + svm_conf_matrix[10][10] + svm_conf_matrix[11][11] + svm_conf_matrix[12][12] + svm_conf_matrix[13][13] + svm_conf_matrix[14][14] + svm_conf_matrix[15][15] + svm_conf_matrix[16][16] + svm_conf_matrix[17][17] + svm_conf_matrix[18][18] + svm_conf_matrix[19][19]) / (
                sum(svm_conf_matrix[0]) + sum(svm_conf_matrix[1]) + sum(svm_conf_matrix[2]) + sum(svm_conf_matrix[3]) + sum(svm_conf_matrix[4]) + sum(svm_conf_matrix[5]) + sum(svm_conf_matrix[6]) + sum(svm_conf_matrix[7]) + sum(svm_conf_matrix[8]) + sum(svm_conf_matrix[9]) + sum(svm_conf_matrix[10]) + sum(svm_conf_matrix[11]) + sum(svm_conf_matrix[12]) + sum(svm_conf_matrix[13]) + sum(svm_conf_matrix[14]) + sum(svm_conf_matrix[15]) + sum(svm_conf_matrix[16]) + sum(svm_conf_matrix[17]) + sum(svm_conf_matrix[18]) + sum(svm_conf_matrix[19]))
    print("svm_f1 score of pen 1 is:")
    print(svm_f1_score_pen_1)
    print("svm_f1 score of pen 5 is:")
    print(svm_f1_score_pen_5)
    print("svm_overall accuracy is:")
    print(svm_ovr_accuracy)
    svm_conf_matrix = pd.DataFrame(svm_conf_matrix)
    svm_conf_matrix.to_csv('conf_matrix_' + imb_technique + '_svm_production_' + str(nsplits) + 'foldcv_' + str(repeat+1) + '.csv', header=False, index=False)  # First repetition
    #svm_conf_matrix.to_csv('conf_matrix_' + imb_technique + '_penalty_' + str(penalty) + '_svm_production_' + str(nsplits) + 'foldcv_' + str(repeat+6) + '.csv', header=False, index=False)  # First repetition
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
#dnn_f1_score_kfoldcv.to_csv('f1_score_' + imb_technique + '_dnn_production_' + str(nsplits) + 'foldcv_1~5.csv', header=False, index=False)  # First repetition
dnn_ovr_accuracy_kfoldcv[5] = "Average" #To let users figure out that f1_score_kfoldcv[6] value is the average when seeing the csv file
dnn_ovr_accuracy_kfoldcv[6] = (dnn_ovr_accuracy_kfoldcv[0]+dnn_ovr_accuracy_kfoldcv[1]+dnn_ovr_accuracy_kfoldcv[2]+dnn_ovr_accuracy_kfoldcv[3]+dnn_ovr_accuracy_kfoldcv[4])/5
dnn_ovr_accuracy_kfoldcv = pd.DataFrame(dnn_ovr_accuracy_kfoldcv)
dnn_ovr_accuracy_kfoldcv.to_csv('ovr_accuracy_' + imb_technique + '_dnn_production_' + str(nsplits) + 'foldcv_1~5.csv', header=False, index=False)  # First repetition
#dnn_ovr_accuracy_kfoldcv.to_csv('ovr_accuracy_' + imb_technique + '_dnn_production_' + str(nsplits) + 'foldcv_6~10.csv', header=False, index=False)  # Second repetition

lr_f1_score_pen_1_kfoldcv[5] = "Average" #To let users figure out that f1_score_kfoldcv[6] value is the average when seeing the csv file
lr_f1_score_pen_1_kfoldcv[6] = (lr_f1_score_pen_1_kfoldcv[0]+lr_f1_score_pen_1_kfoldcv[1]+lr_f1_score_pen_1_kfoldcv[2]+lr_f1_score_pen_1_kfoldcv[3]+lr_f1_score_pen_1_kfoldcv[4])/5
lr_f1_score_pen_1_kfoldcv = pd.DataFrame(lr_f1_score_pen_1_kfoldcv)
lr_f1_score_pen_1_kfoldcv.to_csv('f1_score_'+imb_technique+'_penalty_1_lr_bpic2013_closed_' + str(nsplits) + 'foldcv_1~5.csv',header=False,index=False) #First repetition
lr_f1_score_pen_5_kfoldcv[5] = "Average" #To let users figure out that f1_score_kfoldcv[6] value is the average when seeing the csv file
lr_f1_score_pen_5_kfoldcv[6] = (lr_f1_score_pen_5_kfoldcv[0]+lr_f1_score_pen_5_kfoldcv[1]+lr_f1_score_pen_5_kfoldcv[2]+lr_f1_score_pen_5_kfoldcv[3]+lr_f1_score_pen_5_kfoldcv[4])/5
lr_f1_score_pen_5_kfoldcv = pd.DataFrame(lr_f1_score_pen_5_kfoldcv)
lr_f1_score_pen_5_kfoldcv.to_csv('f1_score_'+imb_technique+'_penalty_5_lr_bpic2013_closed_' + str(nsplits) + 'foldcv_1~5.csv',header=False,index=False) #First repetition
#lr_f1_score_kfoldcv.to_csv('f1_score_' + imb_technique + '_lr_production_' + str(nsplits) + 'foldcv_1~5.csv', header=False, index=False)  # First repetition
lr_ovr_accuracy_kfoldcv[5] = "Average" #To let users figure out that f1_score_kfoldcv[6] value is the average when seeing the csv file
lr_ovr_accuracy_kfoldcv[6] = (lr_ovr_accuracy_kfoldcv[0]+lr_ovr_accuracy_kfoldcv[1]+lr_ovr_accuracy_kfoldcv[2]+lr_ovr_accuracy_kfoldcv[3]+lr_ovr_accuracy_kfoldcv[4])/5
lr_ovr_accuracy_kfoldcv = pd.DataFrame(lr_ovr_accuracy_kfoldcv)
lr_ovr_accuracy_kfoldcv.to_csv('ovr_accuracy_' + imb_technique + '_lr_production_' + str(nsplits) + 'foldcv_1~5.csv', header=False, index=False)  # First repetition
#lr_ovr_accuracy_kfoldcv.to_csv('ovr_accuracy_' + imb_technique + '_lr_production_' + str(nsplits) + 'foldcv_6~10.csv', header=False, index=False)  # Second repetition

nb_f1_score_pen_1_kfoldcv[5] = "Average" #To let users figure out that f1_score_kfoldcv[6] value is the average when seeing the csv file
nb_f1_score_pen_1_kfoldcv[6] = (nb_f1_score_pen_1_kfoldcv[0]+nb_f1_score_pen_1_kfoldcv[1]+nb_f1_score_pen_1_kfoldcv[2]+nb_f1_score_pen_1_kfoldcv[3]+nb_f1_score_pen_1_kfoldcv[4])/5
nb_f1_score_pen_1_kfoldcv = pd.DataFrame(nb_f1_score_pen_1_kfoldcv)
nb_f1_score_pen_1_kfoldcv.to_csv('f1_score_'+imb_technique+'_penalty_1_nb_bpic2013_closed_' + str(nsplits) + 'foldcv_1~5.csv',header=False,index=False) #First repetition
nb_f1_score_pen_5_kfoldcv[5] = "Average" #To let users figure out that f1_score_kfoldcv[6] value is the average when seeing the csv file
nb_f1_score_pen_5_kfoldcv[6] = (nb_f1_score_pen_5_kfoldcv[0]+nb_f1_score_pen_5_kfoldcv[1]+nb_f1_score_pen_5_kfoldcv[2]+nb_f1_score_pen_5_kfoldcv[3]+nb_f1_score_pen_5_kfoldcv[4])/5
nb_f1_score_pen_5_kfoldcv = pd.DataFrame(nb_f1_score_pen_5_kfoldcv)
nb_f1_score_pen_5_kfoldcv.to_csv('f1_score_'+imb_technique+'_penalty_5_nb_bpic2013_closed_' + str(nsplits) + 'foldcv_1~5.csv',header=False,index=False) #First repetition
#nb_f1_score_kfoldcv.to_csv('f1_score_' + imb_technique + '_nb_production_' + str(nsplits) + 'foldcv_1~5.csv', header=False, index=False)  # First repetition
nb_ovr_accuracy_kfoldcv[5] = "Average" #To let users figure out that f1_score_kfoldcv[6] value is the average when seeing the csv file
nb_ovr_accuracy_kfoldcv[6] = (nb_ovr_accuracy_kfoldcv[0]+nb_ovr_accuracy_kfoldcv[1]+nb_ovr_accuracy_kfoldcv[2]+nb_ovr_accuracy_kfoldcv[3]+nb_ovr_accuracy_kfoldcv[4])/5
nb_ovr_accuracy_kfoldcv = pd.DataFrame(nb_ovr_accuracy_kfoldcv)
nb_ovr_accuracy_kfoldcv.to_csv('ovr_accuracy_' + imb_technique + '_nb_production_' + str(nsplits) + 'foldcv_1~5.csv', header=False, index=False)  # First repetition
#nb_ovr_accuracy_kfoldcv.to_csv('ovr_accuracy_' + imb_technique + '_nb_production_' + str(nsplits) + 'foldcv_6~10.csv', header=False, index=False)  # Second repetition

rf_f1_score_pen_1_kfoldcv[5] = "Average" #To let users figure out that f1_score_kfoldcv[6] value is the average when seeing the csv file
rf_f1_score_pen_1_kfoldcv[6] = (rf_f1_score_pen_1_kfoldcv[0]+rf_f1_score_pen_1_kfoldcv[1]+rf_f1_score_pen_1_kfoldcv[2]+rf_f1_score_pen_1_kfoldcv[3]+rf_f1_score_pen_1_kfoldcv[4])/5
rf_f1_score_pen_1_kfoldcv = pd.DataFrame(rf_f1_score_pen_1_kfoldcv)
rf_f1_score_pen_1_kfoldcv.to_csv('f1_score_'+imb_technique+'_penalty_1_rf_bpic2013_closed_' + str(nsplits) + 'foldcv_1~5.csv',header=False,index=False) #First repetition
rf_f1_score_pen_5_kfoldcv[5] = "Average" #To let users figure out that f1_score_kfoldcv[6] value is the average when seeing the csv file
rf_f1_score_pen_5_kfoldcv[6] = (rf_f1_score_pen_5_kfoldcv[0]+rf_f1_score_pen_5_kfoldcv[1]+rf_f1_score_pen_5_kfoldcv[2]+rf_f1_score_pen_5_kfoldcv[3]+rf_f1_score_pen_5_kfoldcv[4])/5
rf_f1_score_pen_5_kfoldcv = pd.DataFrame(rf_f1_score_pen_5_kfoldcv)
rf_f1_score_pen_5_kfoldcv.to_csv('f1_score_'+imb_technique+'_penalty_5_rf_bpic2013_closed_' + str(nsplits) + 'foldcv_1~5.csv',header=False,index=False) #First repetition
#rf_f1_score_kfoldcv.to_csv('f1_score_' + imb_technique + '_rf_production_' + str(nsplits) + 'foldcv_1~5.csv', header=False, index=False)  # First repetition
rf_ovr_accuracy_kfoldcv[5] = "Average" #To let users figure out that f1_score_kfoldcv[6] value is the average when seeing the csv file
rf_ovr_accuracy_kfoldcv[6] = (rf_ovr_accuracy_kfoldcv[0]+rf_ovr_accuracy_kfoldcv[1]+rf_ovr_accuracy_kfoldcv[2]+rf_ovr_accuracy_kfoldcv[3]+rf_ovr_accuracy_kfoldcv[4])/5
rf_ovr_accuracy_kfoldcv = pd.DataFrame(rf_ovr_accuracy_kfoldcv)
rf_ovr_accuracy_kfoldcv.to_csv('ovr_accuracy_' + imb_technique + '_rf_production_' + str(nsplits) + 'foldcv_1~5.csv', header=False, index=False)  # First repetition
#rf_ovr_accuracy_kfoldcv.to_csv('ovr_accuracy_' + imb_technique + '_rf_production_' + str(nsplits) + 'foldcv_6~10.csv', header=False, index=False)  # Second repetition

svm_f1_score_pen_1_kfoldcv[5] = "Average" #To let users figure out that f1_score_kfoldcv[6] value is the average when seeing the csv file
svm_f1_score_pen_1_kfoldcv[6] = (svm_f1_score_pen_1_kfoldcv[0]+svm_f1_score_pen_1_kfoldcv[1]+svm_f1_score_pen_1_kfoldcv[2]+svm_f1_score_pen_1_kfoldcv[3]+svm_f1_score_pen_1_kfoldcv[4])/5
svm_f1_score_pen_1_kfoldcv = pd.DataFrame(svm_f1_score_pen_1_kfoldcv)
svm_f1_score_pen_1_kfoldcv.to_csv('f1_score_'+imb_technique+'_penalty_1_svm_bpic2013_closed_' + str(nsplits) + 'foldcv_1~5.csv',header=False,index=False) #First repetition
svm_f1_score_pen_5_kfoldcv[5] = "Average" #To let users figure out that f1_score_kfoldcv[6] value is the average when seeing the csv file
svm_f1_score_pen_5_kfoldcv[6] = (svm_f1_score_pen_5_kfoldcv[0]+svm_f1_score_pen_5_kfoldcv[1]+svm_f1_score_pen_5_kfoldcv[2]+svm_f1_score_pen_5_kfoldcv[3]+svm_f1_score_pen_5_kfoldcv[4])/5
svm_f1_score_pen_5_kfoldcv = pd.DataFrame(svm_f1_score_pen_5_kfoldcv)
svm_f1_score_pen_5_kfoldcv.to_csv('f1_score_'+imb_technique+'_penalty_5_svm_bpic2013_closed_' + str(nsplits) + 'foldcv_1~5.csv',header=False,index=False) #First repetition
#svm_f1_score_kfoldcv.to_csv('f1_score_' + imb_technique + '_svm_production_' + str(nsplits) + 'foldcv_1~5.csv', header=False, index=False)  # First repetition
svm_ovr_accuracy_kfoldcv[5] = "Average" #To let users figure out that f1_score_kfoldcv[6] value is the average when seeing the csv file
svm_ovr_accuracy_kfoldcv[6] = (svm_ovr_accuracy_kfoldcv[0]+svm_ovr_accuracy_kfoldcv[1]+svm_ovr_accuracy_kfoldcv[2]+svm_ovr_accuracy_kfoldcv[3]+svm_ovr_accuracy_kfoldcv[4])/5
svm_ovr_accuracy_kfoldcv = pd.DataFrame(svm_ovr_accuracy_kfoldcv)
svm_ovr_accuracy_kfoldcv.to_csv('ovr_accuracy_' + imb_technique + '_svm_production_' + str(nsplits) + 'foldcv_1~5.csv', header=False, index=False)  # First repetition
#svm_ovr_accuracy_kfoldcv.to_csv('ovr_accuracy_' + imb_technique + '_svm_production_' + str(nsplits) + 'foldcv_6~10.csv', header=False, index=False)  # Second repetition

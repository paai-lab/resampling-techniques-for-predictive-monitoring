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
from resampling.resampler_production import resampling_assigner
from imblearn.metrics import geometric_mean_score
from collections import Counter

data_dir = "/home/jongchan/Production/window_3_production_preprocessed.csv"
data = pd.read_csv(data_dir, encoding='cp437')
data = data.sample(frac=1).reset_index(drop=True)
X = data[['ACT_1', 'ACT_2', 'ACT_3','Span_in_minutes']]
y = data[['ACT_4']]

imb_technique = argv[1]

# Dummification
X_dummy = pd.get_dummies(X)
X_dummy.iloc[:, 0] = (X_dummy.iloc[:, 0] - X_dummy.iloc[:, 0].mean()) / X_dummy.iloc[:, 0].std()

# X and y here will be used for hyperparameter tuning using random search
X_randomsearch = X.replace(regex=True, to_replace=["Final Inspection Q.C.","Flat Grinding - Machine 11","Grinding Rework - Machine 27","Lapping - Machine 1","Laser Marking - Machine 7","Packing","Round Grinding - Machine 12","Round Grinding - Machine 2","Round Grinding - Machine 3","Round Grinding - Manual","Round Grinding - Q.C.","Turning - Machine 8","Turning & Milling - Machine 10","Turning & Milling - Machine 4","Turning & Milling - Machine 5","Turning & Milling - Machine 6","Turning & Milling - Machine 8","Turning & Milling - Machine 9","Turning & Milling Q.C.","Turning Q.C.","Wire Cut - Machine 13","Deburring - Manual","Stress Relief","Turning - Machine 9","Grinding Rework","Fix EDM","Wire Cut - Machine 18","Turn & Mill. & Screw Assem - Machine 10","Fix - Machine 15M","Nitration Q.C.","Fix - Machine 15","Milling - Machine 16","Turning Rework - Machine 21","Turn & Mill. & Screw Assem - Machine 9","Fix - Machine 3","Round Grinding - Machine 19","Change Version - Machine 22","Milling - Machine 14","Setup - Machine 4","Turning - Machine 21","Turning - Machine 5","Milling - Machine 10","Round  Q.C.","Rework Milling - Machine 28","Milling Q.C.","Turning - Machine 4","Flat Grinding - Machine 26","Fix - Machine 19","Setup - Machine 8"], value=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49])
y_randomsearch = y.replace(regex=True, to_replace=["Final Inspection Q.C.","Flat Grinding - Machine 11","Grinding Rework - Machine 27","Lapping - Machine 1","Laser Marking - Machine 7","Packing","Round Grinding - Machine 12","Round Grinding - Machine 2","Round Grinding - Machine 3","Round Grinding - Manual","Round Grinding - Q.C.","Turning - Machine 8","Turning & Milling - Machine 10","Turning & Milling - Machine 4","Turning & Milling - Machine 5","Turning & Milling - Machine 6","Turning & Milling - Machine 8","Turning & Milling - Machine 9","Turning & Milling Q.C.","Turning Q.C.","Wire Cut - Machine 13","Deburring - Manual","Stress Relief","Turning - Machine 9","Grinding Rework","Fix EDM","Wire Cut - Machine 18","Turn & Mill. & Screw Assem - Machine 10","Fix - Machine 15M","Nitration Q.C.","Fix - Machine 15","Milling - Machine 16","Turning Rework - Machine 21","Turn & Mill. & Screw Assem - Machine 9","Fix - Machine 3","Round Grinding - Machine 19","Change Version - Machine 22","Milling - Machine 14","Setup - Machine 4","Turning - Machine 21","Turning - Machine 5","Milling - Machine 10","Round  Q.C.","Rework Milling - Machine 28","Milling Q.C.","Turning - Machine 4","Flat Grinding - Machine 26","Fix - Machine 19","Setup - Machine 8"], value=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49])

nsplits = 5
kf = KFold(n_splits=nsplits)
kf.get_n_splits(X_dummy)

dnn_f1_score_pen_1_kfoldcv, dnn_f1_score_pen_5_kfoldcv, dnn_ovr_accuracy_kfoldcv, dnn_auc_kfoldcv, dnn_gmean_kfoldcv = [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2)
dnn_params_hls_FI, dnn_params_hls_FG, dnn_params_hls_GR27, dnn_params_hls_LM, dnn_params_hls_LMM, dnn_params_hls_PC, dnn_params_hls_RG12, dnn_params_hls_RG2, dnn_params_hls_RG3, dnn_params_hls_RGM, dnn_params_hls_RGQC, dnn_params_hls_T8, dnn_params_hls_TM10, dnn_params_hls_TM4, dnn_params_hls_TM5, dnn_params_hls_TM6, dnn_params_hls_TM8, dnn_params_hls_TM9, dnn_params_hls_TMQC, dnn_params_hls_TQC = [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2)
dnn_params_lri_FI, dnn_params_lri_FG, dnn_params_lri_GR27, dnn_params_lri_LM, dnn_params_lri_LMM, dnn_params_lri_PC, dnn_params_lri_RG12, dnn_params_lri_RG2, dnn_params_lri_RG3, dnn_params_lri_RGM, dnn_params_lri_RGQC, dnn_params_lri_T8, dnn_params_lri_TM10, dnn_params_lri_TM4, dnn_params_lri_TM5, dnn_params_lri_TM6, dnn_params_lri_TM8, dnn_params_lri_TM9, dnn_params_lri_TMQC, dnn_params_lri_TQC = [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2)

lr_f1_score_pen_1_kfoldcv, lr_f1_score_pen_5_kfoldcv, lr_ovr_accuracy_kfoldcv, lr_auc_kfoldcv, lr_gmean_kfoldcv = [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2)
lr_params_solver_FI, lr_params_solver_FG, lr_params_solver_GR27, lr_params_solver_LM, lr_params_solver_LMM, lr_params_solver_PC, lr_params_solver_RG12, lr_params_solver_RG2, lr_params_solver_RG3, lr_params_solver_RGM, lr_params_solver_RGQC, lr_params_solver_T8, lr_params_solver_TM10, lr_params_solver_TM4, lr_params_solver_TM5, lr_params_solver_TM6, lr_params_solver_TM8, lr_params_solver_TM9, lr_params_solver_TMQC, lr_params_solver_TQC = [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2)
lr_params_tol_FI, lr_params_tol_FG, lr_params_tol_GR27, lr_params_tol_LM, lr_params_tol_LMM, lr_params_tol_PC, lr_params_tol_RG12, lr_params_tol_RG2, lr_params_tol_RG3, lr_params_tol_RGM, lr_params_tol_RGQC, lr_params_tol_T8, lr_params_tol_TM10, lr_params_tol_TM4, lr_params_tol_TM5, lr_params_tol_TM6, lr_params_tol_TM8, lr_params_tol_TM9, lr_params_tol_TMQC, lr_params_tol_TQC = [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2)
lr_params_C_FI, lr_params_C_FG, lr_params_C_GR27, lr_params_C_LM, lr_params_C_LMM, lr_params_C_PC, lr_params_C_RG12, lr_params_C_RG2, lr_params_C_RG3, lr_params_C_RGM, lr_params_C_RGQC, lr_params_C_T8, lr_params_C_TM10, lr_params_C_TM4, lr_params_C_TM5, lr_params_C_TM6, lr_params_C_TM8, lr_params_C_TM9, lr_params_C_TMQC, lr_params_C_TQC = [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2)

nb_f1_score_pen_1_kfoldcv, nb_f1_score_pen_5_kfoldcv, nb_ovr_accuracy_kfoldcv, nb_auc_kfoldcv, nb_gmean_kfoldcv = [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2)
nb_params_vs_FI, nb_params_vs_FG, nb_params_vs_GR27, nb_params_vs_LM, nb_params_vs_LMM, nb_params_vs_PC, nb_params_vs_RG12, nb_params_vs_RG2, nb_params_vs_RG3, nb_params_vs_RGM, nb_params_vs_RGQC, nb_params_vs_T8, nb_params_vs_TM10, nb_params_vs_TM4, nb_params_vs_TM5, nb_params_vs_TM6, nb_params_vs_TM8, nb_params_vs_TM9, nb_params_vs_TMQC, nb_params_vs_TQC = [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2)

rf_f1_score_pen_1_kfoldcv, rf_f1_score_pen_5_kfoldcv, rf_ovr_accuracy_kfoldcv, rf_auc_kfoldcv, rf_gmean_kfoldcv = [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2)
rf_params_est_FI, rf_params_est_FG, rf_params_est_GR27, rf_params_est_LM, rf_params_est_LMM, rf_params_est_PC, rf_params_est_RG12, rf_params_est_RG2, rf_params_est_RG3, rf_params_est_RGM, rf_params_est_RGQC, rf_params_est_T8, rf_params_est_TM10, rf_params_est_TM4, rf_params_est_TM5, rf_params_est_TM6, rf_params_est_TM8, rf_params_est_TM9, rf_params_est_TMQC, rf_params_est_TQC = [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2)
rf_params_md_FI, rf_params_md_FG, rf_params_md_GR27, rf_params_md_LM, rf_params_md_LMM, rf_params_md_PC, rf_params_md_RG12, rf_params_md_RG2, rf_params_md_RG3, rf_params_md_RGM, rf_params_md_RGQC, rf_params_md_T8, rf_params_md_TM10, rf_params_md_TM4, rf_params_md_TM5, rf_params_md_TM6, rf_params_md_TM8, rf_params_md_TM9, rf_params_md_TMQC, rf_params_md_TQC = [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2)
rf_params_mss_FI, rf_params_mss_FG, rf_params_mss_GR27, rf_params_mss_LM, rf_params_mss_LMM, rf_params_mss_PC, rf_params_mss_RG12, rf_params_mss_RG2, rf_params_mss_RG3, rf_params_mss_RGM, rf_params_mss_RGQC, rf_params_mss_T8, rf_params_mss_TM10, rf_params_mss_TM4, rf_params_mss_TM5, rf_params_mss_TM6, rf_params_mss_TM8, rf_params_mss_TM9, rf_params_mss_TMQC, rf_params_mss_TQC = [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2)

svm_f1_score_pen_1_kfoldcv, svm_f1_score_pen_5_kfoldcv, svm_ovr_accuracy_kfoldcv, svm_auc_kfoldcv, svm_gmean_kfoldcv = [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2)
svm_params_tol_FI, svm_params_tol_FG, svm_params_tol_GR27, svm_params_tol_LM, svm_params_tol_LMM, svm_params_tol_PC, svm_params_tol_RG12, svm_params_tol_RG2, svm_params_tol_RG3, svm_params_tol_RGM, svm_params_tol_RGQC, svm_params_tol_T8, svm_params_tol_TM10, svm_params_tol_TM4, svm_params_tol_TM5, svm_params_tol_TM6, svm_params_tol_TM8, svm_params_tol_TM9, svm_params_tol_TMQC, svm_params_tol_TQC = [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2), [None] * (nsplits+2)

outfile = os.path.join("/home/jongchan/Production", "performance_results_%s_%s.csv" % ("Production", imb_technique))
outfile_param_FI = os.path.join("/home/jongchan/Production", "parameters_FI_%s_%s.csv" % ("Production", imb_technique))
outfile_param_FG = os.path.join("/home/jongchan/Production", "parameters_FG_%s_%s.csv" % ("Production", imb_technique))
outfile_param_GR27 = os.path.join("/home/jongchan/Production", "parameters_GR27_%s_%s.csv" % ("Production", imb_technique))
outfile_param_LM = os.path.join("/home/jongchan/Production", "parameters_LM_%s_%s.csv" % ("Production", imb_technique))
outfile_param_LMM = os.path.join("/home/jongchan/Production", "parameters_LMM_%s_%s.csv" % ("Production", imb_technique))
outfile_param_PC = os.path.join("/home/jongchan/Production", "parameters_PC_%s_%s.csv" % ("Production", imb_technique))
outfile_param_RG12 = os.path.join("/home/jongchan/Production", "parameters_RG12_%s_%s.csv" % ("Production", imb_technique))
outfile_param_RG2 = os.path.join("/home/jongchan/Production", "parameters_RG2_%s_%s.csv" % ("Production", imb_technique))
outfile_param_RG3 = os.path.join("/home/jongchan/Production", "parameters_RG3_%s_%s.csv" % ("Production", imb_technique))
outfile_param_RGM = os.path.join("/home/jongchan/Production", "parameters_RGM_%s_%s.csv" % ("Production", imb_technique))
outfile_param_RGQC = os.path.join("/home/jongchan/Production", "parameters_RGQC_%s_%s.csv" % ("Production", imb_technique))
outfile_param_T8 = os.path.join("/home/jongchan/Production", "parameters_T8_%s_%s.csv" % ("Production", imb_technique))
outfile_param_TM10 = os.path.join("/home/jongchan/Production", "parameters_TM10_%s_%s.csv" % ("Production", imb_technique))
outfile_param_TM4 = os.path.join("/home/jongchan/Production", "parameters_TM4_%s_%s.csv" % ("Production", imb_technique))
outfile_param_TM5 = os.path.join("/home/jongchan/Production", "parameters_TM5_%s_%s.csv" % ("Production", imb_technique))
outfile_param_TM6 = os.path.join("/home/jongchan/Production", "parameters_TM6_%s_%s.csv" % ("Production", imb_technique))
outfile_param_TM8 = os.path.join("/home/jongchan/Production", "parameters_TM8_%s_%s.csv" % ("Production", imb_technique))
outfile_param_TM9 = os.path.join("/home/jongchan/Production", "parameters_TM9_%s_%s.csv" % ("Production", imb_technique))
outfile_param_TMQC = os.path.join("/home/jongchan/Production", "parameters_TMQC_%s_%s.csv" % ("Production", imb_technique))
outfile_param_TQC = os.path.join("/home/jongchan/Production", "parameters_TQC_%s_%s.csv" % ("Production", imb_technique))

repeat = 0
for train_index, test_index in kf.split(X_dummy):
    X_train, X_test = X_dummy.iloc[train_index], X_dummy.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    y_test_FI = pd.DataFrame([1 if i == "Final Inspection Q.C." else 0 for i in y_test['ACT_4']])
    y_test_FG = pd.DataFrame([1 if i == "Flat Grinding - Machine 11" else 0 for i in y_test['ACT_4']])
    y_test_GR27 = pd.DataFrame([1 if i == "Grinding Rework - Machine 27" else 0 for i in y_test['ACT_4']])
    y_test_LM = pd.DataFrame([1 if i == "Lapping - Machine 1" else 0 for i in y_test['ACT_4']])
    y_test_LMM = pd.DataFrame([1 if i == "Laser Marking - Machine 7" else 0 for i in y_test['ACT_4']])
    y_test_PC = pd.DataFrame([1 if i == "Packing" else 0 for i in y_test['ACT_4']])
    y_test_RG12 = pd.DataFrame([1 if i == "Round Grinding - Machine 12" else 0 for i in y_test['ACT_4']])
    y_test_RG2 = pd.DataFrame([1 if i == "Round Grinding - Machine 2" else 0 for i in y_test['ACT_4']])
    y_test_RG3 = pd.DataFrame([1 if i == "Round Grinding - Machine 3" else 0 for i in y_test['ACT_4']])
    y_test_RGM = pd.DataFrame([1 if i == "Round Grinding - Manual" else 0 for i in y_test['ACT_4']])
    y_test_RGQC = pd.DataFrame([1 if i == "Round Grinding - Q.C." else 0 for i in y_test['ACT_4']])
    y_test_T8 = pd.DataFrame([1 if i == "Turning - Machine 8" else 0 for i in y_test['ACT_4']])
    y_test_TM10 = pd.DataFrame([1 if i == "Turning & Milling - Machine 10" else 0 for i in y_test['ACT_4']])
    y_test_TM4 = pd.DataFrame([1 if i == "Turning & Milling - Machine 4" else 0 for i in y_test['ACT_4']])
    y_test_TM5 = pd.DataFrame([1 if i == "Turning & Milling - Machine 5" else 0 for i in y_test['ACT_4']])
    y_test_TM6 = pd.DataFrame([1 if i == "Turning & Milling - Machine 6" else 0 for i in y_test['ACT_4']])
    y_test_TM8 = pd.DataFrame([1 if i == "Turning & Milling - Machine 8" else 0 for i in y_test['ACT_4']])
    y_test_TM9 = pd.DataFrame([1 if i == "Turning & Milling - Machine 9" else 0 for i in y_test['ACT_4']])
    y_test_TMQC = pd.DataFrame([1 if i == "Turning & Milling Q.C." else 0 for i in y_test['ACT_4']])
    y_test_TQC = pd.DataFrame([1 if i == "Turning Q.C." else 0 for i in y_test['ACT_4']])
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

    FI, FI_rest, FI_ova, FI_ova_X_train, FI_ova_y_train, FI_X_res, FI_y_res = train_and_test(train, ACT_4_index, "Final Inspection Q.C.")
    FG, FG_rest, FG_ova, FG_ova_X_train, FG_ova_y_train, FG_X_res, FG_y_res = train_and_test(train, ACT_4_index, "Flat Grinding - Machine 11")
    GR27, GR27_rest, GR27_ova, GR27_ova_X_train, GR27_ova_y_train, GR27_X_res, GR27_y_res = train_and_test(train, ACT_4_index, "Grinding Rework - Machine 27")
    LM, LM_rest, LM_ova, LM_ova_X_train, LM_ova_y_train, LM_X_res, LM_y_res = train_and_test(train, ACT_4_index, "Lapping - Machine 1")
    LMM, LMM_rest, LMM_ova, LMM_ova_X_train, LMM_ova_y_train, LMM_X_res, LMM_y_res = train_and_test(train, ACT_4_index, "Laser Marking - Machine 7")
    PC, PC_rest, PC_ova, PC_ova_X_train, PC_ova_y_train, PC_X_res, PC_y_res = train_and_test(train, ACT_4_index, "Packing")
    RG12, RG12_rest, RG12_ova, RG12_ova_X_train, RG12_ova_y_train, RG12_X_res, RG12_y_res = train_and_test(train, ACT_4_index, "Round Grinding - Machine 12")
    RG2, RG2_rest, RG2_ova, RG2_ova_X_train, RG2_ova_y_train, RG2_X_res, RG2_y_res = train_and_test(train, ACT_4_index, "Round Grinding - Machine 2")
    RG3, RG3_rest, RG3_ova, RG3_ova_X_train, RG3_ova_y_train, RG3_X_res, RG3_y_res = train_and_test(train, ACT_4_index, "Round Grinding - Machine 3")
    RGM, RGM_rest, RGM_ova, RGM_ova_X_train, RGM_ova_y_train, RGM_X_res, RGM_y_res = train_and_test(train, ACT_4_index, "Round Grinding - Manual")
    RGQC, RGQC_rest, RGQC_ova, RGQC_ova_X_train, RGQC_ova_y_train, RGQC_X_res, RGQC_y_res = train_and_test(train, ACT_4_index, "Round Grinding - Q.C.")
    T8, T8_rest, T8_ova, T8_ova_X_train, T8_ova_y_train, T8_X_res, T8_y_res = train_and_test(train, ACT_4_index, "Turning - Machine 8")
    TM10, TM10_rest, TM10_ova, TM10_ova_X_train, TM10_ova_y_train, TM10_X_res, TM10_y_res = train_and_test(train, ACT_4_index, "Turning & Milling - Machine 10")
    TM4, TM4_rest, TM4_ova, TM4_ova_X_train, TM4_ova_y_train, TM4_X_res, TM4_y_res = train_and_test(train, ACT_4_index, "Turning & Milling - Machine 4")
    TM5, TM5_rest, TM5_ova, TM5_ova_X_train, TM5_ova_y_train, TM5_X_res, TM5_y_res = train_and_test(train, ACT_4_index, "Turning & Milling - Machine 5")
    TM6, TM6_rest, TM6_ova, TM6_ova_X_train, TM6_ova_y_train, TM6_X_res, TM6_y_res = train_and_test(train, ACT_4_index, "Turning & Milling - Machine 6")
    TM8, TM8_rest, TM8_ova, TM8_ova_X_train, TM8_ova_y_train, TM8_X_res, TM8_y_res = train_and_test(train, ACT_4_index, "Turning & Milling - Machine 8")
    TM9, TM9_rest, TM9_ova, TM9_ova_X_train, TM9_ova_y_train, TM9_X_res, TM9_y_res = train_and_test(train, ACT_4_index, "Turning & Milling - Machine 9")
    TMQC, TMQC_rest, TMQC_ova, TMQC_ova_X_train, TMQC_ova_y_train, TMQC_X_res, TMQC_y_res = train_and_test(train, ACT_4_index, "Turning & Milling Q.C.")
    TQC, TQC_rest, TQC_ova, TQC_ova_X_train, TQC_ova_y_train, TQC_X_res, TQC_y_res = train_and_test(train, ACT_4_index, "Turning Q.C.")

    FI_X_res, FI_y_res, FG_X_res, FG_y_res, GR27_X_res, GR27_y_res, LM_X_res, LM_y_res, LMM_X_res, LMM_y_res, PC_X_res, PC_y_res, RG12_X_res, RG12_y_res, RG2_X_res, RG2_y_res, RG3_X_res, RG3_y_res, RGM_X_res, RGM_y_res, RGQC_X_res, RGQC_y_res, T8_X_res, T8_y_res, TM10_X_res, TM10_y_res, TM4_X_res, TM4_y_res, TM5_X_res, TM5_y_res, TM6_X_res, TM6_y_res, TM8_X_res, TM8_y_res, TM9_X_res, TM9_y_res, TMQC_X_res, TMQC_y_res, TQC_X_res, TQC_y_res = resampling_assigner(imb_technique, FI_ova_X_train,FI_ova_y_train,FG_ova_X_train,FG_ova_y_train,GR27_ova_X_train,GR27_ova_y_train,LM_ova_X_train,LM_ova_y_train,LMM_ova_X_train,LMM_ova_y_train,PC_ova_X_train,PC_ova_y_train,RG12_ova_X_train,RG12_ova_y_train,RG2_ova_X_train,RG2_ova_y_train,RG3_ova_X_train,RG3_ova_y_train,RGM_ova_X_train,RGM_ova_y_train,RGQC_ova_X_train,RGQC_ova_y_train,T8_ova_X_train,T8_ova_y_train,TM10_ova_X_train,TM10_ova_y_train,TM4_ova_X_train,TM4_ova_y_train,TM5_ova_X_train,TM5_ova_y_train,TM6_ova_X_train,TM6_ova_y_train,TM8_ova_X_train,TM8_ova_y_train,TM9_ova_X_train,TM9_ova_y_train,TMQC_ova_X_train,TMQC_ova_y_train,TQC_ova_X_train,TQC_ova_y_train)

    first_digit_parameters = [x for x in itertools.product((5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100), repeat=1)]
    all_digit_parameters = first_digit_parameters
    learning_rate_init_parameters = [0.1, 0.01, 0.001]
    parameters = {'hidden_layer_sizes': all_digit_parameters,
                  'learning_rate_init': learning_rate_init_parameters}
    dnn_FI,dnn_FG,dnn_GR27,dnn_LM,dnn_LMM,dnn_PC,dnn_RG12,dnn_RG2,dnn_RG3,dnn_RGM,dnn_RGQC,dnn_T8,dnn_TM10,dnn_TM4,dnn_TM5,dnn_TM6,dnn_TM8,dnn_TM9,dnn_TMQC,dnn_TQC = MLPClassifier(max_iter=10000, activation='relu'),MLPClassifier(max_iter=10000, activation='relu'),MLPClassifier(max_iter=10000, activation='relu'),MLPClassifier(max_iter=10000, activation='relu'),MLPClassifier(max_iter=10000, activation='relu'),MLPClassifier(max_iter=10000, activation='relu'),MLPClassifier(max_iter=10000, activation='relu'),MLPClassifier(max_iter=10000, activation='relu'),MLPClassifier(max_iter=10000, activation='relu'),MLPClassifier(max_iter=10000, activation='relu'),MLPClassifier(max_iter=10000, activation='relu'),MLPClassifier(max_iter=10000, activation='relu'),MLPClassifier(max_iter=10000, activation='relu'),MLPClassifier(max_iter=10000, activation='relu'),MLPClassifier(max_iter=10000, activation='relu'),MLPClassifier(max_iter=10000, activation='relu'),MLPClassifier(max_iter=10000, activation='relu'),MLPClassifier(max_iter=10000, activation='relu'),MLPClassifier(max_iter=10000, activation='relu'),MLPClassifier(max_iter=10000, activation='relu')
    dnn_FI_clf = RandomizedSearchCV(dnn_FI, parameters, n_jobs=-1, cv=5)
    dnn_FI_clf.fit(FI_X_res, FI_y_res)
    dnn_FG_clf = RandomizedSearchCV(dnn_FG, parameters, n_jobs=-1, cv=5)
    dnn_FG_clf.fit(FG_X_res, FG_y_res)
    dnn_GR27_clf = RandomizedSearchCV(dnn_GR27, parameters, n_jobs=-1, cv=5)
    dnn_GR27_clf.fit(GR27_X_res, GR27_y_res)
    dnn_LM_clf = RandomizedSearchCV(dnn_LM, parameters, n_jobs=-1, cv=5)
    dnn_LM_clf.fit(LM_X_res, LM_y_res)
    dnn_LMM_clf = RandomizedSearchCV(dnn_LMM, parameters, n_jobs=-1, cv=5)
    dnn_LMM_clf.fit(LMM_X_res, LMM_y_res)
    dnn_PC_clf = RandomizedSearchCV(dnn_PC, parameters, n_jobs=-1, cv=5)
    dnn_PC_clf.fit(PC_X_res, PC_y_res)
    dnn_RG12_clf = RandomizedSearchCV(dnn_RG12, parameters, n_jobs=-1, cv=5)
    dnn_RG12_clf.fit(RG12_X_res, RG12_y_res)
    dnn_RG2_clf = RandomizedSearchCV(dnn_RG2, parameters, n_jobs=-1, cv=5)
    dnn_RG2_clf.fit(RG2_X_res, RG2_y_res)
    dnn_RG3_clf = RandomizedSearchCV(dnn_RG3, parameters, n_jobs=-1, cv=5)
    dnn_RG3_clf.fit(RG3_X_res, RG3_y_res)
    dnn_RGM_clf = RandomizedSearchCV(dnn_RGM, parameters, n_jobs=-1, cv=5)
    dnn_RGM_clf.fit(RGM_X_res, RGM_y_res)
    dnn_RGQC_clf = RandomizedSearchCV(dnn_RGQC, parameters, n_jobs=-1, cv=5)
    dnn_RGQC_clf.fit(RGQC_X_res, RGQC_y_res)
    dnn_T8_clf = RandomizedSearchCV(dnn_T8, parameters, n_jobs=-1, cv=5)
    dnn_T8_clf.fit(T8_X_res, T8_y_res)
    dnn_TM10_clf = RandomizedSearchCV(dnn_TM10, parameters, n_jobs=-1, cv=5)
    dnn_TM10_clf.fit(TM10_X_res, TM10_y_res)
    dnn_TM4_clf = RandomizedSearchCV(dnn_TM4, parameters, n_jobs=-1, cv=5)
    dnn_TM4_clf.fit(TM4_X_res, TM4_y_res)
    dnn_TM5_clf = RandomizedSearchCV(dnn_TM5, parameters, n_jobs=-1, cv=5)
    dnn_TM5_clf.fit(TM5_X_res, TM5_y_res)
    dnn_TM6_clf = RandomizedSearchCV(dnn_TM6, parameters, n_jobs=-1, cv=5)
    dnn_TM6_clf.fit(TM6_X_res, TM6_y_res)
    dnn_TM8_clf = RandomizedSearchCV(dnn_TM8, parameters, n_jobs=-1, cv=5)
    dnn_TM8_clf.fit(TM8_X_res, TM8_y_res)
    dnn_TM9_clf = RandomizedSearchCV(dnn_TM9, parameters, n_jobs=-1, cv=5)
    dnn_TM9_clf.fit(TM9_X_res, TM9_y_res)
    dnn_TMQC_clf = RandomizedSearchCV(dnn_TMQC, parameters, n_jobs=-1, cv=5)
    dnn_TMQC_clf.fit(TMQC_X_res, TMQC_y_res)
    dnn_TQC_clf = RandomizedSearchCV(dnn_TQC, parameters, n_jobs=-1, cv=5)
    dnn_TQC_clf.fit(TQC_X_res, TQC_y_res)

    dnn_params_hls_FI[repeat] = dnn_FI_clf.best_params_['hidden_layer_sizes']
    dnn_params_hls_FG[repeat] = dnn_FG_clf.best_params_['hidden_layer_sizes']
    dnn_params_hls_GR27[repeat] = dnn_GR27_clf.best_params_['hidden_layer_sizes']
    dnn_params_hls_LM[repeat] = dnn_LM_clf.best_params_['hidden_layer_sizes'] 
    dnn_params_hls_LMM[repeat] = dnn_LMM_clf.best_params_['hidden_layer_sizes'] 
    dnn_params_hls_PC[repeat] = dnn_PC_clf.best_params_['hidden_layer_sizes'] 
    dnn_params_hls_RG12[repeat] = dnn_RG12_clf.best_params_['hidden_layer_sizes'] 
    dnn_params_hls_RG2[repeat] = dnn_RG2_clf.best_params_['hidden_layer_sizes'] 
    dnn_params_hls_RG3[repeat] = dnn_RG3_clf.best_params_['hidden_layer_sizes'] 
    dnn_params_hls_RGM[repeat] = dnn_RGM_clf.best_params_['hidden_layer_sizes'] 
    dnn_params_hls_RGQC[repeat] = dnn_RGQC_clf.best_params_['hidden_layer_sizes'] 
    dnn_params_hls_T8[repeat] = dnn_T8_clf.best_params_['hidden_layer_sizes'] 
    dnn_params_hls_TM10[repeat] = dnn_TM10_clf.best_params_['hidden_layer_sizes'] 
    dnn_params_hls_TM4[repeat] = dnn_TM4_clf.best_params_['hidden_layer_sizes'] 
    dnn_params_hls_TM5[repeat] = dnn_TM5_clf.best_params_['hidden_layer_sizes'] 
    dnn_params_hls_TM6[repeat] = dnn_TM6_clf.best_params_['hidden_layer_sizes'] 
    dnn_params_hls_TM8[repeat] = dnn_TM8_clf.best_params_['hidden_layer_sizes'] 
    dnn_params_hls_TM9[repeat] = dnn_TM9_clf.best_params_['hidden_layer_sizes'] 
    dnn_params_hls_TMQC[repeat] = dnn_TMQC_clf.best_params_['hidden_layer_sizes'] 
    dnn_params_hls_TQC[repeat] = dnn_TQC_clf.best_params_['hidden_layer_sizes'] 
    dnn_params_lri_FI[repeat] = dnn_FI_clf.best_params_['learning_rate_init']
    dnn_params_lri_FG[repeat] = dnn_FG_clf.best_params_['learning_rate_init']
    dnn_params_lri_GR27[repeat] = dnn_GR27_clf.best_params_['learning_rate_init']
    dnn_params_lri_LM[repeat] = dnn_LM_clf.best_params_['learning_rate_init'] 
    dnn_params_lri_LMM[repeat] = dnn_LMM_clf.best_params_['learning_rate_init'] 
    dnn_params_lri_PC[repeat] = dnn_PC_clf.best_params_['learning_rate_init'] 
    dnn_params_lri_RG12[repeat] = dnn_RG12_clf.best_params_['learning_rate_init'] 
    dnn_params_lri_RG2[repeat] = dnn_RG2_clf.best_params_['learning_rate_init'] 
    dnn_params_lri_RG3[repeat] = dnn_RG3_clf.best_params_['learning_rate_init'] 
    dnn_params_lri_RGM[repeat] = dnn_RGM_clf.best_params_['learning_rate_init'] 
    dnn_params_lri_RGQC[repeat] = dnn_RGQC_clf.best_params_['learning_rate_init'] 
    dnn_params_lri_T8[repeat] = dnn_T8_clf.best_params_['learning_rate_init'] 
    dnn_params_lri_TM10[repeat] = dnn_TM10_clf.best_params_['learning_rate_init'] 
    dnn_params_lri_TM4[repeat] = dnn_TM4_clf.best_params_['learning_rate_init'] 
    dnn_params_lri_TM5[repeat] = dnn_TM5_clf.best_params_['learning_rate_init'] 
    dnn_params_lri_TM6[repeat] = dnn_TM6_clf.best_params_['learning_rate_init'] 
    dnn_params_lri_TM8[repeat] = dnn_TM8_clf.best_params_['learning_rate_init'] 
    dnn_params_lri_TM9[repeat] = dnn_TM9_clf.best_params_['learning_rate_init'] 
    dnn_params_lri_TMQC[repeat] = dnn_TMQC_clf.best_params_['learning_rate_init'] 
    dnn_params_lri_TQC[repeat] = dnn_TQC_clf.best_params_['learning_rate_init'] 

    solver = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
    tol = [1e-2, 1e-3, 1e-4, 1e-5]
    reg_strength = [0.5, 1.0, 1.5]
    parameters = {'solver': solver,
	          'tol': tol,
	          'C': reg_strength}
    lr_FI,lr_FG,lr_GR27,lr_LM,lr_LMM,lr_PC,lr_RG12,lr_RG2,lr_RG3,lr_RGM,lr_RGQC,lr_T8,lr_TM10,lr_TM4,lr_TM5,lr_TM6,lr_TM8,lr_TM9,lr_TMQC,lr_TQC = LogisticRegression(),LogisticRegression(),LogisticRegression(),LogisticRegression(),LogisticRegression(),LogisticRegression(),LogisticRegression(),LogisticRegression(),LogisticRegression(),LogisticRegression(),LogisticRegression(),LogisticRegression(),LogisticRegression(),LogisticRegression(),LogisticRegression(),LogisticRegression(),LogisticRegression(),LogisticRegression(),LogisticRegression(),LogisticRegression()
    lr_FI_clf = RandomizedSearchCV(lr_FI, parameters, n_jobs=-1, cv=5)
    lr_FI_clf.fit(FI_X_res, FI_y_res)
    lr_FG_clf = RandomizedSearchCV(lr_FG, parameters, n_jobs=-1, cv=5)
    lr_FG_clf.fit(FG_X_res, FG_y_res)
    lr_GR27_clf = RandomizedSearchCV(lr_GR27, parameters, n_jobs=-1, cv=5)
    lr_GR27_clf.fit(GR27_X_res, GR27_y_res)
    lr_LM_clf = RandomizedSearchCV(lr_LM, parameters, n_jobs=-1, cv=5)
    lr_LM_clf.fit(LM_X_res, LM_y_res)
    lr_LMM_clf = RandomizedSearchCV(lr_LMM, parameters, n_jobs=-1, cv=5)
    lr_LMM_clf.fit(LMM_X_res, LMM_y_res)
    lr_PC_clf = RandomizedSearchCV(lr_PC, parameters, n_jobs=-1, cv=5)
    lr_PC_clf.fit(PC_X_res, PC_y_res)
    lr_RG12_clf = RandomizedSearchCV(lr_RG12, parameters, n_jobs=-1, cv=5)
    lr_RG12_clf.fit(RG12_X_res, RG12_y_res)
    lr_RG2_clf = RandomizedSearchCV(lr_RG2, parameters, n_jobs=-1, cv=5)
    lr_RG2_clf.fit(RG2_X_res, RG2_y_res)
    lr_RG3_clf = RandomizedSearchCV(lr_RG3, parameters, n_jobs=-1, cv=5)
    lr_RG3_clf.fit(RG3_X_res, RG3_y_res)
    lr_RGM_clf = RandomizedSearchCV(lr_RGM, parameters, n_jobs=-1, cv=5)
    lr_RGM_clf.fit(RGM_X_res, RGM_y_res)
    lr_RGQC_clf = RandomizedSearchCV(lr_RGQC, parameters, n_jobs=-1, cv=5)
    lr_RGQC_clf.fit(RGQC_X_res, RGQC_y_res)
    lr_T8_clf = RandomizedSearchCV(lr_T8, parameters, n_jobs=-1, cv=5)
    lr_T8_clf.fit(T8_X_res, T8_y_res)
    lr_TM10_clf = RandomizedSearchCV(lr_TM10, parameters, n_jobs=-1, cv=5)
    lr_TM10_clf.fit(TM10_X_res, TM10_y_res)
    lr_TM4_clf = RandomizedSearchCV(lr_TM4, parameters, n_jobs=-1, cv=5)
    lr_TM4_clf.fit(TM4_X_res, TM4_y_res)
    lr_TM5_clf = RandomizedSearchCV(lr_TM5, parameters, n_jobs=-1, cv=5)
    lr_TM5_clf.fit(TM5_X_res, TM5_y_res)
    lr_TM6_clf = RandomizedSearchCV(lr_TM6, parameters, n_jobs=-1, cv=5)
    lr_TM6_clf.fit(TM6_X_res, TM6_y_res)
    lr_TM8_clf = RandomizedSearchCV(lr_TM8, parameters, n_jobs=-1, cv=5)
    lr_TM8_clf.fit(TM8_X_res, TM8_y_res)
    lr_TM9_clf = RandomizedSearchCV(lr_TM9, parameters, n_jobs=-1, cv=5)
    lr_TM9_clf.fit(TM9_X_res, TM9_y_res)
    lr_TMQC_clf = RandomizedSearchCV(lr_TMQC, parameters, n_jobs=-1, cv=5)
    lr_TMQC_clf.fit(TMQC_X_res, TMQC_y_res)
    lr_TQC_clf = RandomizedSearchCV(lr_TQC, parameters, n_jobs=-1, cv=5)
    lr_TQC_clf.fit(TQC_X_res, TQC_y_res)

    lr_params_solver_FI[repeat] = lr_FI_clf.best_params_['solver']
    lr_params_solver_FG[repeat] = lr_FG_clf.best_params_['solver']
    lr_params_solver_GR27[repeat] = lr_GR27_clf.best_params_['solver']
    lr_params_solver_LM[repeat] = lr_LM_clf.best_params_['solver'] 
    lr_params_solver_LMM[repeat] = lr_LMM_clf.best_params_['solver'] 
    lr_params_solver_PC[repeat] = lr_PC_clf.best_params_['solver'] 
    lr_params_solver_RG12[repeat] = lr_RG12_clf.best_params_['solver'] 
    lr_params_solver_RG2[repeat] = lr_RG2_clf.best_params_['solver'] 
    lr_params_solver_RG3[repeat] = lr_RG3_clf.best_params_['solver'] 
    lr_params_solver_RGM[repeat] = lr_RGM_clf.best_params_['solver'] 
    lr_params_solver_RGQC[repeat] = lr_RGQC_clf.best_params_['solver'] 
    lr_params_solver_T8[repeat] = lr_T8_clf.best_params_['solver'] 
    lr_params_solver_TM10[repeat] = lr_TM10_clf.best_params_['solver'] 
    lr_params_solver_TM4[repeat] = lr_TM4_clf.best_params_['solver'] 
    lr_params_solver_TM5[repeat] = lr_TM5_clf.best_params_['solver'] 
    lr_params_solver_TM6[repeat] = lr_TM6_clf.best_params_['solver'] 
    lr_params_solver_TM8[repeat] = lr_TM8_clf.best_params_['solver'] 
    lr_params_solver_TM9[repeat] = lr_TM9_clf.best_params_['solver'] 
    lr_params_solver_TMQC[repeat] = lr_TMQC_clf.best_params_['solver'] 
    lr_params_solver_TQC[repeat] = lr_TQC_clf.best_params_['solver'] 
    lr_params_tol_FI[repeat] = lr_FI_clf.best_params_['tol']
    lr_params_tol_FG[repeat] = lr_FG_clf.best_params_['tol']
    lr_params_tol_GR27[repeat] = lr_GR27_clf.best_params_['tol']
    lr_params_tol_LM[repeat] = lr_LM_clf.best_params_['tol'] 
    lr_params_tol_LMM[repeat] = lr_LMM_clf.best_params_['tol'] 
    lr_params_tol_PC[repeat] = lr_PC_clf.best_params_['tol'] 
    lr_params_tol_RG12[repeat] = lr_RG12_clf.best_params_['tol'] 
    lr_params_tol_RG2[repeat] = lr_RG2_clf.best_params_['tol'] 
    lr_params_tol_RG3[repeat] = lr_RG3_clf.best_params_['tol'] 
    lr_params_tol_RGM[repeat] = lr_RGM_clf.best_params_['tol'] 
    lr_params_tol_RGQC[repeat] = lr_RGQC_clf.best_params_['tol'] 
    lr_params_tol_T8[repeat] = lr_T8_clf.best_params_['tol'] 
    lr_params_tol_TM10[repeat] = lr_TM10_clf.best_params_['tol'] 
    lr_params_tol_TM4[repeat] = lr_TM4_clf.best_params_['tol'] 
    lr_params_tol_TM5[repeat] = lr_TM5_clf.best_params_['tol'] 
    lr_params_tol_TM6[repeat] = lr_TM6_clf.best_params_['tol'] 
    lr_params_tol_TM8[repeat] = lr_TM8_clf.best_params_['tol'] 
    lr_params_tol_TM9[repeat] = lr_TM9_clf.best_params_['tol'] 
    lr_params_tol_TMQC[repeat] = lr_TMQC_clf.best_params_['tol'] 
    lr_params_tol_TQC[repeat] = lr_TQC_clf.best_params_['tol'] 
    lr_params_C_FI[repeat] = lr_FI_clf.best_params_['C']
    lr_params_C_FG[repeat] = lr_FG_clf.best_params_['C']
    lr_params_C_GR27[repeat] = lr_GR27_clf.best_params_['C']
    lr_params_C_LM[repeat] = lr_LM_clf.best_params_['C'] 
    lr_params_C_LMM[repeat] = lr_LMM_clf.best_params_['C'] 
    lr_params_C_PC[repeat] = lr_PC_clf.best_params_['C'] 
    lr_params_C_RG12[repeat] = lr_RG12_clf.best_params_['C'] 
    lr_params_C_RG2[repeat] = lr_RG2_clf.best_params_['C'] 
    lr_params_C_RG3[repeat] = lr_RG3_clf.best_params_['C'] 
    lr_params_C_RGM[repeat] = lr_RGM_clf.best_params_['C'] 
    lr_params_C_RGQC[repeat] = lr_RGQC_clf.best_params_['C'] 
    lr_params_C_T8[repeat] = lr_T8_clf.best_params_['C'] 
    lr_params_C_TM10[repeat] = lr_TM10_clf.best_params_['C'] 
    lr_params_C_TM4[repeat] = lr_TM4_clf.best_params_['C'] 
    lr_params_C_TM5[repeat] = lr_TM5_clf.best_params_['C'] 
    lr_params_C_TM6[repeat] = lr_TM6_clf.best_params_['C'] 
    lr_params_C_TM8[repeat] = lr_TM8_clf.best_params_['C'] 
    lr_params_C_TM9[repeat] = lr_TM9_clf.best_params_['C'] 
    lr_params_C_TMQC[repeat] = lr_TMQC_clf.best_params_['C'] 
    lr_params_C_TQC[repeat] = lr_TQC_clf.best_params_['C'] 

    var_smoothing = [1e-07, 1e-08, 1e-09]
    parameters = {'var_smoothing': var_smoothing}
    nb_FI,nb_FG,nb_GR27,nb_LM,nb_LMM,nb_PC,nb_RG12,nb_RG2,nb_RG3,nb_RGM,nb_RGQC,nb_T8,nb_TM10,nb_TM4,nb_TM5,nb_TM6,nb_TM8,nb_TM9,nb_TMQC,nb_TQC = GaussianNB(),GaussianNB(),GaussianNB(),GaussianNB(),GaussianNB(),GaussianNB(),GaussianNB(),GaussianNB(),GaussianNB(),GaussianNB(),GaussianNB(),GaussianNB(),GaussianNB(),GaussianNB(),GaussianNB(),GaussianNB(),GaussianNB(),GaussianNB(),GaussianNB(),GaussianNB()
    nb_FI_clf = RandomizedSearchCV(nb_FI, parameters, n_jobs = -1, cv = 5)
    nb_FI_clf.fit(FI_X_res, FI_y_res)
    nb_FG_clf = RandomizedSearchCV(nb_FG, parameters, n_jobs = -1, cv = 5)
    nb_FG_clf.fit(FG_X_res, FG_y_res)
    nb_GR27_clf = RandomizedSearchCV(nb_GR27, parameters, n_jobs = -1, cv = 5)
    nb_GR27_clf.fit(GR27_X_res, GR27_y_res)
    nb_LM_clf = RandomizedSearchCV(nb_LM, parameters, n_jobs = -1, cv = 5)
    nb_LM_clf.fit(LM_X_res, LM_y_res)
    nb_LMM_clf = RandomizedSearchCV(nb_LMM, parameters, n_jobs = -1, cv = 5)
    nb_LMM_clf.fit(LMM_X_res, LMM_y_res)
    nb_PC_clf = RandomizedSearchCV(nb_PC, parameters, n_jobs = -1, cv = 5)
    nb_PC_clf.fit(PC_X_res, PC_y_res)
    nb_RG12_clf = RandomizedSearchCV(nb_RG12, parameters, n_jobs = -1, cv = 5)
    nb_RG12_clf.fit(RG12_X_res, RG12_y_res)
    nb_RG2_clf = RandomizedSearchCV(nb_RG2, parameters, n_jobs = -1, cv = 5)
    nb_RG2_clf.fit(RG2_X_res, RG2_y_res)
    nb_RG3_clf = RandomizedSearchCV(nb_RG3, parameters, n_jobs = -1, cv = 5)
    nb_RG3_clf.fit(RG3_X_res, RG3_y_res)
    nb_RGM_clf = RandomizedSearchCV(nb_RGM, parameters, n_jobs = -1, cv = 5)
    nb_RGM_clf.fit(RGM_X_res, RGM_y_res)
    nb_RGQC_clf = RandomizedSearchCV(nb_RGQC, parameters, n_jobs = -1, cv = 5)
    nb_RGQC_clf.fit(RGQC_X_res, RGQC_y_res)
    nb_T8_clf = RandomizedSearchCV(nb_T8, parameters, n_jobs = -1, cv = 5)
    nb_T8_clf.fit(T8_X_res, T8_y_res)
    nb_TM10_clf = RandomizedSearchCV(nb_TM10, parameters, n_jobs = -1, cv = 5)
    nb_TM10_clf.fit(TM10_X_res, TM10_y_res)
    nb_TM4_clf = RandomizedSearchCV(nb_TM4, parameters, n_jobs = -1, cv = 5)
    nb_TM4_clf.fit(TM4_X_res, TM4_y_res)
    nb_TM5_clf = RandomizedSearchCV(nb_TM5, parameters, n_jobs = -1, cv = 5)
    nb_TM5_clf.fit(TM5_X_res, TM5_y_res)
    nb_TM6_clf = RandomizedSearchCV(nb_TM6, parameters, n_jobs = -1, cv = 5)
    nb_TM6_clf.fit(TM6_X_res, TM6_y_res)
    nb_TM8_clf = RandomizedSearchCV(nb_TM8, parameters, n_jobs = -1, cv = 5)
    nb_TM8_clf.fit(TM8_X_res, TM8_y_res)
    nb_TM9_clf = RandomizedSearchCV(nb_TM9, parameters, n_jobs = -1, cv = 5)
    nb_TM9_clf.fit(TM9_X_res, TM9_y_res)
    nb_TMQC_clf = RandomizedSearchCV(nb_TMQC, parameters, n_jobs = -1, cv = 5)
    nb_TMQC_clf.fit(TMQC_X_res, TMQC_y_res)
    nb_TQC_clf = RandomizedSearchCV(nb_TQC, parameters, n_jobs = -1, cv = 5)
    nb_TQC_clf.fit(TQC_X_res, TQC_y_res)

    nb_params_vs_FI[repeat] = nb_FI_clf.best_params_['var_smoothing']
    nb_params_vs_FG[repeat] = nb_FG_clf.best_params_['var_smoothing']
    nb_params_vs_GR27[repeat] = nb_GR27_clf.best_params_['var_smoothing']
    nb_params_vs_LM[repeat] = nb_LM_clf.best_params_['var_smoothing'] 
    nb_params_vs_LMM[repeat] = nb_LMM_clf.best_params_['var_smoothing'] 
    nb_params_vs_PC[repeat] = nb_PC_clf.best_params_['var_smoothing'] 
    nb_params_vs_RG12[repeat] = nb_RG12_clf.best_params_['var_smoothing'] 
    nb_params_vs_RG2[repeat] = nb_RG2_clf.best_params_['var_smoothing'] 
    nb_params_vs_RG3[repeat] = nb_RG3_clf.best_params_['var_smoothing'] 
    nb_params_vs_RGM[repeat] = nb_RGM_clf.best_params_['var_smoothing'] 
    nb_params_vs_RGQC[repeat] = nb_RGQC_clf.best_params_['var_smoothing'] 
    nb_params_vs_T8[repeat] = nb_T8_clf.best_params_['var_smoothing'] 
    nb_params_vs_TM10[repeat] = nb_TM10_clf.best_params_['var_smoothing'] 
    nb_params_vs_TM4[repeat] = nb_TM4_clf.best_params_['var_smoothing'] 
    nb_params_vs_TM5[repeat] = nb_TM5_clf.best_params_['var_smoothing'] 
    nb_params_vs_TM6[repeat] = nb_TM6_clf.best_params_['var_smoothing'] 
    nb_params_vs_TM8[repeat] = nb_TM8_clf.best_params_['var_smoothing'] 
    nb_params_vs_TM9[repeat] = nb_TM9_clf.best_params_['var_smoothing'] 
    nb_params_vs_TMQC[repeat] = nb_TMQC_clf.best_params_['var_smoothing'] 
    nb_params_vs_TQC[repeat] = nb_TQC_clf.best_params_['var_smoothing'] 

    n_tree = [50, 100, 200, 300, 400, 500, 600, 700]
    max_depth = [10, 20, 30, 40, 50, 60, 70]
    min_samples_split = [5, 10, 15, 20, 25, 30]
    parameters = {'n_estimators': n_tree,
		  'max_depth': max_depth,
		  'min_samples_split': min_samples_split}
    rf_FI,rf_FG,rf_GR27,rf_LM,rf_LMM,rf_PC,rf_RG12,rf_RG2,rf_RG3,rf_RGM,rf_RGQC,rf_T8,rf_TM10,rf_TM4,rf_TM5,rf_TM6,rf_TM8,rf_TM9,rf_TMQC,rf_TQC = RandomForestClassifier(),RandomForestClassifier(),RandomForestClassifier(),RandomForestClassifier(),RandomForestClassifier(),RandomForestClassifier(),RandomForestClassifier(),RandomForestClassifier(),RandomForestClassifier(),RandomForestClassifier(),RandomForestClassifier(),RandomForestClassifier(),RandomForestClassifier(),RandomForestClassifier(),RandomForestClassifier(),RandomForestClassifier(),RandomForestClassifier(),RandomForestClassifier(),RandomForestClassifier(),RandomForestClassifier()
    rf_FI_clf = RandomizedSearchCV(rf_FI, parameters, n_jobs=-1, cv=5)
    rf_FI_clf.fit(FI_X_res, FI_y_res)
    rf_FG_clf = RandomizedSearchCV(rf_FG, parameters, n_jobs=-1, cv=5)
    rf_FG_clf.fit(FG_X_res, FG_y_res)
    rf_GR27_clf = RandomizedSearchCV(rf_GR27, parameters, n_jobs=-1, cv=5)
    rf_GR27_clf.fit(GR27_X_res, GR27_y_res)
    rf_LM_clf = RandomizedSearchCV(rf_LM, parameters, n_jobs=-1, cv=5)
    rf_LM_clf.fit(LM_X_res, LM_y_res)
    rf_LMM_clf = RandomizedSearchCV(rf_LMM, parameters, n_jobs=-1, cv=5)
    rf_LMM_clf.fit(LMM_X_res, LMM_y_res)
    rf_PC_clf = RandomizedSearchCV(rf_PC, parameters, n_jobs=-1, cv=5)
    rf_PC_clf.fit(PC_X_res, PC_y_res)
    rf_RG12_clf = RandomizedSearchCV(rf_RG12, parameters, n_jobs=-1, cv=5)
    rf_RG12_clf.fit(RG12_X_res, RG12_y_res)
    rf_RG2_clf = RandomizedSearchCV(rf_RG2, parameters, n_jobs=-1, cv=5)
    rf_RG2_clf.fit(RG2_X_res, RG2_y_res)
    rf_RG3_clf = RandomizedSearchCV(rf_RG3, parameters, n_jobs=-1, cv=5)
    rf_RG3_clf.fit(RG3_X_res, RG3_y_res)
    rf_RGM_clf = RandomizedSearchCV(rf_RGM, parameters, n_jobs=-1, cv=5)
    rf_RGM_clf.fit(RGM_X_res, RGM_y_res)
    rf_RGQC_clf = RandomizedSearchCV(rf_RGQC, parameters, n_jobs=-1, cv=5)
    rf_RGQC_clf.fit(RGQC_X_res, RGQC_y_res)
    rf_T8_clf = RandomizedSearchCV(rf_T8, parameters, n_jobs=-1, cv=5)
    rf_T8_clf.fit(T8_X_res, T8_y_res)
    rf_TM10_clf = RandomizedSearchCV(rf_TM10, parameters, n_jobs=-1, cv=5)
    rf_TM10_clf.fit(TM10_X_res, TM10_y_res)
    rf_TM4_clf = RandomizedSearchCV(rf_TM4, parameters, n_jobs=-1, cv=5)
    rf_TM4_clf.fit(TM4_X_res, TM4_y_res)
    rf_TM5_clf = RandomizedSearchCV(rf_TM5, parameters, n_jobs=-1, cv=5)
    rf_TM5_clf.fit(TM5_X_res, TM5_y_res)
    rf_TM6_clf = RandomizedSearchCV(rf_TM6, parameters, n_jobs=-1, cv=5)
    rf_TM6_clf.fit(TM6_X_res, TM6_y_res)
    rf_TM8_clf = RandomizedSearchCV(rf_TM8, parameters, n_jobs=-1, cv=5)
    rf_TM8_clf.fit(TM8_X_res, TM8_y_res)
    rf_TM9_clf = RandomizedSearchCV(rf_TM9, parameters, n_jobs=-1, cv=5)
    rf_TM9_clf.fit(TM9_X_res, TM9_y_res)
    rf_TMQC_clf = RandomizedSearchCV(rf_TMQC, parameters, n_jobs=-1, cv=5)
    rf_TMQC_clf.fit(TMQC_X_res, TMQC_y_res)
    rf_TQC_clf = RandomizedSearchCV(rf_TQC, parameters, n_jobs=-1, cv=5)
    rf_TQC_clf.fit(TQC_X_res, TQC_y_res)

    rf_params_est_FI[repeat] = rf_FI_clf.best_params_['n_estimators']
    rf_params_est_FG[repeat] = rf_FG_clf.best_params_['n_estimators']
    rf_params_est_GR27[repeat] = rf_GR27_clf.best_params_['n_estimators']
    rf_params_est_LM[repeat] = rf_LM_clf.best_params_['n_estimators'] 
    rf_params_est_LMM[repeat] = rf_LMM_clf.best_params_['n_estimators'] 
    rf_params_est_PC[repeat] = rf_PC_clf.best_params_['n_estimators'] 
    rf_params_est_RG12[repeat] = rf_RG12_clf.best_params_['n_estimators'] 
    rf_params_est_RG2[repeat] = rf_RG2_clf.best_params_['n_estimators'] 
    rf_params_est_RG3[repeat] = rf_RG3_clf.best_params_['n_estimators'] 
    rf_params_est_RGM[repeat] = rf_RGM_clf.best_params_['n_estimators'] 
    rf_params_est_RGQC[repeat] = rf_RGQC_clf.best_params_['n_estimators'] 
    rf_params_est_T8[repeat] = rf_T8_clf.best_params_['n_estimators'] 
    rf_params_est_TM10[repeat] = rf_TM10_clf.best_params_['n_estimators'] 
    rf_params_est_TM4[repeat] = rf_TM4_clf.best_params_['n_estimators'] 
    rf_params_est_TM5[repeat] = rf_TM5_clf.best_params_['n_estimators'] 
    rf_params_est_TM6[repeat] = rf_TM6_clf.best_params_['n_estimators'] 
    rf_params_est_TM8[repeat] = rf_TM8_clf.best_params_['n_estimators'] 
    rf_params_est_TM9[repeat] = rf_TM9_clf.best_params_['n_estimators'] 
    rf_params_est_TMQC[repeat] = rf_TMQC_clf.best_params_['n_estimators'] 
    rf_params_est_TQC[repeat] = rf_TQC_clf.best_params_['n_estimators'] 
    rf_params_md_FI[repeat] = rf_FI_clf.best_params_['max_depth']
    rf_params_md_FG[repeat] = rf_FG_clf.best_params_['max_depth']
    rf_params_md_GR27[repeat] = rf_GR27_clf.best_params_['max_depth']
    rf_params_md_LM[repeat] = rf_LM_clf.best_params_['max_depth'] 
    rf_params_md_LMM[repeat] = rf_LMM_clf.best_params_['max_depth'] 
    rf_params_md_PC[repeat] = rf_PC_clf.best_params_['max_depth'] 
    rf_params_md_RG12[repeat] = rf_RG12_clf.best_params_['max_depth'] 
    rf_params_md_RG2[repeat] = rf_RG2_clf.best_params_['max_depth'] 
    rf_params_md_RG3[repeat] = rf_RG3_clf.best_params_['max_depth'] 
    rf_params_md_RGM[repeat] = rf_RGM_clf.best_params_['max_depth'] 
    rf_params_md_RGQC[repeat] = rf_RGQC_clf.best_params_['max_depth'] 
    rf_params_md_T8[repeat] = rf_T8_clf.best_params_['max_depth'] 
    rf_params_md_TM10[repeat] = rf_TM10_clf.best_params_['max_depth'] 
    rf_params_md_TM4[repeat] = rf_TM4_clf.best_params_['max_depth'] 
    rf_params_md_TM5[repeat] = rf_TM5_clf.best_params_['max_depth'] 
    rf_params_md_TM6[repeat] = rf_TM6_clf.best_params_['max_depth'] 
    rf_params_md_TM8[repeat] = rf_TM8_clf.best_params_['max_depth'] 
    rf_params_md_TM9[repeat] = rf_TM9_clf.best_params_['max_depth'] 
    rf_params_md_TMQC[repeat] = rf_TMQC_clf.best_params_['max_depth'] 
    rf_params_md_TQC[repeat] = rf_TQC_clf.best_params_['max_depth'] 
    rf_params_mss_FI[repeat] = rf_FI_clf.best_params_['min_samples_split']
    rf_params_mss_FG[repeat] = rf_FG_clf.best_params_['min_samples_split']
    rf_params_mss_GR27[repeat] = rf_GR27_clf.best_params_['min_samples_split']
    rf_params_mss_LM[repeat] = rf_LM_clf.best_params_['min_samples_split'] 
    rf_params_mss_LMM[repeat] = rf_LMM_clf.best_params_['min_samples_split'] 
    rf_params_mss_PC[repeat] = rf_PC_clf.best_params_['min_samples_split'] 
    rf_params_mss_RG12[repeat] = rf_RG12_clf.best_params_['min_samples_split'] 
    rf_params_mss_RG2[repeat] = rf_RG2_clf.best_params_['min_samples_split'] 
    rf_params_mss_RG3[repeat] = rf_RG3_clf.best_params_['min_samples_split'] 
    rf_params_mss_RGM[repeat] = rf_RGM_clf.best_params_['min_samples_split'] 
    rf_params_mss_RGQC[repeat] = rf_RGQC_clf.best_params_['min_samples_split'] 
    rf_params_mss_T8[repeat] = rf_T8_clf.best_params_['min_samples_split'] 
    rf_params_mss_TM10[repeat] = rf_TM10_clf.best_params_['min_samples_split'] 
    rf_params_mss_TM4[repeat] = rf_TM4_clf.best_params_['min_samples_split'] 
    rf_params_mss_TM5[repeat] = rf_TM5_clf.best_params_['min_samples_split'] 
    rf_params_mss_TM6[repeat] = rf_TM6_clf.best_params_['min_samples_split'] 
    rf_params_mss_TM8[repeat] = rf_TM8_clf.best_params_['min_samples_split'] 
    rf_params_mss_TM9[repeat] = rf_TM9_clf.best_params_['min_samples_split'] 
    rf_params_mss_TMQC[repeat] = rf_TMQC_clf.best_params_['min_samples_split'] 
    rf_params_mss_TQC[repeat] = rf_TQC_clf.best_params_['min_samples_split'] 

    tol = [1e-2, 1e-3, 1e-4]
    parameters = {'tol': tol,
                  'kernel': ['linear'],
                  'probability': [True]}
    svm_FI,svm_FG,svm_GR27,svm_LM,svm_LMM,svm_PC,svm_RG12,svm_RG2,svm_RG3,svm_RGM,svm_RGQC,svm_T8,svm_TM10,svm_TM4,svm_TM5,svm_TM6,svm_TM8,svm_TM9,svm_TMQC,svm_TQC = SVC(),SVC(),SVC(),SVC(),SVC(),SVC(),SVC(),SVC(),SVC(),SVC(),SVC(),SVC(),SVC(),SVC(),SVC(),SVC(),SVC(),SVC(),SVC(),SVC()
    svm_FI_clf = RandomizedSearchCV(svm_FI, parameters, n_jobs=-1, cv=5)
    svm_FI_clf.fit(FI_X_res, FI_y_res)
    svm_FG_clf = RandomizedSearchCV(svm_FG, parameters, n_jobs=-1, cv=5)
    svm_FG_clf.fit(FG_X_res, FG_y_res)
    svm_GR27_clf = RandomizedSearchCV(svm_GR27, parameters, n_jobs=-1, cv=5)
    svm_GR27_clf.fit(GR27_X_res, GR27_y_res)
    svm_LM_clf = RandomizedSearchCV(svm_LM, parameters, n_jobs=-1, cv=5)
    svm_LM_clf.fit(LM_X_res, LM_y_res)
    svm_LMM_clf = RandomizedSearchCV(svm_LMM, parameters, n_jobs=-1, cv=5)
    svm_LMM_clf.fit(LMM_X_res, LMM_y_res)
    svm_PC_clf = RandomizedSearchCV(svm_PC, parameters, n_jobs=-1, cv=5)
    svm_PC_clf.fit(PC_X_res, PC_y_res)
    svm_RG12_clf = RandomizedSearchCV(svm_RG12, parameters, n_jobs=-1, cv=5)
    svm_RG12_clf.fit(RG12_X_res, RG12_y_res)
    svm_RG2_clf = RandomizedSearchCV(svm_RG2, parameters, n_jobs=-1, cv=5)
    svm_RG2_clf.fit(RG2_X_res, RG2_y_res)
    svm_RG3_clf = RandomizedSearchCV(svm_RG3, parameters, n_jobs=-1, cv=5)
    svm_RG3_clf.fit(RG3_X_res, RG3_y_res)
    svm_RGM_clf = RandomizedSearchCV(svm_RGM, parameters, n_jobs=-1, cv=5)
    svm_RGM_clf.fit(RGM_X_res, RGM_y_res)
    svm_RGQC_clf = RandomizedSearchCV(svm_RGQC, parameters, n_jobs=-1, cv=5)
    svm_RGQC_clf.fit(RGQC_X_res, RGQC_y_res)
    svm_T8_clf = RandomizedSearchCV(svm_T8, parameters, n_jobs=-1, cv=5)
    svm_T8_clf.fit(T8_X_res, T8_y_res)
    svm_TM10_clf = RandomizedSearchCV(svm_TM10, parameters, n_jobs=-1, cv=5)
    svm_TM10_clf.fit(TM10_X_res, TM10_y_res)
    svm_TM4_clf = RandomizedSearchCV(svm_TM4, parameters, n_jobs=-1, cv=5)
    svm_TM4_clf.fit(TM4_X_res, TM4_y_res)
    svm_TM5_clf = RandomizedSearchCV(svm_TM5, parameters, n_jobs=-1, cv=5)
    svm_TM5_clf.fit(TM5_X_res, TM5_y_res)
    svm_TM6_clf = RandomizedSearchCV(svm_TM6, parameters, n_jobs=-1, cv=5)
    svm_TM6_clf.fit(TM6_X_res, TM6_y_res)
    svm_TM8_clf = RandomizedSearchCV(svm_TM8, parameters, n_jobs=-1, cv=5)
    svm_TM8_clf.fit(TM8_X_res, TM8_y_res)
    svm_TM9_clf = RandomizedSearchCV(svm_TM9, parameters, n_jobs=-1, cv=5)
    svm_TM9_clf.fit(TM9_X_res, TM9_y_res)
    svm_TMQC_clf = RandomizedSearchCV(svm_TMQC, parameters, n_jobs=-1, cv=5)
    svm_TMQC_clf.fit(TMQC_X_res, TMQC_y_res)
    svm_TQC_clf = RandomizedSearchCV(svm_TQC, parameters, n_jobs=-1, cv=5)
    svm_TQC_clf.fit(TQC_X_res, TQC_y_res)

    svm_params_tol_FI[repeat] = svm_FI_clf.best_params_['tol']
    svm_params_tol_FG[repeat] = svm_FG_clf.best_params_['tol']
    svm_params_tol_GR27[repeat] = svm_GR27_clf.best_params_['tol']
    svm_params_tol_LM[repeat] = svm_LM_clf.best_params_['tol'] 
    svm_params_tol_LMM[repeat] = svm_LMM_clf.best_params_['tol'] 
    svm_params_tol_PC[repeat] = svm_PC_clf.best_params_['tol'] 
    svm_params_tol_RG12[repeat] = svm_RG12_clf.best_params_['tol'] 
    svm_params_tol_RG2[repeat] = svm_RG2_clf.best_params_['tol'] 
    svm_params_tol_RG3[repeat] = svm_RG3_clf.best_params_['tol'] 
    svm_params_tol_RGM[repeat] = svm_RGM_clf.best_params_['tol'] 
    svm_params_tol_RGQC[repeat] = svm_RGQC_clf.best_params_['tol'] 
    svm_params_tol_T8[repeat] = svm_T8_clf.best_params_['tol'] 
    svm_params_tol_TM10[repeat] = svm_TM10_clf.best_params_['tol'] 
    svm_params_tol_TM4[repeat] = svm_TM4_clf.best_params_['tol'] 
    svm_params_tol_TM5[repeat] = svm_TM5_clf.best_params_['tol'] 
    svm_params_tol_TM6[repeat] = svm_TM6_clf.best_params_['tol'] 
    svm_params_tol_TM8[repeat] = svm_TM8_clf.best_params_['tol'] 
    svm_params_tol_TM9[repeat] = svm_TM9_clf.best_params_['tol'] 
    svm_params_tol_TMQC[repeat] = svm_TMQC_clf.best_params_['tol'] 
    svm_params_tol_TQC[repeat] = svm_TQC_clf.best_params_['tol'] 

    def prediction(FI_clf,FG_clf,GR27_clf,LM_clf,LMM_clf,PC_clf,RG12_clf,RG2_clf,RG3_clf,RGM_clf,RGQC_clf,T8_clf,TM10_clf,TM4_clf,TM5_clf,TM6_clf,TM8_clf,TM9_clf,TMQC_clf,TQC_clf, X_test):
        pred_class_FI = FI_clf.predict(X_test)
        pred_prob_FI = FI_clf.predict_proba(X_test)
        pred_class_FG = FG_clf.predict(X_test)
        pred_prob_FG = FG_clf.predict_proba(X_test)
        pred_class_GR27 = GR27_clf.predict(X_test)
        pred_prob_GR27 = GR27_clf.predict_proba(X_test)
        pred_class_LM = LM_clf.predict(X_test)
        pred_prob_LM = LM_clf.predict_proba(X_test)
        pred_class_LMM = LMM_clf.predict(X_test)
        pred_prob_LMM = LMM_clf.predict_proba(X_test)
        pred_class_PC = PC_clf.predict(X_test)
        pred_prob_PC = PC_clf.predict_proba(X_test)
        pred_class_RG12 = RG12_clf.predict(X_test)
        pred_prob_RG12 = RG12_clf.predict_proba(X_test)
        pred_class_RG2 = RG2_clf.predict(X_test)
        pred_prob_RG2 = RG2_clf.predict_proba(X_test)
        pred_class_RG3 = RG3_clf.predict(X_test)
        pred_prob_RG3 = RG3_clf.predict_proba(X_test)
        pred_class_RGM = RGM_clf.predict(X_test)
        pred_prob_RGM = RGM_clf.predict_proba(X_test)
        pred_class_RGQC = RGQC_clf.predict(X_test)
        pred_prob_RGQC = RGQC_clf.predict_proba(X_test)
        pred_class_T8 = T8_clf.predict(X_test)
        pred_prob_T8 = T8_clf.predict_proba(X_test)
        pred_class_TM10 = TM10_clf.predict(X_test)
        pred_prob_TM10 = TM10_clf.predict_proba(X_test)
        pred_class_TM4 = TM4_clf.predict(X_test)
        pred_prob_TM4 = TM4_clf.predict_proba(X_test)
        pred_class_TM5 = TM5_clf.predict(X_test)
        pred_prob_TM5 = TM5_clf.predict_proba(X_test)
        pred_class_TM6 = TM6_clf.predict(X_test)
        pred_prob_TM6 = TM6_clf.predict_proba(X_test)
        pred_class_TM8 = TM8_clf.predict(X_test)
        pred_prob_TM8 = TM8_clf.predict_proba(X_test)
        pred_class_TM9 = TM9_clf.predict(X_test)
        pred_prob_TM9 = TM9_clf.predict_proba(X_test)
        pred_class_TMQC = TMQC_clf.predict(X_test)
        pred_prob_TMQC = TMQC_clf.predict_proba(X_test)
        pred_class_TQC = TQC_clf.predict(X_test)
        pred_prob_TQC = TQC_clf.predict_proba(X_test)
        return pred_class_FI,pred_prob_FI,pred_class_FG,pred_prob_FG,pred_class_GR27,pred_prob_GR27,pred_class_LM,pred_prob_LM,pred_class_LMM,pred_prob_LMM,pred_class_PC,pred_prob_PC,pred_class_RG12,pred_prob_RG12,pred_class_RG2,pred_prob_RG2,pred_class_RG3,pred_prob_RG3,pred_class_RGM,pred_prob_RGM,pred_class_RGQC,pred_prob_RGQC,pred_class_T8,pred_prob_T8,pred_class_TM10,pred_prob_TM10,pred_class_TM4,pred_prob_TM4,pred_class_TM5,pred_prob_TM5,pred_class_TM6,pred_prob_TM6,pred_class_TM8,pred_prob_TM8,pred_class_TM9,pred_prob_TM9,pred_class_TMQC,pred_prob_TMQC,pred_class_TQC,pred_prob_TQC

    dnn_pred_class_FI,dnn_pred_prob_FI,dnn_pred_class_FG,dnn_pred_prob_FG,dnn_pred_class_GR27,dnn_pred_prob_GR27,dnn_pred_class_LM,dnn_pred_prob_LM,dnn_pred_class_LMM,dnn_pred_prob_LMM,dnn_pred_class_PC,dnn_pred_prob_PC,dnn_pred_class_RG12,dnn_pred_prob_RG12,dnn_pred_class_RG2,dnn_pred_prob_RG2,dnn_pred_class_RG3,dnn_pred_prob_RG3,dnn_pred_class_RGM,dnn_pred_prob_RGM,dnn_pred_class_RGQC,dnn_pred_prob_RGQC,dnn_pred_class_T8,dnn_pred_prob_T8,dnn_pred_class_TM10,dnn_pred_prob_TM10,dnn_pred_class_TM4,dnn_pred_prob_TM4,dnn_pred_class_TM5,dnn_pred_prob_TM5,dnn_pred_class_TM6,dnn_pred_prob_TM6,dnn_pred_class_TM8,dnn_pred_prob_TM8,dnn_pred_class_TM9,dnn_pred_prob_TM9,dnn_pred_class_TMQC,dnn_pred_prob_TMQC,dnn_pred_class_TQC,dnn_pred_prob_TQC = prediction(dnn_FI_clf,dnn_FG_clf,dnn_GR27_clf,dnn_LM_clf,dnn_LMM_clf,dnn_PC_clf,dnn_RG12_clf,dnn_RG2_clf,dnn_RG3_clf,dnn_RGM_clf,dnn_RGQC_clf,dnn_T8_clf,dnn_TM10_clf,dnn_TM4_clf,dnn_TM5_clf,dnn_TM6_clf,dnn_TM8_clf,dnn_TM9_clf,dnn_TMQC_clf,dnn_TQC_clf,X_test)
    lr_pred_class_FI,lr_pred_prob_FI,lr_pred_class_FG,lr_pred_prob_FG,lr_pred_class_GR27,lr_pred_prob_GR27,lr_pred_class_LM,lr_pred_prob_LM,lr_pred_class_LMM,lr_pred_prob_LMM,lr_pred_class_PC,lr_pred_prob_PC,lr_pred_class_RG12,lr_pred_prob_RG12,lr_pred_class_RG2,lr_pred_prob_RG2,lr_pred_class_RG3,lr_pred_prob_RG3,lr_pred_class_RGM,lr_pred_prob_RGM,lr_pred_class_RGQC,lr_pred_prob_RGQC,lr_pred_class_T8,lr_pred_prob_T8,lr_pred_class_TM10,lr_pred_prob_TM10,lr_pred_class_TM4,lr_pred_prob_TM4,lr_pred_class_TM5,lr_pred_prob_TM5,lr_pred_class_TM6,lr_pred_prob_TM6,lr_pred_class_TM8,lr_pred_prob_TM8,lr_pred_class_TM9,lr_pred_prob_TM9,lr_pred_class_TMQC,lr_pred_prob_TMQC,lr_pred_class_TQC,lr_pred_prob_TQC = prediction(lr_FI_clf,lr_FG_clf,lr_GR27_clf,lr_LM_clf,lr_LMM_clf,lr_PC_clf,lr_RG12_clf,lr_RG2_clf,lr_RG3_clf,lr_RGM_clf,lr_RGQC_clf,lr_T8_clf,lr_TM10_clf,lr_TM4_clf,lr_TM5_clf,lr_TM6_clf,lr_TM8_clf,lr_TM9_clf,lr_TMQC_clf,lr_TQC_clf,X_test)
    nb_pred_class_FI,nb_pred_prob_FI,nb_pred_class_FG,nb_pred_prob_FG,nb_pred_class_GR27,nb_pred_prob_GR27,nb_pred_class_LM,nb_pred_prob_LM,nb_pred_class_LMM,nb_pred_prob_LMM,nb_pred_class_PC,nb_pred_prob_PC,nb_pred_class_RG12,nb_pred_prob_RG12,nb_pred_class_RG2,nb_pred_prob_RG2,nb_pred_class_RG3,nb_pred_prob_RG3,nb_pred_class_RGM,nb_pred_prob_RGM,nb_pred_class_RGQC,nb_pred_prob_RGQC,nb_pred_class_T8,nb_pred_prob_T8,nb_pred_class_TM10,nb_pred_prob_TM10,nb_pred_class_TM4,nb_pred_prob_TM4,nb_pred_class_TM5,nb_pred_prob_TM5,nb_pred_class_TM6,nb_pred_prob_TM6,nb_pred_class_TM8,nb_pred_prob_TM8,nb_pred_class_TM9,nb_pred_prob_TM9,nb_pred_class_TMQC,nb_pred_prob_TMQC,nb_pred_class_TQC,nb_pred_prob_TQC = prediction(nb_FI_clf,nb_FG_clf,nb_GR27_clf,nb_LM_clf,nb_LMM_clf,nb_PC_clf,nb_RG12_clf,nb_RG2_clf,nb_RG3_clf,nb_RGM_clf,nb_RGQC_clf,nb_T8_clf,nb_TM10_clf,nb_TM4_clf,nb_TM5_clf,nb_TM6_clf,nb_TM8_clf,nb_TM9_clf,nb_TMQC_clf,nb_TQC_clf,X_test)
    rf_pred_class_FI,rf_pred_prob_FI,rf_pred_class_FG,rf_pred_prob_FG,rf_pred_class_GR27,rf_pred_prob_GR27,rf_pred_class_LM,rf_pred_prob_LM,rf_pred_class_LMM,rf_pred_prob_LMM,rf_pred_class_PC,rf_pred_prob_PC,rf_pred_class_RG12,rf_pred_prob_RG12,rf_pred_class_RG2,rf_pred_prob_RG2,rf_pred_class_RG3,rf_pred_prob_RG3,rf_pred_class_RGM,rf_pred_prob_RGM,rf_pred_class_RGQC,rf_pred_prob_RGQC,rf_pred_class_T8,rf_pred_prob_T8,rf_pred_class_TM10,rf_pred_prob_TM10,rf_pred_class_TM4,rf_pred_prob_TM4,rf_pred_class_TM5,rf_pred_prob_TM5,rf_pred_class_TM6,rf_pred_prob_TM6,rf_pred_class_TM8,rf_pred_prob_TM8,rf_pred_class_TM9,rf_pred_prob_TM9,rf_pred_class_TMQC,rf_pred_prob_TMQC,rf_pred_class_TQC,rf_pred_prob_TQC = prediction(rf_FI_clf,rf_FG_clf,rf_GR27_clf,rf_LM_clf,rf_LMM_clf,rf_PC_clf,rf_RG12_clf,rf_RG2_clf,rf_RG3_clf,rf_RGM_clf,rf_RGQC_clf,rf_T8_clf,rf_TM10_clf,rf_TM4_clf,rf_TM5_clf,rf_TM6_clf,rf_TM8_clf,rf_TM9_clf,rf_TMQC_clf,rf_TQC_clf,X_test)
    svm_pred_class_FI,svm_pred_prob_FI,svm_pred_class_FG,svm_pred_prob_FG,svm_pred_class_GR27,svm_pred_prob_GR27,svm_pred_class_LM,svm_pred_prob_LM,svm_pred_class_LMM,svm_pred_prob_LMM,svm_pred_class_PC,svm_pred_prob_PC,svm_pred_class_RG12,svm_pred_prob_RG12,svm_pred_class_RG2,svm_pred_prob_RG2,svm_pred_class_RG3,svm_pred_prob_RG3,svm_pred_class_RGM,svm_pred_prob_RGM,svm_pred_class_RGQC,svm_pred_prob_RGQC,svm_pred_class_T8,svm_pred_prob_T8,svm_pred_class_TM10,svm_pred_prob_TM10,svm_pred_class_TM4,svm_pred_prob_TM4,svm_pred_class_TM5,svm_pred_prob_TM5,svm_pred_class_TM6,svm_pred_prob_TM6,svm_pred_class_TM8,svm_pred_prob_TM8,svm_pred_class_TM9,svm_pred_prob_TM9,svm_pred_class_TMQC,svm_pred_prob_TMQC,svm_pred_class_TQC,svm_pred_prob_TQC = prediction(svm_FI_clf,svm_FG_clf,svm_GR27_clf,svm_LM_clf,svm_LMM_clf,svm_PC_clf,svm_RG12_clf,svm_RG2_clf,svm_RG3_clf,svm_RGM_clf,svm_RGQC_clf,svm_T8_clf,svm_TM10_clf,svm_TM4_clf,svm_TM5_clf,svm_TM6_clf,svm_TM8_clf,svm_TM9_clf,svm_TMQC_clf,svm_TQC_clf,X_test)

    dnn_prediction, lr_prediction, nb_prediction, rf_prediction, svm_prediction = pd.DataFrame(columns=['Prediction']),pd.DataFrame(columns=['Prediction']),pd.DataFrame(columns=['Prediction']),pd.DataFrame(columns=['Prediction']),pd.DataFrame(columns=['Prediction'])

    def label_assigner(y_test, prediction, pred_class_FI,pred_class_FG,pred_class_GR27,pred_class_LM,pred_class_LMM,pred_class_PC,pred_class_RG12,pred_class_RG2,pred_class_RG3,pred_class_RGM,pred_class_RGQC,pred_class_T8,pred_class_TM10,pred_class_TM4,pred_class_TM5,pred_class_TM6,pred_class_TM8,pred_class_TM9,pred_class_TMQC,pred_class_TQC,pred_prob_FI,pred_prob_FG,pred_prob_GR27,pred_prob_LM,pred_prob_LMM,pred_prob_PC,pred_prob_RG12,pred_prob_RG2,pred_prob_RG3,pred_prob_RGM,pred_prob_RGQC,pred_prob_T8,pred_prob_TM10,pred_prob_TM4,pred_prob_TM5,pred_prob_TM6,pred_prob_TM8,pred_prob_TM9,pred_prob_TMQC,pred_prob_TQC):
        for i in range(0, len(y_test)):
            FI_index,FG_index,GR27_index,LM_index,LMM_index,PC_index,RG12_index,RG2_index,RG3_index,RGM_index,RGQC_index,T8_index,TM10_index,TM4_index,TM5_index,TM6_index,TM8_index,TM9_index,TMQC_index,TQC_index = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            if pred_class_FI[i] == "Final Inspection Q.C.":
                FI_index = 0 if pred_prob_FI[i][0] >= 0.5 else 1
            elif pred_class_FI[i] == "Others":
                FI_index = 0 if pred_prob_FI[i][0] < 0.5 else 1
            if pred_class_FG[i] == "Flat Grinding - Machine 11":
                FG_index = 0 if pred_prob_FG[i][0] >= 0.5 else 1
            elif pred_class_FG[i] == "Others":
                FG_index = 0 if pred_prob_FG[i][0] < 0.5 else 1
            if pred_class_GR27[i] == "Grinding Rework - Machine 27":
                GR27_index = 0 if pred_prob_GR27[i][0] >= 0.5 else 1
            elif pred_class_GR27[i] == "Others":
                GR27_index = 0 if pred_prob_GR27[i][0] < 0.5 else 1
            if pred_class_LM[i] == "Lapping - Machine 1":
                LM_index = 0 if pred_prob_LM[i][0] >= 0.5 else 1
            elif pred_class_LM[i] == "Others":
                LM_index = 0 if pred_prob_LM[i][0] < 0.5 else 1
            if pred_class_LMM[i] == "Laser Marking - Machine 7":
                LMM_index = 0 if pred_prob_LMM[i][0] >= 0.5 else 1
            elif pred_class_LMM[i] == "Others":
                LMM_index = 0 if pred_prob_LMM[i][0] < 0.5 else 1
            if pred_class_PC[i] == "Packing":
                PC_index = 0 if pred_prob_PC[i][0] >= 0.5 else 1
            elif pred_class_PC[i] == "Others":
                PC_index = 0 if pred_prob_PC[i][0] < 0.5 else 1
            if pred_class_RG12[i] == "Round Grinding - Machine 12":
                RG12_index = 0 if pred_prob_RG12[i][0] >= 0.5 else 1
            elif pred_class_RG12[i] == "Others":
                RG12_index = 0 if pred_prob_RG12[i][0] < 0.5 else 1
            if pred_class_RG2[i] == "Round Grinding - Machine 2":
                RG2_index = 0 if pred_prob_RG2[i][0] >= 0.5 else 1
            elif pred_class_RG2[i] == "Others":
                RG2_index = 0 if pred_prob_RG2[i][0] < 0.5 else 1
            if pred_class_RG3[i] == "Round Grinding - Machine 3":
                RG3_index = 0 if pred_prob_RG3[i][0] >= 0.5 else 1
            elif pred_class_RG3[i] == "Others":
                RG3_index = 0 if pred_prob_RG3[i][0] < 0.5 else 1
            if pred_class_RGM[i] == "Round Grinding - Manual":
                RGM_index = 0 if pred_prob_RGM[i][0] >= 0.5 else 1
            elif pred_class_RGM[i] == "Others":
                RGM_index = 0 if pred_prob_RGM[i][0] < 0.5 else 1
            if pred_class_RGQC[i] == "Round Grinding - Q.C.":
                RGQC_index = 0 if pred_prob_RGQC[i][0] >= 0.5 else 1
            elif pred_class_RGQC[i] == "Others":
                RGQC_index = 0 if pred_prob_RGQC[i][0] < 0.5 else 1
            if pred_class_T8[i] == "Turning - Machine 8":
                T8_index = 0 if pred_prob_T8[i][0] >= 0.5 else 1
            elif pred_class_T8[i] == "Others":
                T8_index = 0 if pred_prob_T8[i][0] < 0.5 else 1
            if pred_class_TM10[i] == "Turning & Milling - Machine 10":
                TM10_index = 0 if pred_prob_TM10[i][0] >= 0.5 else 1
            elif pred_class_TM10[i] == "Others":
                TM10_index = 0 if pred_prob_TM10[i][0] < 0.5 else 1
            if pred_class_TM4[i] == "Turning & Milling - Machine 4":
                TM4_index = 0 if pred_prob_TM4[i][0] >= 0.5 else 1
            elif pred_class_TM4[i] == "Others":
                TM4_index = 0 if pred_prob_TM4[i][0] < 0.5 else 1
            if pred_class_TM5[i] == "Turning & Milling - Machine 5":
                TM5_index = 0 if pred_prob_TM5[i][0] >= 0.5 else 1
            elif pred_class_TM5[i] == "Others":
                TM5_index = 0 if pred_prob_TM5[i][0] < 0.5 else 1
            if pred_class_TM6[i] == "Turning & Milling - Machine 6":
                TM6_index = 0 if pred_prob_TM6[i][0] >= 0.5 else 1
            elif pred_class_TM6[i] == "Others":
                TM6_index = 0 if pred_prob_TM6[i][0] < 0.5 else 1
            if pred_class_TM8[i] == "Turning & Milling - Machine 8":
                TM8_index = 0 if pred_prob_TM8[i][0] >= 0.5 else 1
            elif pred_class_TM8[i] == "Others":
                TM8_index = 0 if pred_prob_TM8[i][0] < 0.5 else 1
            if pred_class_TM9[i] == "Turning & Milling - Machine 9":
                TM9_index = 0 if pred_prob_TM9[i][0] >= 0.5 else 1
            elif pred_class_TM9[i] == "Others":
                TM9_index = 0 if pred_prob_TM9[i][0] < 0.5 else 1
            if pred_class_TMQC[i] == "Turning & Milling Q.C.":
                TMQC_index = 0 if pred_prob_TMQC[i][0] >= 0.5 else 1
            elif pred_class_TMQC[i] == "Others":
                TMQC_index = 0 if pred_prob_TMQC[i][0] < 0.5 else 1
            if pred_class_TQC[i] == "Turning Q.C.":
                TQC_index = 0 if pred_prob_TQC[i][0] >= 0.5 else 1
            elif pred_class_TQC[i] == "Others":
                TQC_index = 0 if pred_prob_TQC[i][0] < 0.5 else 1
            if pred_prob_FI[i][FI_index] == max(pred_prob_FI[i][FI_index], pred_prob_FG[i][FG_index],pred_prob_GR27[i][GR27_index],
                                                        pred_prob_LM[i][LM_index], pred_prob_LMM[i][LMM_index],
                                                        pred_prob_PC[i][PC_index], pred_prob_RG12[i][RG12_index],
                                                        pred_prob_RG2[i][RG2_index], pred_prob_RG3[i][RG3_index],
                                                        pred_prob_RGM[i][RGM_index], pred_prob_RGQC[i][RGQC_index],
                                                        pred_prob_T8[i][T8_index], pred_prob_TM10[i][TM10_index],
                                                        pred_prob_TM4[i][TM4_index], pred_prob_TM5[i][TM5_index],
                                                        pred_prob_TM6[i][TM6_index], pred_prob_TM8[i][TM8_index],
                                                        pred_prob_TM9[i][TM9_index], pred_prob_TMQC[i][TMQC_index],
                                                        pred_prob_TQC[i][TQC_index]):
                prediction.loc[i] = "Final Inspection Q.C."
            elif pred_prob_FG[i][FG_index] == max(pred_prob_FI[i][FI_index], pred_prob_FG[i][FG_index],pred_prob_GR27[i][GR27_index],
                                                        pred_prob_LM[i][LM_index], pred_prob_LMM[i][LMM_index],
                                                        pred_prob_PC[i][PC_index], pred_prob_RG12[i][RG12_index],
                                                        pred_prob_RG2[i][RG2_index], pred_prob_RG3[i][RG3_index],
                                                        pred_prob_RGM[i][RGM_index], pred_prob_RGQC[i][RGQC_index],
                                                        pred_prob_T8[i][T8_index], pred_prob_TM10[i][TM10_index],
                                                        pred_prob_TM4[i][TM4_index], pred_prob_TM5[i][TM5_index],
                                                        pred_prob_TM6[i][TM6_index], pred_prob_TM8[i][TM8_index],
                                                        pred_prob_TM9[i][TM9_index], pred_prob_TMQC[i][TMQC_index],
                                                        pred_prob_TQC[i][TQC_index]):
                prediction.loc[i] = "Flat Grinding - Machine 11"
            elif pred_prob_GR27[i][GR27_index] == max(pred_prob_FI[i][FI_index], pred_prob_FG[i][FG_index],pred_prob_GR27[i][GR27_index],
                                                        pred_prob_LM[i][LM_index], pred_prob_LMM[i][LMM_index],
                                                        pred_prob_PC[i][PC_index], pred_prob_RG12[i][RG12_index],
                                                        pred_prob_RG2[i][RG2_index], pred_prob_RG3[i][RG3_index],
                                                        pred_prob_RGM[i][RGM_index], pred_prob_RGQC[i][RGQC_index],
                                                        pred_prob_T8[i][T8_index], pred_prob_TM10[i][TM10_index],
                                                        pred_prob_TM4[i][TM4_index], pred_prob_TM5[i][TM5_index],
                                                        pred_prob_TM6[i][TM6_index], pred_prob_TM8[i][TM8_index],
                                                        pred_prob_TM9[i][TM9_index], pred_prob_TMQC[i][TMQC_index],
                                                        pred_prob_TQC[i][TQC_index]):
                prediction.loc[i] = "Grinding Rework - Machine 27"
            elif pred_prob_LM[i][LM_index] == max(pred_prob_FI[i][FI_index], pred_prob_FG[i][FG_index],pred_prob_GR27[i][GR27_index],
                                                        pred_prob_LM[i][LM_index], pred_prob_LMM[i][LMM_index],
                                                        pred_prob_PC[i][PC_index], pred_prob_RG12[i][RG12_index],
                                                        pred_prob_RG2[i][RG2_index], pred_prob_RG3[i][RG3_index],
                                                        pred_prob_RGM[i][RGM_index], pred_prob_RGQC[i][RGQC_index],
                                                        pred_prob_T8[i][T8_index], pred_prob_TM10[i][TM10_index],
                                                        pred_prob_TM4[i][TM4_index], pred_prob_TM5[i][TM5_index],
                                                        pred_prob_TM6[i][TM6_index], pred_prob_TM8[i][TM8_index],
                                                        pred_prob_TM9[i][TM9_index], pred_prob_TMQC[i][TMQC_index],
                                                        pred_prob_TQC[i][TQC_index]):
                prediction.loc[i] = "Lapping - Machine 1"
            elif pred_prob_LMM[i][LMM_index] == max(pred_prob_FI[i][FI_index], pred_prob_FG[i][FG_index],pred_prob_GR27[i][GR27_index],
                                                        pred_prob_LM[i][LM_index], pred_prob_LMM[i][LMM_index],
                                                        pred_prob_PC[i][PC_index], pred_prob_RG12[i][RG12_index],
                                                        pred_prob_RG2[i][RG2_index], pred_prob_RG3[i][RG3_index],
                                                        pred_prob_RGM[i][RGM_index], pred_prob_RGQC[i][RGQC_index],
                                                        pred_prob_T8[i][T8_index], pred_prob_TM10[i][TM10_index],
                                                        pred_prob_TM4[i][TM4_index], pred_prob_TM5[i][TM5_index],
                                                        pred_prob_TM6[i][TM6_index], pred_prob_TM8[i][TM8_index],
                                                        pred_prob_TM9[i][TM9_index], pred_prob_TMQC[i][TMQC_index],
                                                        pred_prob_TQC[i][TQC_index]):
                prediction.loc[i] = "Laser Marking - Machine 7"
            elif pred_prob_PC[i][PC_index] == max(pred_prob_FI[i][FI_index], pred_prob_FG[i][FG_index],pred_prob_GR27[i][GR27_index],
                                                        pred_prob_LM[i][LM_index], pred_prob_LMM[i][LMM_index],
                                                        pred_prob_PC[i][PC_index], pred_prob_RG12[i][RG12_index],
                                                        pred_prob_RG2[i][RG2_index], pred_prob_RG3[i][RG3_index],
                                                        pred_prob_RGM[i][RGM_index], pred_prob_RGQC[i][RGQC_index],
                                                        pred_prob_T8[i][T8_index], pred_prob_TM10[i][TM10_index],
                                                        pred_prob_TM4[i][TM4_index], pred_prob_TM5[i][TM5_index],
                                                        pred_prob_TM6[i][TM6_index], pred_prob_TM8[i][TM8_index],
                                                        pred_prob_TM9[i][TM9_index], pred_prob_TMQC[i][TMQC_index],
                                                        pred_prob_TQC[i][TQC_index]):
                prediction.loc[i] = "Packing"
            elif pred_prob_RG12[i][RG12_index] == max(pred_prob_FI[i][FI_index], pred_prob_FG[i][FG_index],pred_prob_GR27[i][GR27_index],
                                                        pred_prob_LM[i][LM_index], pred_prob_LMM[i][LMM_index],
                                                        pred_prob_PC[i][PC_index], pred_prob_RG12[i][RG12_index],
                                                        pred_prob_RG2[i][RG2_index], pred_prob_RG3[i][RG3_index],
                                                        pred_prob_RGM[i][RGM_index], pred_prob_RGQC[i][RGQC_index],
                                                        pred_prob_T8[i][T8_index], pred_prob_TM10[i][TM10_index],
                                                        pred_prob_TM4[i][TM4_index], pred_prob_TM5[i][TM5_index],
                                                        pred_prob_TM6[i][TM6_index], pred_prob_TM8[i][TM8_index],
                                                        pred_prob_TM9[i][TM9_index], pred_prob_TMQC[i][TMQC_index],
                                                        pred_prob_TQC[i][TQC_index]):
                prediction.loc[i] = "Round Grinding - Machine 12"
            elif pred_prob_RG2[i][RG2_index] == max(pred_prob_FI[i][FI_index], pred_prob_FG[i][FG_index],pred_prob_GR27[i][GR27_index],
                                                        pred_prob_LM[i][LM_index], pred_prob_LMM[i][LMM_index],
                                                        pred_prob_PC[i][PC_index], pred_prob_RG12[i][RG12_index],
                                                        pred_prob_RG2[i][RG2_index], pred_prob_RG3[i][RG3_index],
                                                        pred_prob_RGM[i][RGM_index], pred_prob_RGQC[i][RGQC_index],
                                                        pred_prob_T8[i][T8_index], pred_prob_TM10[i][TM10_index],
                                                        pred_prob_TM4[i][TM4_index], pred_prob_TM5[i][TM5_index],
                                                        pred_prob_TM6[i][TM6_index], pred_prob_TM8[i][TM8_index],
                                                        pred_prob_TM9[i][TM9_index], pred_prob_TMQC[i][TMQC_index],
                                                        pred_prob_TQC[i][TQC_index]):
                prediction.loc[i] = "Round Grinding - Machine 2"
            elif pred_prob_RG3[i][RG3_index] == max(pred_prob_FI[i][FI_index], pred_prob_FG[i][FG_index],pred_prob_GR27[i][GR27_index],
                                                        pred_prob_LM[i][LM_index], pred_prob_LMM[i][LMM_index],
                                                        pred_prob_PC[i][PC_index], pred_prob_RG12[i][RG12_index],
                                                        pred_prob_RG2[i][RG2_index], pred_prob_RG3[i][RG3_index],
                                                        pred_prob_RGM[i][RGM_index], pred_prob_RGQC[i][RGQC_index],
                                                        pred_prob_T8[i][T8_index], pred_prob_TM10[i][TM10_index],
                                                        pred_prob_TM4[i][TM4_index], pred_prob_TM5[i][TM5_index],
                                                        pred_prob_TM6[i][TM6_index], pred_prob_TM8[i][TM8_index],
                                                        pred_prob_TM9[i][TM9_index], pred_prob_TMQC[i][TMQC_index],
                                                        pred_prob_TQC[i][TQC_index]):
                prediction.loc[i] = "Round Grinding - Machine 3"
            elif pred_prob_RGM[i][RGM_index] == max(pred_prob_FI[i][FI_index], pred_prob_FG[i][FG_index],pred_prob_GR27[i][GR27_index],
                                                        pred_prob_LM[i][LM_index], pred_prob_LMM[i][LMM_index],
                                                        pred_prob_PC[i][PC_index], pred_prob_RG12[i][RG12_index],
                                                        pred_prob_RG2[i][RG2_index], pred_prob_RG3[i][RG3_index],
                                                        pred_prob_RGM[i][RGM_index], pred_prob_RGQC[i][RGQC_index],
                                                        pred_prob_T8[i][T8_index], pred_prob_TM10[i][TM10_index],
                                                        pred_prob_TM4[i][TM4_index], pred_prob_TM5[i][TM5_index],
                                                        pred_prob_TM6[i][TM6_index], pred_prob_TM8[i][TM8_index],
                                                        pred_prob_TM9[i][TM9_index], pred_prob_TMQC[i][TMQC_index],
                                                        pred_prob_TQC[i][TQC_index]):
                prediction.loc[i] = "Round Grinding - Manual"
            elif pred_prob_RGQC[i][RGQC_index] == max(pred_prob_FI[i][FI_index], pred_prob_FG[i][FG_index],pred_prob_GR27[i][GR27_index],
                                                        pred_prob_LM[i][LM_index], pred_prob_LMM[i][LMM_index],
                                                        pred_prob_PC[i][PC_index], pred_prob_RG12[i][RG12_index],
                                                        pred_prob_RG2[i][RG2_index], pred_prob_RG3[i][RG3_index],
                                                        pred_prob_RGM[i][RGM_index], pred_prob_RGQC[i][RGQC_index],
                                                        pred_prob_T8[i][T8_index], pred_prob_TM10[i][TM10_index],
                                                        pred_prob_TM4[i][TM4_index], pred_prob_TM5[i][TM5_index],
                                                        pred_prob_TM6[i][TM6_index], pred_prob_TM8[i][TM8_index],
                                                        pred_prob_TM9[i][TM9_index], pred_prob_TMQC[i][TMQC_index],
                                                        pred_prob_TQC[i][TQC_index]):
                prediction.loc[i] = "Round Grinding - Q.C."
            elif pred_prob_T8[i][T8_index] == max(pred_prob_FI[i][FI_index], pred_prob_FG[i][FG_index],pred_prob_GR27[i][GR27_index],
                                                        pred_prob_LM[i][LM_index], pred_prob_LMM[i][LMM_index],
                                                        pred_prob_PC[i][PC_index], pred_prob_RG12[i][RG12_index],
                                                        pred_prob_RG2[i][RG2_index], pred_prob_RG3[i][RG3_index],
                                                        pred_prob_RGM[i][RGM_index], pred_prob_RGQC[i][RGQC_index],
                                                        pred_prob_T8[i][T8_index], pred_prob_TM10[i][TM10_index],
                                                        pred_prob_TM4[i][TM4_index], pred_prob_TM5[i][TM5_index],
                                                        pred_prob_TM6[i][TM6_index], pred_prob_TM8[i][TM8_index],
                                                        pred_prob_TM9[i][TM9_index], pred_prob_TMQC[i][TMQC_index],
                                                        pred_prob_TQC[i][TQC_index]):
                prediction.loc[i] = "Turning - Machine 8"
            elif pred_prob_TM10[i][TM10_index] == max(pred_prob_FI[i][FI_index], pred_prob_FG[i][FG_index],pred_prob_GR27[i][GR27_index],
                                                        pred_prob_LM[i][LM_index], pred_prob_LMM[i][LMM_index],
                                                        pred_prob_PC[i][PC_index], pred_prob_RG12[i][RG12_index],
                                                        pred_prob_RG2[i][RG2_index], pred_prob_RG3[i][RG3_index],
                                                        pred_prob_RGM[i][RGM_index], pred_prob_RGQC[i][RGQC_index],
                                                        pred_prob_T8[i][T8_index], pred_prob_TM10[i][TM10_index],
                                                        pred_prob_TM4[i][TM4_index], pred_prob_TM5[i][TM5_index],
                                                        pred_prob_TM6[i][TM6_index], pred_prob_TM8[i][TM8_index],
                                                        pred_prob_TM9[i][TM9_index], pred_prob_TMQC[i][TMQC_index],
                                                        pred_prob_TQC[i][TQC_index]):
                prediction.loc[i] = "Turning & Milling - Machine 10"
            elif pred_prob_TM4[i][TM4_index] == max(pred_prob_FI[i][FI_index], pred_prob_FG[i][FG_index],pred_prob_GR27[i][GR27_index],
                                                        pred_prob_LM[i][LM_index], pred_prob_LMM[i][LMM_index],
                                                        pred_prob_PC[i][PC_index], pred_prob_RG12[i][RG12_index],
                                                        pred_prob_RG2[i][RG2_index], pred_prob_RG3[i][RG3_index],
                                                        pred_prob_RGM[i][RGM_index], pred_prob_RGQC[i][RGQC_index],
                                                        pred_prob_T8[i][T8_index], pred_prob_TM10[i][TM10_index],
                                                        pred_prob_TM4[i][TM4_index], pred_prob_TM5[i][TM5_index],
                                                        pred_prob_TM6[i][TM6_index], pred_prob_TM8[i][TM8_index],
                                                        pred_prob_TM9[i][TM9_index], pred_prob_TMQC[i][TMQC_index],
                                                        pred_prob_TQC[i][TQC_index]):
                prediction.loc[i] = "Turning & Milling - Machine 4"
            elif pred_prob_TM5[i][TM5_index] == max(pred_prob_FI[i][FI_index], pred_prob_FG[i][FG_index],pred_prob_GR27[i][GR27_index],
                                                        pred_prob_LM[i][LM_index], pred_prob_LMM[i][LMM_index],
                                                        pred_prob_PC[i][PC_index], pred_prob_RG12[i][RG12_index],
                                                        pred_prob_RG2[i][RG2_index], pred_prob_RG3[i][RG3_index],
                                                        pred_prob_RGM[i][RGM_index], pred_prob_RGQC[i][RGQC_index],
                                                        pred_prob_T8[i][T8_index], pred_prob_TM10[i][TM10_index],
                                                        pred_prob_TM4[i][TM4_index], pred_prob_TM5[i][TM5_index],
                                                        pred_prob_TM6[i][TM6_index], pred_prob_TM8[i][TM8_index],
                                                        pred_prob_TM9[i][TM9_index], pred_prob_TMQC[i][TMQC_index],
                                                        pred_prob_TQC[i][TQC_index]):
                prediction.loc[i] = "Turning & Milling - Machine 5"
            elif pred_prob_TM6[i][TM6_index] == max(pred_prob_FI[i][FI_index], pred_prob_FG[i][FG_index],pred_prob_GR27[i][GR27_index],
                                                        pred_prob_LM[i][LM_index], pred_prob_LMM[i][LMM_index],
                                                        pred_prob_PC[i][PC_index], pred_prob_RG12[i][RG12_index],
                                                        pred_prob_RG2[i][RG2_index], pred_prob_RG3[i][RG3_index],
                                                        pred_prob_RGM[i][RGM_index], pred_prob_RGQC[i][RGQC_index],
                                                        pred_prob_T8[i][T8_index], pred_prob_TM10[i][TM10_index],
                                                        pred_prob_TM4[i][TM4_index], pred_prob_TM5[i][TM5_index],
                                                        pred_prob_TM6[i][TM6_index], pred_prob_TM8[i][TM8_index],
                                                        pred_prob_TM9[i][TM9_index], pred_prob_TMQC[i][TMQC_index],
                                                        pred_prob_TQC[i][TQC_index]):
                prediction.loc[i] = "Turning & Milling - Machine 6"
            elif pred_prob_TM8[i][TM8_index] == max(pred_prob_FI[i][FI_index], pred_prob_FG[i][FG_index],pred_prob_GR27[i][GR27_index],
                                                        pred_prob_LM[i][LM_index], pred_prob_LMM[i][LMM_index],
                                                        pred_prob_PC[i][PC_index], pred_prob_RG12[i][RG12_index],
                                                        pred_prob_RG2[i][RG2_index], pred_prob_RG3[i][RG3_index],
                                                        pred_prob_RGM[i][RGM_index], pred_prob_RGQC[i][RGQC_index],
                                                        pred_prob_T8[i][T8_index], pred_prob_TM10[i][TM10_index],
                                                        pred_prob_TM4[i][TM4_index], pred_prob_TM5[i][TM5_index],
                                                        pred_prob_TM6[i][TM6_index], pred_prob_TM8[i][TM8_index],
                                                        pred_prob_TM9[i][TM9_index], pred_prob_TMQC[i][TMQC_index],
                                                        pred_prob_TQC[i][TQC_index]):
                prediction.loc[i] = "Turning & Milling - Machine 8"
            elif pred_prob_TM9[i][TM9_index] == max(pred_prob_FI[i][FI_index], pred_prob_FG[i][FG_index],pred_prob_GR27[i][GR27_index],
                                                        pred_prob_LM[i][LM_index], pred_prob_LMM[i][LMM_index],
                                                        pred_prob_PC[i][PC_index], pred_prob_RG12[i][RG12_index],
                                                        pred_prob_RG2[i][RG2_index], pred_prob_RG3[i][RG3_index],
                                                        pred_prob_RGM[i][RGM_index], pred_prob_RGQC[i][RGQC_index],
                                                        pred_prob_T8[i][T8_index], pred_prob_TM10[i][TM10_index],
                                                        pred_prob_TM4[i][TM4_index], pred_prob_TM5[i][TM5_index],
                                                        pred_prob_TM6[i][TM6_index], pred_prob_TM8[i][TM8_index],
                                                        pred_prob_TM9[i][TM9_index], pred_prob_TMQC[i][TMQC_index],
                                                        pred_prob_TQC[i][TQC_index]):
                prediction.loc[i] = "Turning & Milling - Machine 9"
            elif pred_prob_TMQC[i][TMQC_index] == max(pred_prob_FI[i][FI_index], pred_prob_FG[i][FG_index],pred_prob_GR27[i][GR27_index],
                                                        pred_prob_LM[i][LM_index], pred_prob_LMM[i][LMM_index],
                                                        pred_prob_PC[i][PC_index], pred_prob_RG12[i][RG12_index],
                                                        pred_prob_RG2[i][RG2_index], pred_prob_RG3[i][RG3_index],
                                                        pred_prob_RGM[i][RGM_index], pred_prob_RGQC[i][RGQC_index],
                                                        pred_prob_T8[i][T8_index], pred_prob_TM10[i][TM10_index],
                                                        pred_prob_TM4[i][TM4_index], pred_prob_TM5[i][TM5_index],
                                                        pred_prob_TM6[i][TM6_index], pred_prob_TM8[i][TM8_index],
                                                        pred_prob_TM9[i][TM9_index], pred_prob_TMQC[i][TMQC_index],
                                                        pred_prob_TQC[i][TQC_index]):
                prediction.loc[i] = "Turning & Milling Q.C."
            elif pred_prob_TQC[i][TQC_index] == max(pred_prob_FI[i][FI_index], pred_prob_FG[i][FG_index],pred_prob_GR27[i][GR27_index],
                                                        pred_prob_LM[i][LM_index], pred_prob_LMM[i][LMM_index],
                                                        pred_prob_PC[i][PC_index], pred_prob_RG12[i][RG12_index],
                                                        pred_prob_RG2[i][RG2_index], pred_prob_RG3[i][RG3_index],
                                                        pred_prob_RGM[i][RGM_index], pred_prob_RGQC[i][RGQC_index],
                                                        pred_prob_T8[i][T8_index], pred_prob_TM10[i][TM10_index],
                                                        pred_prob_TM4[i][TM4_index], pred_prob_TM5[i][TM5_index],
                                                        pred_prob_TM6[i][TM6_index], pred_prob_TM8[i][TM8_index],
                                                        pred_prob_TM9[i][TM9_index], pred_prob_TMQC[i][TMQC_index],
                                                        pred_prob_TQC[i][TQC_index]):
                prediction.loc[i] = "Turning Q.C."
        return pred_prob_FI,pred_prob_FG,pred_prob_GR27,pred_prob_LM,pred_prob_LMM,pred_prob_PC,pred_prob_RG12,pred_prob_RG2,pred_prob_RG3,pred_prob_RGM,pred_prob_RGQC,pred_prob_T8,pred_prob_TM10,pred_prob_TM4,pred_prob_TM5,pred_prob_TM6,pred_prob_TM8,pred_prob_TM9,pred_prob_TMQC,pred_prob_TQC,prediction

    def get_precision(conf_matrix):
        tp_1 = conf_matrix[0][0]
        tp_2 = conf_matrix[1][1]
        tp_3 = conf_matrix[2][2]
        tp_4 = conf_matrix[3][3]
        tp_5 = conf_matrix[4][4]
        tp_6 = conf_matrix[5][5]
        tp_7 = conf_matrix[6][6]
        tp_8 = conf_matrix[7][7]
        tp_9 = conf_matrix[8][8]
        tp_10 = conf_matrix[9][9]
        tp_11 = conf_matrix[10][10]
        tp_12 = conf_matrix[11][11]
        tp_13 = conf_matrix[12][12]
        tp_14 = conf_matrix[13][13]
        tp_15 = conf_matrix[14][14]
        tp_16 = conf_matrix[15][15]
        tp_17 = conf_matrix[16][16]
        tp_18 = conf_matrix[17][17]
        tp_19 = conf_matrix[18][18]
        tp_20 = conf_matrix[19][19]

        fp_1 = conf_matrix[1][0] + conf_matrix[2][0] + conf_matrix[3][0] + conf_matrix[4][0] + conf_matrix[5][0] + conf_matrix[6][0] + conf_matrix[7][0] + conf_matrix[8][0] + conf_matrix[9][0] + conf_matrix[10][0] + conf_matrix[11][0] + conf_matrix[12][0] + conf_matrix[13][0] + conf_matrix[14][0] + conf_matrix[15][0] + conf_matrix[16][0] + conf_matrix[17][0] + conf_matrix[18][0] + conf_matrix[19][0]
        fp_2 = conf_matrix[0][1] + conf_matrix[2][1] + conf_matrix[3][1] + conf_matrix[4][1] + conf_matrix[5][1] + conf_matrix[6][1] + conf_matrix[7][1] + conf_matrix[8][1] + conf_matrix[9][1] + conf_matrix[10][1] + conf_matrix[11][1] + conf_matrix[12][1] + conf_matrix[13][1] + conf_matrix[14][1] + conf_matrix[15][1] + conf_matrix[16][1] + conf_matrix[17][1] + conf_matrix[18][1] + conf_matrix[19][1]
        fp_3 = conf_matrix[0][2] + conf_matrix[1][2] + conf_matrix[3][2] + conf_matrix[4][2] + conf_matrix[5][2] + conf_matrix[6][2] + conf_matrix[7][2] + conf_matrix[8][2] + conf_matrix[9][2] + conf_matrix[10][2] + conf_matrix[11][2] + conf_matrix[12][2] + conf_matrix[13][2] + conf_matrix[14][2] + conf_matrix[15][2] + conf_matrix[16][2] + conf_matrix[17][2] + conf_matrix[18][2] + conf_matrix[19][2]
        fp_4 = conf_matrix[0][3] + conf_matrix[1][3] + conf_matrix[2][3] + conf_matrix[4][3] + conf_matrix[5][3] + conf_matrix[6][3] + conf_matrix[7][3] + conf_matrix[8][3] + conf_matrix[9][3] + conf_matrix[10][3] + conf_matrix[11][3] + conf_matrix[12][3] + conf_matrix[13][3] + conf_matrix[14][3] + conf_matrix[15][3] + conf_matrix[16][3] + conf_matrix[17][3] + conf_matrix[18][3] + conf_matrix[19][3]
        fp_5 = conf_matrix[0][4] + conf_matrix[1][4] + conf_matrix[2][4] + conf_matrix[3][4] + conf_matrix[5][4] + conf_matrix[6][4] + conf_matrix[7][4] + conf_matrix[8][4] + conf_matrix[9][4] + conf_matrix[10][4] + conf_matrix[11][4] + conf_matrix[12][4] + conf_matrix[13][4] + conf_matrix[14][4] + conf_matrix[15][4] + conf_matrix[16][4] + conf_matrix[17][4] + conf_matrix[18][4] + conf_matrix[19][4]
        fp_6 = conf_matrix[0][5] + conf_matrix[1][5] + conf_matrix[2][5] + conf_matrix[3][5] + conf_matrix[4][5] + conf_matrix[6][5] + conf_matrix[7][5] + conf_matrix[8][5] + conf_matrix[9][5] + conf_matrix[10][5] + conf_matrix[11][5] + conf_matrix[12][5] + conf_matrix[13][5] + conf_matrix[14][5] + conf_matrix[15][5] + conf_matrix[16][5] + conf_matrix[17][5] + conf_matrix[18][5] + conf_matrix[19][5]
        fp_7 = conf_matrix[0][6] + conf_matrix[1][6] + conf_matrix[2][6] + conf_matrix[3][6] + conf_matrix[4][6] + conf_matrix[5][6] + conf_matrix[7][6] + conf_matrix[8][6] + conf_matrix[9][6] + conf_matrix[10][6] + conf_matrix[11][6] + conf_matrix[12][6] + conf_matrix[13][6] + conf_matrix[14][6] + conf_matrix[15][6] + conf_matrix[16][6] + conf_matrix[17][6] + conf_matrix[18][6] + conf_matrix[19][6]
        fp_8 = conf_matrix[0][7] + conf_matrix[1][7] + conf_matrix[2][7] + conf_matrix[3][7] + conf_matrix[4][7] + conf_matrix[5][7] + conf_matrix[6][7] + conf_matrix[8][7] + conf_matrix[9][7] + conf_matrix[10][7] + conf_matrix[11][7] + conf_matrix[12][7] + conf_matrix[13][7] + conf_matrix[14][7] + conf_matrix[15][7] + conf_matrix[16][7] + conf_matrix[17][7] + conf_matrix[18][7] + conf_matrix[19][7]
        fp_9 = conf_matrix[0][8] + conf_matrix[1][8] + conf_matrix[2][8] + conf_matrix[3][8] + conf_matrix[4][8] + conf_matrix[5][8] + conf_matrix[6][8] + conf_matrix[7][8] + conf_matrix[9][8] + conf_matrix[10][8] + conf_matrix[11][8] + conf_matrix[12][8] + conf_matrix[13][8] + conf_matrix[14][8] + conf_matrix[15][8] + conf_matrix[16][8] + conf_matrix[17][8] + conf_matrix[18][8] + conf_matrix[19][8]
        fp_10 = conf_matrix[0][9] + conf_matrix[1][9] + conf_matrix[2][9] + conf_matrix[3][9] + conf_matrix[4][9] + conf_matrix[5][9] + conf_matrix[6][9] + conf_matrix[7][9] + conf_matrix[8][9] + conf_matrix[10][9] + conf_matrix[11][9] + conf_matrix[12][9] + conf_matrix[13][9] + conf_matrix[14][9] + conf_matrix[15][9] + conf_matrix[16][9] + conf_matrix[17][9] + conf_matrix[18][9] + conf_matrix[19][9]
        fp_11 = conf_matrix[0][10] + conf_matrix[1][10] + conf_matrix[2][10] + conf_matrix[3][10] + conf_matrix[4][10] + conf_matrix[5][10] + conf_matrix[6][10] + conf_matrix[7][10] + conf_matrix[8][10] + conf_matrix[9][10] + conf_matrix[11][10] + conf_matrix[12][10] + conf_matrix[13][10] + conf_matrix[14][10] + conf_matrix[15][10] + conf_matrix[16][10] + conf_matrix[17][10] + conf_matrix[18][10] + conf_matrix[19][10]
        fp_12 = conf_matrix[0][11] + conf_matrix[1][11] + conf_matrix[2][11] + conf_matrix[3][11] + conf_matrix[4][11] + conf_matrix[5][11] + conf_matrix[6][11] + conf_matrix[7][11] + conf_matrix[8][11] + conf_matrix[9][11] + conf_matrix[10][11] + conf_matrix[12][11] + conf_matrix[13][11] + conf_matrix[14][11] + conf_matrix[15][11] + conf_matrix[16][11] + conf_matrix[17][11] + conf_matrix[18][11] + conf_matrix[19][11]
        fp_13 = conf_matrix[0][12] + conf_matrix[1][12] + conf_matrix[2][12] + conf_matrix[3][12] + conf_matrix[4][12] + conf_matrix[5][12] + conf_matrix[6][12] + conf_matrix[7][12] + conf_matrix[8][12] + conf_matrix[9][12] + conf_matrix[10][12] + conf_matrix[11][12] + conf_matrix[13][12] + conf_matrix[14][12] + conf_matrix[15][12] + conf_matrix[16][12] + conf_matrix[17][12] + conf_matrix[18][12] + conf_matrix[19][12]
        fp_14 = conf_matrix[0][13] + conf_matrix[1][13] + conf_matrix[2][13] + conf_matrix[3][13] + conf_matrix[4][13] + conf_matrix[5][13] + conf_matrix[6][13] + conf_matrix[7][13] + conf_matrix[8][13] + conf_matrix[9][13] + conf_matrix[10][13] + conf_matrix[11][13] + conf_matrix[12][13] + conf_matrix[14][13] + conf_matrix[15][13] + conf_matrix[16][13] + conf_matrix[17][13] + conf_matrix[18][13] + conf_matrix[19][13]
        fp_15 = conf_matrix[0][14] + conf_matrix[1][14] + conf_matrix[2][14] + conf_matrix[3][14] + conf_matrix[4][14] + conf_matrix[5][14] + conf_matrix[6][14] + conf_matrix[7][14] + conf_matrix[8][14] + conf_matrix[9][14] + conf_matrix[10][14] + conf_matrix[11][14] + conf_matrix[12][14] + conf_matrix[13][14] + conf_matrix[15][14] + conf_matrix[16][14] + conf_matrix[17][14] + conf_matrix[18][14] + conf_matrix[19][14]
        fp_16 = conf_matrix[0][15] + conf_matrix[1][15] + conf_matrix[2][15] + conf_matrix[3][15] + conf_matrix[4][15] + conf_matrix[5][15] + conf_matrix[6][15] + conf_matrix[7][15] + conf_matrix[8][15] + conf_matrix[9][15] + conf_matrix[10][15] + conf_matrix[11][15] + conf_matrix[12][15] + conf_matrix[13][15] + conf_matrix[14][15] + conf_matrix[16][15] + conf_matrix[17][15] + conf_matrix[18][15] + conf_matrix[19][15]
        fp_17 = conf_matrix[0][16] + conf_matrix[1][16] + conf_matrix[2][16] + conf_matrix[3][16] + conf_matrix[4][16] + conf_matrix[5][16] + conf_matrix[6][16] + conf_matrix[7][16] + conf_matrix[8][16] + conf_matrix[9][16] + conf_matrix[10][16] + conf_matrix[11][16] + conf_matrix[12][16] + conf_matrix[13][16] + conf_matrix[14][16] + conf_matrix[15][16] + conf_matrix[17][16] + conf_matrix[18][16] + conf_matrix[19][16]
        fp_18 = conf_matrix[0][17] + conf_matrix[1][17] + conf_matrix[2][17] + conf_matrix[3][17] + conf_matrix[4][17] + conf_matrix[5][17] + conf_matrix[6][17] + conf_matrix[7][17] + conf_matrix[8][17] + conf_matrix[9][17] + conf_matrix[10][17] + conf_matrix[11][17] + conf_matrix[12][17] + conf_matrix[13][17] + conf_matrix[14][17] + conf_matrix[15][17] + conf_matrix[16][17] + conf_matrix[18][17] + conf_matrix[19][17]
        fp_19 = conf_matrix[0][18] + conf_matrix[1][18] + conf_matrix[2][18] + conf_matrix[3][18] + conf_matrix[4][18] + conf_matrix[5][18] + conf_matrix[6][18] + conf_matrix[7][18] + conf_matrix[8][18] + conf_matrix[9][18] + conf_matrix[10][18] + conf_matrix[11][18] + conf_matrix[12][18] + conf_matrix[13][18] + conf_matrix[14][18] + conf_matrix[15][18] + conf_matrix[16][18] + conf_matrix[17][18] + conf_matrix[19][18]
        fp_20 = conf_matrix[0][19] + conf_matrix[1][19] + conf_matrix[2][19] + conf_matrix[3][19] + conf_matrix[4][19] + conf_matrix[5][19] + conf_matrix[6][19] + conf_matrix[7][19] + conf_matrix[8][19] + conf_matrix[9][19] + conf_matrix[10][19] + conf_matrix[11][19] + conf_matrix[12][19] + conf_matrix[13][19] + conf_matrix[14][19] + conf_matrix[15][19] + conf_matrix[16][19] + conf_matrix[17][19] + conf_matrix[18][19]

        precision_1 = 0 if tp_1 + fp_1 == 0 else tp_1 / (tp_1 + fp_1)
        precision_2 = 0 if tp_2 + fp_2 == 0 else tp_2 / (tp_2 + fp_2)
        precision_3 = 0 if tp_3 + fp_3 == 0 else tp_3 / (tp_3 + fp_3)
        precision_4 = 0 if tp_4 + fp_4 == 0 else tp_4 / (tp_4 + fp_4)
        precision_5 = 0 if tp_5 + fp_5 == 0 else tp_5 / (tp_5 + fp_5)
        precision_6 = 0 if tp_6 + fp_6 == 0 else tp_6 / (tp_6 + fp_6)
        precision_7 = 0 if tp_7 + fp_7 == 0 else tp_7 / (tp_7 + fp_7)
        precision_8 = 0 if tp_8 + fp_8 == 0 else tp_8 / (tp_8 + fp_8)
        precision_9 = 0 if tp_9 + fp_9 == 0 else tp_9 / (tp_9 + fp_9)
        precision_10 = 0 if tp_10 + fp_10 == 0 else tp_10 / (tp_10 + fp_10)
        precision_11 = 0 if tp_11 + fp_11 == 0 else tp_11 / (tp_11 + fp_11)
        precision_12 = 0 if tp_12 + fp_12 == 0 else tp_12 / (tp_12 + fp_12)
        precision_13 = 0 if tp_13 + fp_13 == 0 else tp_13 / (tp_13 + fp_13)
        precision_14 = 0 if tp_14 + fp_14 == 0 else tp_14 / (tp_14 + fp_14)
        precision_15 = 0 if tp_15 + fp_15 == 0 else tp_15 / (tp_15 + fp_15)
        precision_16 = 0 if tp_16 + fp_16 == 0 else tp_16 / (tp_16 + fp_16)
        precision_17 = 0 if tp_17 + fp_17 == 0 else tp_17 / (tp_17 + fp_17)
        precision_18 = 0 if tp_18 + fp_18 == 0 else tp_18 / (tp_18 + fp_18)
        precision_19 = 0 if tp_19 + fp_19 == 0 else tp_19 / (tp_19 + fp_19)
        precision_20 = 0 if tp_20 + fp_20 == 0 else tp_20 / (tp_20 + fp_20)
        precision_avg = (precision_1 + precision_2 + precision_3 + precision_4 + precision_5 + precision_6 + precision_7 + precision_8 + precision_9 + precision_10 + precision_11 + precision_12 + precision_13 + precision_14 + precision_15 + precision_16 + precision_17 + precision_18 + precision_19 + precision_20) / 20
        return precision_avg

    def get_recall_pen_1(conf_matrix):
        tp_1 = conf_matrix[0][0]
        tp_2 = conf_matrix[1][1]
        tp_3 = conf_matrix[2][2]
        tp_4 = conf_matrix[3][3]
        tp_5 = conf_matrix[4][4]
        tp_6 = conf_matrix[5][5]
        tp_7 = conf_matrix[6][6]
        tp_8 = conf_matrix[7][7]
        tp_9 = conf_matrix[8][8]
        tp_10 = conf_matrix[9][9]
        tp_11 = conf_matrix[10][10]
        tp_12 = conf_matrix[11][11]
        tp_13 = conf_matrix[12][12]
        tp_14 = conf_matrix[13][13]
        tp_15 = conf_matrix[14][14]
        tp_16 = conf_matrix[15][15]
        tp_17 = conf_matrix[16][16]
        tp_18 = conf_matrix[17][17]
        tp_19 = conf_matrix[18][18]
        tp_20 = conf_matrix[19][19]
        fn_1 = conf_matrix[0][1] + conf_matrix[0][2] + conf_matrix[0][3] + conf_matrix[0][4] + conf_matrix[0][5] + conf_matrix[0][6] + conf_matrix[0][7] + conf_matrix[0][8] + conf_matrix[0][9] + conf_matrix[0][10] + conf_matrix[0][11] + conf_matrix[0][12] + conf_matrix[0][13] + conf_matrix[0][14] + conf_matrix[0][15] + conf_matrix[0][16] + conf_matrix[0][17] + conf_matrix[0][18] + conf_matrix[0][19]
        fn_2 = conf_matrix[1][0] + conf_matrix[1][2] + conf_matrix[1][3] + conf_matrix[1][4] + conf_matrix[1][5] + conf_matrix[1][6] + conf_matrix[1][7] + conf_matrix[1][8] + conf_matrix[1][9] + conf_matrix[1][10] + conf_matrix[1][11] + conf_matrix[1][12] + conf_matrix[1][13] + conf_matrix[1][14] + conf_matrix[1][15] + conf_matrix[1][16] + conf_matrix[1][17] + conf_matrix[1][18] + conf_matrix[1][19]
        fn_3 = conf_matrix[2][0] + conf_matrix[2][1] + conf_matrix[2][3] + conf_matrix[2][4] + conf_matrix[2][5] + conf_matrix[2][6] + conf_matrix[2][7] + conf_matrix[2][8] + conf_matrix[2][9] + conf_matrix[2][10] + conf_matrix[2][11] + conf_matrix[2][12] + conf_matrix[2][13] + conf_matrix[2][14] + conf_matrix[2][15] + conf_matrix[2][16] + conf_matrix[2][17] + conf_matrix[2][18] + conf_matrix[2][19]
        fn_4 = conf_matrix[3][0] + conf_matrix[3][1] + conf_matrix[3][2] + conf_matrix[3][4] + conf_matrix[3][5] + conf_matrix[3][6] + conf_matrix[3][7] + conf_matrix[3][8] + conf_matrix[3][9] + conf_matrix[3][10] + conf_matrix[3][11] + conf_matrix[3][12] + conf_matrix[3][13] + conf_matrix[3][14] + conf_matrix[3][15] + conf_matrix[3][16] + conf_matrix[3][17] + conf_matrix[3][18] + conf_matrix[3][19]
        fn_5 = conf_matrix[4][0] + conf_matrix[4][1] + conf_matrix[4][2] + conf_matrix[4][3] + conf_matrix[4][5] + conf_matrix[4][6] + conf_matrix[4][7] + conf_matrix[4][8] + conf_matrix[4][9] + conf_matrix[4][10] + conf_matrix[4][11] + conf_matrix[4][12] + conf_matrix[4][13] + conf_matrix[4][14] + conf_matrix[4][15] + conf_matrix[4][16] + conf_matrix[4][17] + conf_matrix[4][18] + conf_matrix[4][19]
        fn_6 = conf_matrix[5][0] + conf_matrix[5][1] + conf_matrix[5][2] + conf_matrix[5][3] + conf_matrix[5][4] + conf_matrix[5][6] + conf_matrix[5][7] + conf_matrix[5][8] + conf_matrix[5][9] + conf_matrix[5][10] + conf_matrix[5][11] + conf_matrix[5][12] + conf_matrix[5][13] + conf_matrix[5][14] + conf_matrix[5][15] + conf_matrix[5][16] + conf_matrix[5][17] + conf_matrix[5][18] + conf_matrix[5][19]
        fn_7 = conf_matrix[6][0] + conf_matrix[6][1] + conf_matrix[6][2] + conf_matrix[6][3] + conf_matrix[6][4] + conf_matrix[6][5] + conf_matrix[6][7] + conf_matrix[6][8] + conf_matrix[6][9] + conf_matrix[6][10] + conf_matrix[6][11] + conf_matrix[6][12] + conf_matrix[6][13] + conf_matrix[6][14] + conf_matrix[6][15] + conf_matrix[6][16] + conf_matrix[6][17] + conf_matrix[6][18] + conf_matrix[6][19]
        fn_8 = conf_matrix[7][0] + conf_matrix[7][1] + conf_matrix[7][2] + conf_matrix[7][3] + conf_matrix[7][4] + conf_matrix[7][5] + conf_matrix[7][6] + conf_matrix[7][8] + conf_matrix[7][9] + conf_matrix[7][10] + conf_matrix[7][11] + conf_matrix[7][12] + conf_matrix[7][13] + conf_matrix[7][14] + conf_matrix[7][15] + conf_matrix[7][16] + conf_matrix[7][17] + conf_matrix[7][18] + conf_matrix[7][19]
        fn_9 = conf_matrix[8][0] + conf_matrix[8][1] + conf_matrix[8][2] + conf_matrix[8][3] + conf_matrix[8][4] + conf_matrix[8][5] + conf_matrix[8][6] + conf_matrix[8][7] + conf_matrix[8][9] + conf_matrix[8][10] + conf_matrix[8][11] + conf_matrix[8][12] + conf_matrix[8][13] + conf_matrix[8][14] + conf_matrix[8][15] + conf_matrix[8][16] + conf_matrix[8][17] + conf_matrix[8][18] + conf_matrix[8][19]
        fn_10 = conf_matrix[9][0] + conf_matrix[9][1] + conf_matrix[9][2] + conf_matrix[9][3] + conf_matrix[9][4] + conf_matrix[9][5] + conf_matrix[9][6] + conf_matrix[9][7] + conf_matrix[9][8] + conf_matrix[9][10] + conf_matrix[9][11] + conf_matrix[9][12] + conf_matrix[9][13] + conf_matrix[9][14] + conf_matrix[9][15] + conf_matrix[9][16] + conf_matrix[9][17] + conf_matrix[9][18] + conf_matrix[9][19]
        fn_11 = conf_matrix[10][0] + conf_matrix[10][1] + conf_matrix[10][2] + conf_matrix[10][3] + conf_matrix[10][4] + conf_matrix[10][5] + conf_matrix[10][6] + conf_matrix[10][7] + conf_matrix[10][8] + conf_matrix[10][9] + conf_matrix[10][11] + conf_matrix[10][12] + conf_matrix[10][13] + conf_matrix[10][14] + conf_matrix[10][15] + conf_matrix[10][16] + conf_matrix[10][17] + conf_matrix[10][18] + conf_matrix[10][19]
        fn_12 = conf_matrix[11][0] + conf_matrix[11][1] + conf_matrix[11][2] + conf_matrix[11][3] + conf_matrix[11][4] + conf_matrix[11][5] + conf_matrix[11][6] + conf_matrix[11][7] + conf_matrix[11][8] + conf_matrix[11][9] + conf_matrix[11][10] + conf_matrix[11][12] + conf_matrix[11][13] + conf_matrix[11][14] + conf_matrix[11][15] + conf_matrix[11][16] + conf_matrix[11][17] + conf_matrix[11][18] + conf_matrix[11][19]
        fn_13 = conf_matrix[12][0] + conf_matrix[12][1] + conf_matrix[12][2] + conf_matrix[12][3] + conf_matrix[12][4] + conf_matrix[12][5] + conf_matrix[12][6] + conf_matrix[12][7] + conf_matrix[12][8] + conf_matrix[12][9] + conf_matrix[12][10] + conf_matrix[12][11] + conf_matrix[12][13] + conf_matrix[12][14] + conf_matrix[12][15] + conf_matrix[12][16] + conf_matrix[12][17] + conf_matrix[12][18] + conf_matrix[12][19]
        fn_14 = conf_matrix[13][0] + conf_matrix[13][1] + conf_matrix[13][2] + conf_matrix[13][3] + conf_matrix[13][4] + conf_matrix[13][5] + conf_matrix[13][6] + conf_matrix[13][7] + conf_matrix[13][8] + conf_matrix[13][9] + conf_matrix[13][10] + conf_matrix[13][11] + conf_matrix[13][12] + conf_matrix[13][14] + conf_matrix[13][15] + conf_matrix[13][16] + conf_matrix[13][17] + conf_matrix[13][18] + conf_matrix[13][19]
        fn_15 = conf_matrix[14][0] + conf_matrix[14][1] + conf_matrix[14][2] + conf_matrix[14][3] + conf_matrix[14][4] + conf_matrix[14][5] + conf_matrix[14][6] + conf_matrix[14][7] + conf_matrix[14][8] + conf_matrix[14][9] + conf_matrix[14][10] + conf_matrix[14][11] + conf_matrix[14][12] + conf_matrix[14][13] + conf_matrix[14][15] + conf_matrix[14][16] + conf_matrix[14][17] + conf_matrix[14][18] + conf_matrix[14][19]
        fn_16 = conf_matrix[15][0] + conf_matrix[15][1] + conf_matrix[15][2] + conf_matrix[15][3] + conf_matrix[15][4] + conf_matrix[15][5] + conf_matrix[15][6] + conf_matrix[15][7] + conf_matrix[15][8] + conf_matrix[15][9] + conf_matrix[15][10] + conf_matrix[15][11] + conf_matrix[15][12] + conf_matrix[15][13] + conf_matrix[15][14] + conf_matrix[15][16] + conf_matrix[15][17] + conf_matrix[15][18] + conf_matrix[15][19]
        fn_17 = conf_matrix[16][0] + conf_matrix[16][1] + conf_matrix[16][2] + conf_matrix[16][3] + conf_matrix[16][4] + conf_matrix[16][5] + conf_matrix[16][6] + conf_matrix[16][7] + conf_matrix[16][8] + conf_matrix[16][9] + conf_matrix[16][10] + conf_matrix[16][11] + conf_matrix[16][12] + conf_matrix[16][13] + conf_matrix[16][14] + conf_matrix[16][15] + conf_matrix[16][17] + conf_matrix[16][18] + conf_matrix[16][19]
        fn_18 = conf_matrix[17][0] + conf_matrix[17][1] + conf_matrix[17][2] + conf_matrix[17][3] + conf_matrix[17][4] + conf_matrix[17][5] + conf_matrix[17][6] + conf_matrix[17][7] + conf_matrix[17][8] + conf_matrix[17][9] + conf_matrix[17][10] + conf_matrix[17][11] + conf_matrix[17][12] + conf_matrix[17][13] + conf_matrix[17][14] + conf_matrix[17][15] + conf_matrix[17][16] + conf_matrix[17][18] + conf_matrix[17][19]
        fn_19 = conf_matrix[18][0] + conf_matrix[18][1] + conf_matrix[18][2] + conf_matrix[18][3] + conf_matrix[18][4] + conf_matrix[18][5] + conf_matrix[18][6] + conf_matrix[18][7] + conf_matrix[18][8] + conf_matrix[18][9] + conf_matrix[18][10] + conf_matrix[18][11] + conf_matrix[18][12] + conf_matrix[18][13] + conf_matrix[18][14] + conf_matrix[18][15] + conf_matrix[18][16] + conf_matrix[18][17] + conf_matrix[18][19]
        fn_20 = conf_matrix[19][0] + conf_matrix[19][1] + conf_matrix[19][2] + conf_matrix[19][3] + conf_matrix[19][4] + conf_matrix[19][5] + conf_matrix[19][6] + conf_matrix[19][7] + conf_matrix[19][8] + conf_matrix[19][9] + conf_matrix[19][10] + conf_matrix[19][11] + conf_matrix[19][12] + conf_matrix[19][13] + conf_matrix[19][14] + conf_matrix[19][15] + conf_matrix[19][16] + conf_matrix[19][17] + conf_matrix[19][18]
        recall_1 = 0 if tp_1 + fn_1 == 0 else tp_1 / (tp_1 + fn_1)
        recall_2 = 0 if tp_2 + fn_2 == 0 else tp_2 / (tp_2 + fn_2)
        recall_3 = 0 if tp_3 + fn_3 == 0 else tp_3 / (tp_3 + fn_3)
        recall_4 = 0 if tp_4 + fn_4 == 0 else tp_4 / (tp_4 + fn_4)
        recall_5 = 0 if tp_5 + fn_5 == 0 else tp_5 / (tp_5 + fn_5)
        recall_6 = 0 if tp_6 + fn_6 == 0 else tp_6 / (tp_6 + fn_6)
        recall_7 = 0 if tp_7 + fn_7 == 0 else tp_7 / (tp_7 + fn_7)
        recall_8 = 0 if tp_8 + fn_8 == 0 else tp_8 / (tp_8 + fn_8)
        recall_9 = 0 if tp_9 + fn_9 == 0 else tp_9 / (tp_9 + fn_9)
        recall_10 = 0 if tp_10 + fn_10 == 0 else tp_10 / (tp_10 + fn_10)
        recall_11 = 0 if tp_11 + fn_11 == 0 else tp_11 / (tp_11 + fn_11)
        recall_12 = 0 if tp_12 + fn_12 == 0 else tp_12 / (tp_12 + fn_12)
        recall_13 = 0 if tp_13 + fn_13 == 0 else tp_13 / (tp_13 + fn_13)
        recall_14 = 0 if tp_14 + fn_14 == 0 else tp_14 / (tp_14 + fn_14)
        recall_15 = 0 if tp_15 + fn_15 == 0 else tp_15 / (tp_15 + fn_15)
        recall_16 = 0 if tp_16 + fn_16 == 0 else tp_16 / (tp_16 + fn_16)
        recall_17 = 0 if tp_17 + fn_17 == 0 else tp_17 / (tp_17 + fn_17)
        recall_18 = 0 if tp_18 + fn_18 == 0 else tp_18 / (tp_18 + fn_18)
        recall_19 = 0 if tp_19 + fn_19 == 0 else tp_19 / (tp_19 + fn_19)
        recall_20 = 0 if tp_20 + fn_20 == 0 else tp_20 / (tp_20 + fn_20)
        recall_avg_pen_1 = (recall_1 + recall_2 + recall_3 + recall_4 + recall_5 + recall_6 + recall_7 + recall_8 + recall_9 + recall_10 + recall_11 + recall_12 + recall_13 + recall_14 + recall_15 + recall_16 + recall_17 + recall_18 + recall_19 + recall_20) / (20+1-1)
        return recall_avg_pen_1

    def get_recall_pen_5(conf_matrix):
        tp_1 = conf_matrix[0][0]
        tp_2 = conf_matrix[1][1]
        tp_3 = conf_matrix[2][2]
        tp_4 = conf_matrix[3][3]
        tp_5 = conf_matrix[4][4]
        tp_6 = conf_matrix[5][5]
        tp_7 = conf_matrix[6][6]
        tp_8 = conf_matrix[7][7]
        tp_9 = conf_matrix[8][8]
        tp_10 = conf_matrix[9][9]
        tp_11 = conf_matrix[10][10]
        tp_12 = conf_matrix[11][11]
        tp_13 = conf_matrix[12][12]
        tp_14 = conf_matrix[13][13]
        tp_15 = conf_matrix[14][14]
        tp_16 = conf_matrix[15][15]
        tp_17 = conf_matrix[16][16]
        tp_18 = conf_matrix[17][17]
        tp_19 = conf_matrix[18][18]
        tp_20 = conf_matrix[19][19]
        fn_1 = conf_matrix[0][1] + conf_matrix[0][2] + conf_matrix[0][3] + conf_matrix[0][4] + conf_matrix[0][5] + conf_matrix[0][6] + conf_matrix[0][7] + conf_matrix[0][8] + conf_matrix[0][9] + conf_matrix[0][10] + conf_matrix[0][11] + conf_matrix[0][12] + conf_matrix[0][13] + conf_matrix[0][14] + conf_matrix[0][15] + conf_matrix[0][16] + conf_matrix[0][17] + conf_matrix[0][18] + conf_matrix[0][19]
        fn_2 = conf_matrix[1][0] + conf_matrix[1][2] + conf_matrix[1][3] + conf_matrix[1][4] + conf_matrix[1][5] + conf_matrix[1][6] + conf_matrix[1][7] + conf_matrix[1][8] + conf_matrix[1][9] + conf_matrix[1][10] + conf_matrix[1][11] + conf_matrix[1][12] + conf_matrix[1][13] + conf_matrix[1][14] + conf_matrix[1][15] + conf_matrix[1][16] + conf_matrix[1][17] + conf_matrix[1][18] + conf_matrix[1][19]
        fn_3 = conf_matrix[2][0] + conf_matrix[2][1] + conf_matrix[2][3] + conf_matrix[2][4] + conf_matrix[2][5] + conf_matrix[2][6] + conf_matrix[2][7] + conf_matrix[2][8] + conf_matrix[2][9] + conf_matrix[2][10] + conf_matrix[2][11] + conf_matrix[2][12] + conf_matrix[2][13] + conf_matrix[2][14] + conf_matrix[2][15] + conf_matrix[2][16] + conf_matrix[2][17] + conf_matrix[2][18] + conf_matrix[2][19]
        fn_4 = conf_matrix[3][0] + conf_matrix[3][1] + conf_matrix[3][2] + conf_matrix[3][4] + conf_matrix[3][5] + conf_matrix[3][6] + conf_matrix[3][7] + conf_matrix[3][8] + conf_matrix[3][9] + conf_matrix[3][10] + conf_matrix[3][11] + conf_matrix[3][12] + conf_matrix[3][13] + conf_matrix[3][14] + conf_matrix[3][15] + conf_matrix[3][16] + conf_matrix[3][17] + conf_matrix[3][18] + conf_matrix[3][19]
        fn_5 = conf_matrix[4][0] + conf_matrix[4][1] + conf_matrix[4][2] + conf_matrix[4][3] + conf_matrix[4][5] + conf_matrix[4][6] + conf_matrix[4][7] + conf_matrix[4][8] + conf_matrix[4][9] + conf_matrix[4][10] + conf_matrix[4][11] + conf_matrix[4][12] + conf_matrix[4][13] + conf_matrix[4][14] + conf_matrix[4][15] + conf_matrix[4][16] + conf_matrix[4][17] + conf_matrix[4][18] + conf_matrix[4][19]
        fn_6 = conf_matrix[5][0] + conf_matrix[5][1] + conf_matrix[5][2] + conf_matrix[5][3] + conf_matrix[5][4] + conf_matrix[5][6] + conf_matrix[5][7] + conf_matrix[5][8] + conf_matrix[5][9] + conf_matrix[5][10] + conf_matrix[5][11] + conf_matrix[5][12] + conf_matrix[5][13] + conf_matrix[5][14] + conf_matrix[5][15] + conf_matrix[5][16] + conf_matrix[5][17] + conf_matrix[5][18] + conf_matrix[5][19]
        fn_7 = conf_matrix[6][0] + conf_matrix[6][1] + conf_matrix[6][2] + conf_matrix[6][3] + conf_matrix[6][4] + conf_matrix[6][5] + conf_matrix[6][7] + conf_matrix[6][8] + conf_matrix[6][9] + conf_matrix[6][10] + conf_matrix[6][11] + conf_matrix[6][12] + conf_matrix[6][13] + conf_matrix[6][14] + conf_matrix[6][15] + conf_matrix[6][16] + conf_matrix[6][17] + conf_matrix[6][18] + conf_matrix[6][19]
        fn_8 = conf_matrix[7][0] + conf_matrix[7][1] + conf_matrix[7][2] + conf_matrix[7][3] + conf_matrix[7][4] + conf_matrix[7][5] + conf_matrix[7][6] + conf_matrix[7][8] + conf_matrix[7][9] + conf_matrix[7][10] + conf_matrix[7][11] + conf_matrix[7][12] + conf_matrix[7][13] + conf_matrix[7][14] + conf_matrix[7][15] + conf_matrix[7][16] + conf_matrix[7][17] + conf_matrix[7][18] + conf_matrix[7][19]
        fn_9 = conf_matrix[8][0] + conf_matrix[8][1] + conf_matrix[8][2] + conf_matrix[8][3] + conf_matrix[8][4] + conf_matrix[8][5] + conf_matrix[8][6] + conf_matrix[8][7] + conf_matrix[8][9] + conf_matrix[8][10] + conf_matrix[8][11] + conf_matrix[8][12] + conf_matrix[8][13] + conf_matrix[8][14] + conf_matrix[8][15] + conf_matrix[8][16] + conf_matrix[8][17] + conf_matrix[8][18] + conf_matrix[8][19]
        fn_10 = conf_matrix[9][0] + conf_matrix[9][1] + conf_matrix[9][2] + conf_matrix[9][3] + conf_matrix[9][4] + conf_matrix[9][5] + conf_matrix[9][6] + conf_matrix[9][7] + conf_matrix[9][8] + conf_matrix[9][10] + conf_matrix[9][11] + conf_matrix[9][12] + conf_matrix[9][13] + conf_matrix[9][14] + conf_matrix[9][15] + conf_matrix[9][16] + conf_matrix[9][17] + conf_matrix[9][18] + conf_matrix[9][19]
        fn_11 = conf_matrix[10][0] + conf_matrix[10][1] + conf_matrix[10][2] + conf_matrix[10][3] + conf_matrix[10][4] + conf_matrix[10][5] + conf_matrix[10][6] + conf_matrix[10][7] + conf_matrix[10][8] + conf_matrix[10][9] + conf_matrix[10][11] + conf_matrix[10][12] + conf_matrix[10][13] + conf_matrix[10][14] + conf_matrix[10][15] + conf_matrix[10][16] + conf_matrix[10][17] + conf_matrix[10][18] + conf_matrix[10][19]
        fn_12 = conf_matrix[11][0] + conf_matrix[11][1] + conf_matrix[11][2] + conf_matrix[11][3] + conf_matrix[11][4] + conf_matrix[11][5] + conf_matrix[11][6] + conf_matrix[11][7] + conf_matrix[11][8] + conf_matrix[11][9] + conf_matrix[11][10] + conf_matrix[11][12] + conf_matrix[11][13] + conf_matrix[11][14] + conf_matrix[11][15] + conf_matrix[11][16] + conf_matrix[11][17] + conf_matrix[11][18] + conf_matrix[11][19]
        fn_13 = conf_matrix[12][0] + conf_matrix[12][1] + conf_matrix[12][2] + conf_matrix[12][3] + conf_matrix[12][4] + conf_matrix[12][5] + conf_matrix[12][6] + conf_matrix[12][7] + conf_matrix[12][8] + conf_matrix[12][9] + conf_matrix[12][10] + conf_matrix[12][11] + conf_matrix[12][13] + conf_matrix[12][14] + conf_matrix[12][15] + conf_matrix[12][16] + conf_matrix[12][17] + conf_matrix[12][18] + conf_matrix[12][19]
        fn_14 = conf_matrix[13][0] + conf_matrix[13][1] + conf_matrix[13][2] + conf_matrix[13][3] + conf_matrix[13][4] + conf_matrix[13][5] + conf_matrix[13][6] + conf_matrix[13][7] + conf_matrix[13][8] + conf_matrix[13][9] + conf_matrix[13][10] + conf_matrix[13][11] + conf_matrix[13][12] + conf_matrix[13][14] + conf_matrix[13][15] + conf_matrix[13][16] + conf_matrix[13][17] + conf_matrix[13][18] + conf_matrix[13][19]
        fn_15 = conf_matrix[14][0] + conf_matrix[14][1] + conf_matrix[14][2] + conf_matrix[14][3] + conf_matrix[14][4] + conf_matrix[14][5] + conf_matrix[14][6] + conf_matrix[14][7] + conf_matrix[14][8] + conf_matrix[14][9] + conf_matrix[14][10] + conf_matrix[14][11] + conf_matrix[14][12] + conf_matrix[14][13] + conf_matrix[14][15] + conf_matrix[14][16] + conf_matrix[14][17] + conf_matrix[14][18] + conf_matrix[14][19]
        fn_16 = conf_matrix[15][0] + conf_matrix[15][1] + conf_matrix[15][2] + conf_matrix[15][3] + conf_matrix[15][4] + conf_matrix[15][5] + conf_matrix[15][6] + conf_matrix[15][7] + conf_matrix[15][8] + conf_matrix[15][9] + conf_matrix[15][10] + conf_matrix[15][11] + conf_matrix[15][12] + conf_matrix[15][13] + conf_matrix[15][14] + conf_matrix[15][16] + conf_matrix[15][17] + conf_matrix[15][18] + conf_matrix[15][19]
        fn_17 = conf_matrix[16][0] + conf_matrix[16][1] + conf_matrix[16][2] + conf_matrix[16][3] + conf_matrix[16][4] + conf_matrix[16][5] + conf_matrix[16][6] + conf_matrix[16][7] + conf_matrix[16][8] + conf_matrix[16][9] + conf_matrix[16][10] + conf_matrix[16][11] + conf_matrix[16][12] + conf_matrix[16][13] + conf_matrix[16][14] + conf_matrix[16][15] + conf_matrix[16][17] + conf_matrix[16][18] + conf_matrix[16][19]
        fn_18 = conf_matrix[17][0] + conf_matrix[17][1] + conf_matrix[17][2] + conf_matrix[17][3] + conf_matrix[17][4] + conf_matrix[17][5] + conf_matrix[17][6] + conf_matrix[17][7] + conf_matrix[17][8] + conf_matrix[17][9] + conf_matrix[17][10] + conf_matrix[17][11] + conf_matrix[17][12] + conf_matrix[17][13] + conf_matrix[17][14] + conf_matrix[17][15] + conf_matrix[17][16] + conf_matrix[17][18] + conf_matrix[17][19]
        fn_19 = conf_matrix[18][0] + conf_matrix[18][1] + conf_matrix[18][2] + conf_matrix[18][3] + conf_matrix[18][4] + conf_matrix[18][5] + conf_matrix[18][6] + conf_matrix[18][7] + conf_matrix[18][8] + conf_matrix[18][9] + conf_matrix[18][10] + conf_matrix[18][11] + conf_matrix[18][12] + conf_matrix[18][13] + conf_matrix[18][14] + conf_matrix[18][15] + conf_matrix[18][16] + conf_matrix[18][17] + conf_matrix[18][19]
        fn_20 = conf_matrix[19][0] + conf_matrix[19][1] + conf_matrix[19][2] + conf_matrix[19][3] + conf_matrix[19][4] + conf_matrix[19][5] + conf_matrix[19][6] + conf_matrix[19][7] + conf_matrix[19][8] + conf_matrix[19][9] + conf_matrix[19][10] + conf_matrix[19][11] + conf_matrix[19][12] + conf_matrix[19][13] + conf_matrix[19][14] + conf_matrix[19][15] + conf_matrix[19][16] + conf_matrix[19][17] + conf_matrix[19][18]
        recall_1 = 0 if tp_1 + fn_1 == 0 else tp_1 / (tp_1 + fn_1)
        recall_2 = 0 if tp_2 + fn_2 == 0 else tp_2 / (tp_2 + fn_2)
        recall_3 = 0 if tp_3 + fn_3 == 0 else tp_3 / (tp_3 + fn_3)
        recall_4 = 0 if tp_4 + fn_4 == 0 else tp_4 / (tp_4 + fn_4)
        recall_5 = 0 if tp_5 + fn_5 == 0 else tp_5 / (tp_5 + fn_5)
        recall_6 = 0 if tp_6 + fn_6 == 0 else tp_6 / (tp_6 + fn_6)
        recall_7 = 0 if tp_7 + fn_7 == 0 else tp_7 / (tp_7 + fn_7)
        recall_8 = 0 if tp_8 + fn_8 == 0 else tp_8 / (tp_8 + fn_8)
        recall_9 = 0 if tp_9 + fn_9 == 0 else tp_9 / (tp_9 + fn_9)
        recall_10 = 0 if tp_10 + fn_10 == 0 else tp_10 / (tp_10 + fn_10)
        recall_11 = 0 if tp_11 + fn_11 == 0 else tp_11 / (tp_11 + fn_11)
        recall_12 = 0 if tp_12 + fn_12 == 0 else tp_12 / (tp_12 + fn_12)
        recall_13 = 0 if tp_13 + fn_13 == 0 else tp_13 / (tp_13 + fn_13)
        recall_14 = 0 if tp_14 + fn_14 == 0 else tp_14 / (tp_14 + fn_14)
        recall_15 = 0 if tp_15 + fn_15 == 0 else tp_15 / (tp_15 + fn_15)
        recall_16 = 0 if tp_16 + fn_16 == 0 else tp_16 / (tp_16 + fn_16)
        recall_17 = 0 if tp_17 + fn_17 == 0 else tp_17 / (tp_17 + fn_17)
        recall_18 = 0 if tp_18 + fn_18 == 0 else tp_18 / (tp_18 + fn_18)
        recall_19 = 0 if tp_19 + fn_19 == 0 else tp_19 / (tp_19 + fn_19)
        recall_20 = 0 if tp_20 + fn_20 == 0 else tp_20 / (tp_20 + fn_20)
        recall_avg_pen_5 = (recall_1 + recall_2 + recall_3 + recall_4 + recall_5 + recall_6 + recall_7 + recall_8 + recall_9 + recall_10 + recall_11 + recall_12 + recall_13 + recall_14 + recall_15 + recall_16 + recall_17 + recall_18 + recall_19 + (5*recall_20)) / (20+5-1)
        return recall_avg_pen_5

    dnn_pred_prob_FI,dnn_pred_prob_FG,dnn_pred_prob_GR27,dnn_pred_prob_LM,dnn_pred_prob_LMM,dnn_pred_prob_PC,dnn_pred_prob_RG12,dnn_pred_prob_RG2,dnn_pred_prob_RG3,dnn_pred_prob_RGM,dnn_pred_prob_RGQC,dnn_pred_prob_T8,dnn_pred_prob_TM10,dnn_pred_prob_TM4,dnn_pred_prob_TM5,dnn_pred_prob_TM6,dnn_pred_prob_TM8,dnn_pred_prob_TM9,dnn_pred_prob_TMQC,dnn_pred_prob_TQC,dnn_prediction = label_assigner(y_test, dnn_prediction, dnn_pred_class_FI,dnn_pred_class_FG,dnn_pred_class_GR27,dnn_pred_class_LM,dnn_pred_class_LMM,dnn_pred_class_PC,dnn_pred_class_RG12,dnn_pred_class_RG2,dnn_pred_class_RG3,dnn_pred_class_RGM,dnn_pred_class_RGQC,dnn_pred_class_T8,dnn_pred_class_TM10,dnn_pred_class_TM4,dnn_pred_class_TM5,dnn_pred_class_TM6,dnn_pred_class_TM8,dnn_pred_class_TM9,dnn_pred_class_TMQC,dnn_pred_class_TQC,dnn_pred_prob_FI,dnn_pred_prob_FG,dnn_pred_prob_GR27,dnn_pred_prob_LM,dnn_pred_prob_LMM,dnn_pred_prob_PC,dnn_pred_prob_RG12,dnn_pred_prob_RG2,dnn_pred_prob_RG3,dnn_pred_prob_RGM,dnn_pred_prob_RGQC,dnn_pred_prob_T8,dnn_pred_prob_TM10,dnn_pred_prob_TM4,dnn_pred_prob_TM5,dnn_pred_prob_TM6,dnn_pred_prob_TM8,dnn_pred_prob_TM9,dnn_pred_prob_TMQC,dnn_pred_prob_TQC)
    lr_pred_prob_FI,lr_pred_prob_FG,lr_pred_prob_GR27,lr_pred_prob_LM,lr_pred_prob_LMM,lr_pred_prob_PC,lr_pred_prob_RG12,lr_pred_prob_RG2,lr_pred_prob_RG3,lr_pred_prob_RGM,lr_pred_prob_RGQC,lr_pred_prob_T8,lr_pred_prob_TM10,lr_pred_prob_TM4,lr_pred_prob_TM5,lr_pred_prob_TM6,lr_pred_prob_TM8,lr_pred_prob_TM9,lr_pred_prob_TMQC,lr_pred_prob_TQC,lr_prediction = label_assigner(y_test, lr_prediction, lr_pred_class_FI,lr_pred_class_FG,lr_pred_class_GR27,lr_pred_class_LM,lr_pred_class_LMM,lr_pred_class_PC,lr_pred_class_RG12,lr_pred_class_RG2,lr_pred_class_RG3,lr_pred_class_RGM,lr_pred_class_RGQC,lr_pred_class_T8,lr_pred_class_TM10,lr_pred_class_TM4,lr_pred_class_TM5,lr_pred_class_TM6,lr_pred_class_TM8,lr_pred_class_TM9,lr_pred_class_TMQC,lr_pred_class_TQC,lr_pred_prob_FI,lr_pred_prob_FG,lr_pred_prob_GR27,lr_pred_prob_LM,lr_pred_prob_LMM,lr_pred_prob_PC,lr_pred_prob_RG12,lr_pred_prob_RG2,lr_pred_prob_RG3,lr_pred_prob_RGM,lr_pred_prob_RGQC,lr_pred_prob_T8,lr_pred_prob_TM10,lr_pred_prob_TM4,lr_pred_prob_TM5,lr_pred_prob_TM6,lr_pred_prob_TM8,lr_pred_prob_TM9,lr_pred_prob_TMQC,lr_pred_prob_TQC)
    nb_pred_prob_FI,nb_pred_prob_FG,nb_pred_prob_GR27,nb_pred_prob_LM,nb_pred_prob_LMM,nb_pred_prob_PC,nb_pred_prob_RG12,nb_pred_prob_RG2,nb_pred_prob_RG3,nb_pred_prob_RGM,nb_pred_prob_RGQC,nb_pred_prob_T8,nb_pred_prob_TM10,nb_pred_prob_TM4,nb_pred_prob_TM5,nb_pred_prob_TM6,nb_pred_prob_TM8,nb_pred_prob_TM9,nb_pred_prob_TMQC,nb_pred_prob_TQC,nb_prediction = label_assigner(y_test, nb_prediction, nb_pred_class_FI,nb_pred_class_FG,nb_pred_class_GR27,nb_pred_class_LM,nb_pred_class_LMM,nb_pred_class_PC,nb_pred_class_RG12,nb_pred_class_RG2,nb_pred_class_RG3,nb_pred_class_RGM,nb_pred_class_RGQC,nb_pred_class_T8,nb_pred_class_TM10,nb_pred_class_TM4,nb_pred_class_TM5,nb_pred_class_TM6,nb_pred_class_TM8,nb_pred_class_TM9,nb_pred_class_TMQC,nb_pred_class_TQC,nb_pred_prob_FI,nb_pred_prob_FG,nb_pred_prob_GR27,nb_pred_prob_LM,nb_pred_prob_LMM,nb_pred_prob_PC,nb_pred_prob_RG12,nb_pred_prob_RG2,nb_pred_prob_RG3,nb_pred_prob_RGM,nb_pred_prob_RGQC,nb_pred_prob_T8,nb_pred_prob_TM10,nb_pred_prob_TM4,nb_pred_prob_TM5,nb_pred_prob_TM6,nb_pred_prob_TM8,nb_pred_prob_TM9,nb_pred_prob_TMQC,nb_pred_prob_TQC)
    rf_pred_prob_FI,rf_pred_prob_FG,rf_pred_prob_GR27,rf_pred_prob_LM,rf_pred_prob_LMM,rf_pred_prob_PC,rf_pred_prob_RG12,rf_pred_prob_RG2,rf_pred_prob_RG3,rf_pred_prob_RGM,rf_pred_prob_RGQC,rf_pred_prob_T8,rf_pred_prob_TM10,rf_pred_prob_TM4,rf_pred_prob_TM5,rf_pred_prob_TM6,rf_pred_prob_TM8,rf_pred_prob_TM9,rf_pred_prob_TMQC,rf_pred_prob_TQC,rf_prediction = label_assigner(y_test, rf_prediction, rf_pred_class_FI,rf_pred_class_FG,rf_pred_class_GR27,rf_pred_class_LM,rf_pred_class_LMM,rf_pred_class_PC,rf_pred_class_RG12,rf_pred_class_RG2,rf_pred_class_RG3,rf_pred_class_RGM,rf_pred_class_RGQC,rf_pred_class_T8,rf_pred_class_TM10,rf_pred_class_TM4,rf_pred_class_TM5,rf_pred_class_TM6,rf_pred_class_TM8,rf_pred_class_TM9,rf_pred_class_TMQC,rf_pred_class_TQC,rf_pred_prob_FI,rf_pred_prob_FG,rf_pred_prob_GR27,rf_pred_prob_LM,rf_pred_prob_LMM,rf_pred_prob_PC,rf_pred_prob_RG12,rf_pred_prob_RG2,rf_pred_prob_RG3,rf_pred_prob_RGM,rf_pred_prob_RGQC,rf_pred_prob_T8,rf_pred_prob_TM10,rf_pred_prob_TM4,rf_pred_prob_TM5,rf_pred_prob_TM6,rf_pred_prob_TM8,rf_pred_prob_TM9,rf_pred_prob_TMQC,rf_pred_prob_TQC)
    svm_pred_prob_FI,svm_pred_prob_FG,svm_pred_prob_GR27,svm_pred_prob_LM,svm_pred_prob_LMM,svm_pred_prob_PC,svm_pred_prob_RG12,svm_pred_prob_RG2,svm_pred_prob_RG3,svm_pred_prob_RGM,svm_pred_prob_RGQC,svm_pred_prob_T8,svm_pred_prob_TM10,svm_pred_prob_TM4,svm_pred_prob_TM5,svm_pred_prob_TM6,svm_pred_prob_TM8,svm_pred_prob_TM9,svm_pred_prob_TMQC,svm_pred_prob_TQC,svm_prediction = label_assigner(y_test, svm_prediction, svm_pred_class_FI,svm_pred_class_FG,svm_pred_class_GR27,svm_pred_class_LM,svm_pred_class_LMM,svm_pred_class_PC,svm_pred_class_RG12,svm_pred_class_RG2,svm_pred_class_RG3,svm_pred_class_RGM,svm_pred_class_RGQC,svm_pred_class_T8,svm_pred_class_TM10,svm_pred_class_TM4,svm_pred_class_TM5,svm_pred_class_TM6,svm_pred_class_TM8,svm_pred_class_TM9,svm_pred_class_TMQC,svm_pred_class_TQC,svm_pred_prob_FI,svm_pred_prob_FG,svm_pred_prob_GR27,svm_pred_prob_LM,svm_pred_prob_LMM,svm_pred_prob_PC,svm_pred_prob_RG12,svm_pred_prob_RG2,svm_pred_prob_RG3,svm_pred_prob_RGM,svm_pred_prob_RGQC,svm_pred_prob_T8,svm_pred_prob_TM10,svm_pred_prob_TM4,svm_pred_prob_TM5,svm_pred_prob_TM6,svm_pred_prob_TM8,svm_pred_prob_TM9,svm_pred_prob_TMQC,svm_pred_prob_TQC)

    def perf_evaluator(y_test, pred_prob_FI,pred_prob_FG,pred_prob_GR27,pred_prob_LM,pred_prob_LMM,pred_prob_PC,pred_prob_RG12,pred_prob_RG2,pred_prob_RG3,pred_prob_RGM,pred_prob_RGQC,pred_prob_T8,pred_prob_TM10,pred_prob_TM4,pred_prob_TM5,pred_prob_TM6,pred_prob_TM8,pred_prob_TM9,pred_prob_TMQC,pred_prob_TQC,prediction, f1_score_pen_1_kfoldcv, f1_score_pen_5_kfoldcv, ovr_accuracy_kfoldcv, auc_kfoldcv, gmean_kfoldcv, clf, repeat):
        conf_matrix = confusion_matrix(y_test, prediction)
        precision = get_precision(conf_matrix)
        recall_pen_1 = get_recall_pen_1(conf_matrix)
        recall_pen_5 = get_recall_pen_5(conf_matrix)
        f1_score_pen_1 = 2 * (precision * recall_pen_1) / (precision + recall_pen_1)
        f1_score_pen_5 = 2 * (precision * recall_pen_5) / (precision + recall_pen_5)
        ovr_accuracy = (conf_matrix[0][0] + conf_matrix[1][1] + conf_matrix[2][2] + conf_matrix[3][3] + conf_matrix[4][4] + conf_matrix[5][5] + conf_matrix[6][6] + conf_matrix[7][7] + conf_matrix[8][8] + conf_matrix[9][9] + conf_matrix[10][10] + conf_matrix[11][11] + conf_matrix[12][12] + conf_matrix[13][13] + conf_matrix[14][14] + conf_matrix[15][15] + conf_matrix[16][16] + conf_matrix[17][17] + conf_matrix[18][18] + conf_matrix[19][19]) / (
                sum(conf_matrix[0]) + sum(conf_matrix[1]) + sum(conf_matrix[2]) + sum(conf_matrix[3]) + sum(conf_matrix[4]) + sum(conf_matrix[5]) + sum(conf_matrix[6]) + sum(conf_matrix[7]) + sum(conf_matrix[8]) + sum(conf_matrix[9]) + sum(conf_matrix[10]) + sum(conf_matrix[11]) + sum(conf_matrix[12]) + sum(conf_matrix[13]) + sum(conf_matrix[14]) + sum(conf_matrix[15]) + sum(conf_matrix[16]) + sum(conf_matrix[17]) + sum(conf_matrix[18]) + sum(conf_matrix[19]))
        auc_FI = roc_auc_score(y_true = y_test_FI, y_score = pd.DataFrame(pred_prob_FI).iloc[:,0])
        auc_FG = roc_auc_score(y_true = y_test_FG, y_score = pd.DataFrame(pred_prob_FG).iloc[:,0])
        auc_GR27 = roc_auc_score(y_true = y_test_GR27, y_score = pd.DataFrame(pred_prob_GR27).iloc[:,0])
        auc_LM = roc_auc_score(y_true = y_test_LM, y_score = pd.DataFrame(pred_prob_LM).iloc[:,0])
        auc_LMM = roc_auc_score(y_true = y_test_LMM, y_score = pd.DataFrame(pred_prob_LMM).iloc[:,0])
        auc_PC = roc_auc_score(y_true = y_test_PC, y_score = pd.DataFrame(pred_prob_PC).iloc[:,0])
        auc_RG12 = roc_auc_score(y_true = y_test_RG12, y_score = pd.DataFrame(pred_prob_RG12).iloc[:,0])
        auc_RG2 = roc_auc_score(y_true = y_test_RG2, y_score = pd.DataFrame(pred_prob_RG2).iloc[:,0])
        auc_RG3 = roc_auc_score(y_true = y_test_RG3, y_score = pd.DataFrame(pred_prob_RG3).iloc[:,0])
        auc_RGM = roc_auc_score(y_true = y_test_RGM, y_score = pd.DataFrame(pred_prob_RGM).iloc[:,0])
        auc_RGQC = roc_auc_score(y_true = y_test_RGQC, y_score = pd.DataFrame(pred_prob_RGQC).iloc[:,0])
        auc_T8 = roc_auc_score(y_true = y_test_T8, y_score = pd.DataFrame(pred_prob_T8).iloc[:,0])
        auc_TM10 = roc_auc_score(y_true = y_test_TM10, y_score = pd.DataFrame(pred_prob_TM10).iloc[:,0])
        auc_TM4 = roc_auc_score(y_true = y_test_TM4, y_score = pd.DataFrame(pred_prob_TM4).iloc[:,0])
        auc_TM5 = roc_auc_score(y_true = y_test_TM5, y_score = pd.DataFrame(pred_prob_TM5).iloc[:,0])
        auc_TM6 = roc_auc_score(y_true = y_test_TM6, y_score = pd.DataFrame(pred_prob_TM6).iloc[:,0])
        auc_TM8 = roc_auc_score(y_true = y_test_TM8, y_score = pd.DataFrame(pred_prob_TM8).iloc[:,0])
        auc_TM9 = roc_auc_score(y_true = y_test_TM9, y_score = pd.DataFrame(pred_prob_TM9).iloc[:,0])
        auc_TMQC = roc_auc_score(y_true = y_test_TMQC, y_score = pd.DataFrame(pred_prob_TMQC).iloc[:,0])
        auc_TQC = roc_auc_score(y_true = y_test_TQC, y_score = pd.DataFrame(pred_prob_TQC).iloc[:,0])
        gmean = geometric_mean_score(y_true = y_test, y_pred = prediction, average = 'macro')
        conf_matrix = pd.DataFrame(conf_matrix)               
        conf_matrix.to_csv('conf_matrix_'+imb_technique+'_' + clf + '_production_'+ str(nsplits) +'foldcv_' + str(repeat+1)+'.csv',header=False,index=False) #First repetition
        #conf_matrix.to_csv('conf_matrix_'+imb_technique+'_penalty_' + str(penalty) + '_' + clf + '_production_'+ str(nsplits) +'foldcv_' + str(repeat+6)+'.csv',header=False,index=False) #Second repetition
        f1_score_pen_1_kfoldcv[repeat] = f1_score_pen_1
        f1_score_pen_5_kfoldcv[repeat] = f1_score_pen_5
        ovr_accuracy_kfoldcv[repeat] = ovr_accuracy
        auc_kfoldcv[repeat] = (auc_FI + auc_FG + auc_GR27 + auc_LM + auc_LMM + auc_PC + auc_RG12 + auc_RG2 + auc_RG3 + auc_RGM + auc_RGQC + auc_T8 + auc_TM10 + auc_TM4 + auc_TM5 + auc_TM6 + auc_TM8 + auc_TM9 + auc_TMQC + auc_TQC)/20
        gmean_kfoldcv[repeat] = gmean
        return conf_matrix, f1_score_pen_1_kfoldcv, f1_score_pen_5_kfoldcv, ovr_accuracy_kfoldcv, auc_kfoldcv, gmean_kfoldcv

    dnn_conf_matrix, dnn_f1_score_pen_1_kfoldcv, dnn_f1_score_pen_5_kfoldcv, dnn_ovr_accuracy_kfoldcv, dnn_auc_kfoldcv, dnn_gmean_kfoldcv = perf_evaluator(y_test, dnn_pred_prob_FI,dnn_pred_prob_FG,dnn_pred_prob_GR27,dnn_pred_prob_LM,dnn_pred_prob_LMM,dnn_pred_prob_PC,dnn_pred_prob_RG12,dnn_pred_prob_RG2,dnn_pred_prob_RG3,dnn_pred_prob_RGM,dnn_pred_prob_RGQC,dnn_pred_prob_T8,dnn_pred_prob_TM10,dnn_pred_prob_TM4,dnn_pred_prob_TM5,dnn_pred_prob_TM6,dnn_pred_prob_TM8,dnn_pred_prob_TM9,dnn_pred_prob_TMQC,dnn_pred_prob_TQC,dnn_prediction, dnn_f1_score_pen_1_kfoldcv, dnn_f1_score_pen_5_kfoldcv, dnn_ovr_accuracy_kfoldcv, dnn_auc_kfoldcv, dnn_gmean_kfoldcv, "dnn", repeat)
    lr_conf_matrix, lr_f1_score_pen_1_kfoldcv, lr_f1_score_pen_5_kfoldcv, lr_ovr_accuracy_kfoldcv, lr_auc_kfoldcv, lr_gmean_kfoldcv = perf_evaluator(y_test, lr_pred_prob_FI,lr_pred_prob_FG,lr_pred_prob_GR27,lr_pred_prob_LM,lr_pred_prob_LMM,lr_pred_prob_PC,lr_pred_prob_RG12,lr_pred_prob_RG2,lr_pred_prob_RG3,lr_pred_prob_RGM,lr_pred_prob_RGQC,lr_pred_prob_T8,lr_pred_prob_TM10,lr_pred_prob_TM4,lr_pred_prob_TM5,lr_pred_prob_TM6,lr_pred_prob_TM8,lr_pred_prob_TM9,lr_pred_prob_TMQC,lr_pred_prob_TQC,lr_prediction, lr_f1_score_pen_1_kfoldcv, lr_f1_score_pen_5_kfoldcv, lr_ovr_accuracy_kfoldcv, lr_auc_kfoldcv, lr_gmean_kfoldcv, "lr", repeat)
    nb_conf_matrix, nb_f1_score_pen_1_kfoldcv, nb_f1_score_pen_5_kfoldcv, nb_ovr_accuracy_kfoldcv, nb_auc_kfoldcv, nb_gmean_kfoldcv = perf_evaluator(y_test, nb_pred_prob_FI,nb_pred_prob_FG,nb_pred_prob_GR27,nb_pred_prob_LM,nb_pred_prob_LMM,nb_pred_prob_PC,nb_pred_prob_RG12,nb_pred_prob_RG2,nb_pred_prob_RG3,nb_pred_prob_RGM,nb_pred_prob_RGQC,nb_pred_prob_T8,nb_pred_prob_TM10,nb_pred_prob_TM4,nb_pred_prob_TM5,nb_pred_prob_TM6,nb_pred_prob_TM8,nb_pred_prob_TM9,nb_pred_prob_TMQC,nb_pred_prob_TQC,nb_prediction, nb_f1_score_pen_1_kfoldcv, nb_f1_score_pen_5_kfoldcv, nb_ovr_accuracy_kfoldcv, nb_auc_kfoldcv, nb_gmean_kfoldcv, "nb", repeat)
    rf_conf_matrix, rf_f1_score_pen_1_kfoldcv, rf_f1_score_pen_5_kfoldcv, rf_ovr_accuracy_kfoldcv, rf_auc_kfoldcv, rf_gmean_kfoldcv = perf_evaluator(y_test, rf_pred_prob_FI,rf_pred_prob_FG,rf_pred_prob_GR27,rf_pred_prob_LM,rf_pred_prob_LMM,rf_pred_prob_PC,rf_pred_prob_RG12,rf_pred_prob_RG2,rf_pred_prob_RG3,rf_pred_prob_RGM,rf_pred_prob_RGQC,rf_pred_prob_T8,rf_pred_prob_TM10,rf_pred_prob_TM4,rf_pred_prob_TM5,rf_pred_prob_TM6,rf_pred_prob_TM8,rf_pred_prob_TM9,rf_pred_prob_TMQC,rf_pred_prob_TQC,rf_prediction, rf_f1_score_pen_1_kfoldcv, rf_f1_score_pen_5_kfoldcv, rf_ovr_accuracy_kfoldcv, rf_auc_kfoldcv, rf_gmean_kfoldcv, "rf", repeat)
    svm_conf_matrix, svm_f1_score_pen_1_kfoldcv, svm_f1_score_pen_5_kfoldcv, svm_ovr_accuracy_kfoldcv, svm_auc_kfoldcv, svm_gmean_kfoldcv = perf_evaluator(y_test, svm_pred_prob_FI,svm_pred_prob_FG,svm_pred_prob_GR27,svm_pred_prob_LM,svm_pred_prob_LMM,svm_pred_prob_PC,svm_pred_prob_RG12,svm_pred_prob_RG2,svm_pred_prob_RG3,svm_pred_prob_RGM,svm_pred_prob_RGQC,svm_pred_prob_T8,svm_pred_prob_TM10,svm_pred_prob_TM4,svm_pred_prob_TM5,svm_pred_prob_TM6,svm_pred_prob_TM8,svm_pred_prob_TM9,svm_pred_prob_TMQC,svm_pred_prob_TQC,svm_prediction, svm_f1_score_pen_1_kfoldcv, svm_f1_score_pen_5_kfoldcv, svm_ovr_accuracy_kfoldcv, svm_auc_kfoldcv, svm_gmean_kfoldcv, "svm", repeat)

    repeat = repeat + 1

with open(outfile, 'w') as fout:
    fout.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % ("dnn_f1_pen1", "dnn_f1_pen5", "dnn_ovr_acc", "dnn_auc", "dnn_gmean","lr_f1_pen1", "lr_f1_pen5", "lr_ovr_acc", "lr_auc", "lr_gmean","nb_f1_pen1", "nb_f1_pen5", "nb_ovr_acc", "nb_auc", "nb_gmean", "rf_f1_pen1", "rf_f1_pen5", "rf_ovr_acc", "rf_auc", "rf_gmean","svm_f1_pen1", "svm_f1_pen5", "svm_ovr_acc", "svm_auc", "svm_gmean"))
    fout.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_f1_score_pen_1_kfoldcv[0], dnn_f1_score_pen_5_kfoldcv[0], dnn_ovr_accuracy_kfoldcv[0], dnn_auc_kfoldcv[0], dnn_gmean_kfoldcv[0], lr_f1_score_pen_1_kfoldcv[0], lr_f1_score_pen_5_kfoldcv[0], lr_ovr_accuracy_kfoldcv[0], lr_auc_kfoldcv[0], lr_gmean_kfoldcv[0], nb_f1_score_pen_1_kfoldcv[0], nb_f1_score_pen_5_kfoldcv[0], nb_ovr_accuracy_kfoldcv[0], nb_auc_kfoldcv[0], nb_gmean_kfoldcv[0], rf_f1_score_pen_1_kfoldcv[0], rf_f1_score_pen_5_kfoldcv[0], rf_ovr_accuracy_kfoldcv[0], rf_auc_kfoldcv[0], rf_gmean_kfoldcv[0], svm_f1_score_pen_1_kfoldcv[0], svm_f1_score_pen_5_kfoldcv[0], svm_ovr_accuracy_kfoldcv[0], svm_auc_kfoldcv[0], svm_gmean_kfoldcv[0]))
    fout.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_f1_score_pen_1_kfoldcv[1], dnn_f1_score_pen_5_kfoldcv[1], dnn_ovr_accuracy_kfoldcv[1], dnn_auc_kfoldcv[1], dnn_gmean_kfoldcv[1], lr_f1_score_pen_1_kfoldcv[1], lr_f1_score_pen_5_kfoldcv[1], lr_ovr_accuracy_kfoldcv[1], lr_auc_kfoldcv[1], lr_gmean_kfoldcv[1], nb_f1_score_pen_1_kfoldcv[1], nb_f1_score_pen_5_kfoldcv[1], nb_ovr_accuracy_kfoldcv[1], nb_auc_kfoldcv[1], nb_gmean_kfoldcv[1], rf_f1_score_pen_1_kfoldcv[1], rf_f1_score_pen_5_kfoldcv[1], rf_ovr_accuracy_kfoldcv[1], rf_auc_kfoldcv[1], rf_gmean_kfoldcv[1], svm_f1_score_pen_1_kfoldcv[1], svm_f1_score_pen_5_kfoldcv[1], svm_ovr_accuracy_kfoldcv[1], svm_auc_kfoldcv[1], svm_gmean_kfoldcv[1]))
    fout.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_f1_score_pen_1_kfoldcv[2], dnn_f1_score_pen_5_kfoldcv[2], dnn_ovr_accuracy_kfoldcv[2], dnn_auc_kfoldcv[2], dnn_gmean_kfoldcv[2], lr_f1_score_pen_1_kfoldcv[2], lr_f1_score_pen_5_kfoldcv[2], lr_ovr_accuracy_kfoldcv[2], lr_auc_kfoldcv[2], lr_gmean_kfoldcv[2], nb_f1_score_pen_1_kfoldcv[2], nb_f1_score_pen_5_kfoldcv[2], nb_ovr_accuracy_kfoldcv[2], nb_auc_kfoldcv[2], nb_gmean_kfoldcv[2], rf_f1_score_pen_1_kfoldcv[2], rf_f1_score_pen_5_kfoldcv[2], rf_ovr_accuracy_kfoldcv[2], rf_auc_kfoldcv[2], rf_gmean_kfoldcv[2], svm_f1_score_pen_1_kfoldcv[2], svm_f1_score_pen_5_kfoldcv[2], svm_ovr_accuracy_kfoldcv[2], svm_auc_kfoldcv[2], svm_gmean_kfoldcv[2]))
    fout.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_f1_score_pen_1_kfoldcv[3], dnn_f1_score_pen_5_kfoldcv[3], dnn_ovr_accuracy_kfoldcv[3], dnn_auc_kfoldcv[3], dnn_gmean_kfoldcv[3], lr_f1_score_pen_1_kfoldcv[3], lr_f1_score_pen_5_kfoldcv[3], lr_ovr_accuracy_kfoldcv[3], lr_auc_kfoldcv[3], lr_gmean_kfoldcv[3], nb_f1_score_pen_1_kfoldcv[3], nb_f1_score_pen_5_kfoldcv[3], nb_ovr_accuracy_kfoldcv[3], nb_auc_kfoldcv[3], nb_gmean_kfoldcv[3], rf_f1_score_pen_1_kfoldcv[3], rf_f1_score_pen_5_kfoldcv[3], rf_ovr_accuracy_kfoldcv[3], rf_auc_kfoldcv[3], rf_gmean_kfoldcv[3], svm_f1_score_pen_1_kfoldcv[3], svm_f1_score_pen_5_kfoldcv[3], svm_ovr_accuracy_kfoldcv[3], svm_auc_kfoldcv[3], svm_gmean_kfoldcv[3]))
    fout.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_f1_score_pen_1_kfoldcv[4], dnn_f1_score_pen_5_kfoldcv[4], dnn_ovr_accuracy_kfoldcv[4], dnn_auc_kfoldcv[4], dnn_gmean_kfoldcv[4], lr_f1_score_pen_1_kfoldcv[4], lr_f1_score_pen_5_kfoldcv[4], lr_ovr_accuracy_kfoldcv[4], lr_auc_kfoldcv[4], lr_gmean_kfoldcv[4], nb_f1_score_pen_1_kfoldcv[4], nb_f1_score_pen_5_kfoldcv[4], nb_ovr_accuracy_kfoldcv[4], nb_auc_kfoldcv[4], nb_gmean_kfoldcv[4], rf_f1_score_pen_1_kfoldcv[4], rf_f1_score_pen_5_kfoldcv[4], rf_ovr_accuracy_kfoldcv[4], rf_auc_kfoldcv[4], rf_gmean_kfoldcv[4], svm_f1_score_pen_1_kfoldcv[4], svm_f1_score_pen_5_kfoldcv[4], svm_ovr_accuracy_kfoldcv[4], svm_auc_kfoldcv[4], svm_gmean_kfoldcv[4]))
with open(outfile_param_FI, 'w') as fout_param_FI:
    fout_param_FI.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % ("dnn_hidden_layer_sizes", "dnn_learning_rate_init", "lr_solver", "lr_tol", "lr_C", "nb_var_smoothing", "rf_n_estimator", "rf_max_depth", "rf_min_sample_split", "svm_tol"))
    fout_param_FI.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_FI[0], dnn_params_lri_FI[0], lr_params_solver_FI[0], lr_params_tol_FI[0], lr_params_C_FI[0], nb_params_vs_FI[0], rf_params_est_FI[0], rf_params_md_FI[0], rf_params_mss_FI[0], svm_params_tol_FI[0]))
    fout_param_FI.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_FI[1], dnn_params_lri_FI[1], lr_params_solver_FI[1], lr_params_tol_FI[1], lr_params_C_FI[1], nb_params_vs_FI[1], rf_params_est_FI[1], rf_params_md_FI[1], rf_params_mss_FI[1], svm_params_tol_FI[1]))
    fout_param_FI.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_FI[2], dnn_params_lri_FI[2], lr_params_solver_FI[2], lr_params_tol_FI[2], lr_params_C_FI[2], nb_params_vs_FI[2], rf_params_est_FI[2], rf_params_md_FI[2], rf_params_mss_FI[2], svm_params_tol_FI[2]))
    fout_param_FI.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_FI[3], dnn_params_lri_FI[3], lr_params_solver_FI[3], lr_params_tol_FI[3], lr_params_C_FI[3], nb_params_vs_FI[3], rf_params_est_FI[3], rf_params_md_FI[3], rf_params_mss_FI[3], svm_params_tol_FI[3]))
    fout_param_FI.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_FI[4], dnn_params_lri_FI[4], lr_params_solver_FI[4], lr_params_tol_FI[4], lr_params_C_FI[4], nb_params_vs_FI[4], rf_params_est_FI[4], rf_params_md_FI[4], rf_params_mss_FI[4], svm_params_tol_FI[4]))
with open(outfile_param_FG, 'w') as fout_param_FG:
    fout_param_FG.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % ("dnn_hidden_layer_sizes", "dnn_learning_rate_init", "lr_solver", "lr_tol", "lr_C", "nb_var_smoothing", "rf_n_estimator", "rf_max_depth", "rf_min_sample_split", "svm_tol"))
    fout_param_FG.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_FG[0], dnn_params_lri_FG[0], lr_params_solver_FG[0], lr_params_tol_FG[0], lr_params_C_FG[0], nb_params_vs_FG[0], rf_params_est_FG[0], rf_params_md_FG[0], rf_params_mss_FG[0], svm_params_tol_FG[0]))
    fout_param_FG.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_FG[1], dnn_params_lri_FG[1], lr_params_solver_FG[1], lr_params_tol_FG[1], lr_params_C_FG[1], nb_params_vs_FG[1], rf_params_est_FG[1], rf_params_md_FG[1], rf_params_mss_FG[1], svm_params_tol_FG[1]))
    fout_param_FG.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_FG[2], dnn_params_lri_FG[2], lr_params_solver_FG[2], lr_params_tol_FG[2], lr_params_C_FG[2], nb_params_vs_FG[2], rf_params_est_FG[2], rf_params_md_FG[2], rf_params_mss_FG[2], svm_params_tol_FG[2]))
    fout_param_FG.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_FG[3], dnn_params_lri_FG[3], lr_params_solver_FG[3], lr_params_tol_FG[3], lr_params_C_FG[3], nb_params_vs_FG[3], rf_params_est_FG[3], rf_params_md_FG[3], rf_params_mss_FG[3], svm_params_tol_FG[3]))
    fout_param_FG.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_FG[4], dnn_params_lri_FG[4], lr_params_solver_FG[4], lr_params_tol_FG[4], lr_params_C_FG[4], nb_params_vs_FG[4], rf_params_est_FG[4], rf_params_md_FG[4], rf_params_mss_FG[4], svm_params_tol_FG[4]))
with open(outfile_param_GR27, 'w') as fout_param_GR27:
    fout_param_GR27.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % ("dnn_hidden_layer_sizes", "dnn_learning_rate_init", "lr_solver", "lr_tol", "lr_C", "nb_var_smoothing", "rf_n_estimator", "rf_max_depth", "rf_min_sample_split", "svm_tol"))
    fout_param_GR27.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_GR27[0], dnn_params_lri_GR27[0], lr_params_solver_GR27[0], lr_params_tol_GR27[0], lr_params_C_GR27[0], nb_params_vs_GR27[0], rf_params_est_GR27[0], rf_params_md_GR27[0], rf_params_mss_GR27[0], svm_params_tol_GR27[0]))
    fout_param_GR27.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_GR27[1], dnn_params_lri_GR27[1], lr_params_solver_GR27[1], lr_params_tol_GR27[1], lr_params_C_GR27[1], nb_params_vs_GR27[1], rf_params_est_GR27[1], rf_params_md_GR27[1], rf_params_mss_GR27[1], svm_params_tol_GR27[1]))
    fout_param_GR27.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_GR27[2], dnn_params_lri_GR27[2], lr_params_solver_GR27[2], lr_params_tol_GR27[2], lr_params_C_GR27[2], nb_params_vs_GR27[2], rf_params_est_GR27[2], rf_params_md_GR27[2], rf_params_mss_GR27[2], svm_params_tol_GR27[2]))
    fout_param_GR27.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_GR27[3], dnn_params_lri_GR27[3], lr_params_solver_GR27[3], lr_params_tol_GR27[3], lr_params_C_GR27[3], nb_params_vs_GR27[3], rf_params_est_GR27[3], rf_params_md_GR27[3], rf_params_mss_GR27[3], svm_params_tol_GR27[3]))
    fout_param_GR27.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_GR27[4], dnn_params_lri_GR27[4], lr_params_solver_GR27[4], lr_params_tol_GR27[4], lr_params_C_GR27[4], nb_params_vs_GR27[4], rf_params_est_GR27[4], rf_params_md_GR27[4], rf_params_mss_GR27[4], svm_params_tol_GR27[4]))
with open(outfile_param_LM, 'w') as fout_param_LM:
    fout_param_LM.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % ("dnn_hidden_layer_sizes", "dnn_learning_rate_init", "lr_solver", "lr_tol", "lr_C", "nb_var_smoothing", "rf_n_estimator", "rf_max_depth", "rf_min_sample_split", "svm_tol"))
    fout_param_LM.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_LM[0], dnn_params_lri_LM[0], lr_params_solver_LM[0], lr_params_tol_LM[0], lr_params_C_LM[0], nb_params_vs_LM[0], rf_params_est_LM[0], rf_params_md_LM[0], rf_params_mss_LM[0], svm_params_tol_LM[0]))
    fout_param_LM.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_LM[1], dnn_params_lri_LM[1], lr_params_solver_LM[1], lr_params_tol_LM[1], lr_params_C_LM[1], nb_params_vs_LM[1], rf_params_est_LM[1], rf_params_md_LM[1], rf_params_mss_LM[1], svm_params_tol_LM[1]))
    fout_param_LM.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_LM[2], dnn_params_lri_LM[2], lr_params_solver_LM[2], lr_params_tol_LM[2], lr_params_C_LM[2], nb_params_vs_LM[2], rf_params_est_LM[2], rf_params_md_LM[2], rf_params_mss_LM[2], svm_params_tol_LM[2]))
    fout_param_LM.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_LM[3], dnn_params_lri_LM[3], lr_params_solver_LM[3], lr_params_tol_LM[3], lr_params_C_LM[3], nb_params_vs_LM[3], rf_params_est_LM[3], rf_params_md_LM[3], rf_params_mss_LM[3], svm_params_tol_LM[3]))
    fout_param_LM.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_LM[4], dnn_params_lri_LM[4], lr_params_solver_LM[4], lr_params_tol_LM[4], lr_params_C_LM[4], nb_params_vs_LM[4], rf_params_est_LM[4], rf_params_md_LM[4], rf_params_mss_LM[4], svm_params_tol_LM[4]))
with open(outfile_param_LMM, 'w') as fout_param_LMM:
    fout_param_LMM.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % ("dnn_hidden_layer_sizes", "dnn_learning_rate_init", "lr_solver", "lr_tol", "lr_C", "nb_var_smoothing", "rf_n_estimator", "rf_max_depth", "rf_min_sample_split", "svm_tol"))
    fout_param_LMM.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_LMM[0], dnn_params_lri_LMM[0], lr_params_solver_LMM[0], lr_params_tol_LMM[0], lr_params_C_LMM[0], nb_params_vs_LMM[0], rf_params_est_LMM[0], rf_params_md_LMM[0], rf_params_mss_LMM[0], svm_params_tol_LMM[0]))
    fout_param_LMM.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_LMM[1], dnn_params_lri_LMM[1], lr_params_solver_LMM[1], lr_params_tol_LMM[1], lr_params_C_LMM[1], nb_params_vs_LMM[1], rf_params_est_LMM[1], rf_params_md_LMM[1], rf_params_mss_LMM[1], svm_params_tol_LMM[1]))
    fout_param_LMM.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_LMM[2], dnn_params_lri_LMM[2], lr_params_solver_LMM[2], lr_params_tol_LMM[2], lr_params_C_LMM[2], nb_params_vs_LMM[2], rf_params_est_LMM[2], rf_params_md_LMM[2], rf_params_mss_LMM[2], svm_params_tol_LMM[2]))
    fout_param_LMM.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_LMM[3], dnn_params_lri_LMM[3], lr_params_solver_LMM[3], lr_params_tol_LMM[3], lr_params_C_LMM[3], nb_params_vs_LMM[3], rf_params_est_LMM[3], rf_params_md_LMM[3], rf_params_mss_LMM[3], svm_params_tol_LMM[3]))
    fout_param_LMM.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_LMM[4], dnn_params_lri_LMM[4], lr_params_solver_LMM[4], lr_params_tol_LMM[4], lr_params_C_LMM[4], nb_params_vs_LMM[4], rf_params_est_LMM[4], rf_params_md_LMM[4], rf_params_mss_LMM[4], svm_params_tol_LMM[4]))
with open(outfile_param_PC, 'w') as fout_param_PC:
    fout_param_PC.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % ("dnn_hidden_layer_sizes", "dnn_learning_rate_init", "lr_solver", "lr_tol", "lr_C", "nb_var_smoothing", "rf_n_estimator", "rf_max_depth", "rf_min_sample_split", "svm_tol"))
    fout_param_PC.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_PC[0], dnn_params_lri_PC[0], lr_params_solver_PC[0], lr_params_tol_PC[0], lr_params_C_PC[0], nb_params_vs_PC[0], rf_params_est_PC[0], rf_params_md_PC[0], rf_params_mss_PC[0], svm_params_tol_PC[0]))
    fout_param_PC.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_PC[1], dnn_params_lri_PC[1], lr_params_solver_PC[1], lr_params_tol_PC[1], lr_params_C_PC[1], nb_params_vs_PC[1], rf_params_est_PC[1], rf_params_md_PC[1], rf_params_mss_PC[1], svm_params_tol_PC[1]))
    fout_param_PC.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_PC[2], dnn_params_lri_PC[2], lr_params_solver_PC[2], lr_params_tol_PC[2], lr_params_C_PC[2], nb_params_vs_PC[2], rf_params_est_PC[2], rf_params_md_PC[2], rf_params_mss_PC[2], svm_params_tol_PC[2]))
    fout_param_PC.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_PC[3], dnn_params_lri_PC[3], lr_params_solver_PC[3], lr_params_tol_PC[3], lr_params_C_PC[3], nb_params_vs_PC[3], rf_params_est_PC[3], rf_params_md_PC[3], rf_params_mss_PC[3], svm_params_tol_PC[3]))
    fout_param_PC.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_PC[4], dnn_params_lri_PC[4], lr_params_solver_PC[4], lr_params_tol_PC[4], lr_params_C_PC[4], nb_params_vs_PC[4], rf_params_est_PC[4], rf_params_md_PC[4], rf_params_mss_PC[4], svm_params_tol_PC[4]))
with open(outfile_param_RG12, 'w') as fout_param_RG12:
    fout_param_RG12.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % ("dnn_hidden_layer_sizes", "dnn_learning_rate_init", "lr_solver", "lr_tol", "lr_C", "nb_var_smoothing", "rf_n_estimator", "rf_max_depth", "rf_min_sample_split", "svm_tol"))
    fout_param_RG12.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_RG12[0], dnn_params_lri_RG12[0], lr_params_solver_RG12[0], lr_params_tol_RG12[0], lr_params_C_RG12[0], nb_params_vs_RG12[0], rf_params_est_RG12[0], rf_params_md_RG12[0], rf_params_mss_RG12[0], svm_params_tol_RG12[0]))
    fout_param_RG12.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_RG12[1], dnn_params_lri_RG12[1], lr_params_solver_RG12[1], lr_params_tol_RG12[1], lr_params_C_RG12[1], nb_params_vs_RG12[1], rf_params_est_RG12[1], rf_params_md_RG12[1], rf_params_mss_RG12[1], svm_params_tol_RG12[1]))
    fout_param_RG12.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_RG12[2], dnn_params_lri_RG12[2], lr_params_solver_RG12[2], lr_params_tol_RG12[2], lr_params_C_RG12[2], nb_params_vs_RG12[2], rf_params_est_RG12[2], rf_params_md_RG12[2], rf_params_mss_RG12[2], svm_params_tol_RG12[2]))
    fout_param_RG12.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_RG12[3], dnn_params_lri_RG12[3], lr_params_solver_RG12[3], lr_params_tol_RG12[3], lr_params_C_RG12[3], nb_params_vs_RG12[3], rf_params_est_RG12[3], rf_params_md_RG12[3], rf_params_mss_RG12[3], svm_params_tol_RG12[3]))
    fout_param_RG12.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_RG12[4], dnn_params_lri_RG12[4], lr_params_solver_RG12[4], lr_params_tol_RG12[4], lr_params_C_RG12[4], nb_params_vs_RG12[4], rf_params_est_RG12[4], rf_params_md_RG12[4], rf_params_mss_RG12[4], svm_params_tol_RG12[4]))
with open(outfile_param_RG2, 'w') as fout_param_RG2:
    fout_param_RG2.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % ("dnn_hidden_layer_sizes", "dnn_learning_rate_init", "lr_solver", "lr_tol", "lr_C", "nb_var_smoothing", "rf_n_estimator", "rf_max_depth", "rf_min_sample_split", "svm_tol"))
    fout_param_RG2.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_RG2[0], dnn_params_lri_RG2[0], lr_params_solver_RG2[0], lr_params_tol_RG2[0], lr_params_C_RG2[0], nb_params_vs_RG2[0], rf_params_est_RG2[0], rf_params_md_RG2[0], rf_params_mss_RG2[0], svm_params_tol_RG2[0]))
    fout_param_RG2.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_RG2[1], dnn_params_lri_RG2[1], lr_params_solver_RG2[1], lr_params_tol_RG2[1], lr_params_C_RG2[1], nb_params_vs_RG2[1], rf_params_est_RG2[1], rf_params_md_RG2[1], rf_params_mss_RG2[1], svm_params_tol_RG2[1]))
    fout_param_RG2.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_RG2[2], dnn_params_lri_RG2[2], lr_params_solver_RG2[2], lr_params_tol_RG2[2], lr_params_C_RG2[2], nb_params_vs_RG2[2], rf_params_est_RG2[2], rf_params_md_RG2[2], rf_params_mss_RG2[2], svm_params_tol_RG2[2]))
    fout_param_RG2.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_RG2[3], dnn_params_lri_RG2[3], lr_params_solver_RG2[3], lr_params_tol_RG2[3], lr_params_C_RG2[3], nb_params_vs_RG2[3], rf_params_est_RG2[3], rf_params_md_RG2[3], rf_params_mss_RG2[3], svm_params_tol_RG2[3]))
    fout_param_RG2.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_RG2[4], dnn_params_lri_RG2[4], lr_params_solver_RG2[4], lr_params_tol_RG2[4], lr_params_C_RG2[4], nb_params_vs_RG2[4], rf_params_est_RG2[4], rf_params_md_RG2[4], rf_params_mss_RG2[4], svm_params_tol_RG2[4]))
with open(outfile_param_RG3, 'w') as fout_param_RG3:
    fout_param_RG3.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % ("dnn_hidden_layer_sizes", "dnn_learning_rate_init", "lr_solver", "lr_tol", "lr_C", "nb_var_smoothing", "rf_n_estimator", "rf_max_depth", "rf_min_sample_split", "svm_tol"))
    fout_param_RG3.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_RG3[0], dnn_params_lri_RG3[0], lr_params_solver_RG3[0], lr_params_tol_RG3[0], lr_params_C_RG3[0], nb_params_vs_RG3[0], rf_params_est_RG3[0], rf_params_md_RG3[0], rf_params_mss_RG3[0], svm_params_tol_RG3[0]))
    fout_param_RG3.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_RG3[1], dnn_params_lri_RG3[1], lr_params_solver_RG3[1], lr_params_tol_RG3[1], lr_params_C_RG3[1], nb_params_vs_RG3[1], rf_params_est_RG3[1], rf_params_md_RG3[1], rf_params_mss_RG3[1], svm_params_tol_RG3[1]))
    fout_param_RG3.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_RG3[2], dnn_params_lri_RG3[2], lr_params_solver_RG3[2], lr_params_tol_RG3[2], lr_params_C_RG3[2], nb_params_vs_RG3[2], rf_params_est_RG3[2], rf_params_md_RG3[2], rf_params_mss_RG3[2], svm_params_tol_RG3[2]))
    fout_param_RG3.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_RG3[3], dnn_params_lri_RG3[3], lr_params_solver_RG3[3], lr_params_tol_RG3[3], lr_params_C_RG3[3], nb_params_vs_RG3[3], rf_params_est_RG3[3], rf_params_md_RG3[3], rf_params_mss_RG3[3], svm_params_tol_RG3[3]))
    fout_param_RG3.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_RG3[4], dnn_params_lri_RG3[4], lr_params_solver_RG3[4], lr_params_tol_RG3[4], lr_params_C_RG3[4], nb_params_vs_RG3[4], rf_params_est_RG3[4], rf_params_md_RG3[4], rf_params_mss_RG3[4], svm_params_tol_RG3[4]))
with open(outfile_param_RGM, 'w') as fout_param_RGM:
    fout_param_RGM.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % ("dnn_hidden_layer_sizes", "dnn_learning_rate_init", "lr_solver", "lr_tol", "lr_C", "nb_var_smoothing", "rf_n_estimator", "rf_max_depth", "rf_min_sample_split", "svm_tol"))
    fout_param_RGM.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_RGM[0], dnn_params_lri_RGM[0], lr_params_solver_RGM[0], lr_params_tol_RGM[0], lr_params_C_RGM[0], nb_params_vs_RGM[0], rf_params_est_RGM[0], rf_params_md_RGM[0], rf_params_mss_RGM[0], svm_params_tol_RGM[0]))
    fout_param_RGM.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_RGM[1], dnn_params_lri_RGM[1], lr_params_solver_RGM[1], lr_params_tol_RGM[1], lr_params_C_RGM[1], nb_params_vs_RGM[1], rf_params_est_RGM[1], rf_params_md_RGM[1], rf_params_mss_RGM[1], svm_params_tol_RGM[1]))
    fout_param_RGM.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_RGM[2], dnn_params_lri_RGM[2], lr_params_solver_RGM[2], lr_params_tol_RGM[2], lr_params_C_RGM[2], nb_params_vs_RGM[2], rf_params_est_RGM[2], rf_params_md_RGM[2], rf_params_mss_RGM[2], svm_params_tol_RGM[2]))
    fout_param_RGM.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_RGM[3], dnn_params_lri_RGM[3], lr_params_solver_RGM[3], lr_params_tol_RGM[3], lr_params_C_RGM[3], nb_params_vs_RGM[3], rf_params_est_RGM[3], rf_params_md_RGM[3], rf_params_mss_RGM[3], svm_params_tol_RGM[3]))
    fout_param_RGM.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_RGM[4], dnn_params_lri_RGM[4], lr_params_solver_RGM[4], lr_params_tol_RGM[4], lr_params_C_RGM[4], nb_params_vs_RGM[4], rf_params_est_RGM[4], rf_params_md_RGM[4], rf_params_mss_RGM[4], svm_params_tol_RGM[4]))
with open(outfile_param_RGQC, 'w') as fout_param_RGQC:
    fout_param_RGQC.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % ("dnn_hidden_layer_sizes", "dnn_learning_rate_init", "lr_solver", "lr_tol", "lr_C", "nb_var_smoothing", "rf_n_estimator", "rf_max_depth", "rf_min_sample_split", "svm_tol"))
    fout_param_RGQC.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_RGQC[0], dnn_params_lri_RGQC[0], lr_params_solver_RGQC[0], lr_params_tol_RGQC[0], lr_params_C_RGQC[0], nb_params_vs_RGQC[0], rf_params_est_RGQC[0], rf_params_md_RGQC[0], rf_params_mss_RGQC[0], svm_params_tol_RGQC[0]))
    fout_param_RGQC.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_RGQC[1], dnn_params_lri_RGQC[1], lr_params_solver_RGQC[1], lr_params_tol_RGQC[1], lr_params_C_RGQC[1], nb_params_vs_RGQC[1], rf_params_est_RGQC[1], rf_params_md_RGQC[1], rf_params_mss_RGQC[1], svm_params_tol_RGQC[1]))
    fout_param_RGQC.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_RGQC[2], dnn_params_lri_RGQC[2], lr_params_solver_RGQC[2], lr_params_tol_RGQC[2], lr_params_C_RGQC[2], nb_params_vs_RGQC[2], rf_params_est_RGQC[2], rf_params_md_RGQC[2], rf_params_mss_RGQC[2], svm_params_tol_RGQC[2]))
    fout_param_RGQC.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_RGQC[3], dnn_params_lri_RGQC[3], lr_params_solver_RGQC[3], lr_params_tol_RGQC[3], lr_params_C_RGQC[3], nb_params_vs_RGQC[3], rf_params_est_RGQC[3], rf_params_md_RGQC[3], rf_params_mss_RGQC[3], svm_params_tol_RGQC[3]))
    fout_param_RGQC.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_RGQC[4], dnn_params_lri_RGQC[4], lr_params_solver_RGQC[4], lr_params_tol_RGQC[4], lr_params_C_RGQC[4], nb_params_vs_RGQC[4], rf_params_est_RGQC[4], rf_params_md_RGQC[4], rf_params_mss_RGQC[4], svm_params_tol_RGQC[4]))
with open(outfile_param_T8, 'w') as fout_param_T8:
    fout_param_T8.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % ("dnn_hidden_layer_sizes", "dnn_learning_rate_init", "lr_solver", "lr_tol", "lr_C", "nb_var_smoothing", "rf_n_estimator", "rf_max_depth", "rf_min_sample_split", "svm_tol"))
    fout_param_T8.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_T8[0], dnn_params_lri_T8[0], lr_params_solver_T8[0], lr_params_tol_T8[0], lr_params_C_T8[0], nb_params_vs_T8[0], rf_params_est_T8[0], rf_params_md_T8[0], rf_params_mss_T8[0], svm_params_tol_T8[0]))
    fout_param_T8.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_T8[1], dnn_params_lri_T8[1], lr_params_solver_T8[1], lr_params_tol_T8[1], lr_params_C_T8[1], nb_params_vs_T8[1], rf_params_est_T8[1], rf_params_md_T8[1], rf_params_mss_T8[1], svm_params_tol_T8[1]))
    fout_param_T8.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_T8[2], dnn_params_lri_T8[2], lr_params_solver_T8[2], lr_params_tol_T8[2], lr_params_C_T8[2], nb_params_vs_T8[2], rf_params_est_T8[2], rf_params_md_T8[2], rf_params_mss_T8[2], svm_params_tol_T8[2]))
    fout_param_T8.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_T8[3], dnn_params_lri_T8[3], lr_params_solver_T8[3], lr_params_tol_T8[3], lr_params_C_T8[3], nb_params_vs_T8[3], rf_params_est_T8[3], rf_params_md_T8[3], rf_params_mss_T8[3], svm_params_tol_T8[3]))
    fout_param_T8.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_T8[4], dnn_params_lri_T8[4], lr_params_solver_T8[4], lr_params_tol_T8[4], lr_params_C_T8[4], nb_params_vs_T8[4], rf_params_est_T8[4], rf_params_md_T8[4], rf_params_mss_T8[4], svm_params_tol_T8[4]))
with open(outfile_param_TM10, 'w') as fout_param_TM10:
    fout_param_TM10.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % ("dnn_hidden_layer_sizes", "dnn_learning_rate_init", "lr_solver", "lr_tol", "lr_C", "nb_var_smoothing", "rf_n_estimator", "rf_max_depth", "rf_min_sample_split", "svm_tol"))
    fout_param_TM10.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_TM10[0], dnn_params_lri_TM10[0], lr_params_solver_TM10[0], lr_params_tol_TM10[0], lr_params_C_TM10[0], nb_params_vs_TM10[0], rf_params_est_TM10[0], rf_params_md_TM10[0], rf_params_mss_TM10[0], svm_params_tol_TM10[0]))
    fout_param_TM10.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_TM10[1], dnn_params_lri_TM10[1], lr_params_solver_TM10[1], lr_params_tol_TM10[1], lr_params_C_TM10[1], nb_params_vs_TM10[1], rf_params_est_TM10[1], rf_params_md_TM10[1], rf_params_mss_TM10[1], svm_params_tol_TM10[1]))
    fout_param_TM10.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_TM10[2], dnn_params_lri_TM10[2], lr_params_solver_TM10[2], lr_params_tol_TM10[2], lr_params_C_TM10[2], nb_params_vs_TM10[2], rf_params_est_TM10[2], rf_params_md_TM10[2], rf_params_mss_TM10[2], svm_params_tol_TM10[2]))
    fout_param_TM10.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_TM10[3], dnn_params_lri_TM10[3], lr_params_solver_TM10[3], lr_params_tol_TM10[3], lr_params_C_TM10[3], nb_params_vs_TM10[3], rf_params_est_TM10[3], rf_params_md_TM10[3], rf_params_mss_TM10[3], svm_params_tol_TM10[3]))
    fout_param_TM10.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_TM10[4], dnn_params_lri_TM10[4], lr_params_solver_TM10[4], lr_params_tol_TM10[4], lr_params_C_TM10[4], nb_params_vs_TM10[4], rf_params_est_TM10[4], rf_params_md_TM10[4], rf_params_mss_TM10[4], svm_params_tol_TM10[4]))
with open(outfile_param_TM4, 'w') as fout_param_TM4:
    fout_param_TM4.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % ("dnn_hidden_layer_sizes", "dnn_learning_rate_init", "lr_solver", "lr_tol", "lr_C", "nb_var_smoothing", "rf_n_estimator", "rf_max_depth", "rf_min_sample_split", "svm_tol"))
    fout_param_TM4.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_TM4[0], dnn_params_lri_TM4[0], lr_params_solver_TM4[0], lr_params_tol_TM4[0], lr_params_C_TM4[0], nb_params_vs_TM4[0], rf_params_est_TM4[0], rf_params_md_TM4[0], rf_params_mss_TM4[0], svm_params_tol_TM4[0]))
    fout_param_TM4.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_TM4[1], dnn_params_lri_TM4[1], lr_params_solver_TM4[1], lr_params_tol_TM4[1], lr_params_C_TM4[1], nb_params_vs_TM4[1], rf_params_est_TM4[1], rf_params_md_TM4[1], rf_params_mss_TM4[1], svm_params_tol_TM4[1]))
    fout_param_TM4.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_TM4[2], dnn_params_lri_TM4[2], lr_params_solver_TM4[2], lr_params_tol_TM4[2], lr_params_C_TM4[2], nb_params_vs_TM4[2], rf_params_est_TM4[2], rf_params_md_TM4[2], rf_params_mss_TM4[2], svm_params_tol_TM4[2]))
    fout_param_TM4.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_TM4[3], dnn_params_lri_TM4[3], lr_params_solver_TM4[3], lr_params_tol_TM4[3], lr_params_C_TM4[3], nb_params_vs_TM4[3], rf_params_est_TM4[3], rf_params_md_TM4[3], rf_params_mss_TM4[3], svm_params_tol_TM4[3]))
    fout_param_TM4.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_TM4[4], dnn_params_lri_TM4[4], lr_params_solver_TM4[4], lr_params_tol_TM4[4], lr_params_C_TM4[4], nb_params_vs_TM4[4], rf_params_est_TM4[4], rf_params_md_TM4[4], rf_params_mss_TM4[4], svm_params_tol_TM4[4]))
with open(outfile_param_TM5, 'w') as fout_param_TM5:
    fout_param_TM5.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % ("dnn_hidden_layer_sizes", "dnn_learning_rate_init", "lr_solver", "lr_tol", "lr_C", "nb_var_smoothing", "rf_n_estimator", "rf_max_depth", "rf_min_sample_split", "svm_tol"))
    fout_param_TM5.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_TM5[0], dnn_params_lri_TM5[0], lr_params_solver_TM5[0], lr_params_tol_TM5[0], lr_params_C_TM5[0], nb_params_vs_TM5[0], rf_params_est_TM5[0], rf_params_md_TM5[0], rf_params_mss_TM5[0], svm_params_tol_TM5[0]))
    fout_param_TM5.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_TM5[1], dnn_params_lri_TM5[1], lr_params_solver_TM5[1], lr_params_tol_TM5[1], lr_params_C_TM5[1], nb_params_vs_TM5[1], rf_params_est_TM5[1], rf_params_md_TM5[1], rf_params_mss_TM5[1], svm_params_tol_TM5[1]))
    fout_param_TM5.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_TM5[2], dnn_params_lri_TM5[2], lr_params_solver_TM5[2], lr_params_tol_TM5[2], lr_params_C_TM5[2], nb_params_vs_TM5[2], rf_params_est_TM5[2], rf_params_md_TM5[2], rf_params_mss_TM5[2], svm_params_tol_TM5[2]))
    fout_param_TM5.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_TM5[3], dnn_params_lri_TM5[3], lr_params_solver_TM5[3], lr_params_tol_TM5[3], lr_params_C_TM5[3], nb_params_vs_TM5[3], rf_params_est_TM5[3], rf_params_md_TM5[3], rf_params_mss_TM5[3], svm_params_tol_TM5[3]))
    fout_param_TM5.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_TM5[4], dnn_params_lri_TM5[4], lr_params_solver_TM5[4], lr_params_tol_TM5[4], lr_params_C_TM5[4], nb_params_vs_TM5[4], rf_params_est_TM5[4], rf_params_md_TM5[4], rf_params_mss_TM5[4], svm_params_tol_TM5[4]))
with open(outfile_param_TM6, 'w') as fout_param_TM6:
    fout_param_TM6.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % ("dnn_hidden_layer_sizes", "dnn_learning_rate_init", "lr_solver", "lr_tol", "lr_C", "nb_var_smoothing", "rf_n_estimator", "rf_max_depth", "rf_min_sample_split", "svm_tol"))
    fout_param_TM6.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_TM6[0], dnn_params_lri_TM6[0], lr_params_solver_TM6[0], lr_params_tol_TM6[0], lr_params_C_TM6[0], nb_params_vs_TM6[0], rf_params_est_TM6[0], rf_params_md_TM6[0], rf_params_mss_TM6[0], svm_params_tol_TM6[0]))
    fout_param_TM6.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_TM6[1], dnn_params_lri_TM6[1], lr_params_solver_TM6[1], lr_params_tol_TM6[1], lr_params_C_TM6[1], nb_params_vs_TM6[1], rf_params_est_TM6[1], rf_params_md_TM6[1], rf_params_mss_TM6[1], svm_params_tol_TM6[1]))
    fout_param_TM6.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_TM6[2], dnn_params_lri_TM6[2], lr_params_solver_TM6[2], lr_params_tol_TM6[2], lr_params_C_TM6[2], nb_params_vs_TM6[2], rf_params_est_TM6[2], rf_params_md_TM6[2], rf_params_mss_TM6[2], svm_params_tol_TM6[2]))
    fout_param_TM6.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_TM6[3], dnn_params_lri_TM6[3], lr_params_solver_TM6[3], lr_params_tol_TM6[3], lr_params_C_TM6[3], nb_params_vs_TM6[3], rf_params_est_TM6[3], rf_params_md_TM6[3], rf_params_mss_TM6[3], svm_params_tol_TM6[3]))
    fout_param_TM6.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_TM6[4], dnn_params_lri_TM6[4], lr_params_solver_TM6[4], lr_params_tol_TM6[4], lr_params_C_TM6[4], nb_params_vs_TM6[4], rf_params_est_TM6[4], rf_params_md_TM6[4], rf_params_mss_TM6[4], svm_params_tol_TM6[4]))
with open(outfile_param_TM8, 'w') as fout_param_TM8:
    fout_param_TM8.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % ("dnn_hidden_layer_sizes", "dnn_learning_rate_init", "lr_solver", "lr_tol", "lr_C", "nb_var_smoothing", "rf_n_estimator", "rf_max_depth", "rf_min_sample_split", "svm_tol"))
    fout_param_TM8.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_TM8[0], dnn_params_lri_TM8[0], lr_params_solver_TM8[0], lr_params_tol_TM8[0], lr_params_C_TM8[0], nb_params_vs_TM8[0], rf_params_est_TM8[0], rf_params_md_TM8[0], rf_params_mss_TM8[0], svm_params_tol_TM8[0]))
    fout_param_TM8.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_TM8[1], dnn_params_lri_TM8[1], lr_params_solver_TM8[1], lr_params_tol_TM8[1], lr_params_C_TM8[1], nb_params_vs_TM8[1], rf_params_est_TM8[1], rf_params_md_TM8[1], rf_params_mss_TM8[1], svm_params_tol_TM8[1]))
    fout_param_TM8.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_TM8[2], dnn_params_lri_TM8[2], lr_params_solver_TM8[2], lr_params_tol_TM8[2], lr_params_C_TM8[2], nb_params_vs_TM8[2], rf_params_est_TM8[2], rf_params_md_TM8[2], rf_params_mss_TM8[2], svm_params_tol_TM8[2]))
    fout_param_TM8.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_TM8[3], dnn_params_lri_TM8[3], lr_params_solver_TM8[3], lr_params_tol_TM8[3], lr_params_C_TM8[3], nb_params_vs_TM8[3], rf_params_est_TM8[3], rf_params_md_TM8[3], rf_params_mss_TM8[3], svm_params_tol_TM8[3]))
    fout_param_TM8.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_TM8[4], dnn_params_lri_TM8[4], lr_params_solver_TM8[4], lr_params_tol_TM8[4], lr_params_C_TM8[4], nb_params_vs_TM8[4], rf_params_est_TM8[4], rf_params_md_TM8[4], rf_params_mss_TM8[4], svm_params_tol_TM8[4]))
with open(outfile_param_TM9, 'w') as fout_param_TM9:
    fout_param_TM9.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % ("dnn_hidden_layer_sizes", "dnn_learning_rate_init", "lr_solver", "lr_tol", "lr_C", "nb_var_smoothing", "rf_n_estimator", "rf_max_depth", "rf_min_sample_split", "svm_tol"))
    fout_param_TM9.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_TM9[0], dnn_params_lri_TM9[0], lr_params_solver_TM9[0], lr_params_tol_TM9[0], lr_params_C_TM9[0], nb_params_vs_TM9[0], rf_params_est_TM9[0], rf_params_md_TM9[0], rf_params_mss_TM9[0], svm_params_tol_TM9[0]))
    fout_param_TM9.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_TM9[1], dnn_params_lri_TM9[1], lr_params_solver_TM9[1], lr_params_tol_TM9[1], lr_params_C_TM9[1], nb_params_vs_TM9[1], rf_params_est_TM9[1], rf_params_md_TM9[1], rf_params_mss_TM9[1], svm_params_tol_TM9[1]))
    fout_param_TM9.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_TM9[2], dnn_params_lri_TM9[2], lr_params_solver_TM9[2], lr_params_tol_TM9[2], lr_params_C_TM9[2], nb_params_vs_TM9[2], rf_params_est_TM9[2], rf_params_md_TM9[2], rf_params_mss_TM9[2], svm_params_tol_TM9[2]))
    fout_param_TM9.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_TM9[3], dnn_params_lri_TM9[3], lr_params_solver_TM9[3], lr_params_tol_TM9[3], lr_params_C_TM9[3], nb_params_vs_TM9[3], rf_params_est_TM9[3], rf_params_md_TM9[3], rf_params_mss_TM9[3], svm_params_tol_TM9[3]))
    fout_param_TM9.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_TM9[4], dnn_params_lri_TM9[4], lr_params_solver_TM9[4], lr_params_tol_TM9[4], lr_params_C_TM9[4], nb_params_vs_TM9[4], rf_params_est_TM9[4], rf_params_md_TM9[4], rf_params_mss_TM9[4], svm_params_tol_TM9[4]))
with open(outfile_param_TMQC, 'w') as fout_param_TMQC:
    fout_param_TMQC.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % ("dnn_hidden_layer_sizes", "dnn_learning_rate_init", "lr_solver", "lr_tol", "lr_C", "nb_var_smoothing", "rf_n_estimator", "rf_max_depth", "rf_min_sample_split", "svm_tol"))
    fout_param_TMQC.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_TMQC[0], dnn_params_lri_TMQC[0], lr_params_solver_TMQC[0], lr_params_tol_TMQC[0], lr_params_C_TMQC[0], nb_params_vs_TMQC[0], rf_params_est_TMQC[0], rf_params_md_TMQC[0], rf_params_mss_TMQC[0], svm_params_tol_TMQC[0]))
    fout_param_TMQC.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_TMQC[1], dnn_params_lri_TMQC[1], lr_params_solver_TMQC[1], lr_params_tol_TMQC[1], lr_params_C_TMQC[1], nb_params_vs_TMQC[1], rf_params_est_TMQC[1], rf_params_md_TMQC[1], rf_params_mss_TMQC[1], svm_params_tol_TMQC[1]))
    fout_param_TMQC.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_TMQC[2], dnn_params_lri_TMQC[2], lr_params_solver_TMQC[2], lr_params_tol_TMQC[2], lr_params_C_TMQC[2], nb_params_vs_TMQC[2], rf_params_est_TMQC[2], rf_params_md_TMQC[2], rf_params_mss_TMQC[2], svm_params_tol_TMQC[2]))
    fout_param_TMQC.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_TMQC[3], dnn_params_lri_TMQC[3], lr_params_solver_TMQC[3], lr_params_tol_TMQC[3], lr_params_C_TMQC[3], nb_params_vs_TMQC[3], rf_params_est_TMQC[3], rf_params_md_TMQC[3], rf_params_mss_TMQC[3], svm_params_tol_TMQC[3]))
    fout_param_TMQC.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_TMQC[4], dnn_params_lri_TMQC[4], lr_params_solver_TMQC[4], lr_params_tol_TMQC[4], lr_params_C_TMQC[4], nb_params_vs_TMQC[4], rf_params_est_TMQC[4], rf_params_md_TMQC[4], rf_params_mss_TMQC[4], svm_params_tol_TMQC[4]))
with open(outfile_param_TQC, 'w') as fout_param_TQC:
    fout_param_TQC.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % ("dnn_hidden_layer_sizes", "dnn_learning_rate_init", "lr_solver", "lr_tol", "lr_C", "nb_var_smoothing", "rf_n_estimator", "rf_max_depth", "rf_min_sample_split", "svm_tol"))
    fout_param_TQC.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_TQC[0], dnn_params_lri_TQC[0], lr_params_solver_TQC[0], lr_params_tol_TQC[0], lr_params_C_TQC[0], nb_params_vs_TQC[0], rf_params_est_TQC[0], rf_params_md_TQC[0], rf_params_mss_TQC[0], svm_params_tol_TQC[0]))
    fout_param_TQC.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_TQC[1], dnn_params_lri_TQC[1], lr_params_solver_TQC[1], lr_params_tol_TQC[1], lr_params_C_TQC[1], nb_params_vs_TQC[1], rf_params_est_TQC[1], rf_params_md_TQC[1], rf_params_mss_TQC[1], svm_params_tol_TQC[1]))
    fout_param_TQC.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_TQC[2], dnn_params_lri_TQC[2], lr_params_solver_TQC[2], lr_params_tol_TQC[2], lr_params_C_TQC[2], nb_params_vs_TQC[2], rf_params_est_TQC[2], rf_params_md_TQC[2], rf_params_mss_TQC[2], svm_params_tol_TQC[2]))
    fout_param_TQC.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_TQC[3], dnn_params_lri_TQC[3], lr_params_solver_TQC[3], lr_params_tol_TQC[3], lr_params_C_TQC[3], nb_params_vs_TQC[3], rf_params_est_TQC[3], rf_params_md_TQC[3], rf_params_mss_TQC[3], svm_params_tol_TQC[3]))
    fout_param_TQC.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (dnn_params_hls_TQC[4], dnn_params_lri_TQC[4], lr_params_solver_TQC[4], lr_params_tol_TQC[4], lr_params_C_TQC[4], nb_params_vs_TQC[4], rf_params_est_TQC[4], rf_params_md_TQC[4], rf_params_mss_TQC[4], svm_params_tol_TQC[4]))
dnn_f1_score_pen_1_kfoldcv[5] = "Average" #To let users figure out that f1_score_kfoldcv[6] value is the average when seeing the csv file
dnn_f1_score_pen_1_kfoldcv[6] = (dnn_f1_score_pen_1_kfoldcv[0]+dnn_f1_score_pen_1_kfoldcv[1]+dnn_f1_score_pen_1_kfoldcv[2]+dnn_f1_score_pen_1_kfoldcv[3]+dnn_f1_score_pen_1_kfoldcv[4])/5
dnn_f1_score_pen_1_kfoldcv = pd.DataFrame(dnn_f1_score_pen_1_kfoldcv)
dnn_f1_score_pen_1_kfoldcv.to_csv('f1_score_'+imb_technique+'_penalty_1_dnn_production_' + str(nsplits) + 'foldcv_1~5.csv',header=False,index=False) #First repetition
dnn_f1_score_pen_5_kfoldcv[5] = "Average" #To let users figure out that f1_score_kfoldcv[6] value is the average when seeing the csv file
dnn_f1_score_pen_5_kfoldcv[6] = (dnn_f1_score_pen_5_kfoldcv[0]+dnn_f1_score_pen_5_kfoldcv[1]+dnn_f1_score_pen_5_kfoldcv[2]+dnn_f1_score_pen_5_kfoldcv[3]+dnn_f1_score_pen_5_kfoldcv[4])/5
dnn_f1_score_pen_5_kfoldcv = pd.DataFrame(dnn_f1_score_pen_5_kfoldcv)
dnn_f1_score_pen_5_kfoldcv.to_csv('f1_score_'+imb_technique+'_penalty_5_dnn_production_' + str(nsplits) + 'foldcv_1~5.csv',header=False,index=False) #First repetition
#dnn_f1_score_kfoldcv.to_csv('f1_score_'+imb_technique+'_penalty_' + str(penalty) + '_dnn_production_' + str(nsplits) + 'foldcv_6~10.csv',header=False,index=False) #Second repetition
dnn_ovr_accuracy_kfoldcv[5] = "Average" #To let users figure out that f1_score_kfoldcv[6] value is the average when seeing the csv file
dnn_ovr_accuracy_kfoldcv[6] = (dnn_ovr_accuracy_kfoldcv[0]+dnn_ovr_accuracy_kfoldcv[1]+dnn_ovr_accuracy_kfoldcv[2]+dnn_ovr_accuracy_kfoldcv[3]+dnn_ovr_accuracy_kfoldcv[4])/5
dnn_ovr_accuracy_kfoldcv = pd.DataFrame(dnn_ovr_accuracy_kfoldcv)
dnn_ovr_accuracy_kfoldcv.to_csv('ovr_accuracy_'+imb_technique + '_dnn_production_' + str(nsplits) + 'foldcv_1~5.csv',header=False,index=False) #First repetition
#dnn_ovr_accuracy_kfoldcv.to_csv('ovr_accuracy_'+imb_technique+'_penalty_' + str(penalty) + '_dnn_production_' + str(nsplits) + 'foldcv_6~10.csv',header=False,index=False) #Second repetition
dnn_auc_kfoldcv[5] = "Average" #To let users figure out that f1_score_kfoldcv[6] value is the average when seeing the csv file
dnn_auc_kfoldcv[6] = (dnn_auc_kfoldcv[0]+dnn_auc_kfoldcv[1]+dnn_auc_kfoldcv[2]+dnn_auc_kfoldcv[3]+dnn_auc_kfoldcv[4])/5
dnn_auc_kfoldcv = pd.DataFrame(dnn_auc_kfoldcv)
dnn_auc_kfoldcv.to_csv('auc_'+imb_technique+'_dnn_production_' + str(nsplits) + 'foldcv_1~5.csv',header=False,index=False)
dnn_gmean_kfoldcv[5] = "Average" #To let users figure out that f1_score_kfoldcv[6] value is the average when seeing the csv file
dnn_gmean_kfoldcv[6] = (dnn_gmean_kfoldcv[0]+dnn_gmean_kfoldcv[1]+dnn_gmean_kfoldcv[2]+dnn_gmean_kfoldcv[3]+dnn_gmean_kfoldcv[4])/5
dnn_gmean_kfoldcv = pd.DataFrame(dnn_gmean_kfoldcv)
dnn_gmean_kfoldcv.to_csv('gmean_'+imb_technique+'_dnn_production_' + str(nsplits) + 'foldcv_1~5.csv',header=False,index=False)

lr_f1_score_pen_1_kfoldcv[5] = "Average" #To let users figure out that f1_score_kfoldcv[6] value is the average when seeing the csv file
lr_f1_score_pen_1_kfoldcv[6] = (lr_f1_score_pen_1_kfoldcv[0]+lr_f1_score_pen_1_kfoldcv[1]+lr_f1_score_pen_1_kfoldcv[2]+lr_f1_score_pen_1_kfoldcv[3]+lr_f1_score_pen_1_kfoldcv[4])/5
lr_f1_score_pen_1_kfoldcv = pd.DataFrame(lr_f1_score_pen_1_kfoldcv)
lr_f1_score_pen_1_kfoldcv.to_csv('f1_score_'+imb_technique+'_penalty_1_lr_production_' + str(nsplits) + 'foldcv_1~5.csv',header=False,index=False) #First repetition
lr_f1_score_pen_5_kfoldcv[5] = "Average" #To let users figure out that f1_score_kfoldcv[6] value is the average when seeing the csv file
lr_f1_score_pen_5_kfoldcv[6] = (lr_f1_score_pen_5_kfoldcv[0]+lr_f1_score_pen_5_kfoldcv[1]+lr_f1_score_pen_5_kfoldcv[2]+lr_f1_score_pen_5_kfoldcv[3]+lr_f1_score_pen_5_kfoldcv[4])/5
lr_f1_score_pen_5_kfoldcv = pd.DataFrame(lr_f1_score_pen_5_kfoldcv)
lr_f1_score_pen_5_kfoldcv.to_csv('f1_score_'+imb_technique+'_penalty_5_lr_production_' + str(nsplits) + 'foldcv_1~5.csv',header=False,index=False) #First repetition
#lr_f1_score_kfoldcv.to_csv('f1_score_'+imb_technique+'_penalty_' + str(penalty) + '_lr_production_' + str(nsplits) + 'foldcv_6~10.csv',header=False,index=False) #Second repetition
lr_ovr_accuracy_kfoldcv[5] = "Average" #To let users figure out that f1_score_kfoldcv[6] value is the average when seeing the csv file
lr_ovr_accuracy_kfoldcv[6] = (lr_ovr_accuracy_kfoldcv[0]+lr_ovr_accuracy_kfoldcv[1]+lr_ovr_accuracy_kfoldcv[2]+lr_ovr_accuracy_kfoldcv[3]+lr_ovr_accuracy_kfoldcv[4])/5
lr_ovr_accuracy_kfoldcv = pd.DataFrame(lr_ovr_accuracy_kfoldcv)
lr_ovr_accuracy_kfoldcv.to_csv('ovr_accuracy_' + imb_technique + '_lr_production_' + str(nsplits) + 'foldcv_1~5.csv',header=False,index=False) #First repetition
#lr_ovr_accuracy_kfoldcv.to_csv('ovr_accuracy_'+imb_technique+'_penalty_' + str(penalty) + '_lr_production_' + str(nsplits) + 'foldcv_6~10.csv',header=False,index=False) #Second repetition
lr_auc_kfoldcv[5] = "Average" #To let users figure out that f1_score_kfoldcv[6] value is the average when seeing the csv file
lr_auc_kfoldcv[6] = (lr_auc_kfoldcv[0]+lr_auc_kfoldcv[1]+lr_auc_kfoldcv[2]+lr_auc_kfoldcv[3]+lr_auc_kfoldcv[4])/5
lr_auc_kfoldcv = pd.DataFrame(lr_auc_kfoldcv)
lr_auc_kfoldcv.to_csv('auc_'+imb_technique+'_lr_production_' + str(nsplits) + 'foldcv_1~5.csv',header=False,index=False)
lr_gmean_kfoldcv[5] = "Average" #To let users figure out that f1_score_kfoldcv[6] value is the average when seeing the csv file
lr_gmean_kfoldcv[6] = (lr_gmean_kfoldcv[0]+lr_gmean_kfoldcv[1]+lr_gmean_kfoldcv[2]+lr_gmean_kfoldcv[3]+lr_gmean_kfoldcv[4])/5
lr_gmean_kfoldcv = pd.DataFrame(lr_gmean_kfoldcv)
lr_gmean_kfoldcv.to_csv('gmean_'+imb_technique+'_lr_production_' + str(nsplits) + 'foldcv_1~5.csv',header=False,index=False)

nb_f1_score_pen_1_kfoldcv[5] = "Average" #To let users figure out that f1_score_kfoldcv[6] value is the average when seeing the csv file
nb_f1_score_pen_1_kfoldcv[6] = (nb_f1_score_pen_1_kfoldcv[0]+nb_f1_score_pen_1_kfoldcv[1]+nb_f1_score_pen_1_kfoldcv[2]+nb_f1_score_pen_1_kfoldcv[3]+nb_f1_score_pen_1_kfoldcv[4])/5
nb_f1_score_pen_1_kfoldcv = pd.DataFrame(nb_f1_score_pen_1_kfoldcv)
nb_f1_score_pen_1_kfoldcv.to_csv('f1_score_'+imb_technique+'_penalty_1_nb_production_' + str(nsplits) + 'foldcv_1~5.csv',header=False,index=False) #First repetition
nb_f1_score_pen_5_kfoldcv[5] = "Average" #To let users figure out that f1_score_kfoldcv[6] value is the average when seeing the csv file
nb_f1_score_pen_5_kfoldcv[6] = (nb_f1_score_pen_5_kfoldcv[0]+nb_f1_score_pen_5_kfoldcv[1]+nb_f1_score_pen_5_kfoldcv[2]+nb_f1_score_pen_5_kfoldcv[3]+nb_f1_score_pen_5_kfoldcv[4])/5
nb_f1_score_pen_5_kfoldcv = pd.DataFrame(nb_f1_score_pen_5_kfoldcv)
nb_f1_score_pen_5_kfoldcv.to_csv('f1_score_'+imb_technique+'_penalty_5_nb_production_' + str(nsplits) + 'foldcv_1~5.csv',header=False,index=False) #First repetition
#nb_f1_score_kfoldcv.to_csv('f1_score_'+imb_technique+'_penalty_' + str(penalty) + '_nb_production_' + str(nsplits) + 'foldcv_6~10.csv',header=False,index=False) #Second repetition
nb_ovr_accuracy_kfoldcv[5] = "Average" #To let users figure out that f1_score_kfoldcv[6] value is the average when seeing the csv file
nb_ovr_accuracy_kfoldcv[6] = (nb_ovr_accuracy_kfoldcv[0]+nb_ovr_accuracy_kfoldcv[1]+nb_ovr_accuracy_kfoldcv[2]+nb_ovr_accuracy_kfoldcv[3]+nb_ovr_accuracy_kfoldcv[4])/5
nb_ovr_accuracy_kfoldcv = pd.DataFrame(nb_ovr_accuracy_kfoldcv)
nb_ovr_accuracy_kfoldcv.to_csv('ovr_accuracy_' + imb_technique + '_nb_production_' + str(nsplits) + 'foldcv_1~5.csv',header=False,index=False) #First repetition
#nb_ovr_accuracy_kfoldcv.to_csv('ovr_accuracy_'+imb_technique+'_penalty_' + str(penalty) + '_nb_production_' + str(nsplits) + 'foldcv_6~10.csv',header=False,index=False) #Second repetition
nb_auc_kfoldcv[5] = "Average" #To let users figure out that f1_score_kfoldcv[6] value is the average when seeing the csv file
nb_auc_kfoldcv[6] = (nb_auc_kfoldcv[0]+nb_auc_kfoldcv[1]+nb_auc_kfoldcv[2]+nb_auc_kfoldcv[3]+nb_auc_kfoldcv[4])/5
nb_auc_kfoldcv = pd.DataFrame(nb_auc_kfoldcv)
nb_auc_kfoldcv.to_csv('auc_'+imb_technique+'_nb_production_' + str(nsplits) + 'foldcv_1~5.csv',header=False,index=False)
nb_gmean_kfoldcv[5] = "Average" #To let users figure out that f1_score_kfoldcv[6] value is the average when seeing the csv file
nb_gmean_kfoldcv[6] = (nb_gmean_kfoldcv[0]+nb_gmean_kfoldcv[1]+nb_gmean_kfoldcv[2]+nb_gmean_kfoldcv[3]+nb_gmean_kfoldcv[4])/5
nb_gmean_kfoldcv = pd.DataFrame(nb_gmean_kfoldcv)
nb_gmean_kfoldcv.to_csv('gmean_'+imb_technique+'_nb_production_' + str(nsplits) + 'foldcv_1~5.csv',header=False,index=False)

rf_f1_score_pen_1_kfoldcv[5] = "Average" #To let users figure out that f1_score_kfoldcv[6] value is the average when seeing the csv file
rf_f1_score_pen_1_kfoldcv[6] = (rf_f1_score_pen_1_kfoldcv[0]+rf_f1_score_pen_1_kfoldcv[1]+rf_f1_score_pen_1_kfoldcv[2]+rf_f1_score_pen_1_kfoldcv[3]+rf_f1_score_pen_1_kfoldcv[4])/5
rf_f1_score_pen_1_kfoldcv = pd.DataFrame(rf_f1_score_pen_1_kfoldcv)
rf_f1_score_pen_1_kfoldcv.to_csv('f1_score_'+imb_technique+'_penalty_1_rf_production_' + str(nsplits) + 'foldcv_1~5.csv',header=False,index=False) #First repetition
rf_f1_score_pen_5_kfoldcv[5] = "Average" #To let users figure out that f1_score_kfoldcv[6] value is the average when seeing the csv file
rf_f1_score_pen_5_kfoldcv[6] = (rf_f1_score_pen_5_kfoldcv[0]+rf_f1_score_pen_5_kfoldcv[1]+rf_f1_score_pen_5_kfoldcv[2]+rf_f1_score_pen_5_kfoldcv[3]+rf_f1_score_pen_5_kfoldcv[4])/5
rf_f1_score_pen_5_kfoldcv = pd.DataFrame(rf_f1_score_pen_5_kfoldcv)
rf_f1_score_pen_5_kfoldcv.to_csv('f1_score_'+imb_technique+'_penalty_5_rf_production_' + str(nsplits) + 'foldcv_1~5.csv',header=False,index=False) #First repetition
#rf_f1_score_kfoldcv.to_csv('f1_score_'+imb_technique+'_penalty_' + str(penalty) + '_rf_production_' + str(nsplits) + 'foldcv_6~10.csv',header=False,index=False) #Second repetition
rf_ovr_accuracy_kfoldcv[5] = "Average" #To let users figure out that f1_score_kfoldcv[6] value is the average when seeing the csv file
rf_ovr_accuracy_kfoldcv[6] = (rf_ovr_accuracy_kfoldcv[0]+rf_ovr_accuracy_kfoldcv[1]+rf_ovr_accuracy_kfoldcv[2]+rf_ovr_accuracy_kfoldcv[3]+rf_ovr_accuracy_kfoldcv[4])/5
rf_ovr_accuracy_kfoldcv = pd.DataFrame(rf_ovr_accuracy_kfoldcv)
rf_ovr_accuracy_kfoldcv.to_csv('ovr_accuracy_' + imb_technique + '_rf_production_' + str(nsplits) + 'foldcv_1~5.csv',header=False,index=False) #First repetition
#rf_ovr_accuracy_kfoldcv.to_csv('ovr_accuracy_'+imb_technique+'_penalty_' + str(penalty) + '_rf_production_' + str(nsplits) + 'foldcv_6~10.csv',header=False,index=False) #Second repetition
rf_auc_kfoldcv[5] = "Average" #To let users figure out that f1_score_kfoldcv[6] value is the average when seeing the csv file
rf_auc_kfoldcv[6] = (rf_auc_kfoldcv[0]+rf_auc_kfoldcv[1]+rf_auc_kfoldcv[2]+rf_auc_kfoldcv[3]+rf_auc_kfoldcv[4])/5
rf_auc_kfoldcv = pd.DataFrame(rf_auc_kfoldcv)
rf_auc_kfoldcv.to_csv('auc_'+imb_technique+'_rf_production_' + str(nsplits) + 'foldcv_1~5.csv',header=False,index=False)
rf_gmean_kfoldcv[5] = "Average" #To let users figure out that f1_score_kfoldcv[6] value is the average when seeing the csv file
rf_gmean_kfoldcv[6] = (rf_gmean_kfoldcv[0]+rf_gmean_kfoldcv[1]+rf_gmean_kfoldcv[2]+rf_gmean_kfoldcv[3]+rf_gmean_kfoldcv[4])/5
rf_gmean_kfoldcv = pd.DataFrame(rf_gmean_kfoldcv)
rf_gmean_kfoldcv.to_csv('gmean_'+imb_technique+'_rf_production_' + str(nsplits) + 'foldcv_1~5.csv',header=False,index=False)

svm_f1_score_pen_1_kfoldcv[5] = "Average" #To let users figure out that f1_score_kfoldcv[6] value is the average when seeing the csv file
svm_f1_score_pen_1_kfoldcv[6] = (svm_f1_score_pen_1_kfoldcv[0]+svm_f1_score_pen_1_kfoldcv[1]+svm_f1_score_pen_1_kfoldcv[2]+svm_f1_score_pen_1_kfoldcv[3]+svm_f1_score_pen_1_kfoldcv[4])/5
svm_f1_score_pen_1_kfoldcv = pd.DataFrame(svm_f1_score_pen_1_kfoldcv)
svm_f1_score_pen_1_kfoldcv.to_csv('f1_score_'+imb_technique+'_penalty_1_svm_production_' + str(nsplits) + 'foldcv_1~5.csv',header=False,index=False) #First repetition
svm_f1_score_pen_5_kfoldcv[5] = "Average" #To let users figure out that f1_score_kfoldcv[6] value is the average when seeing the csv file
svm_f1_score_pen_5_kfoldcv[6] = (svm_f1_score_pen_5_kfoldcv[0]+svm_f1_score_pen_5_kfoldcv[1]+svm_f1_score_pen_5_kfoldcv[2]+svm_f1_score_pen_5_kfoldcv[3]+svm_f1_score_pen_5_kfoldcv[4])/5
svm_f1_score_pen_5_kfoldcv = pd.DataFrame(svm_f1_score_pen_5_kfoldcv)
svm_f1_score_pen_5_kfoldcv.to_csv('f1_score_'+imb_technique+'_penalty_5_svm_production_' + str(nsplits) + 'foldcv_1~5.csv',header=False,index=False) #First repetition
#svm_f1_score_kfoldcv.to_csv('f1_score_'+imb_technique+'_penalty_' + str(penalty) + '_svm_production_' + str(nsplits) + 'foldcv_6~10.csv',header=False,index=False) #Second repetition
svm_ovr_accuracy_kfoldcv[5] = "Average" #To let users figure out that f1_score_kfoldcv[6] value is the average when seeing the csv file
svm_ovr_accuracy_kfoldcv[6] = (svm_ovr_accuracy_kfoldcv[0]+svm_ovr_accuracy_kfoldcv[1]+svm_ovr_accuracy_kfoldcv[2]+svm_ovr_accuracy_kfoldcv[3]+svm_ovr_accuracy_kfoldcv[4])/5
svm_ovr_accuracy_kfoldcv = pd.DataFrame(svm_ovr_accuracy_kfoldcv)
svm_ovr_accuracy_kfoldcv.to_csv('ovr_accuracy_'+ imb_technique + '_svm_production_' + str(nsplits) + 'foldcv_1~5.csv',header=False,index=False) #First repetition
#svm_ovr_accuracy_kfoldcv.to_csv('ovr_accuracy_'+imb_technique+'_penalty_' + str(penalty) + '_svm_production_' + str(nsplits) + 'foldcv_6~10.csv',header=False,index=False) #Second repetition
svm_auc_kfoldcv[5] = "Average" #To let users figure out that f1_score_kfoldcv[6] value is the average when seeing the csv file
svm_auc_kfoldcv[6] = (svm_auc_kfoldcv[0]+svm_auc_kfoldcv[1]+svm_auc_kfoldcv[2]+svm_auc_kfoldcv[3]+svm_auc_kfoldcv[4])/5
svm_auc_kfoldcv = pd.DataFrame(svm_auc_kfoldcv)
svm_auc_kfoldcv.to_csv('auc_'+imb_technique+'_svm_production_' + str(nsplits) + 'foldcv_1~5.csv',header=False,index=False)
svm_gmean_kfoldcv[5] = "Average" #To let users figure out that f1_score_kfoldcv[6] value is the average when seeing the csv file
svm_gmean_kfoldcv[6] = (svm_gmean_kfoldcv[0]+svm_gmean_kfoldcv[1]+svm_gmean_kfoldcv[2]+svm_gmean_kfoldcv[3]+svm_gmean_kfoldcv[4])/5
svm_gmean_kfoldcv = pd.DataFrame(svm_gmean_kfoldcv)
svm_gmean_kfoldcv.to_csv('gmean_'+imb_technique+'_svm_production_' + str(nsplits) + 'foldcv_1~5.csv',header=False,index=False)

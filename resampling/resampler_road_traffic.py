import numpy as np
import pandas as pd
import sys
import six
sys.modules['sklearn.externals.six'] = six
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import AllKNN 
from imblearn.under_sampling import CondensedNearestNeighbour
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.under_sampling import InstanceHardnessThreshold
from imblearn.under_sampling import NeighbourhoodCleaningRule
from imblearn.under_sampling import NearMiss
from imblearn.under_sampling import OneSidedSelection
from imblearn.under_sampling import RepeatedEditedNearestNeighbours
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.combine import SMOTEENN    
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks    
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

def resampling_assigner(imb_technique, AP_ova_X_train, AP_ova_y_train, PM_ova_X_train, PM_ova_y_train, SC_ova_X_train, SC_ova_y_train):
    print(imb_technique)
    if imb_technique == "ADASYN":
        AP_ada, PM_ada, SC_ada = ADASYN(), ADASYN(), ADASYN()
        AP_X_res, AP_y_res = AP_ada.fit_resample(AP_ova_X_train, AP_ova_y_train)
        PM_X_res, PM_y_res = PM_ada.fit_resample(PM_ova_X_train, PM_ova_y_train)
        SC_X_res, SC_y_res = SC_ada.fit_resample(SC_ova_X_train, SC_ova_y_train)
    elif imb_technique == "ALLKNN":
        AP_allknn, PM_allknn, SC_allknn = AllKNN(), AllKNN(), AllKNN()
        AP_X_res, AP_y_res = AP_allknn.fit_resample(AP_ova_X_train, AP_ova_y_train)
        PM_X_res, PM_y_res = PM_allknn.fit_resample(PM_ova_X_train, PM_ova_y_train)
        SC_X_res, SC_y_res = SC_allknn.fit_resample(SC_ova_X_train, SC_ova_y_train)
    elif imb_technique == "CNN":
        AP_cnn, PM_cnn, SC_cnn = CondensedNearestNeighbour(), CondensedNearestNeighbour(), CondensedNearestNeighbour()
        AP_X_res, AP_y_res = AP_cnn.fit_resample(AP_ova_X_train, AP_ova_y_train)
        PM_X_res, PM_y_res = PM_cnn.fit_resample(PM_ova_X_train, PM_ova_y_train)
        SC_X_res, SC_y_res = SC_cnn.fit_resample(SC_ova_X_train, SC_ova_y_train)
    elif imb_technique == "ENN":
        AP_enn, PM_enn, SC_enn = EditedNearestNeighbours(), EditedNearestNeighbours(), EditedNearestNeighbours()
        AP_X_res, AP_y_res = AP_enn.fit_resample(AP_ova_X_train, AP_ova_y_train)
        PM_X_res, PM_y_res = PM_enn.fit_resample(PM_ova_X_train, PM_ova_y_train)
        SC_X_res, SC_y_res = SC_enn.fit_resample(SC_ova_X_train, SC_ova_y_train)
    elif imb_technique == "IHT":
        AP_iht, PM_iht, SC_iht = InstanceHardnessThreshold(), InstanceHardnessThreshold(), InstanceHardnessThreshold()
        AP_X_res, AP_y_res = AP_iht.fit_resample(AP_ova_X_train, AP_ova_y_train)
        PM_X_res, PM_y_res = PM_iht.fit_resample(PM_ova_X_train, PM_ova_y_train)
        SC_X_res, SC_y_res = SC_iht.fit_resample(SC_ova_X_train, SC_ova_y_train)
    elif imb_technique == "NCR":
        AP_iht, PM_iht, SC_iht = NeighbourhoodCleaningRule(), NeighbourhoodCleaningRule(), NeighbourhoodCleaningRule()
        AP_ova_y_train = [0 if i == "Add penalty" else 1 for i in AP_ova_y_train]
        AP_X_res, AP_y_res = AP_ncr.fit_resample(AP_ova_X_train, AP_ova_y_train)
        PM_ova_y_train = [0 if i == "Payment" else 1 for i in PM_ova_y_train]
        PM_X_res, PM_y_res = PM_ncr.fit_resample(PM_ova_X_train, PM_ova_y_train)
        SC_ova_y_train = [0 if i == "Send for Credit Collection" else 1 for i in SC_ova_y_train]
        SC_X_res, SC_y_res = SC_ncr.fit_resample(SC_ova_X_train, SC_ova_y_train)
    elif imb_technique == "NM":
        AP_nm, PM_nm, SC_nm = NearMiss(), NearMiss(), NearMiss()
        AP_X_res, AP_y_res = AP_nm.fit_resample(AP_ova_X_train, AP_ova_y_train)
        PM_X_res, PM_y_res = PM_nm.fit_resample(PM_ova_X_train, PM_ova_y_train)
        SC_X_res, SC_y_res = SC_nm.fit_resample(SC_ova_X_train, SC_ova_y_train)
    elif imb_technique == "OSS":
        AP_oss, PM_oss, SC_oss = OneSidedSelection(), OneSidedSelection(), OneSidedSelection()
        AP_X_res, AP_y_res = AP_oss.fit_resample(AP_ova_X_train, AP_ova_y_train)
        PM_X_res, PM_y_res = PM_oss.fit_resample(PM_ova_X_train, PM_ova_y_train)
        SC_X_res, SC_y_res = SC_oss.fit_resample(SC_ova_X_train, SC_ova_y_train)
    elif imb_technique == "RENN":
        AP_renn, PM_renn, SC_renn = RepeatedEditedNearestNeighbours(), RepeatedEditedNearestNeighbours(), RepeatedEditedNearestNeighbours()
        AP_X_res, AP_y_res = AP_renn.fit_resample(AP_ova_X_train, AP_ova_y_train)
        PM_X_res, PM_y_res = PM_renn.fit_resample(PM_ova_X_train, PM_ova_y_train)
        SC_X_res, SC_y_res = SC_renn.fit_resample(SC_ova_X_train, SC_ova_y_train)
    elif imb_technique == "SMOTE":
        AP_sm, PM_sm, SC_sm = SMOTE(), SMOTE(), SMOTE()
        AP_X_res, AP_y_res = AP_sm.fit_resample(AP_ova_X_train, AP_ova_y_train)
        PM_X_res, PM_y_res = PM_sm.fit_resample(PM_ova_X_train, PM_ova_y_train)
        SC_X_res, SC_y_res = SC_sm.fit_resample(SC_ova_X_train, SC_ova_y_train)
    elif imb_technique == "BSMOTE":
        AP_bsm, PM_bsm, SC_bsm = BorderlineSMOTE(), BorderlineSMOTE(), BorderlineSMOTE()
        AP_X_res, AP_y_res = AP_bsm.fit_resample(AP_ova_X_train, AP_ova_y_train)
        PM_X_res, PM_y_res = PM_bsm.fit_resample(PM_ova_X_train, PM_ova_y_train)
        SC_X_res, SC_y_res = SC_bsm.fit_resample(SC_ova_X_train, SC_ova_y_train)
    elif imb_technique == "SMOTEENN":
        AP_smenn, PM_smenn, SC_smenn = SMOTEENN(), SMOTEENN(), SMOTEENN()
        AP_X_res, AP_y_res = AP_smenn.fit_resample(AP_ova_X_train, AP_ova_y_train)
        PM_X_res, PM_y_res = PM_smenn.fit_resample(PM_ova_X_train, PM_ova_y_train)
        SC_X_res, SC_y_res = SC_smenn.fit_resample(SC_ova_X_train, SC_ova_y_train)
    elif imb_technique == "SMOTETOMEK":
        AP_smtm, PM_smtm, SC_smtm = SMOTETomek(), SMOTETomek(), SMOTETomek()
        AP_X_res, AP_y_res = AP_smtm.fit_resample(AP_ova_X_train, AP_ova_y_train)
        PM_X_res, PM_y_res = PM_smtm.fit_resample(PM_ova_X_train, PM_ova_y_train)
        SC_X_res, SC_y_res = SC_smtm.fit_resample(SC_ova_X_train, SC_ova_y_train)
    elif imb_technique == "TOMEK":
        AP_tm, PM_tm, SC_tm = TomekLinks(), TomekLinks(), TomekLinks()
        AP_X_res, AP_y_res = AP_tm.fit_resample(AP_ova_X_train, AP_ova_y_train)
        PM_X_res, PM_y_res = PM_tm.fit_resample(PM_ova_X_train, PM_ova_y_train)
        SC_X_res, SC_y_res = SC_tm.fit_resample(SC_ova_X_train, SC_ova_y_train)
    elif imb_technique == "ROS":
        AP_ros, PM_ros, SC_ros = RandomOverSampler(), RandomOverSampler(), RandomOverSampler()
        AP_X_res, AP_y_res = AP_ros.fit_resample(AP_ova_X_train, AP_ova_y_train)
        PM_X_res, PM_y_res = PM_ros.fit_resample(PM_ova_X_train, PM_ova_y_train)
        SC_X_res, SC_y_res = SC_ros.fit_resample(SC_ova_X_train, SC_ova_y_train)
    elif imb_technique == "RUS":
        AP_rus, PM_rus, SC_rus = RandomUnderSampler(), RandomUnderSampler(), RandomUnderSampler()
        AP_X_res, AP_y_res = AP_rus.fit_resample(AP_ova_X_train, AP_ova_y_train)
        PM_X_res, PM_y_res = PM_rus.fit_resample(PM_ova_X_train, PM_ova_y_train)
        SC_X_res, SC_y_res = SC_rus.fit_resample(SC_ova_X_train, SC_ova_y_train)
    return AP_X_res, AP_y_res, PM_X_res, PM_y_res, SC_X_res, SC_y_res

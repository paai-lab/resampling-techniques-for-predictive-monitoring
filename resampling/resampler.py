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

print("HI")
def resampling_assigner(imb_technique, AA_ova_X_train, AA_ova_y_train, AI_ova_X_train, AI_ova_y_train, AW_ova_X_train, AW_ova_y_train, CC_ova_X_train, CC_ova_y_train, QA_ova_X_train, QA_ova_y_train):
    print(imb_technique)
    if imb_technique == "ADASYN":
        AA_ada, AI_ada, AW_ada, CC_ada, QA_ada = ADASYN(), ADASYN(), ADASYN(), ADASYN(), ADASYN()
        AA_X_res, AA_y_res = AA_ada.fit_resample(AA_ova_X_train, AA_ova_y_train)
        AI_X_res, AI_y_res = AI_ada.fit_resample(AI_ova_X_train, AI_ova_y_train)
        AW_X_res, AW_y_res = AW_ada.fit_resample(AW_ova_X_train, AW_ova_y_train)
        CC_X_res, CC_y_res = CC_ada.fit_resample(CC_ova_X_train, CC_ova_y_train)
        QA_X_res, QA_y_res = QA_ada.fit_resample(QA_ova_X_train, QA_ova_y_train)
    elif imb_technique == "ALLKNN":
        AA_allknn, AI_allknn, AW_allknn, CC_allknn, QA_allknn = AllKNN(), AllKNN(), AllKNN(), AllKNN(), AllKNN()
        AA_X_res, AA_y_res = AA_allknn.fit_resample(AA_ova_X_train, AA_ova_y_train)
        AI_X_res, AI_y_res = AI_allknn.fit_resample(AI_ova_X_train, AI_ova_y_train)
        AW_X_res, AW_y_res = AW_allknn.fit_resample(AW_ova_X_train, AW_ova_y_train)
        CC_X_res, CC_y_res = CC_allknn.fit_resample(CC_ova_X_train, CC_ova_y_train)
        QA_X_res, QA_y_res = QA_allknn.fit_resample(QA_ova_X_train, QA_ova_y_train)    
    elif imb_technique == "CNN":
        AA_cnn, AI_cnn, AW_cnn, CC_cnn, QA_cnn = CondensedNearestNeighbour(), CondensedNearestNeighbour(), CondensedNearestNeighbour(), CondensedNearestNeighbour(), CondensedNearestNeighbour()
        AA_X_res, AA_y_res = AA_cnn.fit_resample(AA_ova_X_train, AA_ova_y_train)
        AI_X_res, AI_y_res = AI_cnn.fit_resample(AI_ova_X_train, AI_ova_y_train)
        AW_X_res, AW_y_res = AW_cnn.fit_resample(AW_ova_X_train, AW_ova_y_train)
        CC_X_res, CC_y_res = CC_cnn.fit_resample(CC_ova_X_train, CC_ova_y_train)
        QA_X_res, QA_y_res = QA_cnn.fit_resample(QA_ova_X_train, QA_ova_y_train)    
    elif imb_technique == "ENN":
        AA_enn, AI_enn, AW_enn, CC_enn, QA_enn = EditedNearestNeighbours(), EditedNearestNeighbours(), EditedNearestNeighbours(), EditedNearestNeighbours(), EditedNearestNeighbours()
        AA_X_res, AA_y_res = AA_enn.fit_resample(AA_ova_X_train, AA_ova_y_train)
        AI_X_res, AI_y_res = AI_enn.fit_resample(AI_ova_X_train, AI_ova_y_train)
        AW_X_res, AW_y_res = AW_enn.fit_resample(AW_ova_X_train, AW_ova_y_train)
        CC_X_res, CC_y_res = CC_enn.fit_resample(CC_ova_X_train, CC_ova_y_train)
        QA_X_res, QA_y_res = QA_enn.fit_resample(QA_ova_X_train, QA_ova_y_train)
    elif imb_technique == "IHT":
        AA_iht, AI_iht, AW_iht, CC_iht, QA_iht = InstanceHardnessThreshold(), InstanceHardnessThreshold(), InstanceHardnessThreshold(), InstanceHardnessThreshold(), InstanceHardnessThreshold()
        AA_X_res, AA_y_res = AA_iht.fit_resample(AA_ova_X_train, AA_ova_y_train)
        AI_X_res, AI_y_res = AI_iht.fit_resample(AI_ova_X_train, AI_ova_y_train)
        AW_X_res, AW_y_res = AW_iht.fit_resample(AW_ova_X_train, AW_ova_y_train)
        CC_X_res, CC_y_res = CC_iht.fit_resample(CC_ova_X_train, CC_ova_y_train)
        QA_X_res, QA_y_res = QA_iht.fit_resample(QA_ova_X_train, QA_ova_y_train)
    elif imb_technique == "NCR":
        AA_ncr, AI_ncr, AW_ncr, CC_ncr, QA_ncr = NeighbourhoodCleaningRule(), NeighbourhoodCleaningRule(), NeighbourhoodCleaningRule(), NeighbourhoodCleaningRule(), NeighbourhoodCleaningRule()
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
    elif imb_technique == "NM":
        AA_nm, AI_nm, AW_nm, CC_nm, QA_nm = NearMiss(), NearMiss(), NearMiss(), NearMiss(), NearMiss()
        AA_X_res, AA_y_res = AA_nm.fit_resample(AA_ova_X_train, AA_ova_y_train)
        AI_X_res, AI_y_res = AI_nm.fit_resample(AI_ova_X_train, AI_ova_y_train)
        AW_X_res, AW_y_res = AW_nm.fit_resample(AW_ova_X_train, AW_ova_y_train)
        CC_X_res, CC_y_res = CC_nm.fit_resample(CC_ova_X_train, CC_ova_y_train)
        QA_X_res, QA_y_res = QA_nm.fit_resample(QA_ova_X_train, QA_ova_y_train)
    elif imb_technique == "OSS":
        AA_oss, AI_oss, AW_oss, CC_oss, QA_oss = OneSidedSelection(), OneSidedSelection(), OneSidedSelection(), OneSidedSelection(), OneSidedSelection()
        AA_X_res, AA_y_res = AA_oss.fit_resample(AA_ova_X_train, AA_ova_y_train)
        AI_X_res, AI_y_res = AI_oss.fit_resample(AI_ova_X_train, AI_ova_y_train)
        AW_X_res, AW_y_res = AW_oss.fit_resample(AW_ova_X_train, AW_ova_y_train)
        CC_X_res, CC_y_res = CC_oss.fit_resample(CC_ova_X_train, CC_ova_y_train)
        QA_X_res, QA_y_res = QA_oss.fit_resample(QA_ova_X_train, QA_ova_y_train)
    elif imb_technique == "RENN":
        AA_renn, AI_renn, AW_renn, CC_renn, QA_renn = RepeatedEditedNearestNeighbours(), RepeatedEditedNearestNeighbours(), RepeatedEditedNearestNeighbours(), RepeatedEditedNearestNeighbours(), RepeatedEditedNearestNeighbours()
        AA_X_res, AA_y_res = AA_renn.fit_resample(AA_ova_X_train, AA_ova_y_train)
        AI_X_res, AI_y_res = AI_renn.fit_resample(AI_ova_X_train, AI_ova_y_train)
        AW_X_res, AW_y_res = AW_renn.fit_resample(AW_ova_X_train, AW_ova_y_train)
        CC_X_res, CC_y_res = CC_renn.fit_resample(CC_ova_X_train, CC_ova_y_train)
        QA_X_res, QA_y_res = QA_renn.fit_resample(QA_ova_X_train, QA_ova_y_train)
    elif imb_technique == "SMOTE":
        AA_sm, AI_sm, AW_sm, CC_sm, QA_sm = SMOTE(), SMOTE(), SMOTE(), SMOTE(), SMOTE()
        AA_X_res, AA_y_res = AA_sm.fit_resample(AA_ova_X_train, AA_ova_y_train)
        AI_X_res, AI_y_res = AI_sm.fit_resample(AI_ova_X_train, AI_ova_y_train)
        AW_X_res, AW_y_res = AW_sm.fit_resample(AW_ova_X_train, AW_ova_y_train)
        CC_X_res, CC_y_res = CC_sm.fit_resample(CC_ova_X_train, CC_ova_y_train)
        QA_X_res, QA_y_res = QA_sm.fit_resample(QA_ova_X_train, QA_ova_y_train)
    elif imb_technique == "BSMOTE":
        AA_bsm, AI_bsm, AW_bsm, CC_bsm, QA_bsm = BorderlineSMOTE(), BorderlineSMOTE(), BorderlineSMOTE(), BorderlineSMOTE(), BorderlineSMOTE()
        AA_X_res, AA_y_res = AA_bsm.fit_resample(AA_ova_X_train, AA_ova_y_train)
        AI_X_res, AI_y_res = AI_bsm.fit_resample(AI_ova_X_train, AI_ova_y_train)
        AW_X_res, AW_y_res = AW_bsm.fit_resample(AW_ova_X_train, AW_ova_y_train)
        CC_X_res, CC_y_res = CC_bsm.fit_resample(CC_ova_X_train, CC_ova_y_train)
        QA_X_res, QA_y_res = QA_bsm.fit_resample(QA_ova_X_train, QA_ova_y_train)    
    elif imb_technique == "SMOTEENN":
        AA_smenn, AI_smenn, AW_smenn, CC_smenn, QA_smenn = SMOTEENN(), SMOTEENN(), SMOTEENN(), SMOTEENN(), SMOTEENN()
        AA_X_res, AA_y_res = AA_smenn.fit_resample(AA_ova_X_train, AA_ova_y_train)
        AI_X_res, AI_y_res = AI_smenn.fit_resample(AI_ova_X_train, AI_ova_y_train)
        AW_X_res, AW_y_res = AW_smenn.fit_resample(AW_ova_X_train, AW_ova_y_train)
        CC_X_res, CC_y_res = CC_smenn.fit_resample(CC_ova_X_train, CC_ova_y_train)
        QA_X_res, QA_y_res = QA_smenn.fit_resample(QA_ova_X_train, QA_ova_y_train)
    elif imb_technique == "SMOTETOMEK":
        AA_smtm, AI_smtm, AW_smtm, CC_smtm, QA_smtm = SMOTETomek(), SMOTETomek(), SMOTETomek(), SMOTETomek(), SMOTETomek()
        AA_X_res, AA_y_res = AA_smtm.fit_resample(AA_ova_X_train, AA_ova_y_train)
        AI_X_res, AI_y_res = AI_smtm.fit_resample(AI_ova_X_train, AI_ova_y_train)
        AW_X_res, AW_y_res = AW_smtm.fit_resample(AW_ova_X_train, AW_ova_y_train)
        CC_X_res, CC_y_res = CC_smtm.fit_resample(CC_ova_X_train, CC_ova_y_train)
        QA_X_res, QA_y_res = QA_smtm.fit_resample(QA_ova_X_train, QA_ova_y_train)
    elif imb_technique == "TOMEK":
        AA_tm, AI_tm, AW_tm, CC_tm, QA_tm = TomekLinks(), TomekLinks(), TomekLinks(), TomekLinks(), TomekLinks()
        AA_X_res, AA_y_res = AA_tm.fit_resample(AA_ova_X_train, AA_ova_y_train)
        AI_X_res, AI_y_res = AI_tm.fit_resample(AI_ova_X_train, AI_ova_y_train)
        AW_X_res, AW_y_res = AW_tm.fit_resample(AW_ova_X_train, AW_ova_y_train)
        CC_X_res, CC_y_res = CC_tm.fit_resample(CC_ova_X_train, CC_ova_y_train)
        QA_X_res, QA_y_res = QA_tm.fit_resample(QA_ova_X_train, QA_ova_y_train)    
    elif imb_technique == "ROS":
        AA_ros, AI_ros, AW_ros, CC_ros, QA_ros = RandomOverSampler(), RandomOverSampler(), RandomOverSampler(), RandomOverSampler(), RandomOverSampler()
        AA_X_res, AA_y_res = AA_ros.fit_resample(AA_ova_X_train, AA_ova_y_train)
        AI_X_res, AI_y_res = AI_ros.fit_resample(AI_ova_X_train, AI_ova_y_train)
        AW_X_res, AW_y_res = AW_ros.fit_resample(AW_ova_X_train, AW_ova_y_train)
        CC_X_res, CC_y_res = CC_ros.fit_resample(CC_ova_X_train, CC_ova_y_train)
        QA_X_res, QA_y_res = QA_ros.fit_resample(QA_ova_X_train, QA_ova_y_train)
    elif imb_technique == "RUS":
        AA_rus, AI_rus, AW_rus, CC_rus, QA_rus = RandomUnderSampler(), RandomUnderSampler(), RandomUnderSampler(), RandomUnderSampler(), RandomUnderSampler()
        AA_X_res, AA_y_res = AA_rus.fit_resample(AA_ova_X_train, AA_ova_y_train)
        AI_X_res, AI_y_res = AI_rus.fit_resample(AI_ova_X_train, AI_ova_y_train)
        AW_X_res, AW_y_res = AW_rus.fit_resample(AW_ova_X_train, AW_ova_y_train)
        CC_X_res, CC_y_res = CC_rus.fit_resample(CC_ova_X_train, CC_ova_y_train)
        QA_X_res, QA_y_res = QA_rus.fit_resample(QA_ova_X_train, QA_ova_y_train)
    return AA_X_res, AA_y_res, AI_X_res, AI_y_res, AW_X_res, AW_y_res, CC_X_res, CC_y_res, QA_X_res, QA_y_res

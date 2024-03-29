import datetime
import numpy as np

import os
import sys
sys.path.insert(0, os.getcwd())

from hyperopt import hp, fmin, tpe

import qml_workdir.classes.config
from qml.cv import QCV
from qml.models import QXgb, QAvg, QAvgOneModelData
from qml_workdir.classes.models import qm

trash wrong model

cv = QCV(qm)

model_id = qm.add_by_params(
    QXgb(
** {"alpha": 0.008, "booster": "gbtree", "colsample_bylevel": 0.9, "colsample_bytree": 0.9, "eta": 0.024, "eval_metric": "logloss",
    "gamma": 0.04, "max_depth": 4, "num_boost_round": 261, "objective": "binary:logistic", "subsample": 0.7, "tree_method": "hist"}
    ),
    'hyperopt xgb',
)

model_id =qm.add_by_params(QAvgOneModelData(model_id, 8), level=-2)

cv.features_sel_add(model_id, 60, [
'age', 'height', 'weight', 'ap_hi', 'ap_lo', 'smoke', 'alco', 'active',
       'gender_male', 'height_low', 'weight_low', 'cholesterol_all',
       'gluc_all', 'cholesterol_1', 'cholesterol_2', 'cholesterol_3', 'gluc_1',
       'gluc_2', 'gluc_3', 'ap_error', 'ap_error_swap', 'imt', 'imt_class_all',
       'imt_class_0', 'imt_class_1', 'imt_class_2', 'imt_class_3',
       'imt_class_4', 'imt_class_5', 'imt_class_6', 'x__age__gluc_all',
       'x__ap_hi__cholesterol_all', 'div6__height__gluc_all__imt',
       'plus__age_norm__ap_hi_norm__gluc_all_norm', 'x__age__weight',
       'div1__age__weight__cholesterol_all',
       'div6__age__weight__cholesterol_all',
       'plus__height_norm__weight_norm__gluc_all_norm',
       'div1__ap_hi__ap_lo__cholesterol_all',
       'div6__ap_hi__ap_lo__cholesterol_all',
       'plus__age_norm__gluc_all_norm__imt_norm',
       'minus6__ap_hi_norm__ap_lo_norm__cholesterol_all_norm',
       'minus1__ap_hi_norm__ap_lo_norm__cholesterol_all_norm',
       'minus6__age_norm__ap_lo_norm__cholesterol_all_norm',
       'minus1__age_norm__ap_lo_norm__cholesterol_all_norm',
       'div6__height__weight__ap_lo', 'div2__ap_lo__cholesterol_all__gluc_all',
       'x__age__ap_hi__gluc_all', 'div5__ap_lo__cholesterol_all__gluc_all',
       'score_scale_val', 'div6__height__gluc_all__imt___gender__scale',
       'k15_0', 'k15_1', 'k15_2', 'k15_3', 'k15_4', 'k15_5', 'k15_6', 'k15_7',
       'k15_8', 'k15_9', 'k15_10', 'k15_11', 'k15_12', 'k15_13', 'k15_14',
       'k7_0', 'k7_1', 'k7_2', 'k7_3', 'k7_4', 'k7_5', 'k7_6', 'k3_0', 'k3_1',
       'k3_2'
],
[
 'minus6__age_norm__ap_lo_norm__cholesterol_all_norm___gender_gluc__scale',
 'minus6__age_norm__ap_lo_norm__cholesterol_all_norm___gender_age__scale',
 'minus1__age_norm__ap_lo_norm__cholesterol_all_norm___gender_chol__scale',
 'minus1__age_norm__ap_lo_norm__cholesterol_all_norm___gender_gluc__scale',
 'minus1__age_norm__ap_lo_norm__cholesterol_all_norm___gender_age__scale',
 'div6__height__weight__ap_lo___gender_chol__scale',
 'div6__height__weight__ap_lo___gender_gluc__scale',
 'div6__height__weight__ap_lo___gender_age__scale',
 'div2__ap_lo__cholesterol_all__gluc_all___gender_chol__scale',
 'div2__ap_lo__cholesterol_all__gluc_all___gender_gluc__scale',
 'div2__ap_lo__cholesterol_all__gluc_all___gender_age__scale',
 'x__age__ap_hi__gluc_all___gender_chol__scale',
 'x__age__ap_hi__gluc_all___gender_gluc__scale',
 'x__age__ap_hi__gluc_all___gender_age__scale',
 'div5__ap_lo__cholesterol_all__gluc_all___gender_chol__scale',
 'div5__ap_lo__cholesterol_all__gluc_all___gender_gluc__scale',
 'div5__ap_lo__cholesterol_all__gluc_all___gender_age__scale',
 'score_scale_val___gender_chol__scale',
 'score_scale_val___gender_gluc__scale',
 'score_scale_val___gender_age__scale',
'cholesterol_all___gender_age__scale',
'gluc_all___gender_age__scale',
'minus6__ap_hi_norm__ap_lo_norm__cholesterol_all_norm___gender_age__scale',
'ap_lo___gender_age__scale',
'ap_hi___gender_chol__scale',
'x__ap_hi__cholesterol_all___gender_chol__scale',
'cholesterol_all___gender_gluc__scale',
'minus1__ap_hi_norm__ap_lo_norm__cholesterol_all_norm___gender_age__scale',
'imt___gender_age__scale',
'plus__age_norm__gluc_all_norm__imt_norm___gender_chol__scale',
'div6__ap_hi__ap_lo__cholesterol_all___gender_age__scale',
'imt_class_all___gender_age__scale',
'x__ap_hi__cholesterol_all___gender_age__scale',
'gluc_all___gender_chol__scale',
'imt_class_all___gender_chol__scale',
'x__age__gluc_all___gender_chol__scale',
'ap_lo___gender_gluc__scale',
'x__age__weight___gender_gluc__scale',
'plus__height_norm__weight_norm__gluc_all_norm___gender_age__scale',
'weight___gender_gluc__scale',
'div6__height__gluc_all__imt___gender_chol__scale',
'weight___gender_chol__scale',
'ap_lo___gender_chol__scale',
'minus6__ap_hi_norm__ap_lo_norm__cholesterol_all_norm___gender_gluc__scale',
'imt_class_all___gender_gluc__scale',
'x__age__weight___gender_age__scale',
'imt___gender_chol__scale',
'minus1__ap_hi_norm__ap_lo_norm__cholesterol_all_norm___gender_gluc__scale',
'plus__age_norm__ap_hi_norm__gluc_all_norm___gender_chol__scale',
'div6__ap_hi__ap_lo__cholesterol_all___gender_gluc__scale',
'div1__ap_hi__ap_lo__cholesterol_all___gender_age__scale',
'div6__ap_hi__ap_lo__cholesterol_all___gender_chol__scale',
'x__age__gluc_all___gender_age__scale',
'imt___gender_gluc__scale',
'plus__height_norm__weight_norm__gluc_all_norm___gender_chol__scale',
'plus__age_norm__gluc_all_norm__imt_norm___gender_gluc__scale',
'weight___gender_age__scale',
'div6__age__weight__cholesterol_all___gender_age__scale',
'minus6__ap_hi_norm__ap_lo_norm__cholesterol_all_norm___gender_chol__scale',
'div1__ap_hi__ap_lo__cholesterol_all___gender_gluc__scale',
'minus1__ap_hi_norm__ap_lo_norm__cholesterol_all_norm___gender_chol__scale',
'plus__height_norm__weight_norm__gluc_all_norm___gender_gluc__scale',
'div1__ap_hi__ap_lo__cholesterol_all___gender_chol__scale',
'div6__age__weight__cholesterol_all___gender_chol__scale',
'div6__age__weight__cholesterol_all___gender_gluc__scale',
'div6__height__gluc_all__imt___gender_gluc__scale',
'age___gender_gluc__scale',
'x__age__gluc_all___gender_gluc__scale',
'height___gender_chol__scale',
'minus6__age_norm__ap_lo_norm__cholesterol_all_norm___gender_chol__scale',
'age___gender_chol__scale',
'plus__age_norm__ap_hi_norm__gluc_all_norm___gender_gluc__scale',
'div1__age__weight__cholesterol_all___gender_age__scale',
'plus__age_norm__ap_hi_norm__gluc_all_norm___gender_age__scale',
'ap_hi___gender_age__scale',
'div6__height__gluc_all__imt___gender_age__scale',
'plus__age_norm__gluc_all_norm__imt_norm___gender_age__scale',
'height___gender_gluc__scale',
'div1__age__weight__cholesterol_all___gender_gluc__scale',
'div1__age__weight__cholesterol_all___gender_chol__scale',
'x__age__weight___gender_chol__scale',
'height___gender_age__scale',
'ap_hi___gender_gluc__scale',
'x__ap_hi__cholesterol_all___gender_gluc__scale',


]
, early_stop_cv=lambda x: x > 0.5414, log_file='qml_workdir/logs/feat14.txt')

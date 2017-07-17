import datetime
import numpy as np

import os
import sys
sys.path.insert(0, os.getcwd())

from hyperopt import hp, fmin, tpe

import qml_workdir.classes.config
from qml.cv import QCV
from qml.models import QXgb, QAvg
from qml_workdir.classes.models import qm



cv = QCV(qm)

model_id = qm.add_by_params(
    QXgb(
** {"alpha": 0.008, "booster": "gbtree", "colsample_bylevel": 0.9, "colsample_bytree": 0.9, "eta": 0.0024, "eval_metric": "logloss",
    "gamma": 0.04, "max_depth": 4, "num_boost_round": 2619, "objective": "binary:logistic", "subsample": 0.7, "tree_method": "hist"}
    ),
    'hyperopt xgb'
)

cv.features_sel_add(model_id, 40, ['age', 'height', 'weight', 'ap_hi', 'ap_lo', 'smoke', 'alco', 'active',
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
       'score_scale_val'],
[
'plus__age_norm__cholesterol_all_norm__gluc_all_norm',
'x__age__cholesterol_all__gluc_all',
'minus2__age_norm__cholesterol_all_norm__gluc_all_norm',
'minus5__age_norm__cholesterol_all_norm__gluc_all_norm',
'div5__age__cholesterol_all__gluc_all',
'div1__age__ap_hi__cholesterol_all',
'div2__age__cholesterol_all__gluc_all',
'div6__age__ap_hi__cholesterol_all',
'minus2__ap_lo_norm__cholesterol_all_norm__gluc_all_norm',
'div__cholesterol_all__gluc_all',
'minus1__height_norm__cholesterol_all_norm__imt_norm',
'div6__age__ap_lo__cholesterol_all',
'div1__age__ap_lo__cholesterol_all',
'div1__height__gluc_all__imt',
'div1__age__weight__gluc_all',
'minus2__height_norm__weight_norm__gluc_all_norm',
'div6__age__weight__gluc_all',
'minus6__height_norm__gluc_all_norm__imt_norm',
'div5__height__cholesterol_all__gluc_all',
'div4__age__cholesterol_all__gluc_all',
'div6__age__cholesterol_all__gluc_all',
'minus5__height_norm__weight_norm__gluc_all_norm',
'minus5__height_norm__cholesterol_all_norm__gluc_all_norm',
'div__age__cholesterol_all',
'minus1__height_norm__gluc_all_norm__imt_norm',
'div3__height__weight__gluc_all',
'minus4__age_norm__ap_hi_norm__cholesterol_all_norm',
'x__ap_hi__ap_lo',
'div2__height__cholesterol_all__gluc_all',
'div1__age__cholesterol_all__gluc_all',
'minus5__age_norm__cholesterol_all_norm__imt_norm',
'minus2__height_norm__cholesterol_all_norm__gluc_all_norm',
'minus__cholesterol_all_norm__imt_norm',
'plus__ap_hi_norm__ap_lo_norm',
'minus6__age_norm__cholesterol_all_norm__gluc_all_norm',
'minus1__age_norm__ap_lo_norm__gluc_all_norm',
'minus1__age_norm__cholesterol_all_norm__gluc_all_norm',
'plus__height_norm__weight_norm__ap_lo_norm',
'div4__height__ap_lo__imt',
'minus6__age_norm__ap_lo_norm__gluc_all_norm',
'div2__weight__ap_hi__gluc_all',
'div1__age__ap_lo__imt',
'minus5__age_norm__ap_hi_norm__ap_lo_norm',
'minus3__age_norm__weight_norm__gluc_all_norm',
'minus2__height_norm__weight_norm__cholesterol_all_norm',
'minus5__weight_norm__ap_hi_norm__gluc_all_norm',
'div__age__ap_hi',
'plus__weight_norm__ap_hi_norm__gluc_all_norm',
'div3__weight__ap_hi__gluc_all',
'minus4__weight_norm__cholesterol_all_norm__imt_norm',
'minus5__cholesterol_all_norm__gluc_all_norm__imt_norm',
'minus1__ap_hi_norm__ap_lo_norm__imt_norm',
'div6__height__weight__cholesterol_all',
'minus2__cholesterol_all_norm__gluc_all_norm__imt_norm',
'x__height__ap_hi__gluc_all',
'plus__ap_hi_norm__ap_lo_norm__gluc_all_norm',
'div5__age__cholesterol_all__imt',
'div__weight__gluc_all',
'minus2__age_norm__height_norm__ap_lo_norm',
'plus__age_norm__weight_norm__ap_lo_norm',
'x__height__cholesterol_all__imt',
'x__age__height__imt',
'minus1__weight_norm__cholesterol_all_norm__imt_norm',
'minus5__age_norm__gluc_all_norm__imt_norm',
'minus2__ap_hi_norm__ap_lo_norm__gluc_all_norm',
'div4__age__ap_hi__imt',
'div4__age__height__gluc_all',
'div1__weight__ap_hi__gluc_all',
'minus6__age_norm__height_norm__weight_norm',
'div3__ap_hi__ap_lo__cholesterol_all',
'div3__age__ap_hi__gluc_all',
'div2__ap_hi__cholesterol_all__imt',
'div6__ap_lo__gluc_all__imt',
'div6__weight__ap_hi__gluc_all',
'minus3__weight_norm__ap_lo_norm__cholesterol_all_norm',
'div__ap_hi__ap_lo',
'div2__height__weight__ap_lo',
'div5__age__weight__ap_lo',
'div6__cholesterol_all__gluc_all__imt',
'div1__weight__cholesterol_all__imt',
'div1__cholesterol_all__gluc_all__imt',
'plus__cholesterol_all_norm__imt_norm',
'minus__weight_norm__ap_lo_norm',
'x__ap_hi__imt',
'div5__age__weight__imt',
'plus__weight_norm__cholesterol_all_norm__imt_norm',
'div__age__ap_lo',
'div2__age__ap_hi__imt',
'div5__age__height__ap_hi',
'minus6__age_norm__ap_hi_norm__ap_lo_norm',
'minus1__age_norm__weight_norm__ap_lo_norm',
'div3__height__ap_lo__cholesterol_all',
'minus5__age_norm__height_norm__weight_norm',


], early_stop_cv=lambda x: x > 0.5414, log_file='qml_workdir/logs/feat09.txt')

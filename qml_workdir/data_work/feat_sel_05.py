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



cv.features_sel_add(1395, 14,
                    [
 'age',
 'height',
 'weight',
 'ap_hi',
 'ap_lo',
 'smoke',
 'alco',
 'active',
 'gender_male',
 'height_low',
 'weight_low',
 'cholesterol_all',
 'gluc_all',
 'cholesterol_1',
 'cholesterol_2',
 'cholesterol_3',
 'gluc_1',
 'gluc_2',
 'gluc_3',
 'ap_error',
 'ap_error_swap',
 'imt',
 'imt_class_all',
 'imt_class_0',
 'imt_class_1',
 'imt_class_2',
 'imt_class_3',
 'imt_class_4',
 'imt_class_5',
 'imt_class_6',
'x__age__gluc_all',
'x__ap_hi__cholesterol_all'
 ],
                    [

'div__ap_hi__active',
'div__cholesterol_all__gluc_all',
'plus__age_years__imt',
'div__smoke__imt_class_all',
'minus__cholesterol_all__gluc_all',
'minus__smoke__imt_class_all',
'plus__ap_hi__ap_lo',
'div__active__cholesterol_all',
'x__height__gluc_all',
'x__weight__age_years',
'div__ap_lo__cholesterol_all',
'div__alco__imt',
'x__weight__imt_class_all',
'x__weight__imt',
'div__ap_lo__alco',
'x__weight__gluc_all',
'plus__cholesterol_all__gluc_all',
'x__cholesterol_all__gluc_all',
'div__active__gluc_all',
'x__ap_hi__gluc_all',
'x__ap_lo__gluc_all',

                    ], early_stop_cv=lambda x: x>0.545, log_file='qml_workdir/logs/feat05.txt')

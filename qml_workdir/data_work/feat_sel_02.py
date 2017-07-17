import datetime
import numpy as np

from hyperopt import hp, fmin, tpe

import qml_workdir.classes.config
from qml.cv import QCV
from qml.models import QXgb, QAvg
from qml_workdir.classes.models import qm



cv = QCV(qm)



cv.features_sel_hyper(1395, 6, early_stop_cv=lambda x: x>0.545, log_file='qml_workdir/logs/feat02.txt')

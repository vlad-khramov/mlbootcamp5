import os
import sys
sys.path.insert(0, os.getcwd())


import qml_workdir.classes.config
from qml.cv import QCV
from qml_workdir.classes.models import qm

# cv = QCV(qm)
# print(cv.cross_val(1, 1))
# qm.qpredict(1, 1)


cv = QCV(qm)
print(cv.cross_val(1, 6))
qm.qpredict(1, 6)



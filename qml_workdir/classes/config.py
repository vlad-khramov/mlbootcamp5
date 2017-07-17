from qml import config

config.QML_DATA_DIR = 'qml_workdir/data/'
config.QML_DB_CONN_STRING = 'mysql+pymysql://root:password@localhost/mlbootcamp5'

config.QML_TRAIN_X_FILE_MASK = 'qml_workdir/data/v{0:0=4d}_train_x.csv'
config.QML_TEST_X_FILE_MASK  = 'qml_workdir/data/v{0:0=4d}_test_x.csv'
config.QML_TRAIN_Y_FILE_MASK = 'qml_workdir/data/train_y.csv'

config.QML_RES_COL   = 'cardio'
#config.QML_RES_COL2  = ['high', 'medium', 'low']
config.QML_INDEX_COL = 'id'


from qml import config

config.QML_DATA_DIR = 'qml_workdir/data/' #папка с данными и сохраненными результатами работы моделей
config.QML_DB_CONN_STRING = 'mysql+pymysql://root:password@localhost/mlbootcamp5' # параметры соединения с БД, в которой хранятся параметры моделей, результаты их работы

config.QML_TRAIN_X_FILE_MASK = 'qml_workdir/data/v{0:0=4d}_train_x.csv' # маска для версий train set
config.QML_TEST_X_FILE_MASK  = 'qml_workdir/data/v{0:0=4d}_test_x.csv' # маска для версий test set
config.QML_TRAIN_Y_FILE_MASK = 'qml_workdir/data/train_y.csv' # файл с метками train set

config.QML_RES_COL   = 'cardio' # колонка с метками
config.QML_INDEX_COL = 'id'     # колонка с id

#config.QML_RES_COL2  = ['high', 'medium', 'low']
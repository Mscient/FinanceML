

from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

X_TRAIN_PATH = DATA_DIR / "X_train.csv"
X_TEST_PATH = DATA_DIR / "X_test.csv"
Y_TRAIN_PATH = DATA_DIR / "y_train.csv"
Y_TEST_PATH = DATA_DIR / "y_test.csv"
y_TRAIN_PATH = Y_TRAIN_PATH
y_TEST_PATH = Y_TEST_PATH
LOAN_DATA_PATH = DATA_DIR / "loans.csv"
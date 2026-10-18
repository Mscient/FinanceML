from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Default classification
X_TRAIN_DEFAULT_PATH = PROCESSED_DATA_DIR / "X_train_default.csv"
X_TEST_DEFAULT_PATH = PROCESSED_DATA_DIR / "X_test_default.csv"
Y_TRAIN_DEFAULT_PATH = PROCESSED_DATA_DIR / "y_train_default.csv"
Y_TEST_DEFAULT_PATH = PROCESSED_DATA_DIR / "y_test_default.csv"

# Fraud classification
X_TRAIN_FRAUD_PATH = PROCESSED_DATA_DIR / "X_train_is_fraud.csv"
X_TEST_FRAUD_PATH = PROCESSED_DATA_DIR / "X_test_is_fraud.csv"
Y_TRAIN_FRAUD_PATH = PROCESSED_DATA_DIR / "y_train_is_fraud.csv"
Y_TEST_FRAUD_PATH = PROCESSED_DATA_DIR / "y_test_is_fraud.csv"

# Original merged dataset
LOAN_DATA_PATH = DATA_DIR / "loans_merged.csv"

# Backward compatibility aliases
X_TRAIN_PATH = X_TRAIN_DEFAULT_PATH
X_TEST_PATH = X_TEST_DEFAULT_PATH
Y_TRAIN_PATH = Y_TRAIN_DEFAULT_PATH
Y_TEST_PATH = Y_TEST_DEFAULT_PATH
y_TRAIN_PATH = Y_TRAIN_PATH
y_TEST_PATH = Y_TEST_PATH




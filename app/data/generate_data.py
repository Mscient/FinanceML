from pathlib import Path
import json
import pandas as pd
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "loans_merged.csv"
OUTPUT_DIR = BASE_DIR / "processed"


def generate_data():
    
    return load_data(DATA_PATH)


def load_data(path=DATA_PATH):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Place loans_merged.csv beside this script or update DATA_PATH."
        )
    return pd.read_csv(path)


def inspect_data(df):
    print("\n--- Dataset info ---")
    df.info()

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    print("\nDescriptive statistics:")
    print(df.describe(include="all").transpose())

    print("\nDuplicate rows:", df.duplicated().sum())
    missing = df.isna().sum()
    print("\nMissing values by column:")
    print(missing[missing > 0])

    print(f"\nShape: {df.shape[0]} rows, {df.shape[1]} columns")

    for col in [
        "employment_type", "loan_purpose", "region",
        "default", "txn_is_fraud",
    ]:
        if col in df.columns:
            print(f"\n{col} distribution:")
            print(df[col].value_counts(dropna=False))


def clean_data(df):
    """Normalize categorical values and coerce expected numeric fields."""
    df = df.copy()

    categorical_columns = {
        "employment_type", "loan_purpose", "region",
        "customer_id", "transaction_id",
    }

    for col in df.columns:
        if col in {"customer_id", "transaction_id"}:
            continue

        if col in categorical_columns or df[col].dtype == "object":
            df[col] = df[col].astype("string").str.lower().str.strip()
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for target in ["default", "txn_is_fraud"]:
        if target in df.columns:
            df[target] = pd.to_numeric(df[target], errors="coerce")

    return df


def validate_data(df):
    """
    Remove duplicate rows and rows with invalid known values.
    Missing values are retained for train-only imputation.
    Missing txn_is_fraud means the customer has no observed transactions.
    """
    df = df.drop_duplicates().copy()

    bounds = {
        "age": (18, 100),
        "credit_score": (300, 850),
        "years_employed": (0, 80),
        "online_transactions_ratio": (0, 1),
        "debt_to_income": (0, 1),
        "credit_util": (0, 1),
        "txn_merchant_risk_score_mean": (0, 1),
    }

    for col, (lower, upper) in bounds.items():
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce")
            invalid = values.notna() & ((values < lower) | (values > upper))
            df = df.loc[~invalid].copy()

    nonnegative_columns = [
        "annual_income", "monthly_income", "monthly_spending",
        "existing_debt", "savings_balance", "loan_amount",
        "loan_term_months", "num_credit_lines",
        "num_transactions_per_month", "avg_transaction_amount",
        "txn_transaction_amount_mean", "txn_transaction_amount_sum",
        "txn_transactions_last_hour_mean", "txn_failed_attempts_24h_mean",
        "txn_distance_from_usual_location_mean",
        "txn_transaction_count", "txn_fraud_transaction_count",
    ]

    for col in nonnegative_columns:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce")
            invalid = values.notna() & (values < 0)
            df = df.loc[~invalid].copy()

    for target in ["default", "txn_is_fraud"]:
        if target in df.columns:
            values = pd.to_numeric(df[target], errors="coerce")
            invalid = values.notna() & ~values.isin([0, 1])
            df = df.loc[~invalid].copy()

    print("\nValidation completed.")
    print("Shape after validation:", df.shape)
    return df


def engineer_features(df):
    """
    Keep existing engineered features from loans_merged.csv.
    Add only missing features, using safe division.
    """
    df = df.copy()

    if "monthly_income" not in df.columns and "annual_income" in df.columns:
        df["monthly_income"] = pd.to_numeric(
            df["annual_income"], errors="coerce"
        ) / 12

    def ratio(numerator, denominator):
        den = pd.to_numeric(df[denominator], errors="coerce").replace(0, float("nan"))
        num = pd.to_numeric(df[numerator], errors="coerce")
        return num / den

    ratio_features = {
        "debt_to_loan": ("existing_debt", "loan_amount"),
        "debt_to_income": ("existing_debt", "annual_income"),
        "salary_to_loan": ("annual_income", "loan_amount"),
        "salary_to_savings": ("annual_income", "savings_balance"),
        "spending_to_income": ("monthly_spending", "monthly_income"),
        "savings_to_income": ("savings_balance", "monthly_income"),
    }

    for feature, (num, den) in ratio_features.items():
        if feature not in df.columns and num in df.columns and den in df.columns:
            df[feature] = ratio(num, den)

    if (
        "debt_to_saving_ratio" not in df.columns
        and {"existing_debt", "loan_amount", "savings_balance"}.issubset(df.columns)
    ):
        den = pd.to_numeric(df["savings_balance"], errors="coerce").replace(0, float("nan"))
        df["debt_to_saving_ratio"] = (
            pd.to_numeric(df["existing_debt"], errors="coerce")
            + pd.to_numeric(df["loan_amount"], errors="coerce")
        ) / den

    return df


def prepare_target(df, target):
    """
    Create features and one target at a time.
    'is_fraud' maps to customer-level target 'txn_is_fraud'.
    Customers with unknown fraud labels are excluded from fraud training.
    """
    target_map = {
        "default": "default",
        "is_fraud": "txn_is_fraud",
        "txn_is_fraud": "txn_is_fraud",
        "loan_amount":"loan_amount",
    }
    if target not in target_map:
        raise ValueError("target must be 'default' or 'is_fraud'.")

    target_column = target_map[target]
    if target_column not in df.columns:
        raise KeyError(f"Missing target column: {target_column}")

    data = df.loc[df[target_column].notna()].copy()
    y = pd.to_numeric(data[target_column], errors="raise").astype(int)

     
    excluded = {
        "customer_id",
        "transaction_id",
        "default",
        "is_fraud",
        "txn_is_fraud",
        "txn_fraud_transaction_count",
        "txn_fraud_transaction_rate",
    }
    
    if target == "loan_amount":
        excluded.update({
            "loan_amount",
            "monthly_emi",
            "loan_term_months",
            "salary_to_loan",
            "debt_to_loan",
            "debt_to_saving_ratio",
        })
    X = data.drop(columns=[c for c in excluded if c in data.columns])

    return X, y


def preprocess_split(X_train, X_test):
    """
    Impute using training data only, one-hot encode categoricals, and align
    test columns to training columns. Output is numeric and ready for
    feature selection/model training. Scaling can be fitted later per model.
    """
    X_train = X_train.copy()
    X_test = X_test.copy()

    # Exclude any unexpected identifier columns defensively.
    X_train = X_train.drop(
        columns=[c for c in ["customer_id", "transaction_id"] if c in X_train.columns],
        errors="ignore",
    )
    X_test = X_test.drop(
        columns=[c for c in ["customer_id", "transaction_id"] if c in X_test.columns],
        errors="ignore",
    )

    # Treat object/string/category columns as categorical.
    categorical_cols = X_train.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()
    numeric_cols = [c for c in X_train.columns if c not in categorical_cols]

    # Numeric imputation: medians learned from training data only.
    numeric_medians = {}
    for col in numeric_cols:
        X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
        X_test[col] = pd.to_numeric(X_test[col], errors="coerce")
        median = X_train[col].median()
        if pd.isna(median):
            median = 0.0
        numeric_medians[col] = float(median)
        X_train[col] = X_train[col].fillna(median)
        X_test[col] = X_test[col].fillna(median)

    # Categorical imputation: most frequent training value, or "unknown".
    categorical_modes = {}
    for col in categorical_cols:
        X_train[col] = X_train[col].astype("string").fillna("unknown")
        X_test[col] = X_test[col].astype("string").fillna("unknown")
        modes = X_train[col].mode(dropna=True)

        mode = str(modes.iloc[0]) if not modes.empty else "unknown"
        categorical_modes[col] = mode

        X_train[col] = X_train[col].replace("<NA>", mode)
        X_test[col] = X_test[col].replace("<NA>", mode)

    # One-hot encode and make test columns identical to training columns.
    X_train = pd.get_dummies(X_train, columns=categorical_cols, drop_first=False, dtype=int)
    X_test = pd.get_dummies(X_test, columns=categorical_cols, drop_first=False, dtype=int)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    # Ensure finite numeric data.
    X_train = X_train.replace([float("inf"), float("-inf")], pd.NA).fillna(0)
    X_test = X_test.replace([float("inf"), float("-inf")], pd.NA).fillna(0)

    metadata = {
        "numeric_medians": numeric_medians,
        "categorical_modes": categorical_modes,
        "feature_columns": X_train.columns.tolist(),
    }
    return X_train, X_test, metadata


def split_data(X, y, is_regression=False):
    """Split data; stratified for classification, plain random for regression."""
    return train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=None if is_regression else y,
    )


def save_processed_data(X_train, X_test, y_train, y_test, target, metadata):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(OUTPUT_DIR / f"X_train_{target}.csv", index=False)
    X_test.to_csv(OUTPUT_DIR / f"X_test_{target}.csv", index=False)
    y_train.to_frame(name=target).to_csv(
        OUTPUT_DIR / f"y_train_{target}.csv", index=False
    )
    y_test.to_frame(name=target).to_csv(
        OUTPUT_DIR / f"y_test_{target}.csv", index=False
    )
    with open(OUTPUT_DIR / f"preprocessing_{target}.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def run_target_pipeline(df, target):
    print(f"\n{'=' * 18} TARGET: {target} {'=' * 18}")
    X, y = prepare_target(df, target)

    # Treat loan_amount (and any future continuous target) as regression
    is_regression = target == "loan_amount"

    print("Raw feature matrix:", X.shape)
    print("Target distribution:")
    print(y.value_counts().sort_index())
    print("Target prevalence:")
    print(y.value_counts(normalize=True).sort_index())

    X_train, X_test, y_train, y_test = split_data(X, y, is_regression=is_regression)
    X_train, X_test, metadata = preprocess_split(X_train, X_test)

    save_processed_data(
        X_train, X_test, y_train, y_test, target, metadata
    )

    print("Processed feature matrix:", X_train.shape)
    print("Train rows:", len(X_train))
    print("Test rows:", len(X_test))
    print("Saved processed files to:", OUTPUT_DIR)


def main():
    df = generate_data()
    inspect_data(df)

    df = clean_data(df)
    df = validate_data(df)
    df = engineer_features(df)
    run_target_pipeline(df, "default")
    run_target_pipeline(df, "is_fraud")
    run_target_pipeline(df,"loan_amount")

    


if __name__ == "__main__":
    main()




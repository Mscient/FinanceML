
import pandas as pd
import numpy as np
import json
from pathlib import Path
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import mutual_info_classif, RFE, RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


 

class Select_feature:

    def __init__(self):
        self.X_train_default=pd.read_csv(PROCESSED_DATA_DIR/"X_train_default.csv")
        self.X_test_default=pd.read_csv(PROCESSED_DATA_DIR/"X_test_default.csv")
        self.y_train_default=pd.read_csv(PROCESSED_DATA_DIR/"y_train_default.csv")
        self.y_test_default=pd.read_csv(PROCESSED_DATA_DIR/"y_test_default.csv")
        self.X_train_fraud=pd.read_csv(PROCESSED_DATA_DIR/"X_train_is_fraud.csv")
        self.X_test_fraud=pd.read_csv(PROCESSED_DATA_DIR/"X_test_is_fraud.csv")
        self.y_train_fraud=pd.read_csv(PROCESSED_DATA_DIR/"y_train_is_fraud.csv")
        self.y_test_fraud=pd.read_csv(PROCESSED_DATA_DIR/"y_test_is_fraud.csv")


    def load_target(self,target):
        if target == "default":
            X = self.X_train_default.copy()
            y = self.y_train_default[target]
        elif target == "is_fraud":
            X = self.X_train_fraud.copy()
            y = self.y_train_fraud[target]
        return X, y

    #correlation
    def correlation_feature(self,X,y):
        correlation=X.corrwith(y)
        return correlation.abs().sort_values(ascending=False)



    #mutual information
    def mutual_information(self,X,y):

        mi=mutual_info_classif(
            X,
            y,
            random_state=42
        )
        result=pd.Series(
        mi,
        index=X.columns
        )
        return result.sort_values(ascending=False)

    def rfe_features(self, X, y):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = LogisticRegression(
            penalty="l2",
            max_iter=1000
        )
        selector = RFE(
            estimator=model,
            n_features_to_select=5
        )

        selector.fit(X_scaled, y)
        features = X.columns[selector.support_].tolist()
        return features

    def rfecv_features(self, X, y):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = LogisticRegression(
            penalty="l2",
            max_iter=1000
        )
        cv = StratifiedKFold(5, shuffle=True, random_state=42)
        selector = RFECV(
            estimator=model,
            step=1,
            cv=cv,
            scoring="f1",
            min_features_to_select=1
        )

        selector.fit(X_scaled, y)
        features = X.columns[selector.support_].tolist()
        return features

    def l1_features(self, X, y):
    
        scaler = StandardScaler()
    
        X_scaled = scaler.fit_transform(X)
    
        model = LogisticRegression(
                penalty="l1",
                solver="liblinear",
                C=1.0,
                max_iter=1000
            )
    
        model.fit(X_scaled, y)
    
        coefficients = np.abs(model.coef_[0])
    
        result = pd.Series(
            coefficients,
            index=X.columns).sort_values(ascending=False)
    
        selected_features = result[result > 0].index.tolist()
    
        return result, selected_features
    



    def random_forest_features(self, X, y):
        model = RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                n_jobs=-1
        )
        model.fit(X, y)
        importance = pd.Series(
            model.feature_importances_,
            index=X.columns
        ).sort_values(ascending=False)
        return importance


    def evidence_table(self, correlation, mi, rfe, rfecv, l1, rf):

        features=correlation.index
        table=pd.DataFrame(index=features)

        table['correlation']=correlation
        table['mutual_information']=mi
        table['rf_importance']=rf
        table['l1_coefficient']=l1

        correlation_threshold=correlation.abs().quantile(0.75)
        mi_threshold=mi.quantile(0.75)
        rf_threshold=rf.quantile(0.75)


        table['corr_support']=(
            correlation.abs()>=correlation_threshold
        )

        table["mi_support"] = (
            mi >= mi_threshold
        )

        table["rfe_support"] = (
            table.index.isin(rfe)
        )

        table["rfecv_support"] = (
            table.index.isin(rfecv)
        )

        table["l1_support"] = (
            l1 > 0
        )

        table["rf_support"] = (
            rf >= rf_threshold
        )


        evidence_columns = [
            "corr_support",
            "mi_support",
            "rfe_support",
            "rfecv_support",
            "l1_support",
            "rf_support"
        ]

        table["evidence_count"] = table[evidence_columns].sum(axis=1)
        return table.sort_values( "evidence_count",ascending=False)


    def create_set(self, evidence_table):

        model_based = (
            evidence_table["rfe_support"]
            | evidence_table["rfecv_support"]
            | evidence_table["l1_support"]
            | evidence_table["rf_support"]
        )

        core_mask = (
        (evidence_table["evidence_count"] >= 4)
        & model_based
        )

        set_d = evidence_table[core_mask].index.tolist()

        core_features = evidence_table.index[core_mask].tolist()

        # Strong candidates
        strong_mask = (
            (evidence_table["evidence_count"] >= 3)
            & model_based
            & ~core_mask
        )

        strong_features = evidence_table.index[strong_mask].tolist()

        set_e = strong_features+ core_features
    
        return set_d, set_e

    def run(self, target):

        print("\n======================================")
        print(f"TARGET: {target}")
        print("======================================")

        X, y = self.load_target(target)
        print("Shape:", X.shape)
        print("Missing values:", X.isna().sum().sum())
        print(
            "Infinite values:",
            np.isinf(X.select_dtypes(include=np.number)).sum().sum()
        )

        # Check for potential fraud leakage
        if target == "is_fraud":
            leakage_candidates = [
                col for col in X.columns
                if "fraud" in col.lower()
            ]
            print("Potential leakage columns:", leakage_candidates)
        print("Target missing:", y.isna().sum())
        correlation = self.correlation_feature(X, y)
        mi = self.mutual_information(X, y)
        rfe = self.rfe_features(X, y)
        rfecv = self.rfecv_features(X, y)
        l1, l1_selected = self.l1_features(X, y)
        rf = self.random_forest_features(X, y)

        ev_table = self.evidence_table(
            correlation,
            mi,
            rfe,
            rfecv,
            l1,
            rf
        )
        set_d,set_e = self.create_set(
            ev_table
        )
        print("\nRFE:")
        print(rfe)

        print("\nRFECV:")
        print(rfecv)

        print("\nL1:")
        print(l1_selected)

        print("\nEvidence Table:")
        print(ev_table)

        print("\nSET D:")
        print(set_d)

        print("\nSET_E")
        print(set_e)

        # Save selected feature lists
        FEATURE_DIR = PROCESSED_DATA_DIR / "selected_features"
        FEATURE_DIR.mkdir(parents=True, exist_ok=True)

        with open(FEATURE_DIR / f"features_{target}_set_d.json", "w") as f:
            json.dump(set_d, f, indent=4)

        with open(FEATURE_DIR / f"features_{target}_set_e.json", "w") as f:
            json.dump(set_e, f, indent=4)

        print(f"Feature lists saved to: {FEATURE_DIR}")

        return rfe,rfecv,set_d,set_e


def main():
    selector = Select_feature()

    fraud_rfe, fraud_rfecv, fraud_set_d, fraud_set_e = selector.run(
        "is_fraud"
    )

    default_rfe, default_rfecv, default_set_d, default_set_e = selector.run(
        "default"
    )


if __name__=="__main__":
    main()  

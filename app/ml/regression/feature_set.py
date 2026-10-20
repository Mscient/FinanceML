import pandas as pd
import numpy as np
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.feature_selection import mutual_info_regression, RFE, RFECV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold



def _scale_y(y):
    """Return (y_scaled, y_std) so that y_scaled = y / y_std."""
    std = y.std()
    return y / std, std


class Select_feature:

    def __init__(self):
        self.X_train = pd.read_csv(PROCESSED_DATA_DIR / "X_train_loan_amount.csv")
        self.X_test  = pd.read_csv(PROCESSED_DATA_DIR / "X_test_loan_amount.csv")
        self.y_train = pd.read_csv(PROCESSED_DATA_DIR / "y_train_loan_amount.csv").squeeze()
        self.y_test  = pd.read_csv(PROCESSED_DATA_DIR / "y_test_loan_amount.csv").squeeze()

    def load_target(self, target):
       
        if target == "loan_amount":
            return self.X_train.copy(), self.y_train.copy()
        elif target == "default":
            X = pd.read_csv(PROCESSED_DATA_DIR / "X_train_default.csv")
            y = pd.read_csv(PROCESSED_DATA_DIR / "y_train_default.csv").squeeze()
            return X, y
        elif target == "is_fraud":
            X = pd.read_csv(PROCESSED_DATA_DIR / "X_train_is_fraud.csv")
            y = pd.read_csv(PROCESSED_DATA_DIR / "y_train_is_fraud.csv").squeeze()
            return X, y
        else:
            raise ValueError(f"Unknown target: {target}")

   
    def correlation_feature(self, X, y):
        return X.corrwith(y).abs().sort_values(ascending=False)

    def mutual_information(self, X, y):
        """Mutual information for regression targets."""
        mi = mutual_info_regression(X, y, random_state=42)
        return pd.Series(mi, index=X.columns).sort_values(ascending=False)

    def rfe_features(self, X, y):
        """RFE with Ridge — Ridge is scale-invariant so no y-scaling needed."""
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = Ridge(alpha=1.0, max_iter=1000)
        selector = RFE(estimator=model, n_features_to_select=5)
        selector.fit(X_scaled, y)
        return X.columns[selector.support_].tolist()

    def rfecv_features(self, X, y):
        """RFECV with Ridge and KFold CV (not stratified — this is regression)."""
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = Ridge(alpha=1.0, max_iter=1000)
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        selector = RFECV(
            estimator=model,
            step=1,
            cv=cv,
            scoring="r2",
            min_features_to_select=1,
        )
        selector.fit(X_scaled, y)
        return X.columns[selector.support_].tolist()

    def l1_features(self, X, y):
        """Ridge coefficients as feature importance proxy (Ridge = L2, never zeroes)."""
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = Ridge(alpha=1.0, max_iter=1000)
        model.fit(X_scaled, y)
        result = pd.Series(np.abs(model.coef_), index=X.columns).sort_values(ascending=False)
        selected = result[result > 0].index.tolist()
        return result, selected

    def random_forest_features(self, X, y):
        """Feature importances from a RandomForestRegressor."""
        model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        model.fit(X, y)
        return pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

    # ---- Lasso -------------------------------------------------------- #
    def lasso(self, X, y):
        """
        Lasso-based feature selection.
        y is scaled to unit-std so that alpha=1.0 is meaningful regardless
        of the target's absolute magnitude.
        """
        scaler_x = StandardScaler()
        X_scaled = scaler_x.fit_transform(X)
        # Scale y — critical for coordinate-descent models with large targets
        y_scaled, _ = _scale_y(y)
        model = Lasso(alpha=1.0, max_iter=10000)
        model.fit(X_scaled, y_scaled)
        # RFE
        rfe_sel = RFE(estimator=Lasso(alpha=1.0, max_iter=10000), n_features_to_select=5)
        rfe_sel.fit(X_scaled, y_scaled)
        rfe = X.columns[rfe_sel.support_].tolist()
        # RFECV
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        rfecv_sel = RFECV(
            estimator=Lasso(alpha=1.0, max_iter=10000),
            step=1, cv=cv, scoring="r2", min_features_to_select=1,
        )
        rfecv_sel.fit(X_scaled, y_scaled)
        rfecv = X.columns[rfecv_sel.support_].tolist()

        # Coefficient-based importance (from the already-fitted model)
        coef = pd.Series(np.abs(model.coef_), index=X.columns).sort_values(ascending=False)
        selected = coef[coef > 0].index.tolist()
        return rfe, rfecv, coef, selected

    # ---- ElasticNet --------------------------------------------------- #

    def elastic(self, X, y):
        
        scaler_x = StandardScaler()
        X_scaled = scaler_x.fit_transform(X)

        y_scaled, _ = _scale_y(y)

        model = ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=10000)
        model.fit(X_scaled, y_scaled)

        # RFE
        rfe_sel = RFE(
            estimator=ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=10000),
            n_features_to_select=5,
        )
        rfe_sel.fit(X_scaled, y_scaled)
        rfe = X.columns[rfe_sel.support_].tolist()

        # RFECV
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        rfecv_sel = RFECV(
            estimator=ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=10000),
            step=1, cv=cv, scoring="r2", min_features_to_select=1,
        )
        rfecv_sel.fit(X_scaled, y_scaled)
        rfecv = X.columns[rfecv_sel.support_].tolist()

        coef = pd.Series(np.abs(model.coef_), index=X.columns).sort_values(ascending=False)
        selected = coef[coef > 0].index.tolist()

        return rfe, rfecv, coef, selected    

   
    # Evidence table & feature set construction                           
     

    def evidence_table(self, correlation, mi, rfe, rfecv, l1_coef, rf):
        table = pd.DataFrame(index=correlation.index)
        table["correlation"]        = correlation
        table["mutual_information"] = mi
        table["rf_importance"]      = rf
        table["l1_coefficient"]     = l1_coef

        table["corr_support"]  = correlation.abs() >= correlation.abs().quantile(0.75)
        table["mi_support"]    = mi >= mi.quantile(0.75)
        table["rfe_support"]   = table.index.isin(rfe)
        table["rfecv_support"] = table.index.isin(rfecv)
        table["l1_support"]    = l1_coef > 0
        table["rf_support"]    = rf >= rf.quantile(0.75)

        evidence_cols = ["corr_support", "mi_support", "rfe_support",
                         "rfecv_support", "l1_support", "rf_support"]
        table["evidence_count"] = table[evidence_cols].sum(axis=1)
        return table.sort_values("evidence_count", ascending=False)

    def create_set(self, ev_table):
        model_based = (
            ev_table["rfe_support"] | ev_table["rfecv_support"]
            | ev_table["l1_support"] | ev_table["rf_support"]
        )
        core_mask   = (ev_table["evidence_count"] >= 4) & model_based
        strong_mask = (ev_table["evidence_count"] >= 3) & model_based & ~core_mask
        set_d = ev_table.index[core_mask].tolist()
        set_e = ev_table.index[strong_mask].tolist() + set_d
        return set_d, set_e
 
    # Main pipeline                                                        #
    
    def run(self, target):
        print("\n======================================")
        print(f"TARGET: {target}")
        print("======================================")

        X, y = self.load_target(target)
        print("Shape:", X.shape)
        print("Missing values:", X.isna().sum().sum())
        print("Infinite values:", np.isinf(X.select_dtypes(include=np.number)).sum().sum())
        print("Target missing:", y.isna().sum())

        # Ridge-based baseline
        correlation       = self.correlation_feature(X, y)
        mi                = self.mutual_information(X, y)
        rfe_ridge         = self.rfe_features(X, y)
        rfecv_ridge       = self.rfecv_features(X, y)
        l1_coef, l1_sel   = self.l1_features(X, y)
        rf                = self.random_forest_features(X, y)

        print("\n--- Ridge ---")
        ev_ridge = self.evidence_table(correlation, mi, rfe_ridge, rfecv_ridge, l1_coef, rf)
        set_d, set_e = self.create_set(ev_ridge)
        print("RFE:", rfe_ridge)
        print("RFECV:", rfecv_ridge)
        print("SET D:", set_d)
        print("SET E:", set_e)

        # Lasso
        print("\n--- Lasso ---")

        rfe_l, rfecv_l, coef_l, sel_l = self.lasso(X, y)
        ev_lasso = self.evidence_table(correlation, mi, rfe_l, rfecv_l, coef_l, rf)
        set_lasso_d, set_lasso_e = self.create_set(ev_lasso)

        print("RFE:", rfe_l)
        print("RFECV:", rfecv_l)
        print("SET D:", set_lasso_d)
        print("SET E:", set_lasso_e)

        # ElasticNet
        print("\n--- ElasticNet ---")

        rfe_e, rfecv_e, coef_e, sel_e = self.elastic(X, y)
        ev_elastic = self.evidence_table(correlation, mi, rfe_e, rfecv_e, coef_e, rf)
        set_elastic_d, set_elastic_e = self.create_set(ev_elastic)

        print("RFE:", rfe_e)
        print("RFECV:", rfecv_e)
        print("SET D:", set_elastic_d)
        print("SET E:", set_elastic_e)

        # Save feature lists
        FEATURE_DIR = PROCESSED_DATA_DIR / "selected_features"
        FEATURE_DIR.mkdir(parents=True, exist_ok=True)

        for name, feat_set in [
            (f"features_{target}_ridge_set_d",   set_d),
            (f"features_{target}_ridge_set_e",   set_e),
            (f"features_{target}_lasso_set_d",   set_lasso_d),
            (f"features_{target}_lasso_set_e",   set_lasso_e),
            (f"features_{target}_elastic_set_d", set_elastic_d),
            (f"features_{target}_elastic_set_e", set_elastic_e),
        ]:
            with open(FEATURE_DIR / f"{name}.json", "w") as f:
                json.dump(feat_set, f, indent=4)

        print(f"\nFeature lists saved to: {FEATURE_DIR}")
        return set_d, set_e, set_lasso_d, set_lasso_e, set_elastic_d, set_elastic_e


def main():
    selector = Select_feature()
    results = selector.run("loan_amount")
    print("\nDone.")


if __name__ == "__main__":
    main()

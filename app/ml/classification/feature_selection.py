# from openpyxl.worksheet import header_footer
# from scipy.spatial.distance import correlation
# import pandas as pd
# from train_random import Random 
# from train_logistic import LogisticRegressionModel
# from sklearn.linear_model import LogisticRegression
# from app.core.config import X_TRAIN_PATH, X_TEST_PATH, Y_TRAIN_PATH, Y_TEST_PATH
# from scipy.stats import pearsonr, spearmanr
# from sklearn.feature_selection import mutual_info_classif
# from sklearn.feature_selection import RFE,RFECV
# from sklearn.model_selection import StratifiedKFold
# from sklearn.feature_selection import SelectFromModel
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import Pipeline



# class Select_feature:

#     def __init__(self):
#         self.X_train=pd.read_csv("BASE_PATH/")


# def is_fraud():
#     model=Random()
#     X_train,X_test,y_train,y_test=model.load_data()
#     y_train=y_train['is_fraud']
#     y_test=y_test['is_fraud']
#     model.train(X_train,y_train)
#     dist1=model.feature(X_train)

#     #correlation
#     correlation1 = {}
#     for col in X_train.columns:
#         pearson_corr, _ = pearsonr(X_train[col], y_train)
#         spearman_corr, _ = spearmanr(X_train[col], y_train)
#         correlation1[col] = {
#             "pearson": pearson_corr,
#             "spearman": spearman_corr
#         }
    
#     #Mutual Information
#     mi_scores = mutual_info_classif(X_train, y_train, discrete_features='auto', random_state=42)
#     mi_df = pd.Series(mi_scores, index=X_train.columns).sort_values(ascending=False)
    
#     #rfe
#     rfe = RFE(model.get_model(), n_features_to_select=5, step=1)
#     rfe.fit(X_train, y_train)
#     selected = X_train.columns[rfe.support_].tolist()
    
#     #rfcv
#     cv = StratifiedKFold(5, shuffle=True, random_state=42)

#     rfecv = RFECV(
#     estimator=model.get_model(),
#     step=1,                        
#     cv=cv,
#     scoring='f1',                 
#     min_features_to_select=1
#         )      


#     rfecv.fit(X_train, y_train)

#     best_features = X_train.columns[rfecv.support_].tolist()
#     optimal_k = rfecv.n_features_

#      # L1 logistic regresion
#     pipe = Pipeline([
#     ('scaler', StandardScaler()),
#     ('selector', SelectFromModel(
#         LogisticRegression(penalty='l1', C=0.1, solver='liblinear', max_iter=1000)
#     ))
#         ])

#     selected_mask = pipe.named_steps['selector'].get_support()
#     selected_names = X_train.columns[selected_mask]

#     return correlation1,mi_df,selected,best_features,selected_names


# def default():
#     model=Random()
#     X_train,X_test,y_train,y_test=model.load_data()
#     y_train=y_train['default']
#     y_test=y_test['default']
#     model.train(X_train,y_train)
#     dist2=model.feature(X_train)

#     #correlation
#     correlation2 = {}
#     for col in X_train.columns:
#         pearson_corr, _ = pearsonr(X_train[col], y_train)
#         spearman_corr, _ = spearmanr(X_train[col], y_train)
#         correlation2[col] = {
#             "pearson": pearson_corr,
#             "spearman": spearman_corr
#         }

#     #Mutual Information
#     mi_scores = mutual_info_classif(X_train, y_train, discrete_features='auto', random_state=42)
#     mi_df = pd.Series(mi_scores, index=X_train.columns).sort_values(ascending=False)


#     #rfe
#     rfe = RFE(model.get_model(), n_features_to_select=5, step=1)
#     rfe.fit(X_train, y_train)
#     selected = X_train.columns[rfe.support_].tolist()

    
#     #rfcv
#     cv = StratifiedKFold(5, shuffle=True, random_state=42)

#     rfecv = RFECV(
#     estimator=model.get_model(),
#     step=1,                        
#     cv=cv,
#     scoring='f1',                 
#     min_features_to_select=1
#         )      
#     rfecv.fit(X_train, y_train)
#     best_features = X_train.columns[rfecv.support_].tolist()
#     optimal_k = rfecv.n_features_

#     # L1 logistic regresion
#     pipe = Pipeline([
#     ('scaler', StandardScaler()),
#     ('selector', SelectFromModel(
#         LogisticRegression(penalty='l1', C=0.1, solver='liblinear', max_iter=1000)
#     ))
#         ])
#     selected_mask = pipe.named_steps['selector'].get_support()
#     selected_names = X_train.columns[selected_mask]

#     return correlation2,mi_df,selected,best_features,selected_names


# def main():
#     print(" hELLO")
#     correlation1,mi_df1,rfe1,best_features1,selected_names1=is_fraud()
#     correlation2,mi_df2,rfe2,best_features2,selected_names2=default()
#     #print("correlation1:",correlation1)
#     #print("correlation2:",correlation2)
#     print("is_fraud")
#     #print("mi_scores1:",mi_df1)
#     print('\n')
#     #print("rfe1:",rfe1)
#     print('\n')
#     print("default")
#     #print("mi_scores2:",mi_df2)
#     print('\n')
#     #print("rfe2:",rfe2)
#     print('\n')
#     #print("best_features1:",best_features1)
#     print('\n')
#     #print("best_features2:",best_features2)
#     print('\n')
#     print("Selected Features:",selected_names1)
#     print("Selected Features:",selected_names2)

#     _fraud={
#         "correlation":correlation1,
#         "mi":mi_df1,
#         "rfe":rfe1,
#         "rfecv":best_features1,
#         "selected_names":selected_names1
#     }

#     _default={
#         "correlation":correlation2,
#         "mi":mi_df2,
#         "rfe":rfe2,
#         "rfecv":best_features2,
#         "selected_names":selected_names2
#     }

    
#     return _fraud,_default
   



# if __name__=="__main__":
#     main()



import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import mutual_info_classif, RFE, RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


BASE_PATH=Path(__file__).parent.parent.parent/"data"


class Select_feature:

    def __init__(self):
        self.X_train=pd.read_csv(BASE_PATH/"X_train.csv")
        self.y_train=pd.read_csv(BASE_PATH/"y_train.csv")

    def load_target(self,target):
        X=self.X_train.copy()
        y=self.y_train[target]

        return X,y

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

        return ev_table, set_d,set_e

        
if __name__ == "__main__":

    selector = Select_feature()

    fraud_table, fraud_set_d, fraud_set_e = selector.run(
        "is_fraud"
    )

    default_table, default_set_d, default_set_e = selector.run(
        "default"
    )
    






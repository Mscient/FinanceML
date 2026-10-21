import pandas as pd
import numpy as np
import json
from pathlib import Path
from feature_selection import Select_feature
from train_logistic import LogisticRegressionModel
from train_random import Random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score

class feature_validation:

    def __init__(self):
        DATA_DIR = Path(__file__).resolve().parents[2] / "data"
        PROCESSED_DATA_DIR = DATA_DIR / "processed"


        #default
        self.X_train_default = pd.read_csv(PROCESSED_DATA_DIR / "X_train_default.csv")
        self.X_test_default  = pd.read_csv(PROCESSED_DATA_DIR / "X_test_default.csv")
        self.y_train_default = pd.read_csv(PROCESSED_DATA_DIR / "y_train_default.csv")
        self.y_test_default  = pd.read_csv(PROCESSED_DATA_DIR / "y_test_default.csv")

        # is_fraud
        self.X_train_fraud = pd.read_csv(PROCESSED_DATA_DIR / "X_train_is_fraud.csv")
        self.X_test_fraud  = pd.read_csv(PROCESSED_DATA_DIR / "X_test_is_fraud.csv")
        self.y_train_fraud = pd.read_csv(PROCESSED_DATA_DIR / "y_train_is_fraud.csv")
        self.y_test_fraud  = pd.read_csv(PROCESSED_DATA_DIR / "y_test_is_fraud.csv")
        self.feature_dir = PROCESSED_DATA_DIR/"selected_features"

        self.set_a=self.X_train_fraud.columns.tolist()
        self.set_b=self.X_train_default.columns.tolist()


    
    def validation(self, feature_set, target):

        if target=="is_fraud":
            X_all = self.X_train_fraud
            y=self.y_train_fraud["is_fraud"]
            
        elif target=="default":
            X_all = self.X_train_default
            y=self.y_train_default['default']

        X = X_all[feature_set]

        X_train_main, X_val, y_train_main, y_val = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )
        
        return X_train_main, y_train_main, X_val, y_val

        
    def analyze_threshold(self,y_val,y_proba):
        threshold=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
        results=[]
        for i in threshold:
            pred=(y_proba>=i).astype(int)
            precision = precision_score(y_val, pred, zero_division=0)
            recall= recall_score(y_val, pred, zero_division=0)

            tn,fp,fn,tp=confusion_matrix(y_val,pred,labels=[0,1]).ravel()

            f1= f1_score(y_val, pred, zero_division=0)
            results.append({
                "precison":precision,
                "recall":recall,
                "f1":f1,
                "tn":int(tn),
                "fp":int(fp),
                "fn":int(fn),
                "tp":int(tp)
            })

        threshold=pd.DataFrame(results)
        return threshold
    
    def train_model(self, model, train_x, train_y, test_x, test_y,target):
        model=model
        from train_logistic import LogisticRegressionModel as LR

        if isinstance(model, LR):
            scaler = StandardScaler()
            train_x = scaler.fit_transform(train_x.values if hasattr(train_x, 'values') else train_x)
            test_x  = scaler.transform(test_x.values if hasattr(test_x, 'values') else test_x)
        model.train(target,model,train_x,train_y)
        pred      = model.predict(test_x)
        accuracy  = accuracy_score(test_y, pred)
        precision = precision_score(test_y, pred, zero_division=0)
        recall    = recall_score(test_y, pred, zero_division=0)
        f1        = f1_score(test_y, pred, zero_division=0)
        conf      = confusion_matrix(test_y, pred)
        proba= model.predict_proba(test_x)
        threshold=[]
        if(target=='is_fraud'):  
            threshold=self.analyze_threshold(test_y,proba[:,1])
        pr_auc = average_precision_score(test_y, proba[:,1])
        return accuracy, precision, recall, f1, conf,pr_auc,threshold

def validate_feature_sets(fv, target, feature_sets):
    print(f"\n{'='*50}")
    print(f"TARGET: {target}")
    print(f"{'='*50}")
    set_names = list(feature_sets.keys())
    splits = {}
    for name, cols in feature_sets.items():
        splits[name] = fv.validation(cols, target)

    # ── Logistic Regression 
    print(f"\n--- Logistic Regression | target={target} ---")
    logistic = LogisticRegressionModel()
    for name in set_names:
        tr_x, tr_y, te_x, te_y = splits[name]
        acc, prec, rec, f1, conf,pr_auc,threshold = fv.train_model(logistic, tr_x, tr_y, te_x, te_y,target)
        print(f"  set_{name}  acc:{acc:.4f}  prec:{prec:.4f}  rec:{rec:.4f}  f1:{f1:.4f} pr_auc:{pr_auc:.4f}") 
        print(f"  conf:\n{conf}\n")
        if(target=='is_fraud'):
            print(threshold)
        


    # ── Random Forest   
    print(f"--- Random Forest       | target={target} ---")
    random = Random()
    for name in set_names:
        tr_x, tr_y, te_x, te_y = splits[name]
        acc, prec, rec, f1, conf,pr_auc,threshold = fv.train_model(random, tr_x, tr_y, te_x, te_y,target)
        print(f"  set_{name}  acc:{acc:.4f}  prec:{prec:.4f}  rec:{rec:.4f}  f1:{f1:.4f} pr_auc:{pr_auc:.4f}")
        print(f"  conf:\n{conf}\n")
        if(target=='is_fraud'):
            print(threshold)


def main():
    fv       = feature_validation()
    selector = Select_feature()

    set_a = fv.set_a
    set_b = fv.set_b
    set_b1, set_c1, set_d1, set_e1 = selector.run("is_fraud")
    set_b2, set_c2, set_d2, set_e2 = selector.run("default")

    fraud_feature_sets   = {"a": set_a, "b": set_b1, "c": set_c1, "d": set_d1, "e": set_e1}
    default_feature_sets = {"a": set_b, "b": set_b2, "c": set_c2, "d": set_d2, "e": set_e2}


    print("SET E")
    fraud_feature_sets={"e":set_e1}
    default_feature_sets={"e":set_e2}
    

    validate_feature_sets(fv, "is_fraud", fraud_feature_sets)
    validate_feature_sets(fv, "default",  default_feature_sets)

    print("SET_D")
    fraud_feature_sets={"d":set_d1}
    default_feature_sets={"d":set_d2}
    validate_feature_sets(fv, "is_fraud", fraud_feature_sets)
    validate_feature_sets(fv, "default",  default_feature_sets)
 

    
if __name__ == "__main__":
    main()
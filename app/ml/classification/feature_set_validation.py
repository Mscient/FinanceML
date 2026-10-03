import pandas as pd
import numpy as np
from pathlib import Path
from feature_selection import Select_feature
from train_random import Random
from train_logistic import LogisticRegressionModel
from sklearn.metrics import accuracy_score, precision_score,recall_score,f1_score,confusion_matrix
 


class feature_validation:

    def __init__(self):
        self.X_train=pd.read_csv("BASE_PATH/X_train.csv")
        self.X_test=pd.read_csv("BASE_PATH/X_test.csv")
        self.y_train=pd.read_csv("BASE_PATH/y_train.csv")
        self.y_test=pd.read_csv("BASE_PATH/y_test.csv")   

    def load_data(self):
        set_a=self.X_train.columns.tolist()
        set_b1,set_c1, set_d1,set_e1=Select_feature.run("is_fraud")
        set_b2,set_c2, set_d2,set_e2=Select_feature.run("default")

    def validation(self,set,target):
         
        train_x=self.X_train[set]
        train_y=self.y_train[target]

        test_x=self.X_test[set]
        test_y=self.y_test[target]

        return train_x,train_y,test_x,test_y

    def train_model(self,model,train_x,train_y,test_x,test_y):

        model.fit(train_x,train_y)
        pred=model.predict(test_x)
        accuracy=accuracy_score(test_y,pred)
        precision=precision_score(test_y,pred)
        recall=recall_score(test_y,pred)
        f1=f1_score(test_y,pred)
        confusion=confusion_matrix(test_y,pred)
        return accuracy,precision,recall,f1,confusion 

         
        
def main():

    set_b1,set_c1, set_d1,set_e1=Select_feature.run("is_fraud")
    set_b2,set_c2, set_d2,set_e2=Select_feature.run("default")

    #is_fraud
    split_data(set_b1,"is_fraud")
    split_data(set_c1,"is_fraud")

    split_data(set_d1,"is_fraud")
    split_data(set_e1,"is_fraud")

    #logistic
    logistic=LogisticRegressionModel()
    acc1,prec1,rec1,f11,conf1=train_model(logistic,set_b1,"is_fraud")
    acc2,prec2,rec2,f12,conf2=train_model(logistic,set_c1,"is_fraud")
    acc3,prec3,rec3,f13,conf3=train_model(logistic,set_d1,"is_fraud")
    acc4,prec4,rec4,f14,conf4=train_model(logistic,set_e1,"is_fraud")

    #random
    random=Random()
    acc1,prec1,rec1,f11,conf1=train_model(random,set_b1,"is_fraud")
    acc2,prec2,rec2,f12,conf2=train_model(random,set_c1,"is_fraud")
    acc3,prec3,rec3,f13,conf3=train_model(random,set_d1,"is_fraud")
    acc4,prec4,rec4,f14,conf4=train_model(random,set_e1,"is_fraud")

    #default
    split_data(set_b2,"default")
    split_data(set_c2,"default")

    split_data(set_d2,"default")
    split_data(set_e2,"default")

    
    #logistic
    logistic=LogisticRegressionModel()
    acc1,prec1,rec1,f11,conf1=train_model(logistic,set_b1,"default")
    acc2,prec2,rec2,f12,conf2=train_model(logistic,set_c1,"default")
    acc3,prec3,rec3,f13,conf3=train_model(logistic,set_d1,"default")
    acc4,prec4,rec4,f14,conf4=train_model(logistic,set_e1,"default")

    #random
    random=Random()
    acc1,prec1,rec1,f11,conf1=train_model(random,set_b1,"default")
    acc2,prec2,rec2,f12,conf2=train_model(random,set_c1,"default")
    acc3,prec3,rec3,f13,conf3=train_model(random,set_d1,"default")
    acc4,prec4,rec4,f14,conf4=train_model(random,set_e1,"default")



if __name__=="__main__":
    main()



        


        


    
        

import pandas as pd
from pathlib import Path


def generate_data():
    pass


def load_data(path):
    df = pd.read_csv(path)
    return df

def inspect_data(df):
   
    df.info()
    print(df.columns.tolist())
    print(df.dtypes)
    print(df.describe())
    print(df.duplicated().sum())
    missing_values=df.isna().sum()
    columns = df.shape[1]
    rows = df.shape[0]
    print(f"\nShape: {rows} rows, {columns} columns")
    print(df['employment_type'].value_counts())
    print(df['loan_purpose'].value_counts())
    print(df['region'].value_counts())
    print(df['default'].value_counts())
    print(df['is_fraud'].value_counts())



def validate_data(df):
    df = df.drop_duplicates()
    df = df.dropna()
        #removing invalid age values
    invalid_age=((df['age']<18) | (df['age']>100))
    df=df[~invalid_age] 
    #removing invalid credi score 
    invalid_credit_score=((df['credit_score']<300) | (df['credit_score']>850))
    df=df[~invalid_credit_score]
    
    #removing invalid values.
    invalid_dti=((df['dti']<0) | (df['dti']>1))
    df=df[~invalid_dti]

    #removing invalid values.
    invalid_monthly_spending=(df['monthly_spending']<0)
    df=df[~invalid_monthly_spending]

    #removing invalid values.
    invalid_annual_income=(df['annual_income']<0)
    df=df[~invalid_annual_income]

    invalid_years_employed=(df['years_employed']<0) |(df['years_employed']>80)
    df=df[~invalid_years_employed]

    invalid_existing_debt=(df['existing_debt']<0)
    df=df[~invalid_existing_debt]

    invalid_savings_balance=(df['savings_balance']<0)
    df=df[~invalid_savings_balance]

    invalid_loan_amount=(df['loan_amount']<0)
    df=df[~invalid_loan_amount]

    invalid_loan_term_months=(df['loan_term_months']<0)
    df=df[~invalid_loan_term_months]

    invalid_num_credit_lines=(df['num_credit_lines']<0)
    df=df[~invalid_num_credit_lines]

    invalid_num_transactions_per_month=(df['num_transactions_per_month']<0)
    df=df[~invalid_num_transactions_per_month]

    invalid_avg_transaction_amount=(df['avg_transaction_amount']<0)
    df=df[~invalid_avg_transaction_amount]

    invalid_online_transactions_ratio=(df['online_transactions_ratio']<0) |(df['online_transactions_ratio']>1)
    df=df[~invalid_online_transactions_ratio]


    print("validation completed.")
    print("Shape: ",df.shape)

    return df


def clean_data(df):

    numeric_columns=df.select_dtypes(include=['int64','float64']).columns.tolist()
    categorical_columns=df.select_dtypes(include=['object']).columns.tolist()

    #cleaning numeric columns
    for coulmns in numeric_columns:
        
        df[coulmns]=pd.to_numeric(df[coulmns],errors='coerce')

    #cleaning categrical values
    for coulmns in categorical_columns:
        df[coulmns]=df[coulmns].str.lower().str.strip()
        df[coulmns]=df[coulmns].astype('object',errors='ignore')

    
    return df


def enginer_features(df):

    #credit_util
    df['credit_util']=df['existing_debt']/df['credit_score']

    #debt_to_loan
    df['debt_to_loan']=df['existing_debt']/df['loan_amount']

    #debt_to_income
    df['debt_to_income']=df['existing_debt']/df['annual_income']

    #salary_to_loan
    df['salary_to_loan']=df['annual_income']/df['loan_amount']

    #salary_to_savings
    df['salary_to_savings']=df['annual_income']/df['savings_balance']

    #monthly_income
    df['monthly_income']=(df['annual_income'])/12

    #debt_to_saving ratio
    df['debt_to_saving_ratio']=(df['existing_debt']+df['loan_amount'])/df['savings_balance']

    #spending_to_income
    df['spending_to_income']=(df['monthly_spending'])/(df['monthly_income'])

    #savings_to_income
    df['savings_to_income']=(df['savings_balance'])/(df['monthly_income'])

    df.head()
    df.shape
    print(df.columns.tolist())
    
    return df



def encode_categorical_features(df):
    df=df.copy()
    df.shape
    df=pd.get_dummies(df,columns=['employment_type','loan_purpose','region'],drop_first=True)
    print(df.shape)
    print(df.columns.tolist())
    return df


def prepare_target(df):
    X=df.copy()
    target_columns=['is_fraud','default']
    
    y=X[target_columns]
    X.drop(columns=target_columns,inplace=True)
    return X,y


def split_Data(X,y):
    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
    return X_train,X_test,y_train,y_test


def save_processed_data(X_train,X_test,y_train,y_test):
    X_train.to_csv(Path(__file__).parent / "X_train.csv",index=False)
    X_test.to_csv(Path(__file__).parent / "X_test.csv",index=False)
    y_train.to_csv(Path(__file__).parent / "y_train.csv",index=False)
    y_test.to_csv(Path(__file__).parent / "y_test.csv",index=False)


def main():
    data_path = Path(__file__).parent / "loans.csv"
    df = load_data(data_path)
    inspect_data(df)
    df = validate_data(df)
    df = clean_data(df)
    df = enginer_features(df)
    df = encode_categorical_features(df)
    X,y = prepare_target(df)
    X_train,X_test,y_train,y_test = split_Data(X,y)
    save_processed_data(X_train,X_test,y_train,y_test)


if __name__ == "__main__":
    main()

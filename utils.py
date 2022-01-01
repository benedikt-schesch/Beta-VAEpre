import pandas as pd
import category_encoders as ce
import numpy as np
import random

def load_data(path):
    if path == "xAPI-Edu-Data.csv":
        df = pd.read_csv(path)
        one_hot_columns = list(df.columns)
        del one_hot_columns[9:13]
        del one_hot_columns[-1]
        encoder_var = ce.OneHotEncoder(cols=one_hot_columns,return_df=True,use_cat_names=True)
        new_df = encoder_var.fit_transform(df[one_hot_columns])
        feature_df = new_df.join(df[df.columns[9:13]])
        X = feature_df.to_numpy()
        Y = (df['Class'] != "L").to_numpy().astype(int)
    elif path == "student-por.csv":
        df = pd.read_csv(path,sep=";")
        one_hot_columns = ["school","sex","address","famsize","Pstatus","Mjob","Fjob","guardian","schoolsup","famsup","paid",\
            "nursery","internet","activities","higher","romantic","reason"]
        encoder_var = ce.OneHotEncoder(cols=one_hot_columns,return_df=True,use_cat_names=True)
        feature_df = encoder_var.fit_transform(df[one_hot_columns])
        feature_df = feature_df.join(df[[col for col in list(df.columns) if col not in one_hot_columns]])
        X = feature_df.to_numpy()
        Y = (df["G3"]>=10).to_numpy().astype(int)
    return X, Y

def balanced_train_test_generator(X,Y):
    at_risk_student_id = np.nonzero(Y == 0)[0]
    non_at_risk_student_id = np.nonzero(Y == 1)[0]

    X_test_risk_id = np.array(random.sample(list(at_risk_student_id), int(len(at_risk_student_id)*0.25)))
    X_test_non_risk_id = np.array(random.sample(list(non_at_risk_student_id), int(len(at_risk_student_id)*0.25)))
    X_train_risk_id = [x for x in at_risk_student_id if x not in X_test_risk_id]
    X_train_non_risk_id = [x for x in non_at_risk_student_id if x not in X_test_non_risk_id]

    X_train = X[np.concatenate((X_train_risk_id,X_train_non_risk_id))].astype(np.float32)
    X_test = X[np.concatenate((X_test_risk_id,X_test_non_risk_id))].astype(np.float32)
    y_train = Y[np.concatenate((X_train_risk_id,X_train_non_risk_id))]
    y_test = Y[np.concatenate((X_test_risk_id,X_test_non_risk_id))]
    return X_train, X_test, y_train, y_test

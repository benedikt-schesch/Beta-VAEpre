import pandas as pd
import category_encoders as ce
import numpy as np
import random
import os
import numpy as np
import torch

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
        feature_df = feature_df.join(df[[col for col in list(df.columns)[:-1] if col not in one_hot_columns]])
        X = feature_df.to_numpy()
        Y = (df["G3"]>=10).to_numpy().astype(int)
    return X, Y, feature_df

def seed_everything(seed: int):    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

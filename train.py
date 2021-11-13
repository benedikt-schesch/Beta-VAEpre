import pandas as pd
import category_encoders as ce


dataset_path = "xAPI-Edu-Data.csv"

df = pd.read_csv(dataset_path)

if dataset_path == "xAPI-Edu-Data.csv":
    one_hot_columns = list(df.columns)
    del one_hot_columns[9:13]
    encoder_var = ce.OneHotEncoder(cols=one_hot_columns,return_df=True,use_cat_names=True)
    new_df = encoder_var.fit_transform(df[one_hot_columns])
    df = new_df.join(df[one_hot_columns[9:13]])
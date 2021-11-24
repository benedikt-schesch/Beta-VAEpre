import pandas as pd
import category_encoders as ce
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
from sklearn import preprocessing
from trainers import *


args = {}
args["batch_size"] = 32
args["epochs"] = 40
args["augmentation_factor"] = 2
args["lr"] = 0.001
args["beta"] = 1
args["dropout"] = 0
args["neurons_num"] = [48,32]
args["weight_decay"] = 1e-5


dataset_path = "xAPI-Edu-Data.csv"

df = pd.read_csv(dataset_path)

if dataset_path == "xAPI-Edu-Data.csv":
    one_hot_columns = list(df.columns)
    del one_hot_columns[9:13]
    del one_hot_columns[-1]
    encoder_var = ce.OneHotEncoder(cols=one_hot_columns,return_df=True,use_cat_names=True)
    new_df = encoder_var.fit_transform(df[one_hot_columns])
    feature_df = new_df.join(df[df.columns[9:13]])
    X = feature_df.to_numpy()
    Y = (df['Class'] != "L").to_numpy().astype(int)


print("Number of datapoint: ",len(X))
print("Data Imbalance: ",100*(1-sum(Y)/len(Y)),"%")

X = preprocessing.normalize(X)
X_train, X_test, y_train, y_test = train_test_split(X.astype(np.float32),Y,test_size=0.25, stratify = Y,random_state=0)
print("Data Train Imbalance: ",100*(1-sum(y_train)/len(y_train)),"%")
print("Data Test Imbalance: ",100*(1-sum(y_test)/len(y_test)),"%")


at_risk_student_id = np.nonzero(y_train == 0)[0]
X_at_risk_student = X_train[at_risk_student_id]

X_at_risk_augmented = augment_betaVAE(X_at_risk_student.astype(np.float32),args=args)

X_train_augmented = np.concatenate([X_at_risk_augmented,np.stack([X_train[i] for i in range(len(X_train)) if i not in at_risk_student_id])])
y_train_augmented = np.array([0 for i in X_at_risk_augmented]+[1 for i in range(len(X_train)) if i not in at_risk_student_id])
args["neurons_num"] = [len(X_train_augmented[0])]+args["neurons_num"]
print("Data Augmented Train Imbalance: ",100*(1-sum(y_train_augmented)/len(y_train_augmented)),"%")

final_acc = train_and_eval_NN(X_train_augmented,y_train_augmented,X_test,y_test,args)

print("Final Accuracy ",final_acc)
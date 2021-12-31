import pandas as pd
import category_encoders as ce
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
from sklearn import preprocessing
from trainers import *
import random

SEED = 0
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

args = {}
args["batch_size"] = 8
args["epochs"] = 1000
args["augmentation_factor"] = 3
args["lr"] = 0.001
args["beta"] = 1
args["dropout"] = 0
args["neurons_num"] = [48,32]
args["weight_decay"] = 0


dataset_path = "xAPI-Edu-Data.csv"

if dataset_path == "xAPI-Edu-Data.csv":
    df = pd.read_csv(dataset_path)
    one_hot_columns = list(df.columns)
    del one_hot_columns[9:13]
    del one_hot_columns[-1]
    encoder_var = ce.OneHotEncoder(cols=one_hot_columns,return_df=True,use_cat_names=True)
    new_df = encoder_var.fit_transform(df[one_hot_columns])
    feature_df = new_df.join(df[df.columns[9:13]])
    X = feature_df.to_numpy()
    Y = (df['Class'] != "L").to_numpy().astype(int)
elif dataset_path == "student-por.csv":
    df = pd.read_csv(dataset_path,sep=";")
    one_hot_columns = ["school","sex","address","famsize","Pstatus","Mjob","Fjob","guardian","schoolsup","famsup","paid",\
           "nursery","internet","activities","higher","romantic","reason"]
    encoder_var = ce.OneHotEncoder(cols=one_hot_columns,return_df=True,use_cat_names=True)
    feature_df = encoder_var.fit_transform(df[one_hot_columns])
    X = feature_df.to_numpy()
    Y = (df["G3"]>=10).to_numpy().astype(int)


print("Number of datapoint: ",len(X))
print("Data Imbalance: ",100*(1-sum(Y)/len(Y)),"%")

X = preprocessing.normalize(X)
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

#X_train, X_test, y_train, y_test = train_test_split(X.astype(np.float32),Y,test_size=0.25, stratify = Y,random_state=SEED)
print("Data Train Imbalance: ",100*(1-sum(y_train)/len(y_train)),"%")
print("Data Test Imbalance: ",100*(1-sum(y_test)/len(y_test)),"%")
print("Proportion of test size: ",100*len(y_test)/(len(y_test)+len(y_train))," %")


at_risk_student_id = np.nonzero(y_train == 0)[0]
X_at_risk_student = X_train[at_risk_student_id]

X_at_risk_augmented, _, _ = augment_betaVAE(X_at_risk_student.astype(np.float32),args=args)

X_train_augmented = np.concatenate([X_at_risk_augmented,np.stack([X_train[i] for i in range(len(X_train)) if i not in at_risk_student_id])])
y_train_augmented = np.array([0 for i in X_at_risk_augmented]+[1 for i in range(len(X_train)) if i not in at_risk_student_id])
args["neurons_num"] = [len(X_train_augmented[0])]+args["neurons_num"]
print("Data Augmented Train Imbalance: ",100*(1-sum(y_train_augmented)/len(y_train_augmented)),"%")

final_acc = train_and_eval_NN(X_train_augmented,y_train_augmented,X_test,y_test,args)
final_acc2 = train_and_eval_NN(X_train,y_train,X_test,y_test,args)


print("Final Scores with augmentation",final_acc)
print("Final Scores w/o augmentation ",final_acc2)
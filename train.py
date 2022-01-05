import numpy as np
from sklearn import preprocessing
from utils import *
from trainers import *
from sklearn.model_selection import train_test_split
import random

SEED = 0
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

args = {}
args["batch_size"] = 8
args["epochs"] = 1
args["augmentation_factor"] = 3
args["lr"] = 0.001
args["beta"] = 1
args["dropout"] = 0
args["neurons_num"] = [48,32]
args["weight_decay"] = 0


dataset_path = "student-por.csv"

X, Y = load_data(dataset_path)

print("Number of datapoint: ",len(X))
print("Data Imbalance: ",100*(1-sum(Y)/len(Y)),"%")

X = preprocessing.normalize(X)

X_train, X_test, y_train, y_test = balanced_train_test_generator(X,Y)
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

final_acc3 = train_and_eval_NN_undersampling(X_train,y_train,X_test,y_test,args)
final_acc3 = train_and_eval_NN_oversampling(X_train,y_train,X_test,y_test,args)
final_acc3 = train_and_eval_NN_class_weights(X_train,y_train,X_test,y_test,args)
final_acc = train_and_eval_NN(X_train_augmented,y_train_augmented,X_test,y_test,args)
final_acc2 = train_and_eval_NN(X_train,y_train,X_test,y_test,args)


print("Final Scores with augmentation",final_acc)
print("Final Scores w/o augmentation ",final_acc2)
import numpy as np
from sklearn import preprocessing
from utils import *
from trainers import *
from sklearn.model_selection import StratifiedKFold

SEED = 42
seed_everything(SEED)

args = {}
args["batch_size"] = 8
args["epochs"] = 1000
args["augmentation_factor"] = 2 #How many times we add an augmented version of
# at risk students to the dataset
args["lr"] = 0.001
args["beta"] = 2
args["neurons_num"] = [] #[48,32]


dataset_path = "xAPI-Edu-Data.csv"
#dataset_path = "student-por.csv"

X, Y, _ = load_data(dataset_path)

print("Number of datapoint: ",len(X))
print("Data Imbalance: ",100*(1-sum(Y)/len(Y)),"%")

X = preprocessing.normalize(X)
X = X.astype(np.float32)
kf = StratifiedKFold(n_splits=4,shuffle=True,random_state=SEED)
results = [[],[],[],[],[]]
for fold_idx, (train_index, test_index) in enumerate(kf.split(X, Y)):
    X_train, y_train = X[train_index], Y[train_index]
    X_test, y_test = X[test_index], Y[test_index] 
    print("Fold ",fold_idx," data Train Imbalance: ",100*(1-sum(y_train)/len(y_train)),"%")
    print("Fold ",fold_idx," data Test Imbalance: ",100*(1-sum(y_test)/len(y_test)),"%")

    at_risk_student_id = np.nonzero(y_train == 0)[0]
    X_at_risk_student = X_train[at_risk_student_id]

    X_at_risk_augmented, _, _ = augment_betaVAE(X_at_risk_student.astype(np.float32),args=args)

    X_train_augmented = np.concatenate([X_at_risk_augmented,np.stack([X_train[i] for i in range(len(X_train)) if i not in at_risk_student_id])])
    y_train_augmented = np.array([0 for i in X_at_risk_augmented]+[1 for i in range(len(X_train)) if i not in at_risk_student_id])
    print("Data Augmented Train Imbalance: ",100*(1-sum(y_train_augmented)/len(y_train_augmented)),"%")

    results[0].append(train_and_eval_NN(X_train_augmented,y_train_augmented,X_test,y_test,args))
    results[1].append(train_and_eval_NN(X_train,y_train,X_test,y_test,args))
    results[2].append(train_and_eval_NN_undersampling(X_train,y_train,X_test,y_test,args))
    results[3].append(train_and_eval_NN_oversampling(X_train,y_train,X_test,y_test,args))
    results[4].append(train_and_eval_NN_class_weights(X_train,y_train,X_test,y_test,args))

for i in range(len(results)):
    dic = {}
    for measure in results[i][0]:
        arr = [j[measure] for j in results[i]]
        dic[measure] = str(np.mean(arr))+' +/- '+str(np.std(arr))
    results[i] = dic

print("Final Scores with augmentation ",results[0])
print("Final Scores w/o augmentation ",results[1])
print("Final Scores Undersampling ",results[2])
print("Final Scores Oversampling ",results[3])
print("Final Scores Class weights ",results[4])
data = results
frame = pd.DataFrame.from_dict(data,orient='columns')
frame.index = ["With augmentation (ours)","Without augmentation","Undersampling","Oversampling","Class weights"]
frame.to_csv("results/RESULT_"+"_".join([str(i) for i in args["neurons_num"]])+"_"+dataset_path)
print(frame)

from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import preprocessing
from sympy import latex
from utils import *
from trainers import *
import random
import matplotlib.pyplot as plt

def TopK(x, k):
    a = dict([(i, j) for i, j in enumerate(x)])
    sorted_a = dict(sorted(a.items(), key = lambda kv:kv[1], reverse=True))
    indices = list(sorted_a.keys())[:k]
    values = list(sorted_a.values())[:k]
    return (indices, values)

SEED = 0
seed_everything(SEED)

args = {}
args["batch_size"] = 8
args["epochs"] = 1000
args["augmentation_factor"] = 2
args["lr"] = 0.001
args["dropout"] = 0
args["beta"] = 2
args["neurons_num"] = [48,32]

dataset_path = "student-por.csv"
#dataset_path = "xAPI-Edu-Data.csv"


X, Y, feature_df = load_data(dataset_path)

print("Number of datapoint: ",len(X))
print("Data Imbalance: ",100*(1-sum(Y)/len(Y)),"%")

X = preprocessing.normalize(X)

X_train, X_test, y_train, y_test = train_test_split(X.astype(np.float32),Y,test_size=0.25, stratify = Y,random_state=SEED)
print("Data Train Imbalance: ",100*(1-sum(y_train)/len(y_train)),"%")
print("Data Test Imbalance: ",100*(1-sum(y_test)/len(y_test)),"%")


at_risk_student_id = np.nonzero(y_train == 0)[0]
X_at_risk_student = X_train[at_risk_student_id]
X_non_at_risk_student = X_train[[i for i in range(len(X_train)) if i not in at_risk_student_id]]

augmented_data, loss, model = augment_betaVAE(X_at_risk_student.astype(np.float32),args=args,eval=True)

plt.figure()
best_perf = set({})
latex_table = "Latent variable"
TOPK = 2
for i in range(TOPK):
    latex_table += " & Feature name (Standard deviation)"
latex_table += "\\\\ \n \\hline \n"
for i in range(model.latent_size):
    res = model.traversal(torch.tensor(X_at_risk_student),i,3,10)
    std = []
    for j in range(len(res[0][0])):
        std.append(np.mean([np.std([i[j].item() for i in data]) for data in res]))
    #std = [np.mean(i[0]) for i in range(len(std[0])))]
    plt.plot(std)
    feature_df.columns[np.argmax(std)]
    top_change = TopK(std,TOPK)
    best_perf.update(top_change[0])
    names = list(feature_df.columns[top_change[0]])
    res = [names[i]+" "+str(round(top_change[1][i],5)) for i in range(len(top_change[1]))]
    print("Component ",i," : ",res)
    latex_table += "$z_"+str(i)+"$"
    for i in range(len(top_change[1])):
        latex_table += " & "+names[i]+" ( "+str(round(top_change[1][i],5))+" ) "
    latex_table += "\\\\ \n"
plt.savefig("results/"+dataset_path[:-4]+"/reconstruct_std.png")
plt.close()
print(best_perf)
print(latex_table)


for i in range(augmented_data.shape[1]):
    fig, ax = plt.subplots(1)
    fig.set_figwidth(10)
    fig.set_figheight(5)
    ax.eventplot(augmented_data[:,i], lineoffsets=[0] ,orientation='vertical', colors='orange',label="Augmented Students")
    ax.eventplot(X_at_risk_student[:,i],lineoffsets=[1], orientation='vertical', colors='r',label="At Risk Students")
    ax.eventplot(X_non_at_risk_student[:,i],lineoffsets=[2], orientation='vertical', colors='g',label="Non At Risk Students")
    ax.axes.get_xaxis().set_visible(False)
    ax.legend()
    ax.set_ylim([0,1])
    plt.savefig("results/"+dataset_path[:-4]+"/dimension__"+feature_df.columns[i]+"__"+str(i)+".png")
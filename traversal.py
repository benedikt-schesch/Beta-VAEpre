from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import preprocessing
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
args["weight_decay"] = 0


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

# res = model.traversal(torch.tensor(X_at_risk_student[0]),0,3,10)
# fig, axs = plt.subplots(len(res[0]))
# fig.set_figheight(150)
# fig.set_figwidth(10)
# ymin, ymax = 10000000, 0
# std = []
# for i in range(len(res[0])):
#     ymin = min(ymin,min([data[i].item() for data in res]))
#     ymax = max(ymax,max([data[i].item() for data in res]))
# for i in range(len(res[0])):
#     axs[i].set_title("Dimension "+str(i))
#     axs[i].plot([data[i].item() for data in res])
#     axs[i].set_ylim([ymin, ymax])
#     std.append(np.std([data[i].item() for data in res]))
# plt.savefig("Reconstructions.png")
# plt.close()
plt.figure()
best_perf = set({})
for i in range(model.latent_size):
    res = model.traversal(torch.tensor(X_at_risk_student),i,3,10)
    std = []
    for j in range(len(res[0][0])):
        std.append(np.mean([np.std([i[j].item() for i in data]) for data in res]))
    #std = [np.mean(i[0]) for i in range(len(std[0])))]
    plt.plot(std)
    feature_df.columns[np.argmax(std)]
    top_change = TopK(std,2)
    best_perf.update(top_change[0])
    names = list(feature_df.columns[top_change[0]])
    res = [names[i]+" "+str(round(top_change[1][i],5)) for i in range(len(top_change[1]))]
    print("Component ",i," : ",res)
plt.savefig("Reconstructions_std.png")
plt.close()
print(best_perf)
for i in range(len(best_perf)):
    fig, ax = plt.subplots(1)
    fig.set_figwidth(10)
    fig.set_figheight(5)
    best_perf = list(best_perf)
    ax.eventplot(augmented_data[:,best_perf[i]], orientation='vertical', colors='orange',label="Augmented Students")
    #ax.eventplot(X_at_risk_student[:,best_perf[i]], orientation='vertical', colors='r',label="At Risk Students")
    #ax.eventplot(X_non_at_risk_student[:,best_perf[i]], orientation='vertical', colors='g',label="Non At Risk Students")
    ax.axes.get_xaxis().set_visible(False)
    ax.legend()
    ax.set_ylim([min(X_train[:,best_perf[i]]),max(X_train[:,best_perf[i]])])
    plt.savefig("test"+str(i)+".png")
import pandas as pd
import category_encoders as ce
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
from sklearn import preprocessing
from trainers import *
import random
import matplotlib.pyplot as plt

SEED = 0
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

args = {}
args["batch_size"] = 8
args["epochs"] = 1000
args["augmentation_factor"] = 0
args["lr"] = 0.001
args["dropout"] = 0
args["beta"] = 100000
args["neurons_num"] = [48,32]
args["weight_decay"] = 0


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
X_train, X_test, y_train, y_test = train_test_split(X.astype(np.float32),Y,test_size=0.25, stratify = Y,random_state=SEED)
print("Data Train Imbalance: ",100*(1-sum(y_train)/len(y_train)),"%")
print("Data Test Imbalance: ",100*(1-sum(y_test)/len(y_test)),"%")


at_risk_student_id = np.nonzero(y_train == 0)[0]
X_at_risk_student = X_train[at_risk_student_id]

_, loss, model = augment_betaVAE(X_at_risk_student.astype(np.float32),args=args)

res = model.traversal(torch.tensor(X_at_risk_student[0]),0,3,10)
fig, axs = plt.subplots(len(res[0]))
fig.set_figheight(150)
fig.set_figwidth(10)
ymin, ymax = 10000000, 0
std = []
for i in range(len(res[0])):
    ymin = min(ymin,min([data[i].item() for data in res]))
    ymax = max(ymax,max([data[i].item() for data in res]))
for i in range(len(res[0])):
    axs[i].set_title("Dimension "+str(i))
    axs[i].plot([data[i].item() for data in res])
    axs[i].set_ylim([ymin, ymax])
    std.append(np.std([data[i].item() for data in res]))
plt.savefig("Reconstructions.png")
plt.close()
plt.figure()
for i in range(model.latent_size):
    res = model.traversal(torch.tensor(X_at_risk_student[0]),i,3,10)
    std = []
    for j in range(len(res[0])):
        std.append(np.std([data[j].item() for data in res]))
    plt.plot(std)
plt.savefig("Reconstructions_std.png")
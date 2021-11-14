import torch
import torch.optim
from  torch.utils.data.dataloader import DataLoader
import pandas as pd
import category_encoders as ce
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
from model import BetaVAE
from sklearn import preprocessing


BATCH_SIZE = 4
NUM_EPOCHS = 200
LR = 0.001
BETA = 1

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

at_risk_student_id = np.nonzero(y_train == 0)[0]
train_loader =  DataLoader([X_train[i] for i in at_risk_student_id], batch_size=BATCH_SIZE)

model = BetaVAE(feature_dim=len(X_train[0]),beta=BETA)
optimizer = torch.optim.Adam(model.parameters(),lr=LR)

for epoch in range(NUM_EPOCHS):
    loop = tqdm(train_loader)
    for idx, x in enumerate(loop):        
        #reset gradients
        optimizer.zero_grad()
        
        #forward propagation through the network
        out, mu, logvar = model(x)
        
        #calculate the loss
        loss = model.loss(out, x, mu, logvar)
        
        #backpropagation
        loss.backward()
        
        #update the parameters
        optimizer.step()
        
        # add stuff to progress bar in the end
        loop.set_description(f"Epoch [{epoch}/{NUM_EPOCHS}]")
        loop.set_postfix(loss=loss.item())
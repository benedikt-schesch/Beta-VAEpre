import torch
import torch.optim
from  torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import numpy as np
from models import *
import sklearn.metrics
import random
from  sklearn.utils import class_weight


def augment_betaVAE(X,args,eval=False):
    train_loader =  DataLoader(X, batch_size=args["batch_size"])
    num_epochs = args["epochs"]
    model = BetaVAE(feature_dim=len(X[0]),beta=args["beta"])
    optimizer = torch.optim.Adam(model.parameters(),lr=args["lr"])
    for epoch in range(args["epochs"]):
        loop = tqdm(train_loader)
        for x in loop:        
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
            loop.set_description(f"Epoch [{epoch}/{num_epochs}]")
            loop.set_postfix(loss=loss.item())
    augmented_samples = []
    total_loss = 0
    model.eval()
    with torch.no_grad():
        for i in range(args["augmentation_factor"]):
            for x in train_loader:
                out, _, _ = model(x)
                augmented_samples.append(out.numpy())
        for x in train_loader:
            out, mu, logvar = model(x)
            loss = model.loss(out, x, mu, logvar)
            total_loss += loss.item()
    model.train()
    if eval:
        return np.concatenate(augmented_samples), total_loss/len(train_loader), model
    return np.concatenate(augmented_samples+[X]), total_loss/len(train_loader), model

def eval_NN(model,X,y,args):
    test_loader =  DataLoader([[X[i],y[i]] for i in range(len(y))], 
                                batch_size=args["batch_size"])
    correct = 0
    y1 = []
    y_true1 = []
    with torch.no_grad():
        for x, y_true in test_loader:
            out = model(x)
            correct += (torch.argmax(out,1) == y_true).sum().item()
            y1.append(torch.argmax(out,1))
            y_true1.append(y_true)
    return {"Accuracy":correct/len(y),
    "F1 score":sklearn.metrics.f1_score(torch.cat(y_true1),torch.cat(y1)), 
    "Balanced Accuracy":sklearn.metrics.balanced_accuracy_score(torch.cat(y_true1),torch.cat(y1)), 
    "Precision":sklearn.metrics.precision_score(torch.cat(y_true1),torch.cat(y1)),
    "Recall":sklearn.metrics.recall_score(torch.cat(y_true1),torch.cat(y1))}

def train_and_eval_NN(X,y,X_test,y_test,args):
    train_loader =  DataLoader([[X[i],y[i]] for i in range(len(y))], 
                                    batch_size=args["batch_size"],
                                    shuffle=True)
    num_epochs = args["epochs"]
    model = DNN(args["neurons_num"])
    optimizer = torch.optim.Adam(model.parameters(),lr=args["lr"])
    f_loss = torch.nn.CrossEntropyLoss()
    for epoch in range(args["epochs"]):
        correct = 0
        num_items = 0
        loop = tqdm(train_loader)
        for x, y_true in loop:        
            #reset gradients
            optimizer.zero_grad()
            
            #forward propagation through the network
            out = model(x)
            
            #calculate the loss
            loss = f_loss(out,y_true)
            
            #backpropagation
            loss.backward()
            
            #update the parameters
            optimizer.step()
            
            # add stuff to progress bar in the end
            correct += (torch.argmax(out,1) == y_true).sum().item()
            num_items += len(y_true)
            loop.set_description(f"Epoch [{epoch}/{num_epochs}]")
            loop.set_postfix(loss=loss.item(),accuracy=100. * correct/num_items)
        print("Epoch ",epoch," Testing Accuracy: ",eval_NN(model,X_test,y_test,args))
                
    return eval_NN(model,X_test,y_test,args)




def train_and_eval_NN_class_weights(X,y,X_test,y_test,args):
    train_loader =  DataLoader([[X[i],y[i]] for i in range(len(y))], 
                                    batch_size=args["batch_size"],
                                    shuffle=True)
    num_epochs = args["epochs"]
    model = DNN(args["neurons_num"])
    class_weights = class_weight.compute_class_weight('balanced',np.unique(y),y)
    class_weights = torch.tensor(class_weights,dtype=torch.float)
    optimizer = torch.optim.Adam(model.parameters(),lr=args["lr"])
    f_loss = torch.nn.CrossEntropyLoss(weight=class_weights)
    for epoch in range(args["epochs"]):
        correct = 0
        num_items = 0
        loop = tqdm(train_loader)
        for x, y_true in loop:        
            #reset gradients
            optimizer.zero_grad()
            
            #forward propagation through the network
            out = model(x)
            
            #calculate the loss
            loss = f_loss(out,y_true)
            
            #backpropagation
            loss.backward()
            
            #update the parameters
            optimizer.step()
            
            # add stuff to progress bar in the end
            correct += (torch.argmax(out,1) == y_true).sum().item()
            num_items += len(y_true)
            loop.set_description(f"Epoch [{epoch}/{num_epochs}]")
            loop.set_postfix(loss=loss.item(),accuracy=100. * correct/num_items)
        print("Epoch ",epoch," Testing Accuracy: ",eval_NN(model,X_test,y_test,args))
                
    return eval_NN(model,X_test,y_test,args)



def train_and_eval_NN_oversampling(X,y,X_test,y_test,args):
    at_risk_student_id = np.nonzero(y == 0)[0]
    train_dataset = [[X[i],y[i]] for i in list(range(len(y)))+random.choices(list(at_risk_student_id),k=len(y)-2*len(at_risk_student_id))]
    train_loader =  DataLoader(train_dataset,
                                    batch_size=args["batch_size"],
                                    shuffle=True)
    num_epochs = args["epochs"]
    model = DNN(args["neurons_num"])
    class_weights = class_weight.compute_class_weight('balanced',np.unique(y),y)
    class_weights = torch.tensor(class_weights,dtype=torch.float)
    optimizer = torch.optim.Adam(model.parameters(),lr=args["lr"])
    f_loss = torch.nn.CrossEntropyLoss(weight=class_weights)
    for epoch in range(args["epochs"]):
        correct = 0
        num_items = 0
        loop = tqdm(train_loader)
        for x, y_true in loop:        
            #reset gradients
            optimizer.zero_grad()
            
            #forward propagation through the network
            out = model(x)
            
            #calculate the loss
            loss = f_loss(out,y_true)
            
            #backpropagation
            loss.backward()
            
            #update the parameters
            optimizer.step()
            
            # add stuff to progress bar in the end
            correct += (torch.argmax(out,1) == y_true).sum().item()
            num_items += len(y_true)
            loop.set_description(f"Epoch [{epoch}/{num_epochs}]")
            loop.set_postfix(loss=loss.item(),accuracy=100. * correct/num_items)
        print("Epoch ",epoch," Testing Accuracy: ",eval_NN(model,X_test,y_test,args))
                
    return eval_NN(model,X_test,y_test,args)



def train_and_eval_NN_undersampling(X,y,X_test,y_test,args):
    non_at_risk_student_id = np.nonzero(y == 1)[0]
    at_risk_student_id = np.nonzero(y == 0)[0]
    train_dataset = [[X[i],y[i]] for i in list(at_risk_student_id)+random.choices(list(non_at_risk_student_id),k=len(at_risk_student_id))]
    train_loader =  DataLoader(train_dataset,
                                    batch_size=args["batch_size"],
                                    shuffle=True)
    num_epochs = args["epochs"]
    model = DNN(args["neurons_num"])
    class_weights = class_weight.compute_class_weight('balanced',np.unique(y),y)
    class_weights = torch.tensor(class_weights,dtype=torch.float)
    optimizer = torch.optim.Adam(model.parameters(),lr=args["lr"])
    f_loss = torch.nn.CrossEntropyLoss(weight=class_weights)
    for epoch in range(args["epochs"]):
        correct = 0
        num_items = 0
        loop = tqdm(train_loader)
        for x, y_true in loop:        
            #reset gradients
            optimizer.zero_grad()
            
            #forward propagation through the network
            out = model(x)
            
            #calculate the loss
            loss = f_loss(out,y_true)
            
            #backpropagation
            loss.backward()
            
            #update the parameters
            optimizer.step()
            
            # add stuff to progress bar in the end
            correct += (torch.argmax(out,1) == y_true).sum().item()
            num_items += len(y_true)
            loop.set_description(f"Epoch [{epoch}/{num_epochs}]")
            loop.set_postfix(loss=loss.item(),accuracy=100. * correct/num_items)
        print("Epoch ",epoch," Testing Accuracy: ",eval_NN(model,X_test,y_test,args))
                
    return eval_NN(model,X_test,y_test,args)

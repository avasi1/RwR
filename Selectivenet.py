'''Functions for the selective net block, training, and testing for different heads'''
import torch.nn as nn
import torch.nn.functional as F
import torch    
import sklearn.datasets
from sklearn.model_selection import train_test_split
import numpy as np
from NN_method import train_head

class Sel_NN(nn.Module):
    def __init__(self, input_dim):
        super(Sel_NN, self).__init__()
        hidden_dim = 64
        output_dim = 1
        self.hidden_dim=hidden_dim
        self.bn0 = nn.BatchNorm1d(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.main= nn.Sequential(self.bn0,self.fc1,self.bn1,nn.ReLU())
        #predcit block
        self.fc_pred = nn.Linear(hidden_dim, output_dim)
        #select block
        self.fc_sel = nn.Linear(hidden_dim, 16)
        self.bn_sel = nn.BatchNorm1d(16)
        self.fc_sel_2 = nn.Linear(16, 1)
        #auxiliary block
        self.fc_aux = nn.Linear(hidden_dim, output_dim)


    def forward(self, x):
        x = self.main(x)
        #x= F.relu(self.bn2(self.fc2(x)))
        #predict
        x_pred = self.fc_pred(x)
        #select
        x_sel = F.relu(self.bn_sel(self.fc_sel(x)))
        x_sel = self.fc_sel_2(x_sel)
        x_sel= torch.sigmoid(x_sel)
        #auxiliary
        x_aux = self.fc_aux(x)
        return x_pred, x_sel, x_aux
    

def train_Sel_NN(data,constraint,cost=0,alpha=0.5):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #hyperparameters
    constraint=1-constraint
    lr=0.0005
    batch_size=256
    epochs=800
    weight_decay=0.0001
    #name of the machine model
    machine_type = 'SelectiveNet'
    if cost==0:
        X_train, X_val, Y_train, Y_val = train_test_split(data['X'], data['Y'],test_size=0.2,random_state=42)
    else:
        X_train=data['X']
        Y_train=data['Y']
    #load data
    X = torch.from_numpy(X_train).float().to(device)
    Y = torch.from_numpy(Y_train).float().to(device)
    
    trainloader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X,Y),batch_size=batch_size,shuffle=True)
    
    
    #define the model
    model = Sel_NN(data['X'].shape[1]).to(device)
    
    #optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # define the loss function
    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, y_target = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            y_pred,y_sel,y_aux = model(inputs)
            y_pred,y_sel,y_aux = y_pred.squeeze().to(device),y_sel.squeeze().to(device),y_aux.squeeze().to(device)
            loss =alpha*torch.mean( (y_pred - y_target)**2* y_sel)/torch.mean(y_sel)+alpha*200*(torch.max(torch.tensor(constraint).to(device)-torch.mean(y_sel),torch.tensor([0]).to(device)) )**2+(1-alpha)*torch.mean((y_aux - y_target)**2)

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
        if epoch % 100 == 99:    # print every 50 epochs
            print(f'[{epoch + 1}] loss: {running_loss/(i+1) :.3f}')
            
    #save the model
    print('Finished Training')
    return model
  
def test_Sel_NN(data,model,cost=0,no_aux=False,thrs=0.5):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    X=torch.from_numpy(data['X']).float().to(device)
    Y=torch.from_numpy(data['Y']).float().to(device)
    model.eval()
    if no_aux:
        y_pred,y_sel = model(X)
        y_pred,y_sel = y_pred.squeeze(),y_sel.squeeze()
    else:
        y_pred,y_sel,y_aux = model(X)
        y_pred,y_sel,y_aux = y_pred.squeeze(),y_sel.squeeze(),y_aux.squeeze()
    flag=y_sel>thrs
    coverage=1-torch.sum(flag)/len(flag)
    print('ratio for giving human:',coverage)
    if cost==0:
        with torch.no_grad():
            loss=torch.mean((y_pred[y_sel>thrs]-Y[y_sel>thrs])**2)
            print('machine loss:',loss)
        return loss.detach().cpu().numpy(),coverage.detach().cpu().numpy()
    else:
        with torch.no_grad():
            loss=(torch.sum((y_pred[y_sel>thrs]-Y[y_sel>thrs])**2)+cost*(len(flag)-torch.sum(flag)))/len(flag)
            print('mean of total loss:',loss)
        return loss.detach().cpu().numpy(),coverage.detach().cpu().numpy()

def train_Sel_NN_cost(data,cost):
    X_train, X_val, Y_train, Y_val = train_test_split(data['X'], data['Y'],test_size=0.2,random_state=42)
    train_data={'X':X_train,'Y':Y_train}
    val_data={'X':X_val,'Y':Y_val}
    loss_list=[]
    constraint_list= [0.01,0.2,0.4,0.6,0.8,0.99]
    for constraint in constraint_list:
        model=train_Sel_NN(train_data,constraint,cost)
        loss,coverage=test_Sel_NN(val_data,model,cost)
        loss_list.append(loss)
    inx=np.argmin(loss_list)
    best_constraint=constraint_list[inx]
    print('best constraint:',best_constraint)
    model=train_Sel_NN(train_data,best_constraint,cost=0)
    return model


def train_val_Sel_NN_budget(train_data,val_data,constraint,alpha,thrs_list=[0.5]):
    loss_list=[]
    coverage_list=[]
    model=train_Sel_NN(train_data,constraint,cost=0,alpha=alpha)

    for thrs in thrs_list:
        loss,coverage=test_Sel_NN(val_data,model,cost=0,thrs=thrs)
        loss_list.append(loss)
        coverage_list.append(coverage)
    ind=np.argmin(np.abs(np.array(coverage_list)-constraint))
    return model,thrs_list[ind]

def train_sel_head_cost(data,pretrain_model,cost):
    X_train, X_val, Y_train, Y_val = train_test_split(data['X'], data['Y'],test_size=0.2,random_state=42)
    train_data={'X':X_train,'Y':Y_train}
    val_data={'X':X_val,'Y':Y_val}
    loss_list=[]
    constraint_list= [0.01,0.2,0.4,0.6,0.8,0.99]
    for constraint in constraint_list:
        model=train_head(train_data,pretrain_model,'sel',constraint,cost)
        loss,coverage=test_Sel_NN(val_data,model,cost,no_aux=True)
        loss_list.append(loss)
    inx=np.argmin(loss_list)
    best_constraint=constraint_list[inx]
    print('best constraint:',best_constraint)
    model=train_head(train_data,pretrain_model,'sel',best_constraint,cost)
    return model

def train_val_sel_head_budget(train_data,val_data,pretrain_model,constraint,thrs_list=[0.5]):

    loss_list=[]
    coverage_list=[]
    model=train_head(train_data,pretrain_model,'sel',constraint,cost=0)

    for thrs in thrs_list:
        loss,coverage=test_Sel_NN(val_data,model,cost=0,no_aux=True,thrs=thrs)
        loss_list.append(loss)
        coverage_list.append(coverage)
    ind=np.argmin(np.abs(np.array(coverage_list)-constraint))
    return model,thrs_list[ind]

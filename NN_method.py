'''
Functions for training and testing the NN-based models. Some functions are based on the following:
    https://github.com/Networks-Learning/differentiable-learning-under-triage/tree/main
'''

import numpy as np
import torch
from NN_block import LinNN,loss_head,logis_head,sel_head
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import copy
import os
import torchvision.models as models
from sklearn.model_selection import train_test_split
import sklearn.datasets
from temperature_scaling import ModelWithTemperature
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import torch
from torch.linalg import norm
from numpy.linalg import norm as npnorm

def find_machine_samples(machine_loss,constraint,cost=0):
    if cost==0:
        diff = machine_loss
        argsorted_diff = torch.clone(torch.argsort(diff))
        num_outsource = int(constraint * machine_loss.shape[0])
        index = -num_outsource

        while index < -1:
            index += 1
        
        if index==0:
            index = -1
        if index == -diff.shape[0]:
            index = 1
        machine_list = argsorted_diff[:index]
    else:
        machine_list = torch.where(machine_loss<=cost)[0]
    return machine_list




def train_full_NN(data):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #hyperparameters
    lr=0.0005
    batch_size=256
    epochs=800
    weight_decay=0.0001
    #name of the machine model
    machine_type = 'FullNet'
    X_train, X_val, Y_train, Y_val = train_test_split(data['X'], data['Y'],test_size=0.2,random_state=42)
    #load data
    X = torch.from_numpy(X_train).float().to(device)
    Y = torch.from_numpy(Y_train).float().to(device)
    trainloader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X,Y),batch_size=batch_size,shuffle=True)
    #define the model
    model = LinNN(data['X'].shape[1]).to(device)
    loss_func = torch.nn.MSELoss(reduction='mean')    
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
            y_pred = model(inputs).to(device).squeeze()

            loss = loss_func(y_pred, y_target)

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
        if epoch % 100 == 99:    # print every 50 epochs
            print(f'[{epoch + 1}] loss: {running_loss/(i+1) :.3f}')
            
    #save the model
    print('Finished Training')
    return model

def train_triage(data,constraint,val=False,cost=0):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('training machine model using constraint:',constraint)
    X_train, X_val, Y_train, Y_val = train_test_split(data['X'], data['Y'],test_size=0.2,random_state=42)

    X = torch.from_numpy(X_train).float()
    Y = torch.from_numpy(Y_train).float()
    if val:
        #make 20% validation set
        val_X = torch.from_numpy(X_val).float()
        val_Y = torch.from_numpy(Y_val).float()

    
    batch_size = 256
    num_batches = int(X.shape[0] / batch_size)
    if val:
        val_num_batches = int(val_X.shape[0] / batch_size)
    
    num_epochs = 800
    mnet = LinNN(X.shape[1]).to(device)
    optimizer = torch.optim.Adam(mnet.parameters(),lr=0.0005,weight_decay=0.0001)
    loss_func = torch.nn.MSELoss(reduction='none')
    train_losses = []
    val_losses = []
    best_val_loss = 1000
    eps = 1e-4
    max_patience = 10
    patience = 0
    res = {}
    res['machine_loss'] = {}
    
    
    for epoch in range(num_epochs):
        #print('----- epoch:',epoch, '-----')
        train_loss = 0
        with torch.no_grad():
            mprim = copy.deepcopy(mnet)
        machine_loss = []
        for i in range(num_batches):





            X_batch = X[i * batch_size: (i + 1) * batch_size].to(device)
            Y_batch = Y[i * batch_size: (i + 1) * batch_size].to(device)



            with torch.no_grad():
                machine_scores_batch = mprim(X_batch).squeeze()
                machine_loss_batch =loss_func(machine_scores_batch,Y_batch)
                machine_loss.extend(machine_loss_batch.detach())
                

            machine_indices = find_machine_samples(machine_loss_batch, constraint,cost)
            if len(machine_indices)<32 and cost!=0:
                machine_indices = find_machine_samples(machine_loss_batch, constraint=1/8,cost=0)
            X_machine = X_batch[machine_indices]
            Y_machine = Y_batch[machine_indices]
            optimizer.zero_grad()
            loss = loss_func(mnet(X_machine).squeeze(),Y_machine)
            loss.sum().backward()
            optimizer.step()
            train_loss += float(loss.mean())

        train_losses.append(train_loss / num_batches)
        #print('machine_loss:', train_loss/num_batches)
        if val:
            with torch.no_grad():
                val_loss = 0
                for i in range(val_num_batches):
                    val_X_batch = val_X[i * batch_size: (i + 1) * batch_size].to(device)
                    val_Y_batch = val_Y[i * batch_size: (i + 1) * batch_size].to(device)
                    val_mscores_batch = mprim(val_X_batch)
                    val_mscores_batch = val_mscores_batch.squeeze()
                    val_mloss_batch = loss_func(val_mscores_batch, val_Y_batch)
                    val_machine_indices = find_machine_samples(val_mloss_batch,constraint,cost)
                    val_loss += float(loss_func(mnet(val_X_batch[val_machine_indices]).squeeze(),val_Y_batch[val_machine_indices]).mean())
                    
                val_loss /= val_num_batches
                #print('val_loss:',val_loss) 

                if val_loss + eps < best_val_loss:
                    best_val_loss = val_loss
                    #print('updated the model')
                    patience = 0
                else:
                    patience += 1
                val_losses.append(val_loss)

            if patience > max_patience:
                print('no progress for 10 epochs... stopping training')
                break



            
    

    '''
    plt.plot(range(len(train_losses)),train_losses,marker='o',label='train')
    if val:
        plt.plot(range(len(val_losses)),val_losses,marker='o',label='validation')
    plt.legend()
    plt.title(' capacity = ' + str(constraint),fontsize=22)
    plt.xlabel(r'Time Step t',fontsize=22)
    plt.ylabel(r'Machine Loss',fontsize=20)
    plt.show()
    '''
    return mnet





def train_g(data,constraint,mnet,val=False,calibrate=False,cost=0):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('started training g using the constraint: ',constraint)  
    with torch.no_grad():
        mnet.to(device)
        mnet.eval()
    X_train, X_val, Y_train, Y_val = train_test_split(data['X'], data['Y'],test_size=0.2,random_state=42)

    train_X = torch.from_numpy(X_train).float()
    train_Y = torch.from_numpy(Y_train).float()
    if val:
        #make 20% validation set
        val_X = torch.from_numpy(X_val).float()
        val_Y = torch.from_numpy(Y_val).float()


    
    batch_size = 128
    num_batches = int(train_X.shape[0] / batch_size)
    if val:
        val_num_batches = int(val_X.shape[0] / batch_size)
    
    num_epochs = 40
    
    gnet = LinNN(train_X.shape[1],2).to(device)



    g_optimizer = torch.optim.Adam(gnet.parameters(),lr=0.001)
    loss_func = torch.nn.CrossEntropyLoss(reduction='none')
    loss_func_machine = torch.nn.MSELoss(reduction='none')

    train_losses = []
    val_losses = []
    best_val_loss = 1000
    max_patience = 8
    patience = 0
    eps = 1e-4
    res = {}
    res['g_train_loss'] = {}
    
    
    for epoch in range(num_epochs):
        machine_loss = []
        val_machine_loss = []
        gprediction = []
        val_gprediction = []
        glabels = []
        val_glabels = []
        rnd = np.random.permutation(train_X.shape[0])
        X, Y = train_X[rnd],train_Y[rnd],
        #print('----- epoch:',epoch, '-----')
        g_train_loss = 0
        for i in range(num_batches):
            X_batch = X[i * batch_size: (i + 1) * batch_size].to(device)
            Y_batch = Y[i * batch_size: (i + 1) * batch_size].to(device)
            with torch.no_grad():
                machine_loss_batch = loss_func_machine(mnet(X_batch).squeeze(),Y_batch)
                machine_indices = find_machine_samples(machine_loss_batch,  constraint,cost)
            g_labels_batch = torch.tensor([0 if j in machine_indices else 1 for j in range(batch_size)]).to(device)
            g_optimizer.zero_grad()
            gpred = gnet(X_batch)
            g_loss = loss_func(gpred,g_labels_batch)
            g_loss.mean().backward()
            g_optimizer.step()
            g_train_loss += float(g_loss.mean())
            machine_loss.extend(machine_loss_batch)
            gprediction.extend(gpred[:,1])
            glabels.append(g_labels_batch)
            
        train_losses.append(g_train_loss/num_batches)
        #print('g_loss:',g_train_loss/num_batches) 
                  
        if val:
            with torch.no_grad():
                val_gloss = 0
                for i in range(val_num_batches):
                    val_X_batch = val_X[i * batch_size: (i + 1) * batch_size].to(device)
                    val_Y_batch = val_Y[i * batch_size: (i + 1) * batch_size].to(device)
                    val_mscores = mnet(val_X_batch).squeeze()
                    val_machine_loss_batch = loss_func_machine(val_mscores,val_Y_batch)
                    val_machine_loss.extend(val_machine_loss_batch)
                    val_machine_indices = find_machine_samples(val_machine_loss_batch,constraint,cost)
                    val_glabels_batch = torch.tensor([0 if j in val_machine_indices else 1 for j in range(val_X_batch.shape[0])]).to(device)
                    val_glabels.extend(val_glabels_batch)
                    val_gpred = gnet(val_X_batch)
                    val_gprediction.extend(val_gpred[:,1])
                    val_loss = loss_func(val_gpred,val_glabels_batch)
                    val_gloss += float(val_loss.mean())
                    
                val_gloss /= val_num_batches
                val_losses.append(val_gloss)
                #print('val_g_loss:',float(val_gloss))

                if val_gloss + eps < best_val_loss:
            
                    best_val_loss = val_gloss
                    #print('updated the model')
                    patience = 0
                else:
                    patience += 1

            if patience > max_patience:
                print('no progress for 10 epochs... stopping training')
                break
        

    
    '''
    plt.plot(range(len(train_losses)),train_losses,marker = 'o',label='train')
    if val:
        plt.plot(range(len(val_losses)),val_losses,marker = 'o',label='validation')
    plt.title('train and validation curve of g using b = ' + str(constraint),fontsize=22)
    plt.xlabel('Epoch',fontsize=22)
    plt.ylabel(r'g Loss',fontsize=20)
    plt.legend()
    plt.show()
    '''
    if calibrate:
        orig_model = gnet # create an uncalibrated model somehow
        # Create a DataLoader from the val_X and val_Y used to train orig_model
        with torch.no_grad():
                machine_loss = loss_func_machine(mnet(val_X.to(device)).squeeze(),val_Y.to(device))
                machine_indices = find_machine_samples(machine_loss,  constraint)
        g_labels = torch.tensor([0 if j in machine_indices else 1 for j in range(val_Y.shape[0])]).to(device)
        valid_loader=DataLoader(TensorDataset(val_X.to(device),g_labels),batch_size=128,shuffle=False)
        scaled_model = ModelWithTemperature(orig_model)
        scaled_model.set_temperature(valid_loader)
        gnet = scaled_model
    return gnet





def get_test_assignments_triage(data,mnet,gnet,constraint,cost=0):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():

            loss_func_machine = torch.nn.MSELoss(reduction='none')
            test_X= torch.tensor(data['X']).float()
            test_Y = torch.tensor(data['Y']).float()
            
            test_X = test_X.to(device)
            test_Y = test_Y.to(device)
            num_machine = int((1.0 - constraint) * test_X.shape[0])
            
            mnet.to(device)
            mnet.eval()


            pred_Y= mnet(test_X)
            machine_loss = loss_func_machine(pred_Y.squeeze(),test_Y)

            gnet.to(device)
            gnet.eval()
            gprediction = torch.exp(gnet(test_X).detach()[:,1])
            human_candidates = torch.argsort(gprediction)[num_machine:]
            to_machine = [i for i in range(test_X.shape[0]) if i not in human_candidates]
            to_human = [i for i in range(test_X.shape[0]) if i not in to_machine]
            if cost==0:
                if len(to_machine)!=0:
                    print('mean of machine error:' ,np.mean(machine_loss[to_machine].cpu().numpy()))
                return np.mean(machine_loss[to_machine].cpu().numpy())
            else:
                if len(to_machine)!=0:
                    total_error=(np.sum(machine_loss[to_machine].cpu().numpy())+cost*len(to_human))/test_X.shape[0]
                else:
                    total_error=cost
                print('mean of total error:',total_error)
                to_human_ratio=len(to_human)/test_X.shape[0]
                return total_error,to_human_ratio

def kernel_var_estimator(train_data,test_data,mnet,base_kernel = lambda X : torch.exp(-norm(X, dim = 2) ** 2),lamb = 1,wid = 1E-1):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        X = train_data['X']
        Y = train_data['Y']
        recal_Z = torch.from_numpy(X).float().to(device)
        recal_Y = torch.from_numpy(Y).float().to(device)
        pred_Y= mnet(recal_Z).squeeze()
        recal_epsilon = (recal_Y - pred_Y)**2
        test_Z= torch.from_numpy(test_data['X']).float().to(device)

        assert len(recal_epsilon.shape) == 1

        sorted_epsi, indices = torch.sort(recal_epsilon, dim = 0)

        sorted_recal_Z = recal_Z[indices]

        test_Z_unsqueezed = test_Z.unsqueeze(1).repeat(1, len(recal_epsilon), 1)
        sorted_recal_Z_unsqueezed = sorted_recal_Z.unsqueeze(0) .repeat(len(test_Z),1,1)

        dist_mat = lamb * base_kernel((sorted_recal_Z_unsqueezed - test_Z_unsqueezed) / wid)

        est_var= torch.matmul(dist_mat, sorted_epsi)/torch.clamp(torch.sum(dist_mat, dim = 1),min=1e-8)
        return est_var     

def get_test_assignments_full(train_data,test_data,mnet,constraint,wid=2,cost=0):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():

            loss_func_machine = torch.nn.MSELoss(reduction='none')

            test_X= torch.tensor(test_data['X']).float().to(device)
            test_Y = torch.tensor(test_data['Y']).float().to(device)
            

            num_machine = int((1.0 - constraint) * test_X.shape[0])
            
            mnet.to(device)
            mnet.eval()


            pred_Y= mnet(test_X)
            machine_loss = loss_func_machine(pred_Y.squeeze(),test_Y)

            gprediction = kernel_var_estimator(train_data,test_data,mnet, lambda X : torch.exp(-norm(X, dim = 2) ** 2),1,wid)
            if cost==0:
                human_candidates = torch.argsort(gprediction)[num_machine:]
                to_machine = [i for i in range(test_X.shape[0]) if i not in human_candidates ]
                if len(to_machine)==0:
                    print('error, no samples to machine')
                return np.mean(machine_loss[to_machine].cpu().numpy())
            else:
                human_candidates =torch.where(machine_loss>cost)[0]
                to_machine = [i for i in range(test_X.shape[0]) if i not in human_candidates ]
                to_human = [i for i in range(test_X.shape[0]) if i not in to_machine]
                if len(to_machine)!=0:
                    total_error=(np.sum(machine_loss[to_machine].cpu().numpy())+cost*len(to_human))/test_X.shape[0]
                else:
                    total_error=cost
                to_human_ratio=len(to_human)/test_X.shape[0]
                return total_error,to_human_ratio
    
            


    
def get_test_assignments_full_no_wid(train_data,test_data,mnet,constraint,cost=0):
    lossl=[]
    wid=[]

    X_train, X, Y_train, Y = train_test_split(train_data['X'],train_data['Y'],test_size=0.2,random_state=42)
    if X.shape[0]<1000:
                val_data={'X':X,'Y':Y}
    else:
        val_data={'X':X[:1000],'Y':Y[:1000]}

    fake_train_data={'X':X_train,'Y':Y_train}
    for m in [1e-3,1e-2,1e-1,1,10,100,1000]:
        for k in [1]:
            loss,_=get_test_assignments_full(fake_train_data,val_data,mnet,constraint,wid=m*k,cost=cost)
            lossl.append(loss)
            wid.append(m*k)
    wid_pick=wid[np.argmin(lossl)]
    loss,to_human_ratio=get_test_assignments_full(fake_train_data,test_data,mnet,constraint,wid=wid_pick,cost=cost)
    return loss,to_human_ratio


def train_head(data,pretrain_model,train_mode,constraint,cost=0):
    device = 'cuda'
    if train_mode=='loss':
        model=loss_head(pretrain_model).cuda()
    if train_mode=='sel':
        model=sel_head(pretrain_model).cuda()

    if train_mode=='logis':
        model=logis_head(pretrain_model).cuda()
    X_train=data['X']
    Y_train=data['Y']
        #freeze the pretrained model
    for param in model.pretrained_model.parameters():
            param.requires_grad = False
    
    #hyperparameters
    constraint=1-constraint #constraint is the ratio of machine now
    lr=0.0005
    batch_size=256
    epochs=800
    weight_decay=0.0001
    #load data
    X = torch.from_numpy(X_train).float().to(device)
    Y = torch.from_numpy(Y_train).float().to(device)
    if train_mode!='sel':
        with torch.no_grad():
            pretrain_model.eval()
            pred_Y= pretrain_model(X).squeeze()
            true_loss = (pred_Y - Y)**2
            if train_mode=='loss':
                Y=true_loss
            else:
                if cost>0:
                    Y=(true_loss>cost).float() #loss>cost triages to human, label as 1 for the logistic head
                else:
                    #choose the top constraint% samples with the largest loss turages to human
                    num_machine = int((1-constraint) * X.shape[0])
                    Y=torch.zeros(X.shape[0])
                    Y[torch.argsort(true_loss)[:num_machine]]=1

    
    trainloader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X,Y),batch_size=batch_size,shuffle=True)
       
    #optimizer
    if train_mode in ['loss','logis']:
        optimizer = torch.optim.Adam(model.fc.parameters(), lr=lr, weight_decay=weight_decay)

    if train_mode=='sel':
        parameters = list(model.sel_head.parameters())+list(model.pred_head.parameters())
        optimizer = torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, y_target = data
            # zero the parameter gradients
            optimizer.zero_grad()
            if train_mode=='loss':
                y_pred = model(inputs)
                y_pred= y_pred.squeeze().to(device)
                loss =torch.mean( (y_pred - y_target)**2)
            if train_mode=='logis':
                y_pred = model(inputs)
                y_pred= y_pred.squeeze().to(device)
                #use CE loss for this biary classification task
                loss =torch.mean( -y_target*torch.log(y_pred+1e-8)-(1-y_target)*torch.log(1-y_pred+1e-8))
            if train_mode=='sel':
                y_pred,y_sel= model(inputs)
                y_pred,y_sel = y_pred.squeeze().to(device),y_sel.squeeze().to(device)
                loss =torch.mean( (y_pred - y_target)**2* y_sel)/torch.mean(y_sel)+200*(torch.max(torch.tensor(constraint).to(device)-torch.mean(y_sel),torch.tensor([0]).to(device)) )**2
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
        if epoch % 100 == 99:    # print every 50 epochs
            print(f'[{epoch + 1}] loss: {running_loss/(i+1) :.3f}')
            
    #save the model
    print('Finished Training')
    if train_mode=='loss':
        with torch.no_grad():
            model.eval()
            loss_pred=model(X).squeeze()
            num_machine = int((constraint) * X.shape[0]) #target number to machine
            #loss> threshold triages to human
            thrs=np.sort(loss_pred.cpu().numpy())[num_machine]
        return model,thrs
    if train_mode=='logis':
        return model
    if train_mode=='sel':
        return model

def train_val_logis_head(data,constraint,cost=0,thrs_list=[0.5]):

    train_data=data
    val_data=data
    mnet=train_full_NN(train_data)
    rej_net=train_head(val_data,mnet,'logis',constraint,cost)
    loss_L=[]
    coverage_L=[]

    
    for thrs in thrs_list:
        loss,human_ratio=test_logis_head(val_data,mnet,rej_net,cost,thrs=thrs)
        loss_L.append(loss)
        coverage_L.append(human_ratio)
    idx=np.argmin(loss_L)
    return mnet,rej_net,thrs_list[idx]
def test_logis_head(test_data,mnet,logis_net,cost,thrs=0.5):
    device='cuda'
    with torch.no_grad():

            loss_func_machine = torch.nn.MSELoss(reduction='none')

            test_X= torch.tensor(test_data['X']).float().to(device)
            test_Y = torch.tensor(test_data['Y']).float().to(device)
            
            
            mnet.to(device)
            mnet.eval()


            pred_Y= mnet(test_X)
            machine_loss = loss_func_machine(pred_Y.squeeze(),test_Y)
            logis_net.to(device)
            logis_net.eval()
            gprediction = logis_net(test_X)
            gprediction=gprediction.squeeze()
            human_candidates =torch.where(gprediction>thrs)[0]
            if cost==0:
                to_machine = [i for i in range(test_X.shape[0]) if i not in human_candidates ]
                if len(to_machine)==0:
                    print('error, no samples to machine')
                return np.mean(machine_loss[to_machine].cpu().numpy())
            else:
                to_machine = [i for i in range(test_X.shape[0]) if i not in human_candidates ]
                to_human = [i for i in range(test_X.shape[0]) if i not in to_machine]
                if len(to_machine)!=0:
                    total_error=(np.sum(machine_loss[to_machine].cpu().numpy())+cost*len(to_human))/test_X.shape[0]
                else:
                    total_error=cost
                to_human_ratio=len(to_human)/test_X.shape[0]
                return total_error,to_human_ratio

def test_loss_head_NN(test_data,mnet,loss_net,thrs,cost=0):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
            test_X= torch.tensor(test_data['X']).float().to(device)
            test_Y = torch.tensor(test_data['Y']).float().to(device)
            loss_net.to(device)
            loss_net.eval()
            mnet.to(device)
            mnet.eval()

            pred_Y= mnet(test_X)
            pred_Y=pred_Y.squeeze()
            machine_loss = torch.mean((pred_Y-test_Y)**2)
            pred_loss= loss_net(test_X)
            pred_loss=pred_loss.squeeze()
    
            #when the pred_loss is larger than thrs, triage to human
            if cost==0:
                flag=pred_loss<thrs
                coverage=1-torch.sum(flag)/len(flag)
                print('ratio for giving human:',coverage)
                loss=torch.mean((pred_Y[pred_loss<thrs]-test_Y[pred_loss<thrs])**2)
                print('machine loss:',loss)
            else:
                flag=pred_loss<cost
                coverage=1-torch.sum(flag)/len(flag)
                print('ratio for giving human:',coverage)
                loss=(torch.sum((pred_Y[pred_loss<cost]-test_Y[pred_loss<cost])**2)+cost*(len(flag)-torch.sum(flag)))/len(flag)
            return loss.detach().cpu().numpy(),coverage.detach().cpu().numpy()

    
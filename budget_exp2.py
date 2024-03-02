'''For Experiment 2 (fixed-budget). Several architectures are compared here:
0. Original SelNet with alpha=1/2
1.SelNet with alpha=1, thus no auxiliary loss and show its importance
2.fullNN+Sel head, an extension of SelNet with alpha=0
3.fullNN+Loss head, an extension of SelNet with alpha=0
'''
from knn import train_KNN_RwR,test_KNN_RwR
from NN_method import train_head,test_loss_head_NN,train_triage,train_g,get_test_assignments_triage,get_test_assignments_full_no_wid,train_full_NN
from uci_datasets import Dataset
from selectivenet import train_Sel_NN,test_Sel_NN,train_val_Sel_NN_budget,train_val_sel_head_budget
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import pandas as pd
#random seed
torch.manual_seed(12)
np.random.seed(12)

for data_name in ['concrete','wine','airfoil','energy','housing','solar','forest','parkinsons']:
    data = Dataset(data_name)
    sel_NN_loss_L=[]    
    sel_NN_coverage_L=[]
    sel_NN_1_loss_L=[]
    sel_NN_1_coverage_L=[]
    full_sel_loss_L=[]
    full_sel_coverage_L=[]
    full_loss_loss_L=[]
    full_loss_coverage_L=[]
    for constraint in [0.1,0.2,0.3]:
        sel_NN_loss=[]
        sel_NN_coverage=[]
        sel_NN_1_loss=[]
        sel_NN_1_coverage=[]
        full_sel_loss=[]
        full_sel_coverage=[]
        full_loss_loss=[]
        full_loss_coverage=[]
        for split in range(10):
            thrs_list=[0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99]
            #thrs_list=[0.5]
            X_train, Y_train, X_test, Y_test = data.get_split(split)
            Y_train=np.squeeze(Y_train)
            Y_test=np.squeeze(Y_test)
            X_train_, X_val, Y_train_, Y_val = train_test_split(X_train, Y_train,test_size=2/9,random_state=42)
            train_full={'X':X_train,'Y':Y_train}
            train_data={'X':X_train_,'Y':Y_train_}
            val_data={'X':X_val,'Y':Y_val}
            test_data={'X':X_test,'Y':Y_test}
       
            #train SelNet with alpha=1/2
            print('---')
            print('SelectiveNet with Alpha=1/2')
            #sel_net=train_Sel_NN(train_data,constraint)
            sel_net,thrs=train_val_Sel_NN_budget(train_data,val_data,constraint,0.5,thrs_list)
            loss,coverage=test_Sel_NN(test_data,sel_net,cost=0,no_aux=False,thrs=thrs)
            sel_NN_loss.append(loss)
            sel_NN_coverage.append(coverage)
            #train SelNet with alpha=1
            print('---')
            print('SelectiveNet with Alpha=1')
            sel_net,thrs=train_val_Sel_NN_budget(train_data,val_data,constraint,1,thrs_list)
            loss,coverage=test_Sel_NN(test_data,sel_net,cost=0,no_aux=False,thrs=thrs)
            sel_NN_1_loss.append(loss)
            sel_NN_1_coverage.append(coverage)
            #fullNN+sel head
            print('---')
            print('fullNN+sel head')
            mnet=train_full_NN(train_data)
            #sel_net=train_head(val_data,mnet,'sel',constraint,cost=0)
            sel_net,thrs=train_val_sel_head_budget(train_data,val_data,mnet,constraint,thrs_list)
            loss,coverage=test_Sel_NN(test_data,sel_net,cost=0,no_aux=True,thrs=thrs)
            full_sel_loss.append(loss)
            full_sel_coverage.append(coverage)
      
            #fullNN+loss head
            print('---')
            print('fullNN+loss head')
            mnet=train_full_NN(train_full)
            loss_net,thrs=train_head(train_full,mnet,'loss',constraint,cost=0)
            loss,coverage=test_loss_head_NN(test_data,mnet,loss_net,thrs,cost=0)
            full_loss_loss.append(loss)
            full_loss_coverage.append(coverage)
        sel_NN_loss_L.append("{:.2f}".format(np.mean(sel_NN_loss))+'+/-'+"{:.2f}".format(np.std(sel_NN_loss)))
        sel_NN_coverage_L.append("{:.2f}".format(np.mean(sel_NN_coverage))+'+/-'+"{:.2f}".format(np.std(sel_NN_coverage)))
        sel_NN_1_loss_L.append("{:.2f}".format(np.mean(sel_NN_1_loss))+'+/-'+"{:.2f}".format(np.std(sel_NN_1_loss)))
        sel_NN_1_coverage_L.append("{:.2f}".format(np.mean(sel_NN_1_coverage))+'+/-'+"{:.2f}".format(np.std(sel_NN_1_coverage)))
        full_sel_loss_L.append("{:.2f}".format(np.mean(full_sel_loss))+'+/-'+"{:.2f}".format(np.std(full_sel_loss)))
        full_sel_coverage_L.append("{:.2f}".format(np.mean(full_sel_coverage))+'+/-'+"{:.2f}".format(np.std(full_sel_coverage)))
        full_loss_loss_L.append("{:.2f}".format(np.mean(full_loss_loss))+'+/-'+"{:.2f}".format(np.std(full_loss_loss)))
        full_loss_coverage_L.append("{:.2f}".format(np.mean(full_loss_coverage))+'+/-'+"{:.2f}".format(np.std(full_loss_coverage)))
    print('_____________________________________________________________')
    print('_____________________________________________________________')
    print('_____________________________________________________________')
    print('____________________Result_________________________')
    print('sel_NN_loss',sel_NN_loss_L)
    print('sel_NN_coverage',sel_NN_coverage_L)
    print('sel_NN_1_loss',sel_NN_1_loss_L)
    print('sel_NN_1_coverage',sel_NN_1_coverage_L)
    print('full_sel_loss',full_sel_loss_L)
    print('full_sel_coverage',full_sel_coverage_L)
    print('full_loss_loss',full_loss_loss_L)
    print('full_loss_coverage',full_loss_coverage_L)
    df=pd.DataFrame({'sel_NN_loss':sel_NN_loss_L,'sel_NN_1_loss':sel_NN_1_loss_L,'full_sel_loss':full_sel_loss_L,'full_loss_loss':full_loss_loss_L,'sel_NN_coverage':sel_NN_coverage_L,'sel_NN_1_coverage':sel_NN_1_coverage_L,'full_sel_coverage':full_sel_coverage_L,'full_loss_coverage':full_loss_coverage_L})
    df.to_csv(data_name+'_comp.csv')

'For Experiment 1 (the fixed-cost case) in the paper'
'The UCI datasets used here are from https://github.com/treforevans/uci_datasets'



from knn import train_KNN_RwR,test_KNN_RwR
from NN_method import test_loss_head_NN,train_triage,train_g,get_test_assignments_triage,get_test_assignments_full_no_wid,train_full_NN,train_head,test_logis_head,train_val_logis_head
from uci_datasets import Dataset
from selectivenet import train_Sel_NN_cost,test_Sel_NN,train_sel_head_cost
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

for data_name in ['concrete','wine','airfoil','energy','housing','solar','forest','parkinsons']:
    data = Dataset(data_name)
    full_NN_loss_L=[]
    NN_triage_loss_L=[]
    sel_NN_loss_L=[]
    knn_loss_L=[]
    full_sel_loss_L=[]
    full_NN_coverage_L=[]
    NN_triage_coverage_L=[]
    sel_NN_coverage_L=[]
    knn_coverage_L=[]
    full_sel_loss_L=[]
    full_sel_coverage_L=[]
    full_logis_loss_L=[]
    full_logis_coverage_L=[]
    full_loss_loss_L=[]
    full_loss_coverage_L=[]
    constraint=1
    cost_list=[0.2,0.5,1,2]
    for cost in cost_list:
        
        full_NN_loss=[]
        full_NN_cal_loss=[]
        NN_triage_loss=[]
        sel_NN_loss=[]
        full_sel_loss=[]
        knn_loss=[]
        sel_NN_coverage=[]
        knn_coverage=[]
        full_NN_coverage=[]
        NN_triage_coverage=[]
        full_sel_coverage=[]
        full_logis_loss=[]
        full_logis_coverage=[]
        full_loss_loss=[]
        full_loss_coverage=[]
        for split in range(10):

            X_train, Y_train, X_test, Y_test = data.get_split(split)
            Y_train=np.squeeze(Y_train)
            Y_test=np.squeeze(Y_test)

            val=False

            X_train_, X_val, Y_train_, Y_val = train_test_split(X_train, Y_train,test_size=2/9,random_state=42)
            train_data={'X':X_train,'Y':Y_train}
            train_part={'X':X_train_,'Y':Y_train_}
            val_data={'X':X_val,'Y':Y_val}
            test_data={'X':X_test,'Y':Y_test}
            mnet_full=train_full_NN(train_data)
            print('---')
            print('NN+knnRej')
            loss,coverage=get_test_assignments_full_no_wid(train_data,test_data,mnet_full,constraint,cost)
            print('loss:',loss)
            full_NN_loss.append(loss)
            full_NN_coverage.append(coverage)
            print('---')

            print('Triage')
            mnet=train_triage(train_data,constraint,val,cost)
            gnet=train_g(train_data,constraint,mnet,val,False,cost)
            loss,coverage=get_test_assignments_triage(test_data,mnet,gnet,constraint,cost)
            NN_triage_loss.append(loss)
            NN_triage_coverage.append(coverage)
            print('---')

            print('KNN')
            knn,var_knn,ecdf=train_KNN_RwR(train_data)
            loss,coverage=test_KNN_RwR(test_data,knn,var_knn,ecdf,constraint,cost)
            knn_loss.append(loss)
            knn_coverage.append(coverage)
            print('---')

            print('SelectiveNet')
            sel_net=train_Sel_NN_cost(train_data,cost)
            loss,coverage=test_Sel_NN(test_data,sel_net,cost)
            sel_NN_loss.append(loss)
            sel_NN_coverage.append(coverage)
            print('---')

            print('NN+SelRej')
            mnet=train_full_NN(train_data)
            sel_net=train_sel_head_cost(train_data,mnet,cost)
            loss,coverage=test_Sel_NN(test_data,sel_net,cost,no_aux=True)
            full_sel_loss.append(loss)
            full_sel_coverage.append(coverage)
            print('---')

            print('NN+LogRej')
            thrs_list=[0.5]
            mnet,rej_net,thrs=train_val_logis_head(train_data,constraint,cost,thrs_list)
            loss,coverage=test_logis_head(test_data,mnet,rej_net,cost,thrs)
            full_logis_loss.append(loss)
            full_logis_coverage.append(coverage)
            print('---')

            print('NN+LossRej')
            mnet=train_full_NN(train_data)
            loss_net,thrs=train_head(train_data,mnet,'loss',constraint,cost)
            loss,coverage=test_loss_head_NN(test_data,mnet,loss_net,thrs,cost)
            full_loss_loss.append(loss)
            full_loss_coverage.append(coverage)
            
        full_NN_loss_L.append("{:.2f}".format(np.mean(full_NN_loss))+'+/-'+"{:.2f}".format(np.std(full_NN_loss)))
        NN_triage_loss_L.append("{:.2f}".format(np.mean(NN_triage_loss))+'+/-'+"{:.2f}".format(np.std(NN_triage_loss)))
        sel_NN_loss_L.append("{:.2f}".format(np.mean(sel_NN_loss))+'+/-'+"{:.2f}".format(np.std(sel_NN_loss)))
        knn_loss_L.append("{:.2f}".format(np.mean(knn_loss))+'+/-'+"{:.2f}".format(np.std(knn_loss)))
        sel_NN_coverage_L.append("{:.2f}".format(np.mean(sel_NN_coverage))+'+/-'+"{:.2f}".format(np.std(sel_NN_coverage)))
        knn_coverage_L.append("{:.2f}".format(np.mean(knn_coverage))+'+/-'+"{:.2f}".format(np.std(knn_coverage)))
        full_NN_coverage_L.append("{:.2f}".format(np.mean(full_NN_coverage))+'+/-'+"{:.2f}".format(np.std(full_NN_coverage)))
        NN_triage_coverage_L.append("{:.2f}".format(np.mean(NN_triage_coverage))+'+/-'+"{:.2f}".format(np.std(NN_triage_coverage)))
        full_sel_loss_L.append("{:.2f}".format(np.mean(full_sel_loss))+'+/-'+"{:.2f}".format(np.std(full_sel_loss)))
        full_logis_loss_L.append("{:.2f}".format(np.mean(full_logis_loss))+'+/-'+"{:.2f}".format(np.std(full_logis_loss)))
        full_logis_coverage_L.append("{:.2f}".format(np.mean(full_logis_coverage))+'+/-'+"{:.2f}".format(np.std(full_logis_coverage)))
        full_sel_coverage_L.append("{:.2f}".format(np.mean(full_sel_coverage))+'+/-'+"{:.2f}".format(np.std(full_sel_coverage)))
        full_loss_loss_L.append("{:.2f}".format(np.mean(full_loss_loss))+'+/-'+"{:.2f}".format(np.std(full_loss_loss)))
        full_loss_coverage_L.append("{:.2f}".format(np.mean(full_loss_coverage))+'+/-'+"{:.2f}".format(np.std(full_loss_coverage)))
    print('_____________________________________________________________')
    print('_____________________________________________________________')
    print('_____________________________________________________________')
    print('____________________Finishied_________________________')
    df=pd.DataFrame({'NN+knnRej':full_NN_loss_L,'NN+SelRej':full_sel_loss_L,'NN+LossRej':full_loss_loss_L,
                     'NN+LogRej':full_logis_loss_L,'Triage':NN_triage_loss_L,
                     'SelNN':sel_NN_loss_L,'kNN':knn_loss_L,
                     'NN+knnRej':full_NN_coverage_L,'NN+SelRej':full_sel_coverage_L,'NN+LossRej':full_logis_coverage_L,
                     'NN+LogRej':full_logis_coverage_L,'Triage':NN_triage_coverage_L,
                     'SelNN':sel_NN_coverage_L,'kNN':knn_coverage_L, },index=cost_list)
    df.to_csv(data_name+'_new.csv')

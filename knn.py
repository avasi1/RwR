'Functions for training and testing KNN methods'
import sklearn
import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF
from sklearn.model_selection import cross_val_score
import sklearn.neighbors
import sklearn.datasets
from sklearn.model_selection import train_test_split
#train knn epsilon-plug-in  method
def train_pred_knn_base(data,n_neighbors):
    X =data['X']
    Y = data['Y']
    knn = sklearn.neighbors.KNeighborsRegressor(n_neighbors)
    scores = cross_val_score(knn, X, Y, cv=10)
    result=np.mean(scores)
    knn.fit(X, Y)
    var_loss =np.zeros(X.shape[0])
    var_loss = (knn.predict(X)-Y)**2
    return knn,var_loss,result



def train_var_knn_base(data,var_loss,n_neighbors):
    X =data['X']
    var_knn = sklearn.neighbors.KNeighborsRegressor(n_neighbors)
    var_knn.fit(X, var_loss)
    return var_knn

def train_KNN(data):
    #make 10 folds cross validation
    knn=[]
    var_loss=[]
    result=[]
    n_list = [5, 10, 15, 20, 30, 50, 70, 100, 150]
    for n in n_list:
        knn_t,var_loss_t,result_t=train_pred_knn_base(data,n)
        knn.append(knn_t)
        var_loss.append(var_loss_t)
        result.append(result_t)
    idx=np.argmax(result)
    print('best n_neighbors:',n_list[idx],'best train loss:',np.mean(var_loss[idx]))
    return knn[idx],var_loss[idx]

def train_var_knn(data,var_loss):
    #make 10 folds cross validation
    var_knn=[]
    result=[]
    n_list = [5, 10, 15, 20, 30, 50, 70, 100, 150]
    for n in n_list:
        var_knn_t=train_var_knn_base(data,var_loss,n)
        var_knn.append(var_knn_t)
        result_t=var_knn_t.score(data['X'],var_loss)
        result.append(result_t)
    idx=np.argmax(result)
    print('best n_neighbors:',n_list[idx])
    return var_knn[idx]


def get_ECDF(val_data,var_knn):
    X =val_data['X']
    vars=var_knn.predict(X)+np.random.uniform(0,0.0001,X.shape[0])
    ecdf = ECDF(vars)
    return ecdf



def train_KNN_RwR(data):
    X_train, X_val, Y_train, Y_val = train_test_split(data['X'], data['Y'],test_size=0.2,random_state=42)
    train_data={'X':X_train,'Y':Y_train}
    val_data={'X':X_val,'Y':Y_val}
    knn,var_loss=train_KNN(train_data)
    var_knn=train_var_knn(train_data,var_loss)
    ecdf=get_ECDF(val_data,var_knn)
    return knn,var_knn,ecdf


def test_KNN_RwR(data,knn,var_knn,ecdf,constraint,cost=0):
    if cost==0:
        X_test=data['X']
        Y_test=data['Y']
        vars=var_knn.predict(X_test)+np.random.uniform(0,0.0001,X_test.shape[0])
        score= ecdf(vars)
        flag=(score<=1-constraint)
        accetp_rate=np.sum(flag)/X_test.shape[0]
        X_test_m=X_test[flag]
        Y_test_m=Y_test[flag]
        Y_pred_m=knn.predict(X_test_m)
        loss=np.mean((Y_pred_m-Y_test_m)**2)
        print("machine loss = ",loss)
        print("human rate = ",1-accetp_rate)
        return loss,1-accetp_rate
    else:
        X_test=data['X']
        Y_test=data['Y']
        vars=var_knn.predict(X_test)
        flag=(vars<cost)

        if np.sum(flag)>0:
            accetp_rate=np.sum(flag)/X_test.shape[0]
            X_test_m=X_test[flag]
            Y_test_m=Y_test[flag]
            Y_pred_m=knn.predict(X_test_m)
            loss=(np.sum((Y_pred_m-Y_test_m)**2)+cost*(X_test.shape[0]-np.sum(flag)))/X_test.shape[0]
            print("total loss = ",loss)
            print("human rate = ",1-accetp_rate)
            return loss,1-accetp_rate
        else:
            print("total loss = ",cost)
            print("human rate = ",1)
            return cost,1



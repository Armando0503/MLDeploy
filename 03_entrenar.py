import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


def training(train_name, filename='mlparams'):
    log_model = LogisticRegression()
    df = pd.read_csv(train_name+'.csv')
    X_train = pd.DataFrame()
    X_train['age'] = df['age']
    X_train['physical_score'] = df['physical_score']
    y= df['test_result']
    log_model.fit(X_train,y)
    y_pred = log_model.predict(X_train)
    
    pickle.dump(log_model, open(filename, 'wb'))
    print('Parametros escritos en archivo mlparams')
    return (y_pred,y)

def accuracy(test, pred):
    return accuracy_score(test, pred)

def conf_matrix(test, pred):
    return confusion_matrix(test, pred)

def validation(val_name,filename='mlparams'):
    log_model=pickle.load(open(filename, 'rb'))
    df = pd.read_csv(val_name+'.csv')
    X_train = pd.DataFrame()
    X_train['age'] = df['age']
    X_train['physical_score'] = df['physical_score']
    y= df['test_result']
    y_pred = log_model.predict(X_train)
    return (y_pred,y)

(yP,yL)=training('train_2')
print(accuracy(yP,yL))
print(conf_matrix(yP,yL))
print(classification_report(yP,yL))
(yP,yL)=validation('val')
print(accuracy(yP,yL))
print(conf_matrix(yP,yL))
print(classification_report(yP,yL))
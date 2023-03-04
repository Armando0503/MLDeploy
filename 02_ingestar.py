#carga
import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(dataframe_name, validation=False):
    df = pd.read_csv(dataframe_name+".csv")
    X = df.drop('test_result',axis=1) 
    y = df['test_result']
    
    if validation == False: 
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)
      dataset_train = pd.DataFrame()
      dataset_train['age']= X_train['age']
      dataset_train['physical_score']= X_train['physical_score']
      dataset_train['test_result']= y_train
      dataset_train.to_csv('train.csv',index=False)
      print("Dataset train.csv listo ...")
      dataset_test = pd.DataFrame()
      dataset_test['age']= X_test['age']
      dataset_test['physical_score']= X_test['physical_score']
      dataset_test['test_result']= y_test
      dataset_test.to_csv('test.csv',index=False)
      print("Dataset test.csv listo ...")
    else:
      X_train2, X_val, y_train2, y_val = train_test_split(X, y, test_size=0.1, random_state=101)
      dataset_train2 = pd.DataFrame()
      dataset_train2['age']= X_train2['age']
      dataset_train2['physical_score']= X_train2['physical_score']
      dataset_train2['test_result']= y_train2
      dataset_train2.to_csv('train_2.csv',index=False)
      print("Dataset train.csv listo ...")
      dataset_val = pd.DataFrame()
      dataset_val['age']= X_val['age']
      dataset_val['physical_score']= X_val['physical_score']
      dataset_val['test_result']= y_val
      dataset_val.to_csv('val.csv',index=False)
      print("Dataset val.csv listo ...")
  
    return
    

split_data('hearing_test', validation = False)

split_data('train', validation = True)
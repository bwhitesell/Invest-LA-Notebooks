import pymongo
from datetime import timedelta, datetime
import numpy as np


eval_date = datetime(year=2018, month=9, day=1)


def build_t_filtered_datasets(eval_date, delta_days, ds, target):
    dat = ds
    train_index=np.where((dat[:,5]<eval_date-timedelta(days=90))&(dat[:,5]>eval_date-timedelta(days=90+delta_days)))
    test_index=np.where((dat[:,5]>=eval_date)&(dat[:,5]<eval_date+timedelta(days=30)))
    dat = np.delete(dat, 5, axis=1)

    train = dat[train_index]
    train_target =  target[train_index]
    test = dat[test_index]
    test_target =  target[test_index]
    return train, test, train_target, test_target
    

def build_dataset():
    mongo_url = 'mongodb://' + '127.0.0.1:27017'
    db = pymongo.MongoClient(mongo_url)['Housing']
    mongo_cursor = db['CTD'].find({})
    features = ['SqftLot', 'SqftMain', 'UsableSqftLot', 'NumOfBaths', 'NumOfBeds', 'RecordingDate',
                 'EffectiveYear', 'Latitude', 'Longitude', 'days_since_last_sale', 'last_sales_price']
    ds = []
    target = []
    for doc in mongo_cursor:
        if doc['ain'] != '4362002013':
            feature_vec = np.array([doc['SqftLot'],
                                    doc['SqftMain'],
                                    doc['UsableSqftLot'],
                                    doc['NumOfBaths'],
                                    doc['NumOfBeds'],
                                    doc['RecordingDate'],
                                    doc['EffectiveYear'],
                                    doc['Latitude'],
                                    doc['Longitude'],
                                    doc['days_since_last_sale'],
                                    doc['last_sales_price'],
                                    ])
            ds.append(feature_vec)
            target.append(np.array(doc['DTTSalePrice']))
    ds = np.array(ds)
    target = np.array(target)
    return ds, target


def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def mean_log_err(y_true, y_pred):
    return np.mean(np.abs(np.log(y_pred) - np.log(y_true)))
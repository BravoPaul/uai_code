 # -*- coding: utf-8 -*
import datetime
import time
from datetime import datetime, timedelta
import datetime
import time
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from pandas import DataFrame
from pandas import Series
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
# from matplotlib import pyplot

# 此模块用于最基本的数据梳理
july_order = pd.read_csv('data/train_July.csv')
july_order['num'] = 1
july_order_25_31 = july_order[july_order['create_date']>'2017-07-24'][['start_geo_id','end_geo_id','create_date','create_hour','num']]
july_order_0725_d = july_order_25_31[(july_order_25_31['create_date']=='2017-07-25')&(july_order_25_31['create_hour']%2 ==0)].groupby(['start_geo_id','end_geo_id','create_date'])['num'].sum()
july_order_0726_s = july_order_25_31[(july_order_25_31['create_date']=='2017-07-26')&(july_order_25_31['create_hour']%2 !=0)].groupby(['start_geo_id','end_geo_id','create_date'])['num'].sum()
july_order_0727_d = july_order_25_31[(july_order_25_31['create_date']=='2017-07-27')&(july_order_25_31['create_hour']%2 ==0)].groupby(['start_geo_id','end_geo_id','create_date'])['num'].sum()
july_order_0728_s = july_order_25_31[(july_order_25_31['create_date']=='2017-07-28')&(july_order_25_31['create_hour']%2 !=0)].groupby(['start_geo_id','end_geo_id','create_date'])['num'].sum()
july_order_0729_d = july_order_25_31[(july_order_25_31['create_date']=='2017-07-29')&(july_order_25_31['create_hour']%2 ==0)].groupby(['start_geo_id','end_geo_id','create_date'])['num'].sum()
july_order_0730_s = july_order_25_31[(july_order_25_31['create_date']=='2017-07-30')&(july_order_25_31['create_hour']%2 !=0)].groupby(['start_geo_id','end_geo_id','create_date'])['num'].sum()
july_order_0731_d = july_order_25_31[(july_order_25_31['create_date']=='2017-07-31')&(july_order_25_31['create_hour']%2 ==0)].groupby(['start_geo_id','end_geo_id','create_date'])['num'].sum()
order_25_31 = pd.concat([july_order_0725_d,july_order_0726_s,july_order_0727_d,july_order_0728_s,july_order_0729_d,july_order_0730_s,july_order_0731_d]).reset_index()
july_order_test = july_order_25_31.groupby(['start_geo_id','end_geo_id','create_date'])['num'].sum().reset_index()
july_order_test_with_label = pd.merge(july_order_test,order_25_31,on = ['start_geo_id','end_geo_id','create_date'],how = 'left')
july_order_train = july_order[july_order['create_date']<='2017-07-24']
july_order_test_with_label = july_order_test_with_label.fillna(0)
july_order_test_with_label['num'] = july_order_test_with_label['num_x'] - july_order_test_with_label['num_y']
del july_order_test['num']
del july_order_test_with_label['num_x']
del july_order_test_with_label['num_y']
print 'train_pre_traitor done'
# print july_order_test_with_label
# 此模块用于最基本的数据梳理
july_order = pd.read_csv('data/train_July.csv')
aug_order = pd.read_csv('data/train_Aug.csv')
july_order = july_order[july_order['create_date']>'2017-07-07']
aug_order_train = pd.concat([july_order,aug_order]).reset_index()
aug_order_train['num'] = 1
order_1_7 = aug_order_train[aug_order_train['create_date']>='2017-08-01'].groupby(['start_geo_id','end_geo_id','create_date'])['num'].sum().reset_index()
aug_order_test = pd.read_csv('data/test_id_Aug_agg_public5k.csv')[['start_geo_id','end_geo_id','create_date']].drop_duplicates()
print 'test_pre_traitor done'


def f1_num(time1,time2,yongtu='xunlian'):
    if yongtu=='xunlian':
        order_train = july_order_train
        order_test = july_order_test
    else:
        order_train = aug_order_train
        order_test = aug_order_test
    # 用中位数衡量
    order_train = order_train[(order_train['create_date']>=time1) & (order_train['create_date']<=time2)]
    order_train_sigle = order_train[['start_geo_id','end_geo_id','create_date']].drop_duplicates()
    f1_tmp_0 = order_train_sigle.groupby(['start_geo_id','end_geo_id'],as_index=False)['create_date'].agg({'day_count'+time1:'count'})
    f1_tmp_1 = order_train.groupby(['start_geo_id','end_geo_id','create_date'],as_index=False)['num'].agg({'hour_count'+time1:'sum'})
    # 这里有个非常牛逼的东西，就是改变mean，median，std ，min，max，分别计算对应的值
    f1_tmp= f1_tmp_1.groupby(['start_geo_id','end_geo_id'],as_index=False)['hour_count'+time1].agg({'hour_median'+time1:'median','hour_std'+time1:'std','hour_min'+time1:'min','hour_max'+time1:'max'})
    # 拼接
    selected_data = order_test.copy()
    feature = pd.merge(selected_data,f1_tmp_0,on=['start_geo_id','end_geo_id'],how='left')
    feature = pd.merge(feature,f1_tmp,on=['start_geo_id','end_geo_id'],how='left')
    return feature


def f1_num_week(time1,time2,yongtu='xunlian'):
    # 求25,26,27,28,29,30,31 这样的日期对应的星期，然后['start_geo_id','end_geo_id','week','create_hour'] 这样的key下的平均
    if yongtu=='xunlian':
        order_train = july_order_train
        order_test = july_order_test
    else:
        order_train = aug_order_train
        order_test = aug_order_test
    order_train = order_train[(order_train['create_date']>=time1) & (order_train['create_date']<=time2)]
    order_train_sigle = order_train[['start_geo_id','end_geo_id','create_date']].drop_duplicates()
    order_train_sigle['week'] = pd.to_datetime(order_train_sigle['create_date']).dt.weekday
    order_train['week'] = pd.to_datetime(order_train['create_date']).dt.weekday
    f1_tmp_0 = order_train_sigle.groupby(['start_geo_id','end_geo_id','week'],as_index=False)['create_date'].agg({'day_week_count'+time1:'count'})
    f1_tmp_1 = order_train.groupby(['start_geo_id','end_geo_id','week','create_date'],as_index=False)['num'].agg({'hour_week_count'+time1:'count'})
    f1_tmp= f1_tmp_1.groupby(['start_geo_id','end_geo_id','week'],as_index=False)['hour_week_count'+time1].agg({'hour_week_median'+time1:'median','hour_week_std'+time1:'std','hour_week_min'+time1:'min','hour_week_max'+time1:'max'})
    
    selected_data = order_test.copy()
    selected_data['week'] = pd.to_datetime(selected_data['create_date']).dt.weekday
    feature = pd.merge(selected_data,f1_tmp_0,on=['start_geo_id','end_geo_id','week'],how='left')
    feature = pd.merge(feature,f1_tmp,on=['start_geo_id','end_geo_id','week'],how='left')
    return feature


def features_num(yongtu='xunlian'):
    if yongtu=='xunlian':
        f1_num_24 = f1_num('2017-07-01', '2017-07-24',yongtu)
        f1_num_7 = f1_num('2017-07-18', '2017-07-24' ,yongtu)
        f1_num_3 = f1_num('2017-07-22', '2017-07-24' ,yongtu)
        f1_num_1 = f1_num('2017-07-24', '2017-07-24' ,yongtu)
        f1_num_week_24 = f1_num_week('2017-07-01', '2017-07-24' ,yongtu) 
        f1_num_week_14 = f1_num_week('2017-07-11', '2017-07-24' ,yongtu)
        f1_num_week_7 = f1_num_week('2017-07-18', '2017-07-24',yongtu)
        f1_num_total = pd.concat([july_order_0725_d,july_order_0726_s,july_order_0727_d,july_order_0728_s,july_order_0729_d,july_order_0730_s,july_order_0731_d]).reset_index()  
#         between_day = f1_num_between_day('2017-07-24', '2017-07-31',yongtu)
#         between_hour = f1_num_between_hour('2017-07-24', '2017-07-31',yongtu)
    else:
        f1_num_24 = f1_num('2017-07-08', '2017-07-31',yongtu)
        f1_num_7 = f1_num('2017-07-25', '2017-07-31' ,yongtu)
        f1_num_3 = f1_num('2017-07-29', '2017-07-31' ,yongtu)
        f1_num_1 = f1_num('2017-07-31', '2017-07-31' ,yongtu)
        f1_num_week_24 = f1_num_week('2017-07-08', '2017-07-31' ,yongtu) 
        f1_num_week_14 = f1_num_week('2017-07-18', '2017-07-31' ,yongtu)
        f1_num_week_7 = f1_num_week('2017-07-24', '2017-07-31',yongtu)
        f1_num_total = order_1_7 
#         between_day = f1_num_between_day('2017-07-31', '2017-08-07',yongtu)
#         between_hour = f1_num_between_hour('2017-07-31', '2017-08-07',yongtu)

    features = pd.merge(f1_num_24,f1_num_7,on=['start_geo_id','end_geo_id','create_date'],how='left')
    features = pd.merge(features,f1_num_3,on=['start_geo_id','end_geo_id','create_date'],how='left')
    features = pd.merge(features,f1_num_1,on=['start_geo_id','end_geo_id','create_date'],how='left')
    features = pd.merge(features,f1_num_week_24,on=['start_geo_id','end_geo_id','create_date'],how='left')
    features = pd.merge(features,f1_num_week_14,on=['start_geo_id','end_geo_id','create_date'],how='left')
    features = pd.merge(features,f1_num_week_7,on=['start_geo_id','end_geo_id','create_date'],how='left')
    features = pd.merge(features,order_25_31,on=['start_geo_id','end_geo_id','create_date'],how='left')
    features = pd.merge(features,f1_num_total,on=['start_geo_id','end_geo_id','create_date'],how='left')    
#     features = pd.merge(features,between_day,on=['start_geo_id','end_geo_id','create_date','create_hour'],how='left')
#     features = pd.merge(features,between_hour,on=['start_geo_id','end_geo_id','create_date','create_hour'],how='left')
    return features


def trainning_test():
    features_train = features_num()
    print features_train.count()
#     print features_train

    del features_train['start_geo_id']
    del features_train['end_geo_id']
    del features_train['create_date']
#   del features_train['create_hour']
    label = july_order_test_with_label['num']

    dtrain, dtest, dtrain_y, dtest_y = train_test_split(features_train, label, test_size=0.2, random_state=42)

    xgb_model = xgb.XGBClassifier(max_depth=10, learning_rate=0.05, n_estimators=500).fit(dtrain, dtrain_y)
    predictions = xgb_model.predict(dtest)
    actuals = dtest_y
    s_prediciotn = pd.Series(predictions)
    s_label = pd.Series(actuals).reset_index()
    del s_label['index']
    result = s_prediciotn.to_frame()
    result['label'] = s_label
    result.to_csv("result.csv")
    print(mean_absolute_error(actuals, predictions))
    return xgb_model

# 训练节测试集构造
def trainning():
    features_train = features_num()
    rows, columns = features_train.shape
    del features_train['start_geo_id']
    del features_train['end_geo_id']
    del features_train['create_date']
    label = july_order_test_with_label['num']
    features_train = features_train.values
    dtrain, dtest, dtrain_y, dtest_y = train_test_split(features_train, label, test_size=0.0, random_state=42)
    xgb_model = xgb.XGBRegressor(max_depth=5, learning_rate=0.05, n_estimators=200).fit(dtrain, dtrain_y)
    return xgb_model


def testing():
    features_test = features_num('ceshi')
    rows, columns = features_test.shape
    features_index = features_test[['start_geo_id','end_geo_id','create_date']]
    del features_test['start_geo_id']
    del features_test['end_geo_id']
    del features_test['create_date']
    return features_test.values,features_index

model = trainning()
test,test_index = testing()
result = model.predict(test)
test_index['num_day'] = result
test_index.to_csv("daydayNum.csv")
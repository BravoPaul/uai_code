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
from matplotlib import pyplot


# 此模块用于最基本的数据梳理
july_order = pd.read_csv('data/train_July.csv')
july_order['num'] = 1
july_order_25_31 = july_order[july_order['create_date']>'2017-07-24'][['start_geo_id','end_geo_id','create_date','create_hour','num']]
july_order_0725_d = july_order_25_31[(july_order_25_31['create_date']=='2017-07-25')&(july_order_25_31['create_hour']%2 !=0)].groupby(['start_geo_id','end_geo_id','create_date','create_hour'])['num'].sum()
july_order_0726_s = july_order_25_31[(july_order_25_31['create_date']=='2017-07-26')&(july_order_25_31['create_hour']%2 ==0)].groupby(['start_geo_id','end_geo_id','create_date','create_hour'])['num'].sum()
july_order_0727_d = july_order_25_31[(july_order_25_31['create_date']=='2017-07-27')&(july_order_25_31['create_hour']%2 !=0)].groupby(['start_geo_id','end_geo_id','create_date','create_hour'])['num'].sum()
july_order_0728_s = july_order_25_31[(july_order_25_31['create_date']=='2017-07-28')&(july_order_25_31['create_hour']%2 ==0)].groupby(['start_geo_id','end_geo_id','create_date','create_hour'])['num'].sum()
july_order_0729_d = july_order_25_31[(july_order_25_31['create_date']=='2017-07-29')&(july_order_25_31['create_hour']%2 !=0)].groupby(['start_geo_id','end_geo_id','create_date','create_hour'])['num'].sum()
july_order_0730_s = july_order_25_31[(july_order_25_31['create_date']=='2017-07-30')&(july_order_25_31['create_hour']%2 ==0)].groupby(['start_geo_id','end_geo_id','create_date','create_hour'])['num'].sum()
july_order_0731_d = july_order_25_31[(july_order_25_31['create_date']=='2017-07-31')&(july_order_25_31['create_hour']%2 !=0)].groupby(['start_geo_id','end_geo_id','create_date','create_hour'])['num'].sum()
july_order_1 = july_order.drop(july_order[(july_order['create_date']=='2017-07-25')&(july_order['create_hour']%2 !=0)].index)
july_order_2 = july_order_1.drop(july_order_1[(july_order_1['create_date']=='2017-07-26')&(july_order_1['create_hour']%2 ==0)].index)
july_order_3 = july_order_2.drop(july_order_2[(july_order_2['create_date']=='2017-07-27')&(july_order_2['create_hour']%2 !=0)].index)
july_order_4 = july_order_3.drop(july_order_3[(july_order_3['create_date']=='2017-07-28')&(july_order_3['create_hour']%2 ==0)].index)
july_order_5 = july_order_4.drop(july_order_4[(july_order_4['create_date']=='2017-07-29')&(july_order_4['create_hour']%2 !=0)].index)
july_order_6 = july_order_5.drop(july_order_5[(july_order_5['create_date']=='2017-07-30')&(july_order_5['create_hour']%2 ==0)].index)
july_order_train = july_order_6.drop(july_order_6[(july_order_6['create_date']=='2017-07-31')&(july_order_6['create_hour']%2 !=0)].index)
july_order_test = pd.concat([july_order_0725_d,july_order_0726_s,july_order_0727_d,july_order_0728_s,july_order_0729_d,july_order_0730_s,july_order_0731_d]).reset_index()
july_order_test_with_label = july_order_test.copy()
del july_order_test['num']
print 'train_pre_traitor done'

# 此模块用于最基本的数据梳理
july_order = pd.read_csv('data/train_July.csv')
aug_order = pd.read_csv('data/train_Aug.csv')
july_order = july_order[july_order['create_date']>'2017-07-07']
aug_order_train = pd.concat([july_order,aug_order]).reset_index()
aug_order_train['num'] = 1
aug_order_test = pd.read_csv('data/test_id_Aug_agg_public5k.csv')
del aug_order_test['test_id']
print 'test_pre_traitor done'


# 如果是训练集time1 = '2017-07-01' , time2 =  '2017-07-24' 
def f1_num(time1,time2,yongtu='xunlian'):
    if yongtu=='xunlian':
        order_train = july_order_train
        order_test = july_order_test
    else:
        order_train = aug_order_train
        order_test = aug_order_test
    # 用中位数衡量
    order_train = order_train[(order_train['create_date']>=time1) & (order_train['create_date']<=time2)]
    order_train_sigle = order_train[['start_geo_id','end_geo_id','create_date','create_hour']].drop_duplicates()
    f1_tmp_0 = order_train_sigle.groupby(['start_geo_id','end_geo_id','create_hour'],as_index=False)['create_date'].agg({'day_count'+time1:'count'})
    f1_tmp_1 = order_train.groupby(['start_geo_id','end_geo_id','create_date','create_hour'],as_index=False)['num'].agg({'hour_count'+time1:'sum'})
    # 这里有个非常牛逼的东西，就是改变mean，median，std ，min，max，分别计算对应的值
    f1_tmp= f1_tmp_1.groupby(['start_geo_id','end_geo_id','create_hour'],as_index=False)['hour_count'+time1].agg({'hour_median'+time1:'median','hour_std'+time1:'std','hour_min'+time1:'min','hour_max'+time1:'max'})
    # 拼接
    selected_data = order_test.copy()
    feature = pd.merge(selected_data,f1_tmp_0,on=['start_geo_id','end_geo_id','create_hour'],how='left')
    feature = pd.merge(feature,f1_tmp,on=['start_geo_id','end_geo_id','create_hour'],how='left')
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
    order_train_sigle = order_train[['start_geo_id','end_geo_id','create_date','create_hour']].drop_duplicates()
    order_train_sigle['week'] = pd.to_datetime(order_train_sigle['create_date']).dt.weekday
    order_train['week'] = pd.to_datetime(order_train['create_date']).dt.weekday
    f1_tmp_0 = order_train_sigle.groupby(['start_geo_id','end_geo_id','week','create_hour'],as_index=False)['create_date'].agg({'day_week_count'+time1:'count'})
    f1_tmp_1 = order_train.groupby(['start_geo_id','end_geo_id','week','create_hour','create_date'],as_index=False)['num'].agg({'hour_week_count'+time1:'count'})
    f1_tmp= f1_tmp_1.groupby(['start_geo_id','end_geo_id','create_hour','week'],as_index=False)['hour_week_count'+time1].agg({'hour_week_median'+time1:'median','hour_week_std'+time1:'std','hour_week_min'+time1:'min','hour_week_max'+time1:'max'})
    
    selected_data = order_test.copy()
    selected_data['week'] = pd.to_datetime(selected_data['create_date']).dt.weekday
    feature = pd.merge(selected_data,f1_tmp_0,on=['start_geo_id','end_geo_id','week','create_hour'],how='left')
    feature = pd.merge(feature,f1_tmp,on=['start_geo_id','end_geo_id','week','create_hour'],how='left')
    return feature

# xunlianji: time1 = '2017-07-24' ,time2 = '2017-07-31'
def f1_num_between_hour(time1,time2,yongtu='xunlian'):
    if yongtu=='xunlian':
        order_train = july_order_train
        order_test = july_order_test
    else:
        order_train = aug_order_train
        order_test = aug_order_test
    order_train_num_25_31 = order_train[(order_train['create_date']>=time1) & (order_train['create_date']<=time2)]
    f2_tem_6 = order_train_num_25_31.copy()
    f2_tem_6['create_hour'] = f2_tem_6['create_hour'].map(lambda x: 23 if x-1<0 else x-1)
    f2_tem_6.loc[f2_tem_6['create_hour'] == 23, 'create_date'] = f2_tem_6['create_date'].map(lambda x: (datetime.strptime(x,'%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d'))
    f2_6 = f2_tem_6.groupby(['start_geo_id','end_geo_id','create_date','create_hour'],as_index = False)['num'].agg({"hou_hour": "count"})
    f2_tem_7 = order_train_num_25_31.copy()
    f2_tem_7['create_hour'] = f2_tem_7['create_hour'].map(lambda x: 0 if x+1>23 else x+1)
    f2_tem_7.loc[f2_tem_7['create_hour'] == 0, 'create_date'] = f2_tem_7['create_date'].map(lambda x: (datetime.strptime(x,'%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d'))
    f2_7 = f2_tem_7.groupby(['start_geo_id','end_geo_id','create_date','create_hour'],as_index = False)['num'].agg({"qian_hour": "count"})

    feature = order_test.copy()
    feature = pd.merge(feature,f2_6,on=['start_geo_id','end_geo_id','create_date','create_hour'],how='left')
    feature = pd.merge(feature,f2_7,on=['start_geo_id','end_geo_id','create_date','create_hour'],how='left')
    features = feature.fillna(0)
    features['between_hour'] = (features['hou_hour']+features['qian_hour'])/2
    return features


# xunlianji: time1 = '2017-07-24' ,time2 = '2017-07-31'
def f1_num_between_day(time1,time2,yongtu='xunlian'):
    if yongtu=='xunlian':
        order_train = july_order_train
        order_train_30 = order_train[order_train['create_date']=='2017-07-30']
        order_train_30['create_date'] = '2017-08-01'
        order_train = order_train.append(order_train_30)
        order_test = july_order_test
    else:
        order_train = aug_order_train
        order_train_30 = order_train[order_train['create_date']=='2017-08-06']
        order_train_30['create_date'] = '2017-08-08'
        order_train = order_train.append(order_train_30)
        order_test = aug_order_test
    order_train_num_25_31 = order_train[(order_train['create_date']>=time1) & (order_train['create_date']<=time2)]
    f2_tem_5 = order_train_num_25_31.copy()
    f2_tem_5
    f2_tem_5['create_date'] = pd.to_datetime(pd.DatetimeIndex(pd.to_datetime(f2_tem_5['create_date']))- pd.DateOffset(1))
    f2_tem_5['create_date'] = f2_tem_5['create_date'].map(lambda x: x.strftime('%Y-%m-%d'))
    f2_5_1 = f2_tem_5.groupby(['start_geo_id','end_geo_id','create_date','create_hour'],as_index = False)['num'].agg({'hou_tian':'count'})
    f2_tem_5 = order_train_num_25_31.copy()
    f2_tem_5['create_date'] = pd.to_datetime(pd.DatetimeIndex(pd.to_datetime(f2_tem_5['create_date']))+ pd.DateOffset(1))
    f2_tem_5['create_date'] = f2_tem_5['create_date'].map(lambda x: x.strftime('%Y-%m-%d'))
    f2_5_2 = f2_tem_5.groupby(['start_geo_id','end_geo_id','create_date','create_hour'],as_index = False)['num'].agg({'qian_tian':'count'})
    feature = july_order_test.copy()
    feature = pd.merge(feature,f2_5_1,on=['start_geo_id','end_geo_id','create_date','create_hour'],how='left')
    feature = pd.merge(feature,f2_5_2,on=['start_geo_id','end_geo_id','create_date','create_hour'],how='left')
    features = feature.fillna(0)
    features['between_tian'] = (features['qian_tian']+features['hou_tian'])/2
    return features





def features_num(yongtu='xunlian'):
    if yongtu=='xunlian':
        f1_num_24 = f1_num('2017-07-01', '2017-07-24',yongtu)
        f1_num_7 = f1_num('2017-07-18', '2017-07-24' ,yongtu)
        f1_num_3 = f1_num('2017-07-22', '2017-07-24' ,yongtu)
        f1_num_1 = f1_num('2017-07-24', '2017-07-24' ,yongtu)
        f1_num_week_24 = f1_num_week('2017-07-01', '2017-07-24' ,yongtu) 
        f1_num_week_14 = f1_num_week('2017-07-11', '2017-07-17' ,yongtu)
        f1_num_week_7 = f1_num_week('2017-07-18', '2017-07-24',yongtu)
        between_day = f1_num_between_day('2017-07-24', '2017-07-31',yongtu)
        between_hour = f1_num_between_hour('2017-07-24', '2017-07-31',yongtu)
    else:
        f1_num_24 = f1_num('2017-07-08', '2017-07-31',yongtu)
        f1_num_7 = f1_num('2017-07-25', '2017-07-31' ,yongtu)
        f1_num_3 = f1_num('2017-07-29', '2017-07-31' ,yongtu)
        f1_num_1 = f1_num('2017-07-31', '2017-07-31' ,yongtu)
        f1_num_week_24 = f1_num_week('2017-07-08', '2017-07-31' ,yongtu) 
        f1_num_week_14 = f1_num_week('2017-07-18', '2017-07-24' ,yongtu)
        f1_num_week_7 = f1_num_week('2017-07-24', '2017-07-31',yongtu)
        between_day = f1_num_between_day('2017-07-31', '2017-08-07',yongtu)
        between_hour = f1_num_between_hour('2017-07-31', '2017-08-07',yongtu)

    features = pd.merge(f1_num_24,f1_num_7,on=['start_geo_id','end_geo_id','create_date','create_hour'],how='left')
    features = pd.merge(features,f1_num_3,on=['start_geo_id','end_geo_id','create_date','create_hour'],how='left')
    features = pd.merge(features,f1_num_1,on=['start_geo_id','end_geo_id','create_date','create_hour'],how='left')
    features = pd.merge(features,f1_num_week_24,on=['start_geo_id','end_geo_id','create_date','create_hour'],how='left')
    features = pd.merge(features,f1_num_week_14,on=['start_geo_id','end_geo_id','create_date','create_hour'],how='left')
    features = pd.merge(features,f1_num_week_7,on=['start_geo_id','end_geo_id','create_date','create_hour'],how='left')
    features = pd.merge(features,between_day,on=['start_geo_id','end_geo_id','create_date','create_hour'],how='left')
    features = pd.merge(features,between_hour,on=['start_geo_id','end_geo_id','create_date','create_hour'],how='left')
    return features


# 训练节测试集构造
def trainning_test():
    features_train = features_num()
    index_get = features_train['day_count'+'2017-07-01']>0
    features_train = features_train[index_get] 
    print features_train

    del features_train['start_geo_id']
    del features_train['end_geo_id']
    del features_train['create_date']
    del features_train['create_hour']
    label = july_order_test_with_label['num']
    label = label[index_get]

    dtrain, dtest, dtrain_y, dtest_y = train_test_split(features_train, label, test_size=0.2, random_state=42)

    xgb_model = xgb.XGBRegressor(max_depth=10, learning_rate=0.05, n_estimators=500).fit(dtrain, dtrain_y)
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
    features_train = features_train[features_train['day_count'+'2017-07-24']>0] 

    del features_train['start_geo_id']
    del features_train['end_geo_id']
    del features_train['create_date']
    del features_train['create_hour']
    label = july_order_test_with_label['num']

    features_train = features_train.values

    dtrain, dtest, dtrain_y, dtest_y = train_test_split(features_train, label, test_size=0.0, random_state=42)

    xgb_model = xgb.XGBRegressor(max_depth=5, learning_rate=0.05, n_estimators=200).fit(dtrain, dtrain_y)
    return xgb_model


def testing():
    features_test = features_num('ceshi')
    rows, columns = features_test.shape
    print columns
    del features_test['start_geo_id']
    del features_test['end_geo_id']
    del features_test['create_date']
    del features_test['create_hour']
    return features_test.values

model = trainning_test()
pyplot.bar(range(len(model.feature_importances_)), model.feature_importances_)
# test = testing()
# result = model.predict(test)
# pd.DataFrame(result).to_csv("result_ture2.csv")



# 灵感灰飞烟灭：
# 总结俩个问题：
# 1. 由于有大量的[出发地，目的地，出发小时]是前面没有出现的,所以以上做的特征会出现大量的空值，这就让算法很难学到强特，因为每个特征都有很多的空值
# 2. 重新定义这个问题：这个问题是给定了某一天的偶数点的时间，然后去预测奇数点的时间
#   2.1. 我拿出全部的偶数日期的奇数小时，然后我用这些做训练集预测偶数日期的偶数小时的订单量
#   2.2. 我拿出全部的奇数日期的偶数小时，然后我用这些做训练集预测奇数日期的奇数小时的订单量 
    
    
    
    
    
    
    


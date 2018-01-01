
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



'''
    u'start_geo_id', u'end_geo_id', u'create_date', u'id',
    u'driver_id', u'member_id', u'create_hour', u'status',
    u'estimate_money', u'estimate_distance', u'estimate_term', u'num'
'''
def train_pre_traitor():
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
    print 'train_pre_traitor done'
    return july_order_train,july_order_test



def test_pre_traitor():
    # 此模块用于最基本的数据梳理
    july_order = pd.read_csv('data/train_July.csv')
    aug_order = pd.read_csv('data/train_Aug.csv')
    july_order = july_order[july_order['create_date']>'2017-07-07']
    order_train = pd.concat([july_order,aug_order]).reset_index()
    order_train['num'] = 1
    order_test = pd.read_csv('data/test_id_Aug_agg_public5k.csv')
    print 'test_pre_traitor done'
    return order_train,order_test

# 此模块是构造出发点热度的特征
# 如果是训练集time1 = '2017-07-24',time2 = '2017-07-21'，代表测试集前一天和测试集前4天
def feature1(time1,time2):
    ###此模块用于构造特征群1
    if time1=='2017-07-24':
        july_order_train,july_order_test = train_pre_traitor()
    else:
        july_order_train,july_order_test = test_pre_traitor()
    f1_tmp_1 = july_order_train[july_order_train['create_date']<=time1].groupby(['start_geo_id'])['num'].sum()
    ### f1_1_1 代表第一个特征群第一个特征
    f1_1_1 = (f1_tmp_1/24).reset_index()
    f1_1_1 = f1_1_1.rename(columns={"num": "f1_1"})
    f1_tmp_2 = july_order_train[(july_order_train['create_date']>time2)&(july_order_train['create_date']<=time1)].groupby(['start_geo_id'])['num'].sum()
    f1_1_2 = (f1_tmp_2/3).reset_index()
    f1_1_2 = f1_1_2.rename(columns={"num": "f1_2"})
    f1_tmp_3 = july_order_train[july_order_train['create_date']==time1].groupby(['start_geo_id'])['num'].sum()
    f1_1_3 = (f1_tmp_3).reset_index()
    f1_1_3 = f1_1_3.rename(columns={"num": "f1_3"})
    f1_tmp_4 = july_order_train.groupby(['start_geo_id','create_date'])['num'].sum()
    f1_1_4 = f1_tmp_4.reset_index()
    f1_1_4 = f1_1_4.rename(columns={"num": "f1_4"})
    f1_1_5 = f1_1_4.copy()
    f1_1_5['create_date2'] = pd.to_datetime(f1_1_5['create_date'])
    f1_1_5['create_date2'] = pd.to_datetime(pd.DatetimeIndex(f1_1_5.create_date2)- pd.DateOffset(1))
    del f1_1_5['create_date']
    f1_1_5['create_date'] = f1_1_5['create_date2'].map(lambda x: x.strftime('%Y-%m-%d'))
    del f1_1_5['create_date2']
    f1_1_5 = f1_1_5.rename(columns={"f1_4": "f1_5"})
    selected_data = july_order_test[['start_geo_id','end_geo_id','create_date','create_hour']]
    feature1 = pd.merge(selected_data,f1_1_1,on=['start_geo_id'],how='left')
    feature1 = pd.merge(feature1,f1_1_2,on=['start_geo_id'],how='left')
    feature1 = pd.merge(feature1,f1_1_3,on=['start_geo_id'],how='left')
    feature1 = pd.merge(feature1,f1_1_4,on=['start_geo_id','create_date'],how='left')
    feature1 = pd.merge(feature1,f1_1_5,on=['start_geo_id','create_date'],how='left')
    feature1 = feature1.fillna(0)
    feature1['f1_6'] = (feature1['f1_5']+feature1['f1_4'])/2
    return feature1



# 此模块是构造目的地热度的特征
# 如果是训练集time1 = '2017-07-24',time2 = '2017-07-21'，代表测试集前一天和测试集前4天
def feature3(time1,time2):
    ###此模块用于构造特征群1
    if time1=='2017-07-24':
        july_order_train,july_order_test = train_pre_traitor()
    else:
        july_order_train,july_order_test = test_pre_traitor()
    f1_tmp_1 = july_order_train[july_order_train['create_date']<=time1].groupby(['end_geo_id'])['num'].sum()
    ### f1_1_1 代表第一个特征群第一个特征
    f1_1_1 = (f1_tmp_1/24).reset_index()
    f1_1_1 = f1_1_1.rename(columns={"num": "f3_1"})
    f1_tmp_2 = july_order_train[(july_order_train['create_date']>time2)&(july_order_train['create_date']<=time1)].groupby(['end_geo_id'])['num'].sum()
    f1_1_2 = (f1_tmp_2/3).reset_index()
    f1_1_2 = f1_1_2.rename(columns={"num": "f3_2"})
    f1_tmp_3 = july_order_train[july_order_train['create_date']==time1].groupby(['end_geo_id'])['num'].sum()
    f1_1_3 = (f1_tmp_3).reset_index()
    f1_1_3 = f1_1_3.rename(columns={"num": "f3_3"})
    f1_tmp_4 = july_order_train.groupby(['end_geo_id','create_date'])['num'].sum()
    f1_1_4 = f1_tmp_4.reset_index()
    f1_1_4 = f1_1_4.rename(columns={"num": "f3_4"})
    f1_1_5 = f1_1_4.copy()
    f1_1_5['create_date2'] = pd.to_datetime(f1_1_5['create_date'])
    f1_1_5['create_date2'] = pd.to_datetime(pd.DatetimeIndex(f1_1_5.create_date2)- pd.DateOffset(1))
    del f1_1_5['create_date']
    f1_1_5['create_date'] = f1_1_5['create_date2'].map(lambda x: x.strftime('%Y-%m-%d'))
    del f1_1_5['create_date2']
    f1_1_5 = f1_1_5.rename(columns={"f3_4": "f3_5"})
    selected_data = july_order_test[['start_geo_id','end_geo_id','create_date','create_hour']]
    feature1 = pd.merge(selected_data,f1_1_1,on=['end_geo_id'],how='left')
    feature1 = pd.merge(feature1,f1_1_2,on=['end_geo_id'],how='left')
    feature1 = pd.merge(feature1,f1_1_3,on=['end_geo_id'],how='left')
    feature1 = pd.merge(feature1,f1_1_4,on=['end_geo_id','create_date'],how='left')
    feature1 = pd.merge(feature1,f1_1_5,on=['end_geo_id','create_date'],how='left')
    feature1 = feature1.fillna(0)
    feature1['f3_6'] = (feature1['f3_5']+feature1['f3_4'])/2
    return feature1


# 此模块是构造出发点和目的地热度的特征
# 如果是训练集time1 = '2017-07-24',time2 = '2017-07-21'，代表测试集前一天和测试集前4天
def feature4(time1,time2):
    ###此模块用于构造特征群1
    if time1=='2017-07-24':
        july_order_train,july_order_test = train_pre_traitor()
    else:
        july_order_train,july_order_test = test_pre_traitor()
    f1_tmp_1 = july_order_train[july_order_train['create_date']<=time1].groupby(['start_geo_id','end_geo_id'])['num'].sum()
    ### f1_1_1 代表第一个特征群第一个特征
    f1_1_1 = (f1_tmp_1/24).reset_index()
    f1_1_1 = f1_1_1.rename(columns={"num": "f4_1"})
    f1_tmp_2 = july_order_train[(july_order_train['create_date']>time2)&(july_order_train['create_date']<=time1)].groupby(['start_geo_id','end_geo_id'])['num'].sum()
    f1_1_2 = (f1_tmp_2/3).reset_index()
    f1_1_2 = f1_1_2.rename(columns={"num": "f4_2"})
    f1_tmp_3 = july_order_train[july_order_train['create_date']==time1].groupby(['start_geo_id','end_geo_id'])['num'].sum()
    f1_1_3 = (f1_tmp_3).reset_index()
    f1_1_3 = f1_1_3.rename(columns={"num": "f4_3"})
    f1_tmp_4 = july_order_train.groupby(['start_geo_id','end_geo_id','create_date'])['num'].sum()
    f1_1_4 = f1_tmp_4.reset_index()
    f1_1_4 = f1_1_4.rename(columns={"num": "f4_4"})
    f1_1_5 = f1_1_4.copy()
    f1_1_5['create_date2'] = pd.to_datetime(f1_1_5['create_date'])
    f1_1_5['create_date2'] = pd.to_datetime(pd.DatetimeIndex(f1_1_5.create_date2)- pd.DateOffset(1))
    del f1_1_5['create_date']
    f1_1_5['create_date'] = f1_1_5['create_date2'].map(lambda x: x.strftime('%Y-%m-%d'))
    del f1_1_5['create_date2']
    f1_1_5 = f1_1_5.rename(columns={"f4_4": "f4_5"})
    selected_data = july_order_test[['start_geo_id','end_geo_id','create_date','create_hour']]
    feature1 = pd.merge(selected_data,f1_1_1,on=['start_geo_id','end_geo_id'],how='left')
    feature1 = pd.merge(feature1,f1_1_2,on=['start_geo_id','end_geo_id'],how='left')
    feature1 = pd.merge(feature1,f1_1_3,on=['start_geo_id','end_geo_id'],how='left')
    feature1 = pd.merge(feature1,f1_1_4,on=['start_geo_id','end_geo_id','create_date'],how='left')
    feature1 = pd.merge(feature1,f1_1_5,on=['start_geo_id','end_geo_id','create_date'],how='left')
    feature1 = feature1.fillna(0)
    feature1['f4_6'] = (feature1['f4_5']+feature1['f4_4'])/2
    return feature1





###此模块用于构造特征群2:个人认为特征群2都是强特
# 如果是训练集，那么time1 = '2017-07-24',time2 = '2017-07-18'，time3 = '2017-07-25'， 代表测试集前一天，测试集前一周，测试集第一天
def feature2(time1,time2,time3):
    # july_order_train
    # july_order_test
    if time1=='2017-07-24':
        july_order_train,july_order_test = train_pre_traitor()
    else:
        july_order_train,july_order_test = test_pre_traitor()
    july_order_train_num = july_order_train[['start_geo_id','end_geo_id','create_date','create_hour','num']]
    july_order_train_num_before_24 = july_order_train_num[july_order_train_num['create_date']<=time1]
    test_select = july_order_test[['start_geo_id','end_geo_id','create_date','create_hour']]
    feature_2 = test_select.copy()
    f2_tmp_1 = pd.merge(test_select,july_order_train_num_before_24,on=['start_geo_id','end_geo_id','create_hour'],how='left')
    f2_1 = f2_tmp_1.groupby(['start_geo_id','end_geo_id','create_hour'])['num'].sum()/24
    f2_1 = f2_1.reset_index().rename(columns={"num": "f2_1"})
    feature_2 = pd.merge(feature_2,f2_1,on=['start_geo_id','end_geo_id','create_hour'],how='left')
    # f1 done
    feature_2['week'] = pd.to_datetime(feature_2['create_date']).dt.weekday
    july_order_train_num_before_24_week = july_order_train_num_before_24.copy()
    july_order_train_num_before_24_week['week'] =  pd.to_datetime(july_order_train_num_before_24_week['create_date']).dt.weekday
    test_select2_week = test_select.copy()
    test_select2_week['week'] = pd.to_datetime(test_select2_week['create_date']).dt.weekday
    del july_order_train_num_before_24_week['create_date']
    f2_tmp_2 = pd.merge(test_select2_week,july_order_train_num_before_24_week,on=['start_geo_id','end_geo_id','week','create_hour'],how='left')
    f2_2 = f2_tmp_2.groupby(['start_geo_id','end_geo_id','week','create_hour'])['num'].sum()/3
    f2_2 = f2_2.reset_index().rename(columns={"num": "f2_2"})
    feature_2 = pd.merge(feature_2,f2_2,on=['start_geo_id','end_geo_id','create_hour','week'],how='left')
    # f2 done
    july_order_train_num_before_7 = july_order_train_num_before_24[july_order_train_num_before_24['create_date']>=time3]
    f2_tmp_3 = pd.merge(test_select,july_order_train_num_before_7,on=['start_geo_id','end_geo_id','create_hour'],how='left')
    f2_3 = f2_tmp_3.groupby(['start_geo_id','end_geo_id','create_hour'])['num'].sum()/7
    f2_3 = f2_3.reset_index().rename(columns={"num": "f2_3"})
    feature_2 = pd.merge(feature_2,f2_3,on=['start_geo_id','end_geo_id','create_hour'],how='left')
    # f3 done
    feature_2['week'] = pd.to_datetime(feature_2['create_date']).dt.weekday
    july_order_train_num_before_7_week = july_order_train_num_before_7.copy()
    july_order_train_num_before_7_week['week'] =  pd.to_datetime(july_order_train_num_before_7_week['create_date']).dt.weekday
    test_select2_week = test_select.copy()
    test_select2_week['week'] = pd.to_datetime(test_select2_week['create_date']).dt.weekday
    del july_order_train_num_before_7_week['create_date']
    f2_tmp_4 = pd.merge(test_select2_week,july_order_train_num_before_7_week,on=['start_geo_id','end_geo_id','week','create_hour'],how='left')
    f2_4 = f2_tmp_4.groupby(['start_geo_id','end_geo_id','week','create_hour'])['num'].sum()
    f2_4 = f2_4.reset_index().rename(columns={"num": "f2_4"})
    feature_2 = pd.merge(feature_2,f2_4,on=['start_geo_id','end_geo_id','create_hour','week'],how='left')
    # f4 done
    # 2017-09-03 变成前一天的 -》2017-09-02
    july_order_train_num_25_31 = july_order_train_num[july_order_train_num['create_date']>time2]
    f2_tem_5 = july_order_train_num_25_31.copy()
    f2_tem_5['create_date'] = pd.to_datetime(pd.DatetimeIndex(pd.to_datetime(f2_tem_5['create_date']))- pd.DateOffset(1))
    f2_tem_5['create_date'] = f2_tem_5['create_date'].map(lambda x: x.strftime('%Y-%m-%d'))
    f2_tem_5_1 = f2_tem_5.groupby(['start_geo_id','end_geo_id','create_date','create_hour'])['num'].sum()
    f2_5 = f2_tem_5_1.reset_index().rename(columns={"num": "f2_5"})
    feature_2 = pd.merge(feature_2,f2_5,on=['start_geo_id','end_geo_id','create_date','create_hour'],how='left')
    # f5 done
    # 后一小时的订单情况
    july_order_train_num_25_31 = july_order_train_num[july_order_train_num['create_date']>=time2]
    f2_tem_6 = july_order_train_num_25_31.copy()
    f2_tem_6['create_hour'] = f2_tem_6['create_hour'].map(lambda x: 23 if x-1<0 else x-1)
    # f2_tem_6_change['create_date'] = f2_tem_6_change['create_date'].map(lambda x: (datetime.strptime(x,'%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d'))
    f2_tem_6.loc[f2_tem_6['create_hour'] == 23, 'create_date'] = f2_tem_6['create_date'].map(lambda x: (datetime.strptime(x,'%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d'))
    f2_tem_6 = f2_tem_6.groupby(['start_geo_id','end_geo_id','create_date','create_hour'])['num'].sum()
    f2_6 = f2_tem_6.reset_index().rename(columns={"num": "f2_6"})
    feature_2 = pd.merge(feature_2,f2_6,on=['start_geo_id','end_geo_id','create_date','create_hour'],how='left')
    july_order_train_num_25_31 = july_order_train_num[july_order_train_num['create_date']>=time1]
    f2_tem_7 = july_order_train_num_25_31.copy()
    f2_tem_7['create_hour'] = f2_tem_7['create_hour'].map(lambda x: 0 if x+1>23 else x+1)
    f2_tem_7.loc[f2_tem_7['create_hour'] == 0, 'create_date'] = f2_tem_7['create_date'].map(lambda x: (datetime.strptime(x,'%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d'))
    f2_tem_7 = f2_tem_7.groupby(['start_geo_id','end_geo_id','create_date','create_hour'])['num'].sum()
    f2_7 = f2_tem_7.reset_index().rename(columns={"num": "f2_7"})
    feature_2 = pd.merge(feature_2,f2_7,on=['start_geo_id','end_geo_id','create_date','create_hour'],how='left')
    feature_2['f2_6_7'] = (feature_2['f2_6']+feature_2['f2_7'])/2
    del feature_2['f2_6']
    del feature_2['f2_7']
    # f6 done
    print feature_2[feature_2['f2_6_7']!=np.nan].count()
    return feature_2


# 训练节测试集构造
def trainning_test():
    features_train_1 = feature1('2017-07-24','2017-07-21')
    features_train_2 = feature2('2017-07-24','2017-07-25','2017-07-18')
    features_train_3 = feature3('2017-07-24','2017-07-21')
    features_train_4 = feature4('2017-07-24','2017-07-21')

    features_train = pd.merge(features_train_1,features_train_2,on=['start_geo_id','end_geo_id','create_date','create_hour'])
    features_train = pd.merge(features_train,features_train_3,on=['start_geo_id','end_geo_id','create_date','create_hour'])
    features_train = pd.merge(features_train,features_train_4,on=['start_geo_id','end_geo_id','create_date','create_hour'])

    del features_train['start_geo_id']
    del features_train['end_geo_id']
    del features_train['create_date']
    del features_train['create_hour']
    _,july_order_test = train_pre_traitor()
    label = july_order_test['num']
    print features_train

    dtrain, dtest, dtrain_y, dtest_y = train_test_split(features_train, label, test_size=0.2, random_state=42)

    # dtrain = xgb.DMatrix(X_train)
    # dtrain_y = xgb.DMatrix(y_train)
    # dtest = xgb.DMatrix(X_test)
    # dtest_y = xgb.DMatrix(y_test)

    xgb_model = xgb.XGBRegressor(max_depth=5, learning_rate=0.05, n_estimators=200).fit(dtrain, dtrain_y).fit(dtrain,dtrain_y)
    predictions = xgb_model.predict(dtest)
    actuals = dtest_y
    s_prediciotn = pd.Series(predictions)
    s_label = pd.Series(actuals).reset_index()
    del s_label['index']
    result = s_prediciotn.to_frame()
    result['label'] = s_label
    result.to_csv("result.csv")
    print(mean_absolute_error(actuals, predictions))


# 训练节测试集构造
def trainning():
    features_train_1 = feature1('2017-07-24','2017-07-21')
    features_train_2 = feature2('2017-07-24','2017-07-25','2017-07-18')
    features_train_3 = feature3('2017-07-24','2017-07-21')
    features_train_4 = feature4('2017-07-24','2017-07-21')

    features_train = pd.merge(features_train_1,features_train_2,on=['start_geo_id','end_geo_id','create_date','create_hour'])
    features_train = pd.merge(features_train,features_train_3,on=['start_geo_id','end_geo_id','create_date','create_hour'])
    features_train = pd.merge(features_train,features_train_4,on=['start_geo_id','end_geo_id','create_date','create_hour'])

    del features_train['start_geo_id']
    del features_train['end_geo_id']
    del features_train['create_date']
    del features_train['create_hour']
    _,july_order_test = train_pre_traitor()
    label = july_order_test['num']

    dtrain, dtest, dtrain_y, dtest_y = train_test_split(features_train, label, test_size=0.0, random_state=42)

    # dtrain = xgb.DMatrix(X_train)
    # dtrain_y = xgb.DMatrix(y_train)
    # dtest = xgb.DMatrix(X_test)
    # dtest_y = xgb.DMatrix(y_test)

    xgb_model = xgb.XGBRegressor(max_depth=5, learning_rate=0.05, n_estimators=200).fit(dtrain, dtrain_y).fit(dtrain,dtrain_y)
    return xgb_model

def testing():
    features_test_1 = feature1('2017-07-31','2017-07-28')
    features_test_2 = feature2('2017-07-31','2017-08-01','2017-07-28')
    features_test_3 = feature3('2017-07-31','2017-07-28')
    features_test_4 = feature4('2017-07-31','2017-07-28')

    features_test = pd.merge(features_test_1,features_test_2,on=['start_geo_id','end_geo_id','create_date','create_hour'])
    features_test = pd.merge(features_test,features_test_3,on=['start_geo_id','end_geo_id','create_date','create_hour'])
    features_test = pd.merge(features_test,features_test_4,on=['start_geo_id','end_geo_id','create_date','create_hour'])

    del features_test['start_geo_id']
    del features_test['end_geo_id']
    del features_test['create_date']
    del features_test['create_hour']
    return features_test

model = trainning()
test = testing()
result = model.predict(test)
pd.DataFrame(result).to_csv("result_ture2.csv")
# trainning_test()


# 思路

# 分类预测。把所有有记录的[start_geo_id,end_geo_id,create_hour] 找出来。而且这个记录数必须大于某个值。
# 对于小于这个值得[start_geo_id,end_geo_id,create_hour]，找这样 [start_geo_id，end_geo_id] 能够匹配到的
# 对于找不到[start_geo_id，end_geo_id]这样的。找[start_geo_id ,create_hour] 这样的和[end_geo_id,create_hour]
# 实在找不到只找[start_geo_id]这样的和[end_geo_id]这样的
# 啥都找不到进行瞎预测的


# 猜测测试集中的数据都是订单量比较大的，例如都是5个，6个，甚至更多的订单，这样的订单，所以在训练的时候，如果把1个订单和2个订单都加上，也许因为测试集和训练集的订单数分布很不一样，训练效果不好




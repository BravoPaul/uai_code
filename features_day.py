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
july_order_25_31 = july_order[july_order['create_date']>'2017-07-24'][['start_geo_id','end_geo_id','create_date','num']]
july_order_test = july_order_25_31.groupby(['start_geo_id','end_geo_id'])['num'].sum().reset_index()
july_order_train = july_order[july_order['create_date']<='2017-07-24']
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
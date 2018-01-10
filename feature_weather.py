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
weather = pd.read_csv('data/weather.csv')
weather['create_date'] = weather['date'].map(lambda x: datetime.strftime(datetime.strptime(x.split(" ")[0],"%Y-%m-%d"),"%Y-%m-%d"))
weather['create_hour'] = weather['date'].map(lambda x: int(x.split(" ")[1].split(":")[0]))
weather_treated = weather.drop_duplicates(["create_date","create_hour"])
del weather_treated["date"]
del weather_treated["text"]
del weather_treated["wind_direction"]
weather_treated.to_csv("data/weatherTreated.csv")


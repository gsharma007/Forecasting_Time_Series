#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 14:22:34 2020

@author: gauravsharma
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import datetime 
from pandas import DataFrame
#import calendar 
from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.tsa.stattools as ts
from sklearn import metrics
import numpy as np
#import chart_studio.plotly as py
#import plotly.graph_objects as go

# =============================================================================
# def parser(x):
#     return datetime.strptime(x, '%/%/%Y')
# =============================================================================

salad_sales = pd.read_csv("/Users/gauravsharma/Documents/Forecasting/salad_data.csv")
                          #index_col=0 ,parse_dates=[0],date_parser=parser)

ADD = salad_sales['Additional']

group_daywise_mean = salad_sales.groupby(['day'], as_index=False).mean()
group_daywise_sum = salad_sales.groupby(['day'], as_index=False).sum()
group_daywise_median = salad_sales.groupby(['day'], as_index=False).median()

group_monthwise_mean = salad_sales.groupby(['month'], as_index=False).mean()
group_monthwise_sum = salad_sales.groupby(['month'], as_index=False).sum()
group_daywise_median = salad_sales.groupby(['month'], as_index=False).median()

# =============================================================================
# plt.figure()
# sns.scatterplot(x= 'date', y= 'salescount', data = salad_sales)
# plt.show()
# plt.clf()
# 
# plt.figure()
# sns.lineplot(x= 'date', y= 'salescount', data = salad_sales)
# plt.show()
# plt.clf()
# =============================================================================

# =============================================================================
# layout = go.Layout(title='salad_sales_data', xaxis=dict(title='Date'),
#                     yaxis=dict(title='(salescount)'))
# 
# fig = go.Figure(data=[salad_sales], layout=layout)
# py.iplot(fig, sharing='public')
# =============================================================================

# =============================================================================
# def findDay(Date): 
#     born = datetime.datetime.strptime(Date, '%d %m %Y').weekday() 
#     return (calendar.day_name[born])
# 
# Date = salad_sales.date
# Date.to_string()
# #salad_sales.day = findDay(Date)
# =============================================================================

plt.figure()
sns.scatterplot(x= 'day', y= 'salescount', data = salad_sales)
plt.show()
plt.clf()

plt.figure()
sns.lineplot(x= 'day', y= 'salescount', data = group_daywise_sum)
plt.show()
plt.clf()

plt.figure()
sns.lineplot(x= 'day', y= 'salescount', data = group_daywise_mean)
plt.show()
plt.clf()

plt.figure()
month_figure_1 = sns.lineplot(x= 'month', y= 'salescount', data = group_monthwise_sum, sort= True)
plt.xticks(rotation=45)
#month_figure_1.set_xticklabels(month_figure_1.get_xticklabels(), rotation=65, horizontalalignment='right')


plt.figure()
month_figure_2= sns.lineplot(x= 'month', y= 'salescount', data = group_monthwise_mean, sort= True)
plt.xticks(rotation=45)
#month_figure_2.set_xticklabels(month_figure_2.get_xticklabels(), rotation=65, horizontalalignment='right')

"""
Model Building
"""

plot_acf(salad_sales.salescount)

#plt.acorr('salescount',data= salad_sales)
#autocorrelation_plot(salad_sales)


x = salad_sales.date
x.to_frame()
y = salad_sales.salescount
y.to_frame()

# =============================================================================
# model = ARIMA(y, order=(7,0,3))
# model_fit = model.fit(disp=0)
# print(model_fit.summary())
# 
# residuals = DataFrame(model_fit.resid)
# residuals.plot()
# plt.show()
# residuals.plot(kind='kde')
# plt.show()
# print(residuals.describe())
# 
# from sklearn.metrics import mean_squared_error
# 
# Y = y.values
# size = int(len(Y) * 0.66)
# train, test = Y[0:size], Y[size:len(Y)] 
# history = [m for m in train]
# predictions = list()
# for t in range(len(test)):
# 	model = ARIMA(history, order=(7,0,3))
# 	model_fit = model.fit(disp=0)
# 	output = model_fit.forecast()
# 	yhat = output[0]
# 	predictions.append(yhat)
# 	obs = test[t]
# 	history.append(obs)
# 	#print('predicted=%f, expected=%f' % (yhat, obs))
# #error = mean_squared_error(test, predictions)
# # plot
# plt.plot(test)
# plt.plot(predictions, color='red')
# plt.show()
# 
# #print('Test MSE: %.3f' % error)
# =============================================================================



results=[]
MAE=[]
MAPE=[]

AR=[]
MA=[]
#salescount += 1
#print(salescount)
Add = Add.astype('float64')
#salescount =+1.0
Add_matrix=Add.as_matrix()
salescount = y.astype('float64')
#salescount =+1.0
salescount_matrix=salescount.as_matrix()
for i in range(0,2,1):
    for j in range(0,3,1):
        model = ARIMA(salescount_matrix, order=(i,0,j),exog=Add_matrix)
        if i==0 and j==0:
            continue
        model_fit = model.fit(disp=0)
        #print(model_fit.summary())
        print("AR="+str(i)+" and MA= " +str(j))
        print(model_fit.summary().tables[1])
        predicted=model_fit.predict(0,1121)
        #print(predicted)
        true=salescount
        #print(true)
        #mse=(sum(predicted-true))
        print("For AR="+str(i)+" and MA="+str(j)+" degree MSE of the model is "+str(metrics.mean_squared_error(true,predicted)))
        print("For AR="+str(i)+" and MA="+str(j)+" degree MAE of the model is "+str(metrics.mean_absolute_error(true,predicted)))
        print("For AR="+str(i)+" and MA="+str(j)+" degree RMSE of the model is "+str(np.sqrt(metrics.mean_squared_error(true,predicted))))
        print("For AR="+str(i)+" and MA="+str(j)+" degree MAPE of the model is "+str(np.mean((np.abs((true - predicted)) / true) * 100)))
        plt.plot(predicted-true)
        plt.show()
        #print(((true-predicted)/true))
        print()
#         results.append(metrics.mean_squared_error(true,predicted))
#         MAE.append(metrics.mean_absolute_error(true,predicted))
#         MAPE.append(np.mean(np.abs((true - predicted) / true)) * 100)
#         AR.append(i)
#         MA.append(j)

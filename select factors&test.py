# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 16:30:06 2019

@author: YUXU
"""

import pandas as pd
import numpy as np
import datetime
import time
import matplotlib.pyplot as plt

#%% calculate daily return rate
'''
data=port_test[1:]
data['rr']=pd.DataFrame((data1['close'][1:].values-data1['close'][:-1].values)/data1['close'][:-1].values)
list=[]
for i in range(1,len(data)-1):
    if(data.iloc[i,1]!=data.iloc[i+1,1]):
        list.append(i+2)
data=data.drop(list)
data=data.reset_index(drop=True)        
data.to_csv('E:/SEU/19-2/Quantitative Investment/Factor Model/YUXU/dataset/dataset.csv')
'''

#%%
#calculate monthly return
data=pd.read_csv('E:/SEU/19-2/Quantitative Investment/Factor Model/YUXU/dataset/dataset.csv',index_col=0,header=0)
data=data.sort_values(by=['code','date'])
data=data.reset_index(drop=True)
data['date']=pd.to_datetime(data['date'],format='%Y-%m-%d')
list=[0]
for i in range(1,len(data)-1):
    if(data['date'][i].month!=data['date'][i+1].month):
        list.append(i+1)      
data_month=data.iloc[list,:]
data_month=data_month.reset_index(drop=True)

data_month_mr=data_month[:-1]
mr=(data_month['close'][1:].values-data_month['close'][:-1].values)/data_month['close'][:-1].values
data_month_mr['mr']=mr
list=[]
for i in range(1,len(data_month_mr)-1):
    if(data_month_mr.iloc[i,1]!=data_month_mr.iloc[i+1,1]):
        list.append(i)
data_month_mr=data_month_mr.drop(list)
data_month_mr=data_month_mr.reset_index(drop=True)        
data_month_mr.to_csv('E:/SEU/19-2/Quantitative Investment/Factor Model/YUXU/dataset/dataset_month.csv')

#%% drop null and normalize
data=pd.read_csv('E:/SEU/19-2/Quantitative Investment/Factor Model/YUXU/dataset/dataset_month.csv',index_col=0,header=0)
data['date']=pd.to_datetime(data['date'],format='%Y-%m-%d')
data['month']=0
for i in range(len(data)):
    data['month'].iloc[i]=str(data['date'][i].year)+str(data['date'][i].month)

#drop null or zero factors
data_drop=data
for j in data.columns.tolist()[8:-1]:
        if (data_drop.loc[:,j].isnull().value_counts()[0]!=len(data_drop) or abs(data_drop.loc[:,j].min())==0):
            data_drop=data_drop.drop([j],axis=1)

#normalize factors
data_normal=pd.DataFrame()
for name,group in data_drop.groupby(['code']):
    temp=data_drop[data_drop.code==name]
    temp2=pd.DataFrame()
    temp2[['date','month','code','mr','close','open','high','low','money','volume']]\
    =temp[['date','month','code','mr','close','open','high','low','money','volume']]
    for i in range(8,temp.shape[1]-2):
        temp1=(temp.iloc[:,i]-temp.iloc[:,i].mean())/temp.iloc[:,i].std()
        temp2=pd.concat([temp2,temp1],axis=1)
    data_normal=pd.concat([data_normal,temp2],axis=0)
data_normal.to_csv('E:/SEU/19-2/Quantitative Investment/Factor Model/YUXU/dataset/dataset_month_normal.csv')


#%%index
index=pd.read_csv('E:/SEU/19-2/Quantitative Investment/Factor Model/YUXU/dataset/index.csv',header=0)
index['date']=pd.to_datetime(index['date'],format='%Y-%m-%d')
index=index.reset_index(drop=True)
list=[0]
for i in range(1,len(index)-1):
    if(index['date'][i].month!=index['date'][i+1].month):
        list.append(i+1)      
index_month=index.iloc[list,:]
index_month=index_month.reset_index(drop=True)

index_month_mr=index_month[:-1]
mr=(index_month['close'][1:].values-index_month['close'][:-1].values)/index_month['close'][:-1].values
index_month_mr['index_mr']=mr
index_month_mr=index_month_mr.reset_index(drop=True)   
index_month_mr['month']=0
for i in range(len(index_month_mr)):
    index_month_mr['month'].iloc[i]=str(index_month_mr['date'][i].year)+str(index_month_mr['date'][i].month)
     
index_month_mr.to_csv('E:/SEU/19-2/Quantitative Investment/Factor Model/YUXU/dataset/index_month.csv')


#%% merge stocks and index
stocks=pd.read_csv('E:/SEU/19-2/Quantitative Investment/Factor Model/YUXU/dataset/dataset_month_normal.csv',index_col=0,header=0)
index=pd.read_csv('E:/SEU/19-2/Quantitative Investment/Factor Model/YUXU/dataset/index_month.csv',index_col=0,header=0)
data=pd.merge(stocks,index.drop(['date','close'],axis=1),how='outer',on='month')
data.to_csv('E:/SEU/19-2/Quantitative Investment/Factor Model/YUXU/dataset/data_final.csv')


#%%select factors
data=pd.read_csv('E:/SEU/19-2/Quantitative Investment/Factor Model/YUXU/dataset/data_final.csv',index_col=0,header=0)
data['date']=pd.to_datetime(data['date'],format='%Y-%m-%d')
corr=pd.DataFrame(np.zeros((0,4)))
corr.columns=['corr','ar1','ar5','ascending']

for factor in data.columns[10:-1].tolist():
    data=data.sort_values(by=['month',factor],axis=0).reset_index(drop=True)
    ar=pd.DataFrame({'num':[1,2,3,4,5]})
    for month,group in data.groupby(['month']):
        temp=data[data['month']==month][['month','code','close','mr','index_mr',factor]]
        ar_temp=pd.DataFrame({'num':[1,2,3,4,5],'Return':[0,0,0,0,0],'ar':[0,0,0,0,0]})
        ar_temp['index_mr']=temp['index_mr'].iloc[0]
        for i in range(0,5):
            ar_temp.iloc[i,1]=(temp.iloc[9*i:9*i+9,2]*temp.iloc[9*i:9*i+9,3]).sum()/temp.iloc[9*i:9*i+9,2].sum()
            ar_temp.iloc[i,2]=ar_temp.iloc[i,1]-ar_temp.iloc[i,3]
            ar[month]=ar_temp['ar']
    ar['total']=pow((ar.iloc[:,1:]+1).cumprod(axis=1).iloc[:,-1],1/(ar.shape[1]-1))-1
    if ar['total'].iloc[0]<ar['total'].iloc[-1]:
        ar=ar.iloc[::-1].reset_index(drop=True)
        ar['num']=[1,2,3,4,5]
        ascending=0
    else:
        ascending=1
    corr.loc[factor,'corr']=ar[['num','total']].corr().iloc[0,1]
    corr.loc[factor,'ar1']=ar['total'].iloc[0]
    corr.loc[factor,'ar5']=ar['total'].iloc[-1]
    corr.loc[factor,'ascending']=ascending

corr_selected=corr[(abs(corr['corr'])>0.5)&(corr['ar1']>0.01)&(corr['ar5']<-0.01)]

corr.to_csv('E:/SEU/19-2/Quantitative Investment/Factor Model/YUXU/dataset/corr.csv')
corr=pd.read_csv('E:/SEU/19-2/Quantitative Investment/Factor Model/YUXU/dataset/corr.csv',index_col=0,header=0)

#remove redundant factors
droplist=[]
factor_selected=data[corr_selected.index.tolist()]
factor_corr=factor_selected.corr()
for i in range(0,17):
    for j in range(i+1,len(factor_corr)):
        if abs(factor_corr.iloc[j,i])>0.9:
            if corr_selected.loc[factor_corr.index[i],'corr']>corr_selected.loc[factor_corr.columns[i],'corr']:
                droplist.append(factor_corr.columns[i])
            else:
                droplist.append(factor_corr.index[i])

corr_selected=corr_selected.drop(droplist,axis=0)
corr_selected.to_csv('E:/SEU/19-2/Quantitative Investment/Factor Model/YUXU/dataset/corr_selected.csv')


plt.bar(x=corr_selected.index.tolist(),height=abs(corr_selected['corr']))
fig = plt.gcf()
fig.set_size_inches(30, 5)
plt.show()

data_selected=pd.concat([data[['month','code','close']],data[corr_selected.index.tolist()]],axis=1)
data_selected.to_csv('E:/SEU/19-2/Quantitative Investment/Factor Model/YUXU/dataset/data_selected.csv')


#%%select stocks
data=pd.read_csv('E:/SEU/19-2/Quantitative Investment/Factor Model/YUXU/dataset/data_selected.csv',index_col=0,header=0)
corr_selected=pd.read_csv('E:/SEU/19-2/Quantitative Investment/Factor Model/YUXU/dataset/corr_selected.csv',index_col=0,header=0)

score=pd.DataFrame()
for month,group in data.groupby(['month']):
    #month=20171
    temp=data[data['month']==month]
    for factor in temp.columns[3:].tolist():
        #factor='circulating_market_cap'
        temp=temp.sort_values(by=factor,axis=0,ascending=corr_selected.loc[factor,'ascending']).reset_index(drop=True)
        for i in range(0,5):
            temp[factor].iloc[9*i:9*i+9]=5-i
    score=pd.concat([score,temp],axis=0)
score['add']=score.iloc[:,3:].cumsum(axis=1).iloc[:,-1]
score=score.sort_values(by=['code','month'])
total_score=pd.DataFrame(score.groupby(by='code').sum()['add']).sort_values(by='add')
stock_list=total_score.index[0:15].tolist()

score.to_csv('E:/SEU/19-2/Quantitative Investment/Factor Model/YUXU/dataset/scores.csv')
pd.DataFrame(stock_list).to_csv('E:/SEU/19-2/Quantitative Investment/Factor Model/YUXU/dataset/selected_stocks_list.csv')

#%%test
stock_test=pd.read_csv('E:/SEU/19-2/Quantitative Investment/Factor Model/YUXU/dataset/stock_test.csv',header=0)
index_test=pd.read_csv('E:/SEU/19-2/Quantitative Investment/Factor Model/YUXU/dataset/index_test.csv',header=0)
stock_test['date']=pd.to_datetime(stock_test['date'],format='%Y-%m-%d')
index_test['date']=pd.to_datetime(index_test['date'],format='%Y-%m-%d')


stock_test=stock_test.sort_values(by=['date','code']).reset_index(drop=True)
portfolio_temp=index_test.iloc[:,:2]
portfolio_temp=portfolio_temp.set_index('date')
portfolio_price=stock_test.groupby(by='date').sum()['close_stock']
port_test=portfolio_temp.merge(portfolio_price,left_index=True,right_index=True)

n_index=10000/port_test['close_index'].iloc[0]
n_stock=10000/port_test['close_stock'].iloc[0]

port_test['index_value']=n_index*port_test['close_index']
port_test['stock_value']=n_stock*port_test['close_stock']

plt.plot(port_test['stock_value'])
plt.plot(port_test['index_value'])
plt.legend(['stock','index'])
fig = plt.gcf()
fig.set_size_inches(10, 4)
plt.show()

diff=pd.DataFrame(port_test['stock_value']-port_test['index_value'])
diff=diff.rename(columns={0:'diff'})
diff_max_date=diff[diff['diff']==diff['diff'].max()].index[0]

#%%future
import tushare as ts
ts.set_token('5ee4d6b13c9826e0a1606ffeadbaeda94eb0abf547c94a336f084d53')
pro=ts.pro_api()
future=pro.fut_daily(ts_code='IHL.CFX', start_date='20190201', end_date='20190331')
future.to_csv('E:/SEU/19-2/Quantitative Investment/Factor Model/YUXU/dataset/future.csv')
future=pd.read_csv('E:/SEU/19-2/Quantitative Investment/Factor Model/YUXU/dataset/future.csv',index_col=0,header=0)
future=future.sort_values(by=['trade_date'])
future=future.reset_index(drop='True')
n_future=10000/future['close'].iloc[0]
port_test=port_test.reset_index()
port_test['future_value']=n_future*future['close']

plt.plot(port_test['date'],port_test['stock_value'])
plt.plot(port_test['date'],port_test['future_value'])
plt.legend(['stock','future'])
fig = plt.gcf()
fig.set_size_inches(10, 4)
plt.show()

profit=pd.DataFrame(port_test['stock_value']-port_test['future_value'])
profit['date']=port_test['date']
profit=profit.rename(columns={0:'profit'})
plt.plot(port_test['date'],profit)
profit_max_date=profit[profit['profit']==profit['profit'].max()]['date']

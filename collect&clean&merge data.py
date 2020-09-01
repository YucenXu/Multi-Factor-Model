# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 10:26:24 2019

@author: YUXU
"""

'''
git test
'''

import pandas as pd
import numpy as np
import datetime
import time
import jqdatasdk
from jqdatasdk import *

#%% get stock list
stocks=get_index_stocks('000016.XSHG')      #获取上证50成分股列表
all_securities=get_all_securities(types=['stock'], date=None)
stocks_info=all_securities.loc[stocks,:]    #成分股基本信息
stocks_old=stocks_info[stocks_info['start_date']<'2017-01-01']      #除去上市未满两年的股票
stocks_list1=stocks_old.index.tolist()

#%% get price & remove ST
is_st=get_extras('is_st', stocks_old.index.tolist(), \
                 start_date='2017-01-01', end_date='2019-01-01', df=True)       #获取ST信息
is_st=is_st.T
not_st=is_st[is_st.iloc[:,0]==False]        #筛选出未被ST的成分股
stocks_list_old_not_st=not_st.index.tolist()

raw_data=get_price(stocks_list_old_not_st, start_date='2017-01-01', end_date='2019-01-01', \
                frequency='daily', fields=None, skip_paused=False, fq='pre')
raw_data=raw_data.rename(columns={'major':'date','minor':'code'})
raw_data.sort_values(by=['date','code'])
raw_data=raw_data.reset_index(drop=True)
raw_data.to_csv('E:/SEU/19-2/Quantitative Investment/Factor Model/YUXU/dataset/raw_data.csv')
pd.DataFrame(stocks_list_old_not_st).to_csv('E:/SEU/19-2/Quantitative Investment/Factor Model/YUXU/dataset/stocks_list.csv')


#%%get factors list

factor_list=get_all_factors()
pd.DataFrame(factor_list).to_csv('E:/SEU/19-2/Quantitative Investment/Factor Model/YUXU/dataset/factors_list.csv')
factor_list=pd.read_csv('E:/SEU/19-2/Quantitative Investment/Factor Model/YUXU/dataset/factors_list.csv',index_col=0,header=0)

#%% get factors
stocks_list_df=pd.read_csv('E:/SEU/19-2/Quantitative Investment/Factor Model/YUXU/dataset/stocks_list.csv',index_col=1)
stocks_list=stocks_list_df.index.tolist()
factors_list=pd.read_csv('E:/SEU/19-2/Quantitative Investment/Factor Model/YUXU/dataset/factors_list.csv',index_col=0)
factors={}
for i in range(1,33):
    factors.update(get_factor_values(securities=stocks_list, factors=factors_list.iloc[8*i-8:8*i,0].tolist(), \
                                     start_date='2017-01-01', end_date='2019-01-01'))
factors.update(get_factor_values(securities=stocks_list, factors=factors_list.iloc[256:259,0].tolist(),\
                                 start_date='2017-01-01', end_date='2019-01-01'))

factors_df=pd.DataFrame(factors['ACCA'].unstack().swaplevel())
factors_df.columns=['ACCA']
for k in factors:
    factors_temp=pd.DataFrame(factors[k].unstack().swaplevel())
    factors_temp.columns=[k]
    factors_df=pd.merge(factors_df,factors_temp,left_index=True,right_index=True)

factors_df.index=pd.to_datetime(factors_df.index)
factors_df.to_csv('E:/SEU/19-2/Quantitative Investment/Factor Model/YUXU/dataset/factors_raw.csv')

factors=pd.read_csv('E:/SEU/19-2/Quantitative Investment/Factor Model/YUXU/dataset/factors_raw.csv')
factors=factors.rename(columns={'Unnamed: 0':'date'})
factors=factors.sort_values(by=['date','code'])
factors=factors.reset_index(drop=True)
factors.to_csv('E:/SEU/19-2/Quantitative Investment/Factor Model/YUXU/dataset/factors.csv')



#%% merge price and factors
price=pd.read_csv('E:/SEU/19-2/Quantitative Investment/Factor Model/YUXU/dataset/raw_data.csv',index_col=0,header=0)
#price.loc[price.loc['minor']=='600007.XSHG']       #'600007XSHG历史行情'
factors=pd.read_csv('E:/SEU/19-2/Quantitative Investment/Factor Model/YUXU/dataset/factors.csv',index_col=0)
dataset=pd.merge(price,factors,on=['date','code'])
dataset.to_csv('E:/SEU/19-2/Quantitative Investment/Factor Model/YUXU/dataset/dataset.csv')


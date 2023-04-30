import os
import sys
import time
import string
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import warnings
import seaborn as sns
import scipy.stats as stats
import math
warnings.filterwarnings("ignore")

import statsmodels.api as sm
import datetime
from multiprocessing import Pool
sys.path.append('C:/User/12632/Desktop/鸣熙实习资料/')#定位到导入库的模块
from functions import load_base_data,load_income_data,load_balance_data,load_cash_data,load_barrar_data
from functions import get_data,get_data_shareincome,get_data_sharebalance,get_data_sharecash,get_IC,get_data_barrar
from functions import plot_cumIC,plot_cnt,plot_corr
from functions import groupby_norm,lr,residualize_multi,cut,add_shift,ts_norm

if __name__ == '__main__':
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    # 读取数据,预处理
    print('download data')
    base_begin_date,base_end_date = 20110104,20230206
    income_begin_date,income_end_date = 20120101,20230207
    cash_begin_date,cash_end_date = 20110104,20230215
    balance_begin_date,balance_end_date = 20120101,20230207
#     base_begin_date,base_end_date = 20170501,20180701
#     barrar_begin_date,barrar_end_date = 20170501,20180701
#     income_begin_date,income_end_date = 20170501,20180701
#     cash_begin_date,cash_end_date = 20170501,20180701
#     balance_begin_date,balance_end_date = 20170501,20180701
    base_data_namelist = ['date','cn_code','cap']
    share_income_namelist  = ['REPORT_PERIOD','ANN_DT','date','S_INFO_WINDCODE','STATEMENT_TYPE','NET_PROFIT_INCL_MIN_INT_INC'] 
    share_balance_namelist = ['REPORT_PERIOD','ANN_DT','date','STATEMENT_TYPE','S_INFO_WINDCODE','CAP_STK','TOT_ASSETS','TOT_LIAB']
    share_cash_namelist = ['REPORT_PERIOD','ANN_DT','S_INFO_WINDCODE','STATEMENT_TYPE','NET_CASH_FLOWS_OPER_ACT']
    base_data=load_base_data(base_data_namelist,base_begin_date,base_end_date)
    share_income_all = load_income_data(share_income_namelist,income_begin_date,income_end_date)
    share_balance_all = load_balance_data(share_balance_namelist,balance_begin_date,balance_end_date)
    share_cash_all = load_cash_data(share_cash_namelist,cash_begin_date,cash_end_date)
    share_income_all['NET_PROFIT_INCL_MIN_INT_INC'] = share_income_all['NET_PROFIT_INCL_MIN_INT_INC'].astype('float32')

   
    
    # 计算因子
    print('calculation begin')
    #截取一部分数据用于后续的计算
    income = share_income_all[['cn_code','date','NET_PROFIT_INCL_MIN_INT_INC']]
    balance = share_balance_all[['cn_code','date','CAP_STK','TOT_ASSETS','TOT_LIAB']]
    balance = add_shift(balance,['CAP_STK'],1)
    cash = share_cash_all[['cn_code','date','NET_CASH_FLOWS_OPER_ACT']]
    
                              
    #分别将计算因子使用到的财务数据merge入base_data
    df = pd.merge_asof(base_data.sort_values(by=['date','cn_code']),income.sort_values(by=['date','cn_code']),on='date',by='cn_code',allow_exact_matches=False)
    df = pd.merge_asof(df.sort_values(by=['date','cn_code']),balance.sort_values(by=['date','cn_code']),on='date',by='cn_code',allow_exact_matches=False)
    df = pd.merge_asof(df.sort_values(by=['date','cn_code']),cash.sort_values(by=['date','cn_code']),on='date',by='cn_code',allow_exact_matches=False)

   
        
         

    #--------------------------------------每股指标类因子---------------------------------------
    #平均每股净资产   净资产/期初、期末市值平均
    df['NAdCAP_ave'] = 2*(df['TOT_ASSETS']-df['TOT_LIAB'])/(df['CAP_STK']+df['shift_CAP_STK1'])
    #最新摊薄每股净资产  净资产/市值
    df['NAdCAP'] = (df['TOT_ASSETS']-df['TOT_LIAB'])/df['cap']
    #平均每股收益  净利润/期初期末市值平均
    df['NPdCAP_ave'] = 2*df['NET_PROFIT_INCL_MIN_INT_INC']/(df['CAP_STK']+df['shift_CAP_STK1'])
    #最新摊薄每股收益  净利润/市值 
    df['NPdCAP'] = df['NET_PROFIT_INCL_MIN_INT_INC']/df['cap']
    #平均每股经营现金流  经营现金流/期初期末市值平均 
    df['CASHdCAP_ave'] = 2*df['NET_CASH_FLOWS_OPER_ACT']/(df['CAP_STK']+df['shift_CAP_STK1'])
    #最新摊薄每股经营现金流  经营现金流/市值
    df['CASHdCAP'] = df['NET_CASH_FLOWS_OPER_ACT']/df['cap']
                              
                              
     
    
    
    #saving
    save_path = 'D:/results_new/'
    for name in ['NAdCAP_ave','NAdCAP','NPdCAP_ave','NPdCAP','CASHdCAP_ave','CASHdCAP']:
        if os.path.exists(save_path+name+'/'):
            print('already exits')
        else:
            os.mkdir(save_path+name+'/')
            df[['date','cn_code',name]].to_csv(save_path+name+'/'+'df.csv',index=False)
    
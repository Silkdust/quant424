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
    share_income_namelist  = ['REPORT_PERIOD','ANN_DT','date','S_INFO_WINDCODE','STATEMENT_TYPE','NET_PROFIT_INCL_MIN_INT_INC','OPER_REV'] 
    share_balance_namelist = ['REPORT_PERIOD','ANN_DT','date','STATEMENT_TYPE','S_INFO_WINDCODE','TOT_ASSETS','TOT_LIAB','DVD_PAYABLE']
    base_data=load_base_data(base_data_namelist,base_begin_date,base_end_date)
    share_income_all = load_income_data(share_income_namelist,income_begin_date,income_end_date)
    share_balance_all = load_balance_data(share_balance_namelist,balance_begin_date,balance_end_date)
    share_income_all['NET_PROFIT_INCL_MIN_INT_INC'] = share_income_all['NET_PROFIT_INCL_MIN_INT_INC'].astype('float32')
    share_income_all['OPER_REV'] = share_income_all['OPER_REV'].astype('float32')
   
    
    # 计算因子
    print('calculation begin')
    #截取一部分数据用于后续的计算
    income = share_income_all[['cn_code','date','NET_PROFIT_INCL_MIN_INT_INC','OPER_REV']]
    income = add_shift(income,['NET_PROFIT_INCL_MIN_INT_INC','OPER_REV'],4)
    
    
    
    balance = share_balance_all[['cn_code','date','TOT_ASSETS','TOT_LIAB','DVD_PAYABLE']]
    for name in ['DVD_PAYABLE']:
        balance[name] = balance[name].replace(np.nan,0)
    balance = add_shift(balance,['DVD_PAYABLE'],4)
    
    

    
    #分别将计算因子使用到的财务数据merge入base_data
    df = pd.merge_asof(base_data.sort_values(by=['date','cn_code']),income.sort_values(by=['date','cn_code']),on='date',by='cn_code',allow_exact_matches=False)
    df = pd.merge_asof(df,balance.sort_values(by=['date','cn_code']),on='date',by='cn_code',allow_exact_matches=False)
    
                
        
         
                           
    #--------------------------------------估值类因子---------------------------------------
    #PB
    df['PB'] = (df['TOT_ASSETS']-df['TOT_LIAB'])/df['cap']
    #PE
    df['PE'] = df['NET_PROFIT_INCL_MIN_INT_INC']/df['cap']
    df['shift_PE1'] = df['shift_NET_PROFIT_INCL_MIN_INT_INC1']/df['cap']
    df['shift_PE2'] = df['shift_NET_PROFIT_INCL_MIN_INT_INC2']/df['cap']
    df['shift_PE3'] = df['shift_NET_PROFIT_INCL_MIN_INT_INC3']/df['cap']
    df['PE_TTM'] = (df['PE']+df['shift_PE1']+df['shift_PE2']+df['shift_PE3'])/4
    #PEG：动态市盈率=市盈率/（净利润增长率）=市值/（净利润*净利润增长率）
    df['NP_G'] = (df['NET_PROFIT_INCL_MIN_INT_INC']-df['shift_NET_PROFIT_INCL_MIN_INT_INC4'])/(abs(df['NET_PROFIT_INCL_MIN_INT_INC'])+abs(df['shift_NET_PROFIT_INCL_MIN_INT_INC4']))
    df['PEG'] = (df['NET_PROFIT_INCL_MIN_INT_INC']*(1+df['NP_G']))/df['cap']
    #PS 市销率=股价/每股销售收入
    df['PS'] = df['OPER_REV']/df['cap']
    df['shift_PS1'] = df['shift_OPER_REV1']/df['cap']
    df['shift_PS2'] = df['shift_OPER_REV2']/df['cap']
    df['shift_PS3'] = df['shift_OPER_REV3']/df['cap']
    df['PS_TTM'] = (df['PS']+df['shift_PS1']+df['shift_PS2']+df['shift_PS3'])/4
    #股息率 = 最近四个季度应付股利求和/总股本
    df['DVD_ratio'] = (df['shift_DVD_PAYABLE0']+df['shift_DVD_PAYABLE1']+df['shift_DVD_PAYABLE2']+df['shift_DVD_PAYABLE3'])/df['cap']
    
    #saving
    save_path = 'D:/results_new/'
    for name in ['PB','PE','PE_TTM','PEG','PS','PS_TTM','DVD_ratio']:
        if os.path.exists(save_path+name+'/'):
            print('already exits')
        else:
            os.mkdir(save_path+name+'/')
            df[['date','cn_code',name]].to_csv(save_path+name+'/'+'df.csv',index=False)
    
    
    
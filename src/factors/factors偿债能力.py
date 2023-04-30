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
    base_data_namelist = ['date','cn_code','pre_close']
    share_income_namelist  = ['REPORT_PERIOD','ANN_DT','date','S_INFO_WINDCODE','STATEMENT_TYPE','NET_PROFIT_INCL_MIN_INT_INC','OPER_REV'] 
    share_balance_namelist = ['REPORT_PERIOD','ANN_DT','date','STATEMENT_TYPE','S_INFO_WINDCODE','MONETARY_CAP','TOT_ASSETS','TOT_LIAB','TOT_CUR_ASSETS','TOT_CUR_LIAB','INVENTORIES']
    share_cash_namelist = ['REPORT_PERIOD','ANN_DT','S_INFO_WINDCODE','STATEMENT_TYPE','CASH_CASH_EQU_END_PERIOD']
    base_data=load_base_data(base_data_namelist,base_begin_date,base_end_date)
    share_income_all = load_income_data(share_income_namelist,income_begin_date,income_end_date)
    share_balance_all = load_balance_data(share_balance_namelist,balance_begin_date,balance_end_date)
    share_cash_all = load_cash_data(share_cash_namelist,cash_begin_date,cash_end_date)
    share_income_all['NET_PROFIT_INCL_MIN_INT_INC'] = share_income_all['NET_PROFIT_INCL_MIN_INT_INC'].astype('float32')
    share_income_all['OPER_REV'] = share_income_all['OPER_REV'].astype('float32')
   
    
    # 计算因子
    print('calculation begin')
    #截取一部分数据用于后续的计算
    income = share_income_all[['cn_code','date','NET_PROFIT_INCL_MIN_INT_INC','OPER_REV']]
    balance = share_balance_all[['cn_code','date','MONETARY_CAP','TOT_ASSETS','TOT_LIAB','TOT_CUR_ASSETS','TOT_CUR_LIAB','INVENTORIES']]  
    cash = share_cash_all[['cn_code','date','CASH_CASH_EQU_END_PERIOD']]
    #分别将计算因子使用到的财务数据merge入base_data
    df = pd.merge_asof(base_data.sort_values(by=['date','cn_code']),income.sort_values(by=['date','cn_code']),on='date',by='cn_code',allow_exact_matches=False)
    df = pd.merge_asof(df,balance.sort_values(by=['date','cn_code']),on='date',by='cn_code',allow_exact_matches=False)
    df = pd.merge_asof(df,cash.sort_values(by=['date','cn_code']),on='date',by='cn_code',allow_exact_matches=False)
    
                
        
         
                           
    #--------------------------------------偿债能力类因子---------------------------------------
    #净负债率=(总负债-现金和现金等价物)/股东权益
    df['NLdEQ'] = (df['TOT_LIAB']-df['CASH_CASH_EQU_END_PERIOD'])/(df['TOT_ASSETS']-df['TOT_LIAB'])
    #流动比率=流动资产/流动负债
    df['WCR'] = df['TOT_CUR_ASSETS']/df['TOT_CUR_LIAB']
    #速动比率=速动资产/流动负债=（流动资产-存货）/流动负债
    df['QR'] = (df['TOT_ASSETS']-df['INVENTORIES'])/df['TOT_CUR_LIAB']
    #现金比率=现金及现金等价物/流动负债
    df['CAR'] = df['CASH_CASH_EQU_END_PERIOD']/df['TOT_CUR_LIAB']
    #资产负债率
    df['LEV'] = df['TOT_LIAB']/df['TOT_ASSETS']
    
    
       
       
       
       
       
    #分母为负数的因子更换为nan   
    df['NLdEQ'][df['TOT_ASSETS']-df['TOT_LIAB']<=0]=np.nan
    
    
    #saving
    save_path = 'D:/results_new/'
    for name in ['NLdEQ','WCR','QR','CAR','LEV']:
        if os.path.exists(save_path+name+'/'):
            print('already exits')
        else:
            os.mkdir(save_path+name+'/')
            df[['date','cn_code',name]].to_csv(save_path+name+'/'+'df.csv',index=False)
    
    
    
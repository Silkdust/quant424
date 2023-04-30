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
from functions import load_base_data,load_income_data,load_balance_data,load_cash_data,load_barrar_data,load_financialindex_data
from functions import get_data,get_data_shareincome,get_data_sharebalance,get_data_sharecash,get_IC,get_data_barrar,get_data_financialindex
from functions import plot_cumIC,plot_cnt,plot_corr
from functions import groupby_norm,lr,residualize_multi,cut,add_shift,ts_norm

if __name__ == '__main__':
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    # 读取数据,预处理
    print('download data')
    base_begin_date,base_end_date = 20110104,20230206
    barrar_begin_date,barrar_end_date = 20130101,20181228
    income_begin_date,income_end_date = 20120101,20230207
    cash_begin_date,cash_end_date = 20110104,20230215
    balance_begin_date,balance_end_date = 20120101,20230207
    financialindex_begin_date,financialindex_end_date = 20120101,20230206
#     base_begin_date,base_end_date = 20170501,20180701
#     barrar_begin_date,barrar_end_date = 20170501,20180701
#     income_begin_date,income_end_date = 20170501,20180701
#     cash_begin_date,cash_end_date = 20170501,20180701
#     balance_begin_date,balance_end_date = 20170501,20180701
#     financialindex_begin_date,financialindex_end_date = 20170501,20180701
    base_data_namelist = ['date','cn_code']
    share_income_namelist  = ['REPORT_PERIOD','ANN_DT','date','S_INFO_WINDCODE','STATEMENT_TYPE','OPER_REV','LESS_SELLING_DIST_EXP','LESS_GERL_ADMIN_EXP','LESS_FIN_EXP'] 
    share_financialindex_namelist = ['REPORT_PERIOD','ANN_DT','date','S_INFO_WINDCODE','RD_EXPENSE']
    base_data=load_base_data(base_data_namelist,base_begin_date,base_end_date)
    RDdata = load_financialindex_data(share_financialindex_namelist,financialindex_begin_date,financialindex_end_date)
    
    share_income_all = load_income_data(share_income_namelist,income_begin_date,income_end_date)
    share_income_all['OPER_REV'] = share_income_all['OPER_REV'].astype('float32')
    share_income_all['LESS_SELLING_DIST_EXP'] = share_income_all['LESS_SELLING_DIST_EXP'].astype('float32')
    share_income_all['LESS_GERL_ADMIN_EXP'] = share_income_all['LESS_GERL_ADMIN_EXP'].astype('float32')
    share_income_all['LESS_FIN_EXP'] = share_income_all['LESS_FIN_EXP'].astype('float32')

   
    
    # 计算因子
    print('calculation begin')
    RDdata['month'] = RDdata['REPORT_PERIOD']%10000//100
    RDdata_pre = RDdata[(RDdata['REPORT_PERIOD']>=20121231)&(RDdata['REPORT_PERIOD']<20180630)]#20180630以前每半年披露一次
    RDdata_after = RDdata[(RDdata['REPORT_PERIOD']>=20180630)]#20180630以后每季度披露一次
    RDdata_pre = RDdata_pre[pd.notna(RDdata_pre['RD_EXPENSE'])]
    RDdata_pre = add_shift(RDdata_pre,['RD_EXPENSE'],1)
    RDdata_after = add_shift(RDdata_after,['RD_EXPENSE'],1)
    RDdata_pre['RD_EXPENSE_sea'] = np.nan
    RDdata_pre['RD_EXPENSE_sea'][RDdata_pre['month']==6] = RDdata_pre['RD_EXPENSE'][RDdata_pre['month']==6]
    RDdata_pre['RD_EXPENSE_sea'][RDdata_pre['month']==12] = RDdata_pre['RD_EXPENSE'][RDdata_pre['month']==12]-RDdata_pre['shift_RD_EXPENSE1'][RDdata_pre['month']==12]
    RDdata_after['RD_EXPENSE_sea'] = np.nan
    RDdata_after['RD_EXPENSE_sea'][RDdata_after['month']==3] = RDdata_after['RD_EXPENSE'][RDdata_after['month']==3]
    RDdata_after['RD_EXPENSE_sea'][RDdata_after['month']!=3] = RDdata_after['RD_EXPENSE'][RDdata_after['month']!=3]-RDdata_after['shift_RD_EXPENSE1'][RDdata_after['month']!=3]
    RDdata = pd.concat([RDdata_pre,RDdata_after],axis=0)
    
    
    #截取一部分数据用于后续的计算
    income = share_income_all[['cn_code','date','OPER_REV','LESS_SELLING_DIST_EXP','LESS_GERL_ADMIN_EXP','LESS_FIN_EXP']]
    #分别将计算因子使用到的财务数据merge入base_data
    df = pd.merge_asof(base_data.sort_values(by=['date','cn_code']),income.sort_values(by=['date','cn_code']),on='date',by='cn_code',allow_exact_matches=False)
    df = pd.merge_asof(df.sort_values(by=['date','cn_code']),RDdata[['cn_code','date','RD_EXPENSE_sea']].sort_values(by=['date','cn_code']),on='date',by='cn_code',allow_exact_matches=False)
   
        
         

    #--------------------------------------费用率类因子---------------------------------------
    #期间费用率：(销售费用+管理费用+财务费用)/营业收入
    df['FEE_ratio'] = (df['LESS_SELLING_DIST_EXP']+df['LESS_GERL_ADMIN_EXP']+df['LESS_FIN_EXP'])/df['OPER_REV']
    #销售费用率
    df['SELL_ratio'] = df['LESS_SELLING_DIST_EXP']/df['OPER_REV']
    #管理费用率
    df['MANAGEMENT_ratio'] = df['LESS_GERL_ADMIN_EXP']/df['OPER_REV']
    #财务费用
    df['FIN_ratio'] = df['LESS_FIN_EXP']/df['OPER_REV']    
    #研发费用率
    df['RD_ratio'] = df['RD_EXPENSE_sea']/df['OPER_REV']
       
       

       
       
    #分母为负数的因子更换为nan   
    df['FEE_ratio'][df['OPER_REV']<=0] = np.nan
    df['SELL_ratio'][df['OPER_REV']<=0] = np.nan
    df['MANAGEMENT_ratio'][df['OPER_REV']<=0] = np.nan
    df['FIN_ratio'][df['OPER_REV']<=0] = np.nan
    df['RD_ratio'][df['OPER_REV']<=0]=np.nan   
    
    
    
    #saving
    save_path = 'D:/results_new/'
    for name in ['FEE_ratio','SELL_ratio','MANAGEMENT_ratio','FIN_ratio','RD_ratio']:
        if os.path.exists(save_path+name+'/'):
            print('already exits')
        else:
            os.mkdir(save_path+name+'/')
            df[['date','cn_code',name]].to_csv(save_path+name+'/'+'df.csv',index=False)
    
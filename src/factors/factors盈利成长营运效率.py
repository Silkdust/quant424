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
    base_data_namelist = ['date','cn_code']
    share_income_namelist  = ['REPORT_PERIOD','ANN_DT','date','S_INFO_WINDCODE','STATEMENT_TYPE','NET_PROFIT_INCL_MIN_INT_INC','LESS_FIN_EXP','INC_TAX','OPER_REV','NET_PROFIT_EXCL_MIN_INT_INC','LESS_OPER_COST','TOT_PROFIT','OPER_PROFIT']
    share_income_namelist  = ['REPORT_PERIOD','ANN_DT','date','S_INFO_WINDCODE','STATEMENT_TYPE','LESS_OPER_COST','OPER_REV']
    share_balance_namelist = ['REPORT_PERIOD','ANN_DT','date','STATEMENT_TYPE','S_INFO_WINDCODE','TOT_ASSETS','TOT_LIAB','TOT_CUR_ASSETS','TOT_CUR_LIAB','FIX_ASSETS','INTANG_ASSETS','INVENTORIES','ACCT_PAYABLE','ACCT_RCV','CAP_STK','MONETARY_CAP','DVD_PAYABLE']
    share_balance_namelist = ['REPORT_PERIOD','ANN_DT','date','STATEMENT_TYPE','S_INFO_WINDCODE','ACCT_PAYABLE','ACCT_RCV']
    share_cash_namelist = ['REPORT_PERIOD','ANN_DT','S_INFO_WINDCODE','STATEMENT_TYPE','AMORT_INTANG_ASSETS','AMORT_LT_DEFERRED_EXP','DEPR_FA_COGA_DPBA','NET_CASH_FLOWS_OPER_ACT','NET_CASH_FLOWS_INV_ACT']
    base_data=load_base_data(base_data_namelist,base_begin_date,base_end_date)
    share_income_all = load_income_data(share_income_namelist,income_begin_date,income_end_date)
    share_balance_all = load_balance_data(share_balance_namelist,balance_begin_date,balance_end_date)
    share_cash_all = load_cash_data(share_cash_namelist,cash_begin_date,cash_end_date)
    share_income_all['NET_PROFIT_INCL_MIN_INT_INC'] = share_income_all['NET_PROFIT_INCL_MIN_INT_INC'].astype('float32')
    share_income_all['LESS_FIN_EXP'] = share_income_all['LESS_FIN_EXP'].astype('float32')
    share_income_all['INC_TAX'] = share_income_all['INC_TAX'].astype('float32')
    share_income_all['OPER_REV'] = share_income_all['OPER_REV'].astype('float32')
#     share_income_all['NET_PROFIT_EXCL_MIN_INT_INC'] = share_income_all['NET_PROFIT_EXCL_MIN_INT_INC'].astype('float32')
    share_income_all['LESS_OPER_COST'] = share_income_all['LESS_OPER_COST'].astype('float32')
#     share_income_all['TOT_PROFIT'] = share_income_all['TOT_PROFIT'].astype('float32')
#     share_income_all['OPER_PROFIT'] = share_income_all['OPER_PROFIT'].astype('float32')
    
    
    # 计算因子
    print('calculation begin')
    #截取一部分数据用于后续的计算
#     income = share_income_all[['cn_code','date','NET_PROFIT_INCL_MIN_INT_INC','LESS_FIN_EXP','INC_TAX','OPER_REV','NET_PROFIT_EXCL_MIN_INT_INC','LESS_OPER_COST','TOT_PROFIT','OPER_PROFIT']]
    income = share_income_all[['cn_code','date','LESS_OPER_COST','OPER_REV']]

#     for name in ['LESS_OPER_COST','OPER_REV']:
#         income[name] = income[name].replace(np.nan,0)
#     income = add_shift(income,['LESS_OPER_COST','OPER_REV'],1)
    
    
    
#     balance = share_balance_all[['cn_code','date','TOT_ASSETS','TOT_LIAB','TOT_CUR_ASSETS','TOT_CUR_LIAB','FIX_ASSETS','INTANG_ASSETS','INVENTORIES','ACCT_PAYABLE','CAP_STK','ACCT_RCV','MONETARY_CAP','DVD_PAYABLE']]
    balance = share_balance_all[['cn_code','date','ACCT_PAYABLE','ACCT_RCV']]
#     for name in ['INTANG_ASSETS']:
#         balance[name] = balance[name].replace(np.nan,0)
    balance = add_shift(balance,['ACCT_PAYABLE','ACCT_RCV'],4)
    
    
    
#     cash = share_cash_all[['cn_code','date','AMORT_INTANG_ASSETS','AMORT_LT_DEFERRED_EXP','DEPR_FA_COGA_DPBA','NET_CASH_FLOWS_OPER_ACT','NET_CASH_FLOWS_INV_ACT']]
#     cash = share_cash_all[['cn_code','date','NET_CASH_FLOWS_OPER_ACT','NET_CASH_FLOWS_INV_ACT']]
    #折旧及摊销数据缺失值替换为0（基本全部缺失值）
#     for name in ['AMORT_INTANG_ASSETS','AMORT_LT_DEFERRED_EXP','DEPR_FA_COGA_DPBA']:
#         cash[name] = cash[name].replace(np.nan,0)
#     cash = add_shift(cash,['AMORT_INTANG_ASSETS','AMORT_LT_DEFERRED_EXP','DEPR_FA_COGA_DPBA'],4)
    
    
    
    
    
    #分别将计算因子使用到的财务数据merge入base_data
    balance = balance[pd.notna(balance['ACCT_PAYABLE'])]
    df = pd.merge_asof(base_data.sort_values(by=['date','cn_code']),income.sort_values(by=['date','cn_code']),on='date',by='cn_code',allow_exact_matches=False)
    df = pd.merge_asof(df,balance.sort_values(by=['date','cn_code']),on='date',by='cn_code',allow_exact_matches=False)
#     df = pd.merge_asof(df,cash.sort_values(by=['date','cn_code']),on='date',by='cn_code',allow_exact_matches=False)
    
                
        
        
        
                              
    #-------------------------------盈利类因子----------------------------------------                       
    #ROA(摊薄）使用期末总资产作为分母
#     df['ROA'] = df['NET_PROFIT_INCL_MIN_INT_INC']/(df['TOT_ASSETS'])
#     df['shift_ROA1'] = df['shift_NET_PROFIT_INCL_MIN_INT_INC1']/(df['shift_TOT_ASSETS1'])
#     df['shift_ROA2'] = df['shift_NET_PROFIT_INCL_MIN_INT_INC2']/(df['shift_TOT_ASSETS2'])
#     df['shift_ROA3'] = df['shift_NET_PROFIT_INCL_MIN_INT_INC3']/(df['shift_TOT_ASSETS3'])
#     df['shift_ROA1'][df['shift_TOT_ASSETS1']<=0]=np.nan
#     df['shift_ROA2'][df['shift_TOT_ASSETS2']<=0]=np.nan
#     df['shift_ROA3'][df['shift_TOT_ASSETS3']<=0]=np.nan
#     df['ROA_TTM'] = (df['ROA']+df['shift_ROA1']+df['shift_ROA2']+df['shift_ROA3'])/4
#     #ROA(平均）使用期末总资产和期初总资产平均值作为分母
#     df['ROA_ave'] = 2*df['NET_PROFIT_INCL_MIN_INT_INC']/(df['TOT_ASSETS']+df['shift_TOT_ASSETS1'])       
#     #ROE(摊薄）使用期末净资产作为分母
#     df['ROE'] = df['NET_PROFIT_INCL_MIN_INT_INC']/(df['TOT_ASSETS']-df['TOT_LIAB'])
#     df['shift_ROE1'] = df['shift_NET_PROFIT_INCL_MIN_INT_INC1']/(df['shift_TOT_ASSETS1']-df['shift_TOT_LIAB1'])
#     df['shift_ROE2'] = df['shift_NET_PROFIT_INCL_MIN_INT_INC2']/(df['shift_TOT_ASSETS2']-df['shift_TOT_LIAB2'])
#     df['shift_ROE3'] = df['shift_NET_PROFIT_INCL_MIN_INT_INC3']/(df['shift_TOT_ASSETS3']-df['shift_TOT_LIAB3'])
#     df['shift_ROE1'][df['shift_TOT_ASSETS1']-df['shift_TOT_LIAB1']<=0]=np.nan
#     df['shift_ROE2'][df['shift_TOT_ASSETS2']-df['shift_TOT_LIAB2']<=0]=np.nan
#     df['shift_ROE3'][df['shift_TOT_ASSETS3']-df['shift_TOT_LIAB3']<=0]=np.nan
#     df['ROE_TTM'] = (df['ROE']+df['shift_ROE1']+df['shift_ROE2']+df['shift_ROE3'])/4
#     #ROE(平均）使用期末净资产和期初净资产平均值作为分母
#     df['ROE_ave'] = 2*df['NET_PROFIT_INCL_MIN_INT_INC']/((df['TOT_ASSETS']-df['TOT_LIAB'])+(df['shift_TOT_ASSETS1']-df['shift_TOT_LIAB1']))
#     #净利率
#     df['NPR'] = df['NET_PROFIT_INCL_MIN_INT_INC']/df['OPER_REV']
#     df['shift_NPR1'] = df['shift_NET_PROFIT_INCL_MIN_INT_INC1']/df['shift_OPER_REV1']
#     df['shift_NPR2'] = df['shift_NET_PROFIT_INCL_MIN_INT_INC2']/df['shift_OPER_REV2']
#     df['shift_NPR3'] = df['shift_NET_PROFIT_INCL_MIN_INT_INC3']/df['shift_OPER_REV3']
#     df['shift_NPR1'][df['shift_OPER_REV1']<=0]=np.nan
#     df['shift_NPR2'][df['shift_OPER_REV2']<=0]=np.nan
#     df['shift_NPR3'][df['shift_OPER_REV3']<=0]=np.nan
#     df['NPR_TTM'] = (df['NPR']+df['shift_NPR1']+df['shift_NPR2']+df['shift_NPR3'])/4
#     #归母净利率
#     df['NPR_main'] = df['NET_PROFIT_EXCL_MIN_INT_INC']/df['OPER_REV']
#     df['shift_NPR_main1'] = df['shift_NET_PROFIT_EXCL_MIN_INT_INC1']/df['shift_OPER_REV1']
#     df['shift_NPR_main2'] = df['shift_NET_PROFIT_EXCL_MIN_INT_INC2']/df['shift_OPER_REV2']
#     df['shift_NPR_main3'] = df['shift_NET_PROFIT_EXCL_MIN_INT_INC3']/df['shift_OPER_REV3']
#     df['shift_NPR_main1'][df['shift_OPER_REV1']<=0]=np.nan
#     df['shift_NPR_main2'][df['shift_OPER_REV2']<=0]=np.nan
#     df['shift_NPR_main3'][df['shift_OPER_REV3']<=0]=np.nan
#     df['NPR_main_TTM'] = (df['NPR_main']+df['shift_NPR_main1']+df['shift_NPR_main3'])/4
#     #毛利率
#     df['GPR'] = (df['OPER_REV']-df['LESS_OPER_COST'])/df['OPER_REV']
#     df['shift_GPR1'] = (df['shift_OPER_REV1']-df['shift_LESS_OPER_COST1'])/df['shift_OPER_REV1']
#     df['shift_GPR2'] = (df['shift_OPER_REV2']-df['shift_LESS_OPER_COST2'])/df['shift_OPER_REV2']
#     df['shift_GPR3'] = (df['shift_OPER_REV3']-df['shift_LESS_OPER_COST3'])/df['shift_OPER_REV3']
#     df['shift_GPR1'][df['shift_OPER_REV1']<=0]=np.nan
#     df['shift_GPR2'][df['shift_OPER_REV2']<=0]=np.nan
#     df['shift_GPR3'][df['shift_OPER_REV3']<=0]=np.nan
#     df['GPR_TTM'] = (df['GPR']+df['shift_GPR1']+df['shift_GPR2']+df['shift_GPR3'])/4
#     #营业费用率
#     df['OCR'] = df['LESS_OPER_COST']/df['OPER_REV']
#     df['shift_OCR1'] = df['shift_LESS_OPER_COST1']/df['shift_OPER_REV1']
#     df['shift_OCR2'] = df['shift_LESS_OPER_COST2']/df['shift_OPER_REV2']
#     df['shift_OCR3'] = df['shift_LESS_OPER_COST3']/df['shift_OPER_REV3']
#     df['shift_OCR1'][df['shift_OPER_REV1']<=0]=np.nan
#     df['shift_OCR2'][df['shift_OPER_REV2']<=0]=np.nan
#     df['shift_OCR3'][df['shift_OPER_REV3']<=0]=np.nan
#     df['OCR_TTM'] =(df['OCR']+df['shift_OCR1']+df['shift_OCR2']+df['shift_OCR3'])/4
#     #ROIC(投入资本回报率）=EBIT(1-税率）/投入资本，其中投入资本=流动资产-流动负债+固定资产+无形资产及商誉），税率（使用所得税费用/利润总额进行估计）
#     df['ROIC'] = ((df['NET_PROFIT_INCL_MIN_INT_INC']+df['LESS_FIN_EXP']+df['INC_TAX'])*(1-df['INC_TAX']/df['TOT_PROFIT']))/(df['TOT_CUR_ASSETS']-df['TOT_CUR_LIAB']+df['FIX_ASSETS']+df['INTANG_ASSETS'])
#     df['shift_ROIC1'] = ((df['shift_NET_PROFIT_INCL_MIN_INT_INC1']+df['shift_LESS_FIN_EXP1']+df['shift_INC_TAX1'])*(1-df['shift_INC_TAX1']/df['shift_TOT_PROFIT1']))/(df['shift_TOT_CUR_ASSETS1']-df['shift_TOT_CUR_LIAB1']+df['shift_FIX_ASSETS1']+df['shift_INTANG_ASSETS1'])
#     df['shift_ROIC2'] = ((df['shift_NET_PROFIT_INCL_MIN_INT_INC2']+df['shift_LESS_FIN_EXP2']+df['shift_INC_TAX2'])*(1-df['shift_INC_TAX2']/df['shift_TOT_PROFIT2']))/(df['shift_TOT_CUR_ASSETS2']-df['shift_TOT_CUR_LIAB2']+df['shift_FIX_ASSETS2']+df['shift_INTANG_ASSETS2'])
#     df['shift_ROIC3'] = ((df['shift_NET_PROFIT_INCL_MIN_INT_INC3']+df['shift_LESS_FIN_EXP3']+df['shift_INC_TAX3'])*(1-df['shift_INC_TAX3']/df['shift_TOT_PROFIT3']))/(df['shift_TOT_CUR_ASSETS3']-df['shift_TOT_CUR_LIAB3']+df['shift_FIX_ASSETS3']+df['shift_INTANG_ASSETS3'])
#     df['shift_ROIC1'][df['shift_TOT_CUR_ASSETS1']-df['shift_TOT_CUR_LIAB1']+df['shift_FIX_ASSETS1']+df['shift_INTANG_ASSETS1']<=0]=np.nan
#     df['shift_ROIC2'][df['shift_TOT_CUR_ASSETS2']-df['shift_TOT_CUR_LIAB2']+df['shift_FIX_ASSETS2']+df['shift_INTANG_ASSETS2']<=0]=np.nan
#     df['shift_ROIC3'][df['shift_TOT_CUR_ASSETS3']-df['shift_TOT_CUR_LIAB3']+df['shift_FIX_ASSETS3']+df['shift_INTANG_ASSETS3']<=0]=np.nan
#     df['ROIC_TTM'] = (df['ROIC']+df['shift_ROIC1']+df['shift_ROIC2']+df['shift_ROIC3'])/4
#     #----------------------------------成长类因子----------------------------------------
#     #净利润同比增长率
#     df['NP_G'] = (df['NET_PROFIT_INCL_MIN_INT_INC']-df['shift_NET_PROFIT_INCL_MIN_INT_INC4'])/(abs(df['NET_PROFIT_INCL_MIN_INT_INC'])+abs(df['shift_NET_PROFIT_INCL_MIN_INT_INC4']))
#     #归母净利润同比增长率
#     df['NP_main_G'] = (df['NET_PROFIT_EXCL_MIN_INT_INC']-df['shift_NET_PROFIT_EXCL_MIN_INT_INC4'])/(abs(df['NET_PROFIT_EXCL_MIN_INT_INC'])+abs(df['shift_NET_PROFIT_EXCL_MIN_INT_INC4']))
#     #净资产同比增长率
#     df['NA_G'] = ((df['TOT_ASSETS']-df['TOT_LIAB'])-(df['shift_TOT_ASSETS4']-df['shift_TOT_LIAB4']))/(abs(df['TOT_ASSETS']-df['TOT_LIAB'])+abs(df['shift_TOT_ASSETS4']-df['shift_TOT_LIAB4']))
#     #总资产同比增长率
#     df['TA_G'] = (df['TOT_ASSETS']-df['shift_TOT_ASSETS4'])/(abs(df['TOT_ASSETS'])+abs(df['shift_TOT_ASSETS4']))
#     #营业利润同比增长率
#     df['OP_G'] = (df['OPER_PROFIT']-df['shift_OPER_PROFIT4'])/(abs(df['OPER_PROFIT'])+abs(df['shift_OPER_PROFIT4']))
#     #营业收入同比增长率
#     df['OR_G'] = (df['OPER_REV']-df['shift_OPER_REV4'])/(abs(df['OPER_REV'])+abs(df['shift_OPER_REV4']))
#     #投入资本同比增长率，投入资本=流动资产-流动负债+不动产与厂房设备净额（固定资产）+无形资产及商誉
#     df['IA_G'] = ((df['TOT_CUR_ASSETS']-df['TOT_CUR_LIAB']+df['FIX_ASSETS']+df['INTANG_ASSETS'])-(df['shift_TOT_CUR_ASSETS4']-df['shift_TOT_CUR_LIAB4']+df['shift_FIX_ASSETS4']+df['shift_INTANG_ASSETS4']))/(abs(df['TOT_CUR_ASSETS']-df['TOT_CUR_LIAB']+df['FIX_ASSETS']+df['INTANG_ASSETS'])+abs(df['shift_TOT_CUR_ASSETS4']-df['shift_TOT_CUR_LIAB4']+df['shift_FIX_ASSETS4']+df['shift_INTANG_ASSETS4']))
#     #NOPLAT同比增长率：NOPLAT（扣除调整税后的净营业利润），ROIC分子
#     df['NOPLAT'] = (df['NET_PROFIT_INCL_MIN_INT_INC']+df['LESS_FIN_EXP']+df['INC_TAX'])*(1-df['INC_TAX']/df['TOT_PROFIT'])
#     df['shift_NOPLAT4'] = (df['shift_NET_PROFIT_INCL_MIN_INT_INC4']+df['shift_LESS_FIN_EXP4']+df['shift_INC_TAX4'])*(1-df['shift_INC_TAX4']/df['shift_TOT_PROFIT4'])
#     df['NOPLAP_G'] = (df['NOPLAT']-df['shift_NOPLAT4'])/(abs(df['NOPLAT'])+abs(df['shift_NOPLAT4']))
#     #------------------------------营运效率类--------------------------------------------------
#     #固定资产周转率：销售收入/期初、期末固定资产均值
#     df['FA_turn'] = 2*df['OPER_REV']/(df['FIX_ASSETS']+df['shift_FIX_ASSETS1'])
#     #总资产周转率
#     df['TA_turn'] = 2*df['OPER_REV']/(df['TOT_ASSETS']+df['shift_TOT_ASSETS1'])
#     #流动资产周转率
#     df['CA_turn'] = 2*df['OPER_REV']/(df['TOT_CUR_ASSETS']+df['shift_TOT_CUR_ASSETS1'])
#     #存货周转率
#     df['INV_turn'] = 2*df['LESS_OPER_COST']/(df['INVENTORIES']+df['shift_INVENTORIES1'])
#     #应付账款周转率
    df['ACCPAY_turn'] = 2*df['LESS_OPER_COST']/(df['ACCT_PAYABLE']+df['shift_ACCT_PAYABLE1'])
#     #应收账款周转率
    df['ACCRCV_turn'] = 2*df['OPER_REV']/(df['ACCT_RCV']+df['shift_ACCT_RCV1'])
#     #资本周转率=销售收入/期初期末平均股本
#     df['CAP_turn'] = 2*df['OPER_REV']/(df['CAP_STK']+df['shift_CAP_STK1'])
#     #营运资金周转率
#     df['OC_turn'] = 2*df['OPER_REV']/((df['TOT_CUR_ASSETS']-df['TOT_CUR_LIAB'])+(df['shift_TOT_CUR_ASSETS1']-df['shift_TOT_CUR_LIAB1']))
#     ---------------------------------------盈余质量类因子---------------
#     sloan_ratio 净利润-经营活动产生的现金流-投资活动产生的现金流/总资产         
#     df['sloan_ratio'] = (df['NET_PROFIT_INCL_MIN_INT_INC']-df['NET_CASH_FLOWS_OPER_ACT']-df['NET_CASH_FLOWS_INV_ACT'])/df['TOT_ASSETS']
    
    


    #分母为负数的ROE/ROE_ave因子值修改为nan
#     df['ROE'][df['TOT_ASSETS']-df['TOT_LIAB']<=0]=np.nan
#     df['ROE_ave'][(df['TOT_ASSETS']-df['TOT_LIAB'])+(df['shift_TOT_ASSETS1']-df['shift_TOT_LIAB1'])<=0]=np.nan
#     df['NPR'][df['OPER_REV']<=0]=np.nan
#     df['NPR_main'][df['OPER_REV']<=0]=np.nan
#     df['GPR'][df['OPER_REV']<=0]=np.nan
#     df['OCR'][df['OPER_REV']<=0]=np.nan
#     df['ROIC'][(df['TOT_CUR_ASSETS']-df['TOT_CUR_LIAB']+df['FIX_ASSETS']+df['INTANG_ASSETS'])<=0]=np.nan 
#     df['OC_turn'][((df['TOT_CUR_ASSETS']-df['TOT_CUR_LIAB'])+(df['shift_TOT_CUR_ASSETS1']-df['shift_TOT_CUR_LIAB1']))<=0]=np.nan
#     df['NOPLAT'][df['TOT_PROFIT']<=0] = np.nan
                              
                                                                                                    #saving
    save_path = 'D:/results_new/'
#     for name in ['NP_G','NP_main_G','NA_G','TA_G','OP_G','OR_G','IA_G','NOPLAP_G','FA_turn','TA_turn','CA_turn','INV_turn','ACCPAY_turn','CAP_turn','OC_turn','sloan_ratio','NOPLAT']:
    for name in ['ACCPAY_turn','ACCRCV_turn']:
#     for name in ['ROA','ROA_TTM','ROA_ave','ROE','ROE_TTM','ROE_ave','NPR','NPR_TTM','NPR_main','NPR_main_TTM','GPR','GPR_TTM','OCR','OCR_TTM','ROIC','ROIC_TTM']:
        if os.path.exists(save_path+name+'/'):
            print('already exits')
        else:
            os.mkdir(save_path+name+'/')
            df[['date','cn_code',name]].to_csv(save_path+name+'/'+'df.csv',index=False)
    
    
    
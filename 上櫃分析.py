# -*- coding: utf-8 -*-
"""
Created on Sun Aug 15 09:38:20 2021

@author: wangshuyou
"""

import numpy as np
import pandas as pd
import pandas_datareader.data as pdr
#抓yahoo finance的資料
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import scipy.optimize as solver
# 類似Excel中的規劃求解，可以求出極小化的狀況
from functools import reduce
# reduce()能讓第二個元素中list中的三個變數，前兩個先套用np.dot矩陣相乘後，再將相乘結果跟和三個元素做矩陣相乘
import warnings
warnings.filterwarnings('ignore')

start = datetime.datetime(2010,1,1)
#end = datetime.datetime(2018,12,30)
df = pdr.get_data_yahoo('00679B.TWO') 

df_00679B = pdr.DataReader('00679B.TWO','yahoo',start = start)
df_00697B = pdr.DataReader('00697B.TWO','yahoo',start = start)
df_00719B = pdr.DataReader('00719B.TWO','yahoo',start = start)
df_00687B = pdr.DataReader('00687B.TWO','yahoo',start = start)
df_00694B = pdr.DataReader('00694B.TWO','yahoo',start = start)
df_00695B = pdr.DataReader('00695B.TWO','yahoo',start = start)
df_00696B = pdr.DataReader('00696B.TWO','yahoo',start = start)
df_00764B = pdr.DataReader('00764B.TWO','yahoo',start = start)
df_00779B = pdr.DataReader('00779B.TWO','yahoo',start = start)

# =============================================================================
# # 每天收盤價與第一天收盤價的比較:投資的價值增加幾倍
# for stock in [df_00679B, df_00697B, df_00719B, df_00687B, df_00694B,df_00695B, df_00696B,df_00764B,df_00779B]:
#     stock['normalized_price'] = stock['Adj Close']/stock['Adj Close'].iloc[0]
# # print(df_00858.head())
# =============================================================================


# 單一個股算術平均報酬
df_total_adjc=pd.concat([df_00679B['Adj Close'],df_00697B['Adj Close'], 
                         df_00719B['Adj Close'],df_00687B['Adj Close'], 
                         df_00694B['Adj Close'],df_00695B['Adj Close'], 
                         df_00696B['Adj Close'],df_00764B['Adj Close'],
                         df_00779B['Adj Close']],axis=1)
df_total_adjc.columns=['df_00679B', 'df_00697B', 'df_00719B', 'df_00687B', 'df_00694B',
                       'df_00695B', 'df_00696B', 'df_00764B', 'df_00779B']
df_total_ret = df_total_adjc.pct_change()


# 針對個股報酬製作敘述統計表 
# skew > 0 右偏， skew = 常態， skew < 0 左偏
# =============================================================================

Annual_Mean = pd.DataFrame((df_total_ret.T.mean(axis = 1)*252)).T
Annual_SD = pd.DataFrame(df_total_ret.T.std(axis = 1) * np.sqrt(252)).T
ms = pd.concat([Annual_Mean,Annual_SD],keys = ['Annual_Mean','Annual_SD']) 
# =============================================================================
# ms = str(ms).replace('Annual_Mean 0','Annual_Mean')
# ms = ms.replace('Annual_SD   0 ','Annual_SD')
# =============================================================================
# concat兩個df：內層要中括號
# keys：最外層的名字


stat = df_total_ret.agg(['mean','median','std','skew','kurt','max','min','count']).round(4)
static = pd.concat([stat,ms])
print('敘述統計表(報酬率)：')
print(static)

# =============================================================================
# # 製作共變異、相關性矩陣
# cov_mat = df_total_ret.cov() * 252
# corr_mat = df_total_ret.corr()
# print('共變異矩陣(年化)：') 
# print(cov_mat)
# print('相關性矩陣： ')
# print(corr_mat)
# 
# 
# # 計算投資組合的報酬率及風險
# total_stocks = len(df_total_ret.columns) # 計算投資組合中股票個數
# stocks_weights = np.array([0.25,]* total_stocks) # 設定一個投資權重matrix
# portfolio_return = round(stocks_weights * Annual_Mean,5).sum(axis =1).tolist()
# 
# # 硬把list的資料轉成str，在用replace的方式拿掉[]，美觀而已
# portfolio_return = str(portfolio_return).replace('[', '')
# portfolio_return = portfolio_return.replace(']', '')
# #print(str_mvp_return)
# 
# 
# # 權重假設各為0.2 
# print('預期報酬率為: ',portfolio_return)
# portfolio_risk = np.sqrt(reduce(np.dot, [stocks_weights, cov_mat, stocks_weights.T]))
# print('風險為： ',portfolio_risk)
# 
# # MVP：透過極小化的方式求解，找出最小變異數的投資組合
# 
# # 定義計算投資組合的風險
# def standard_deviation(weights):
#     return np.sqrt(reduce(np.dot, [weights, cov_mat, weights.T])) 
# 
# x0 = stocks_weights # 為極小化過程的變數，就是各個股票的投資權重
# bounds = tuple((0, 1) for x in range(total_stocks)) # 每個變數(權重)的上下界
# constraints = [{'type': 'eq', 'fun': lambda x: sum(x) - 1}]
# minimize_variance = solver.minimize(standard_deviation, x0=x0, constraints=constraints, bounds=bounds)
# # 使用solver的minimize方法求出最小變異數的投資組合
# 
# mvp_risk = minimize_variance.fun
# mvp_return = (minimize_variance.x * Annual_Mean).sum(axis = 1).round(4).tolist()
# 
# # 硬把list的資料轉成str，在用replace的方式拿掉[]，美觀而已
# str_mvp_return = str(mvp_return).replace('[', '')
# str_mvp_return = str_mvp_return.replace(']', '')
# #print(str_mvp_return)
# 
# print('風險最小化投資組合預期報酬率為:' , str_mvp_return)
# print('風險最小化投資組合風險為:' + str(round(mvp_risk,4)))
# 
# 
# for i in range(total_stocks):
#     stock_symbol = str(df_total_ret.columns[i]) # 抓股票代號轉字串
#     weighted = str(format(minimize_variance.x[i], '.4f')) 
#     # 將浮點數轉為字串，.4f 稱為格式規範，表示輸出應僅顯示小數點後四位，.5f代表後五位
#     print(f'{stock_symbol} 佔投資組合權重 : {weighted}')
# 
# # =============================================================================
#  
# # 資產配置、增加權重表格
# for stock, weight in zip([df_00694B, df_00858, df_00758B, df_006201],[0.9165,0.0327,0.0010,0.0498]):
#      # zip:將迭代的直鏈在一起
#      stock['weighted daily return'] = stock['normalized_price']*weight
# #print(df_006201.info())
# # 
# # # =============================================================================
# # # # MP beta極大的點
# # #     
# # # b = (df_total_ret - 0.0066).fillna(0)
# # # beta_00694B = b['00694B.TW'] / Annual_SD['00694B.TW']
# # # =============================================================================
# # 
# # # =============================================================================
# # # min_vol_port = portfolios.iloc[portfolios['Volatility'].idxmin()]
# # # min_vol_port
# # # plt.subplots(figsize=[10,10])
# # # plt.scatter(portfolios['Volatility'], portfolios['Returns'],marker='o', s=10, alpha=0.3)
# # # plt.scatter(min_vol_port[1], min_vol_port[0], color='r', marker='*', s=50)
# # # 
# # =============================================================================
# # 將加權後總報酬集中一張表
# df_total=pd.concat([df_00694B['weighted daily return'], df_00858['weighted daily return'],df_00758B['weighted daily return'], df_006201['weighted daily return']],axis=1)
# df_total.columns=['00694B.TW', '00858.TW', '00758B.TW', '006201.TW']
# #print(df_total.info())
# #print(df_00858.columns)
#  
# #投資：將收益率乘上投入本金，使用Total Pos 將以上金額加總作為總收益
# df_total_money = df_total * 50000000
# df_total_money['Total Pos'] = df_total_money.sum(axis = 1) #axis = 1，矩陣的每一行向量相加
# #print(df_total_money.head())  # 每日投資組合變化
# 
#  
# #投資組合的算術平均報酬率
# df_total_money['daily return'] = df_total_money['Total Pos'].pct_change() # 投資組合日收益率
# # dataframe.pct_change() 計算當前元素與先前元素之間的百分比變化
# TR = df_total_money['Total Pos'].iloc[-1]/df_total_money['Total Pos'].iloc[0]-1
# #累積收益率:計算投資組合最後一天和第一天的變化的百分比
# #print(df_total_money['daily return'].head())
# print("平均日收益率： ",df_total_money['daily return'].mean())
# print("收益率標準差： ",df_total_money['daily return'].std())
# print('最後一期總資產價值： ',df_total_money['Total Pos'][-1])
#    
#    
# #計算第一天及最後一天的變動百分比
# days=len(df_006201) # 總期數
# print("總收益率: ",TR)
# IRR=(1+TR)**(252/days)-1
# print('年化報酬率： ',IRR)
#   
# fig = plt.figure(figsize=(10,6))
# plt.plot(df_total_money['Total Pos'],'-',label='Total Pos')
# plt.plot(df_total_money['00694B.TW'],'-',label='00694B.TW')
# plt.plot(df_total_money['00858.TW'],'-',label='00858.TW')
# plt.plot(df_total_money['00758B.TW'],'-',label='00758B.TW')
# plt.plot(df_total_money['006201.TW'],'-',label='006201.TW')
# plt.title('Profit Curve',loc='right')
# plt.xlabel('Date')
# plt.ylabel('Asset')
# plt.grid(True, axis='y')
# plt.legend(loc = 2)
#  
# #使用密度表呈現平均日收益率
# plt.rcParams['axes.unicode_minus']=False   # 用來正常顯示負號
# fig = plt.figure(figsize=(10, 6))
# sns.distplot(df_total_money['daily return'].dropna(),bins=100, label="Daily Profit Ratio")
# #同時畫出直方圖(hist)和密度圖(kde) 
# #語法：sns.distplot(a, bins=None, hist=True, kde=True, rug=False, fit=None, hist_kws=None, kde_kws=None, rug_kws=None, fit_kws=None, color=None, vertical=False, norm_hist=False, axlabel=None, label=None, ax=None)
# plt.legend()
# plt.show()   
# #Sharp Ratio = 10% ，代表平均每增加1％風險可帶來10%報酬
# #接近1，就是不錯的策略。超過1.5是非常優秀
# #一年252個交易日
# SR=df_total_money['daily return'].mean()/df_total_money['daily return'].std()
# ASR=np.sqrt(252)*SR
# print('夏普指數',ASR)
# 
# =============================================================================

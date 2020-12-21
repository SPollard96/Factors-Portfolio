# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import scipy.optimize as optimize

#Goal : use funds NAV and AuM to create indexes by morning star categories (Value, Blend and Growth) in order to compare 
#                      Active funds results with the Benchmark (MSCI Europe Total Return) from 31/12/2018 to 30/06/2020
#
#Inputs : funds NAV and AuM dataframes by category, MSCI Total Return Gross Dividends historical value
#Returns : dataframe with the indexes based to 100   
def BuildFundsIndex(df_ValueIndex, df_ValueFundsAuM, df_GrowthIndex, df_GrowthFundsAuM, df_BlendIndex, 
                    df_BlendFundsAuM, df_MsciIndexes):
    #calculate funds weights by size
    df_ValueFundsWeights = df_ValueFundsAuM.div(df_ValueFundsAuM.sum(axis = 1), axis = 0)
    #get funds daily % change 
    df_ValueIndex = df_ValueIndex.pct_change()
    #weighting funds performances
    df_ValueIndex = df_ValueIndex.mul(df_ValueFundsWeights, axis='columns')
    #sum weighted daily returns to get the index index daily % change
    df_ValueIndex['Sum'] = df_ValueIndex.sum(axis = 1)
    
    Index = []
    #append base 100
    Index.append(100)
    #calculate index value with the daily returns 
    for i in range(1,len(df_ValueIndex)):
        Index.append(Index[i - 1] * ( 1 + df_ValueIndex['Sum'][i]))
    #add the index to the dataframe
    df_ValueIndex['Value'] = Index
    del(df_ValueFundsWeights)
    del(df_ValueFundsAuM)
       
    df_GrowthFundsWeights = df_GrowthFundsAuM.div(df_GrowthFundsAuM.sum(axis = 1), axis = 0)
    
    df_GrowthIndex = df_GrowthIndex.pct_change()
    df_GrowthIndex = df_GrowthIndex.mul(df_GrowthFundsWeights, axis = 'columns')
    df_GrowthIndex['Sum'] = df_GrowthIndex.sum(axis = 1)
    
    Index = []
    Index.append(100)
    
    for i in range(1, len(df_GrowthIndex)):
        Index.append(Index[i-1] * ( 1 + df_GrowthIndex['Sum'][i]))
    
    df_GrowthIndex['Growth'] = Index
    del(df_GrowthFundsWeights)
    del(df_GrowthFundsAuM)
    
    df_BlendFundsWeights = df_BlendFundsAuM.div(df_BlendFundsAuM.sum(axis = 1), axis = 0)
    
    df_BlendIndex = df_BlendIndex.pct_change()
    df_BlendIndex = df_BlendIndex.mul(df_BlendFundsWeights, axis='columns')
    df_BlendIndex['Sum'] = df_BlendIndex.sum(axis = 1)
    
    Index = []
    Index.append(100)
    
    for i in range(1,len(df_BlendIndex)):
        Index.append(Index[i-1] * ( 1 + df_BlendIndex['Sum'][i]))
        
    df_BlendIndex['Blend'] = Index
    del(df_BlendFundsWeights)
    del(df_BlendFundsAuM)
    del(Index)
    
    #rebase MSCI indexes to 100
    df_MsciIndexes = 100*(df_MsciIndexes/ df_MsciIndexes.iloc[0, :])
    #join all the indexes 
    df_Indexes = pd.concat([df_BlendIndex['Blend'], df_GrowthIndex['Growth'], df_ValueIndex['Value']], axis = 1)
    df_Indexes = df_Indexes.join(df_MsciIndexes)
    df_Indexes = df_Indexes.dropna()
    
    return(df_Indexes)

#Goal : generate a Price Index with our stocks universe from the 31/12/2018 to 30/06/2020
#Inputs : stocks closing prices in EUR
#Returns : dataframe with Price Index base 100
def PriceIndex(df_Universe_EUR_PX):
    df_PX = df_Universe_EUR_PX.copy()
    #drop data prior to 31/12/2018
    df_PX = df_PX.loc['2018-12-31 00:00:00':,]   
    #sum stocks daily closing prices
    df_PX['Sum'] = df_PX.sum(axis = 1)
    #calculate the index divisor needed to obtain a base 100
    #we won't need to rebalance the divisor because the universe is not going to evolve
    IndexDivisor = df_PX.loc['2018-12-31 00:00:00', 'Sum']/100
    df_Index = pd.DataFrame(index=df_PX.index)
    #divide the prices sum by the divisor to rebase it to 100
    df_Index['Price_Index'] = df_PX['Sum']/IndexDivisor

    return df_Index   
    
#Goal : generate a Capital Weighted Index with our stocks universe from the 31/12/2018 to 30/06/2020
#Inputs : stocks closing prices in EUR and the outstanding number of shares (total or floating)
#Returns : dataframe with Capital Weighted Index base 100
def CapiWeightedIndex(df_PX, df_Shares):
    #drop data prior to 31/12/2018
    df_PX = df_PX.loc['2018-12-31 00:00:00':,]
    df_Shares = df_Shares.loc['2018-12-31 00:00:00':,]
    #calculate each company Market Capitalisation
    df_MarketCap = df_PX.mul(df_Shares, axis='columns')
    #MarketCapShift will be used to rebalance the index divisor considering changes in the number of shares
    #if the quantity of shares changes but not the prices we have to rebalance the divisor to avoid any change in the index value
    df_MarketCapShift = df_Shares.mul(df_PX.shift(1), axis='columns')
    df_MarketCap['Sum'] = df_MarketCap.sum(axis = 1)
    MarketCapSum = df_MarketCap['Sum'].to_numpy()
    df_MarketCapShift['Sum'] = df_MarketCapShift.sum(axis = 1)
    MarketCapShiftedSum = df_MarketCapShift['Sum'].to_numpy()
    
    Index = []
    Divisor = []
    Index.append(100)
    Divisor.append(MarketCapSum[0]/Index[0])
    #each day i, rebalance the divisor and calculate the index value 
    for i in range(1,len(df_MarketCap)):
        Divisor.append(MarketCapShiftedSum[i]/Index[i - 1])
        Index.append(MarketCapSum[i]/Divisor[i])
    
    df_Index = pd.DataFrame(index=df_MarketCap.index)
    df_Index['Capitalization_Weighted_Index'] = Index
    
    return df_Index        

#Goal : generate a Total Return Index with our stocks universe from the 31/12/2018 to 30/06/2020
#Inputs : stocks closing prices in EUR, the dividends history in EUR
#Returns : dataframe with Total Return Index base 100
def TotalReturnIndex(df_Universe_EUR_PX, df_Dividends):
    df_PX = df_Universe_EUR_PX.copy()
    df_PX = df_PX.loc['2018-12-31 00:00:00':,]
    df_Dividends = df_Dividends.loc['2018-12-31 00:00:00':,]
    #calculate the Price Index
    df_PX['Sum'] = df_PX.sum(axis=1)
    IndexDivisor = df_PX.loc['2018-12-31 00:00:00', 'Sum']/100
    df_PriceIndex = pd.DataFrame(index=df_PX.index)
    df_PriceIndex['Price_Index'] = df_PX['Sum']/IndexDivisor
    
    IndexLevel = df_PriceIndex['Price_Index'].to_numpy()
    df_TotalDiv = pd.DataFrame(df_Dividends,index=df_PriceIndex.index, columns=df_Dividends.columns)
    df_TotalDiv.fillna(0, inplace=True)
    #divide the dividends by the divisor to convert them into index points 
    df_TotalDiv['Sum'] = df_TotalDiv.sum(axis=1)/IndexDivisor
    IndexDividend = df_TotalDiv['Sum'].to_numpy()
    #Daily total return
    DTR = []
    DTR.append(100)
    #Total return index
    TRI = []
    TRI.append(100)
    for i in range(1, len(IndexLevel)):
        #update the index with the daily total return
        DTR.append(((IndexLevel[i] + IndexDividend[i])/IndexLevel[i-1])-1)
        TRI.append(TRI[i-1]*(1+DTR[i]))
        
    df_Index = pd.DataFrame(index=df_PX.index)
    df_Index['Total_Return_Index'] = TRI
    
    return df_Index

#Goal : momentum stocks selection 
#Inputs : stocks performances and the number of stocks needed
#Returns : series with the N top performing stocks
def Momentum(df_perfSorted, N):
    #sort stocks by perf
    Selection = df_perfSorted.sort_values(axis=0, ascending = False, inplace=True)
    #Select the N top performers
    Selection = df_perfSorted[0:N]
    return Selection

#Goal : 3 factors stocks selection 
#Inputs : stocks performances, market capitalisation, book to price ratio, and the number of stocks needed
#Returns : series with the N top ranked stocks
def ThreeFactors(Perf, Size, Book_To_Price, N):
    #get the performance ranking
    Perf.sort_values(axis = 0, ascending = False, inplace=True)
    r1 = Perf.rank(ascending = False)
    #rank stocks by size in ascending order (1 being the company with lowest market cap)
    Size.sort_values(axis = 0, ascending = True, inplace=True)
    r2 = Size.rank(ascending = True)
    #rank stocks by book to price ratio (1 being the company with the highest ratio)
    Book_To_Price.sort_values(axis = 0, ascending = False, inplace=True)
    r3 = Book_To_Price.rank(ascending = False)
    #mean ranking with equal weights
    Selection = (r1 + r2 + r3)/3
    Selection.sort_values(axis = 0, ascending = True, inplace=True)
    
    return Selection[0:N]

#Goal : 5 factors stocks selection 
#Inputs : stocks performances, market capitalisation, book to price ratio, return on common equity,
#         asset growth and the number of stocks needed
#Returns : series with the N top ranked stocks
def FiveFactors(Perf, Size, Book_To_Price, Return_Com_Eqy, Asset_Growth, N):
    Perf.sort_values(axis = 0, ascending = False, inplace=True)
    r1 = Perf.rank(ascending = False)
    Size.sort_values(axis = 0, ascending = True, inplace=True)
    r2 = Size.rank(ascending = True)
    Book_To_Price.sort_values(axis = 0, ascending = False, inplace=True)
    r3 = Book_To_Price.rank(ascending = False)
    Return_Com_Eqy.sort_values(axis = 0, ascending = False, inplace=True)
    r4 = Return_Com_Eqy.rank(ascending = False) 
    #rank stocks by asset growth (1 being the company with the lowest Asset growth)
    Asset_Growth.sort_values(axis = 0, ascending = True, inplace=True)
    r5 = Asset_Growth.rank(ascending = True)

    Selection = (r1 + r2 + r3 + r4 + r5)/5
    Selection.sort_values(axis = 0, ascending = True, inplace=True)
    
    return Selection[0:N]

#Goal : 6 factors stocks selection 
#Inputs : stocks performances, market capitalisation, book to price ratio, return on common equity,
#         asset growth, sentiment indicator and the number of stocks needed
#Returns : series with the N top ranked stocks
def SixFactors(Perf, Size, Book_To_Price, Return_Com_Eqy, Asset_Growth, Sentiment_indicator, N):
    Perf.sort_values(axis = 0, ascending = False, inplace=True)
    r1 = Perf.rank(ascending = False)
    Size.sort_values(axis = 0, ascending = True, inplace=True)
    r2 = Size.rank(ascending = True)
    Book_To_Price.sort_values(axis = 0, ascending = False, inplace=True)
    r3 = Book_To_Price.rank(ascending = False)
    Return_Com_Eqy.sort_values(axis = 0, ascending = False, inplace=True)
    r4 = Return_Com_Eqy.rank(ascending = False) 
    Asset_Growth.sort_values(axis = 0, ascending = True, inplace=True)
    r5 = Asset_Growth.rank(ascending = True)
    #rank stock by sentiment (1 being the company with the highest number of positive news - negative news)
    Sentiment_indicator.sort_values(axis = 0, ascending = False, inplace=True)
    r6 = Sentiment_indicator.rank(ascending = False)
    
    Selection = (r1 + r2 + r3 + r4 + r5 + r6)/6
    Selection.sort_values(axis = 0, ascending = True, inplace=True)
    return Selection[0:N]

#Goal : calculate the portfolio sharpe ratio with annualized returns and volatility
#Inputs : portfolio daily value and risk free rates 
#Returns : Portfolio Sharpe Ratio and annualized volatility in %
def SharpeRatio(Ptf, rf):
    #calculate the portfolio daily returns
    NAV = pd.DataFrame(Ptf)
    NAV['Daily Return'] = NAV.pct_change(1)
    #calculate the portfolio expected return
    ExpectedReturn = NAV['Daily Return'].mean()
    #annualize the expected return
    ExpectedReturn = ExpectedReturn * (252)
    #calculate the portfolio volatility
    Volatility = NAV['Daily Return'].std() 
    #annualize the volatility
    Volatility = Volatility * np.sqrt(252)
    #calculate the Sharpe Ratio
    SR = (ExpectedReturn - rf) / Volatility
    
    return SR, Volatility*100

#Goal : calculate the portfolio Max Drawdown
#Inputs : portfolio daily value
#Returns : Portfolio Max DrawDown
def MaxDD(NAV):
    Peak = []
    Peak.append(100)
    DD = []
    for i in range(1,len(NAV)-1):
        #get historical peak for day i
        Peak.append(NAV.iloc[0:i,].max())
        #calculate day i drawdown
        DD.append((NAV.iloc[i]/Peak[i])-1)
         
    return min(min(DD),0)*100

#Goal : calculate the portfolio Beta
#Inputs : portfolio and benchmark daily value
#Returns : Portfolio Beta and Alpha
def Ptf_Beta(Ptf, Benchmark):
    NAV = pd.DataFrame(Ptf)
    NAV['Bench'] = Benchmark
    x = NAV['Index'].pct_change(1)[1:]
    y = NAV['Bench'].pct_change(1)[1:] 

    return stats.linregress(y,x)[0:2]

#Goal : calculate a portfolio sharpe ratio for the allocatio optimizer
#Inputs : portfolio stocks weights and return, the risk free rate
#Returns : Portfolio return, volatility and sharpe ratio
def portfolio_stats(weights, returns, rf):
    weights = np.array(weights)
    port_return = np.sum(returns.mean() * weights) * 252
    port_vol = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    sharpe = (port_return - rf) / port_vol

    return {'return': port_return, 'volatility': port_vol, 'sharpe': sharpe}

#Goal : minimize negative sharpe ratio function for the allocation optimizer
#Inputs : stocks weights and returns, risk free rate
#Returns : portfolio sharpe ratio
def minimize_sharpe(weights, returns, rf):  
    return -portfolio_stats(weights, returns, rf)['sharpe'] 

#Goal : minimize portfolio volatility function for the allocation optimizer
#Inputs : stocks weights and returns, risk free rate
#Returns : portfolio volatility
def minimize_volatility(weights, returns, rf):  
    return portfolio_stats(weights, returns, rf)['volatility']

#Goal : find the portfolio allocation in order to maximize Sharpe Ratio
#Inputs : Portfolio number of assets, stocks weights and returns, risk free rate
#Returns : optimized portfolio weights
def max_sharpe_optimization(NbAssets, stocks, returns, rf):
    #set a maximum weight for each stock depending of the number of assets
    Max_alloc = (100/NbAssets + (100/(2*NbAssets)))/100
    #set a minimum weight for each stock depending of the number of assets
    Min_Alloc = (100/NbAssets - (100/(2*NbAssets)))/100
    #set optimizer constraints with sum of stocks weights = 1
    constraints = ({'type' : 'eq', 'fun': lambda x: np.sum(x) -1})
    bounds = tuple((Min_Alloc,Max_alloc) for x in range(NbAssets))
    #initialize stocks weights as equally weighted
    initializer = NbAssets * [1./NbAssets,]
    #get the optimal portfolio by minimizing the negative sharpe ratio
    optimal_sharpe = optimize.minimize(minimize_sharpe,
                                 initializer,
                                 args = (returns, rf,),
                                 method = 'SLSQP',
                                 bounds = bounds,
                                 constraints = constraints)
    #extract stocks weights
    optimal_sharpe_weights=optimal_sharpe['x'].round(4)

    return optimal_sharpe_weights

#Goal : find the portfolio allocation in order to minimize the portfolio volatility
#Inputs : Portfolio number of assets, stocks weights and returns, risk free rate
#Returns : optimized portfolio weights
def min_vol_optimization(NbAssets, stocks, returns, rf):
    Max_alloc = (100/NbAssets + (100/(2*NbAssets)))/100
    Min_Alloc = (100/NbAssets - (100/(2*NbAssets)))/100
    constraints = ({'type' : 'eq', 'fun': lambda x: np.sum(x) -1})
    bounds = tuple((Min_Alloc,Max_alloc) for x in range(NbAssets))
    initializer = NbAssets * [1./NbAssets,]
    optimal_variance=optimize.minimize(minimize_volatility,
                                   initializer,
                                   args = (returns, rf,),
                                   method = 'SLSQP',
                                   bounds = bounds,
                                   constraints = constraints)


    optimal_variance_weights=optimal_variance['x'].round(4)

    return optimal_variance_weights

def BuildPTF(Cash, N, NbAssets, RebalancingN, df_Universe, df_Dividends, df_MktCap, df_Book_to_Price, df_Return_Com_Eqy, 
             df_Asset_Growth, df_news, df_risk_free_Rate, Selector, Allocation):
    #we keep 2018 data for the closing prices because we are going to use past performances 
    df_Universe = df_Universe.loc['2017-12-29 00:00:00':,]
    #only keep 2019/2020 data for the other dataframes
    df_Dividends = df_Dividends.loc['2018-12-31 00:00:00':,]
    df_MktCap = df_MktCap.loc['2018-12-31 00:00:00':,]
    df_Book_to_Price = df_Book_to_Price.loc['2018-12-31 00:00:00':,]
    df_Return_Com_Eqy = df_Return_Com_Eqy.loc['2018-12-31 00:00:00':,]
    df_Asset_Growth = df_Asset_Growth.loc['2018-12-31 00:00:00':,]
    df_news = df_news.loc['2018-12-31 00:00:00':,]
    #df_strat will store the portfolio holdings by market value
    df_strat = pd.DataFrame(np.zeros_like(df_Universe),index=df_Universe.index, columns=df_Universe.columns)
    df_strat = df_strat.loc['2018-12-31 00:00:00':,]
    df_perfs = df_Universe.copy()
    #df_shares will store the portfolio holdings by number of shares
    df_Shares = df_strat.copy()
    
    #Calculate N performances for the momentum selection
    df_perfs = df_perfs.pct_change(N)
    #set the starting day
    df_perfs = df_perfs.loc['2018-12-31 00:00:00':,]
    
    #Calculate log returns for the allocation optimization
    df_log_return = df_Universe.copy()
    df_log_return = np.log(df_log_return/df_log_return.shift(1))
    
    #Rebalance starts at True to build day 1 portfolio
    Rebalance = True
    for i in range(len(df_perfs)):
        #avoid rebalancing if not asked and don't change cash if first day
        if RebalancingN != 0 and i != 0 :
            #rebalance if i is a multiple of RebalancingN
            Rebalance = (i % RebalancingN == 0)
            #set the cash available if selling all assets to past day closing prices
            Cash =  pd.Series(df_Shares.loc[df_perfs.index[i-1]].multiply(df_Universe.loc[df_perfs.index[i-1]], 
                              fill_value = 0)).sum()
            
        if Rebalance == True:
            #get the stocks selection depending of the strategy
            if Selector == 'Momentum':
                Selection = Momentum(df_perfs.iloc[i,:], NbAssets)
            elif Selector == 'ThreeFactors':
                Selection = ThreeFactors(df_perfs.iloc[i,:], df_MktCap.iloc[i,:], df_Book_to_Price.iloc[i,:], NbAssets)
            elif Selector == 'FiveFactors':
                Selection = FiveFactors(df_perfs.iloc[i,:], df_MktCap.iloc[i,:], df_Book_to_Price.iloc[i,:], 
                                        df_Return_Com_Eqy.iloc[i,:], df_Asset_Growth.iloc[i,:], NbAssets)
            elif Selector == 'SixFactors':
                Selection = SixFactors(df_perfs.iloc[i,:], df_MktCap.iloc[i,:], df_Book_to_Price.iloc[i,:], 
                                        df_Return_Com_Eqy.iloc[i,:], df_Asset_Growth.iloc[i,:], df_news.iloc[i,:], NbAssets)
            #separate selected stocks log returns
            returns = df_log_return[Selection.index.values.tolist()]
            returns = returns.loc[:df_perfs.index[i],]
            rf = df_risk_free_Rate.loc[df_perfs.index[i]]
            #set the number of shares for the selected stocks depending of their weights
            if Allocation == 'Equally Weighted':
                weights = np.full(NbAssets, 1/NbAssets)
            elif Allocation == 'Max Sharpe':
                weights = max_sharpe_optimization(NbAssets, Selection.index.values.tolist(), returns, rf)
            elif Allocation == 'Min Volatility':
                weights = min_vol_optimization(NbAssets, Selection.index.values.tolist(), returns, rf)
            #join stocks names and weights
            Selection[:] = weights
            for index, value in Selection.items():
                df_Shares.at[df_perfs.index[i],index] = (value * Cash)/df_Universe.at[df_perfs.index[i],index]
        else :
            #if no rebalance copy last allocation
            df_Shares.iloc[i] = df_Shares.iloc[i-1]
            #check dividends
            df_DividendsReceived = pd.Series(df_Shares.loc[df_perfs.index[i]].multiply(df_Dividends.loc[df_perfs.index[i]], 
                                             fill_value = 0))
            #buy assets of the paying stocks with the money received
            df_DividendsReceived = df_DividendsReceived.div(df_Universe.loc[df_perfs.index[i]], fill_value = 0)
            df_Shares.iloc[i] = pd.Series(df_Shares.iloc[i]).add(df_DividendsReceived)
        
        Rebalance = False
            
    df_strat = df_Shares.mul(df_Universe.loc['2018-12-31 00:00:00':,], axis='columns')
    df_strat["Index"] = df_strat.sum(axis=1)
    #normalise portfolio values to 100
    df_strat["Index"] = 100*(df_strat["Index"] / df_strat["Index"].iloc[0])
    return df_strat


def main():
    #Load Value Funds data for the Funds index
    df_ValueFundsPX = pd.read_excel(r'C:\Users\polla\OneDrive\Bureau\ESCP\Thèse\Harmonised_Data\Funds2.xlsx', 
                                    sheet_name='Value PX_LAST', index_col = 'Dates')
    df_ValueFundsAuM = pd.read_excel(r'C:\Users\polla\OneDrive\Bureau\ESCP\Thèse\Harmonised_Data\Funds2.xlsx', 
                                     sheet_name='Value AuM', index_col = 'Dates')
    df_ValueFundsPX.fillna(method='ffill', inplace=True)
    df_ValueFundsPX = df_ValueFundsPX.loc['2018-12-31 00:00:00':,]
    df_ValueFundsAuM.fillna(method='ffill', inplace=True)
    df_ValueFundsAuM = df_ValueFundsAuM.reindex(df_ValueFundsPX.index)
    
    #Load Growth Funds data for the Funds index
    df_GrowthFundsPX = pd.read_excel(r'C:\Users\polla\OneDrive\Bureau\ESCP\Thèse\Harmonised_Data\Funds2.xlsx', 
                                     sheet_name='Growth PX_LAST', index_col = 'Dates')
    df_GrowthFundsAuM = pd.read_excel(r'C:\Users\polla\OneDrive\Bureau\ESCP\Thèse\Harmonised_Data\Funds2.xlsx', 
                                      sheet_name='Growth AuM', index_col = 'Dates')
    df_GrowthFundsPX.fillna(method='ffill', inplace=True)
    df_GrowthFundsPX = df_GrowthFundsPX.loc['2018-12-31 00:00:00':,]
    df_GrowthFundsAuM.fillna(method='ffill', inplace=True)
    df_GrowthFundsAuM = df_GrowthFundsAuM.reindex(df_GrowthFundsPX.index)
    
    #Load Blend Funds data for the Funds index
    df_BlendFundsPX = pd.read_excel(r'C:\Users\polla\OneDrive\Bureau\ESCP\Thèse\Harmonised_Data\Funds2.xlsx', 
                                    sheet_name='Blend PX_LAST', index_col = 'Dates')
    df_BlendFundsAuM = pd.read_excel(r'C:\Users\polla\OneDrive\Bureau\ESCP\Thèse\Harmonised_Data\Funds2.xlsx', 
                                     sheet_name='Blend AuM', index_col = 'Dates')
    df_BlendFundsPX.fillna(method='ffill', inplace=True)
    df_BlendFundsPX = df_BlendFundsPX.loc['2018-12-31 00:00:00':,]
    df_BlendFundsAuM.fillna(method='ffill', inplace=True)
    df_BlendFundsAuM = df_BlendFundsAuM.reindex(df_BlendFundsPX.index)
    
    #Load MSCI data to compare with the Funds index
    df_MsciIndexes = pd.read_excel(r'C:\Users\polla\OneDrive\Bureau\ESCP\Thèse\Harmonised_Data\TOT_RETURN_INDEX_GROSS_DVDS.xlsx', 
                                   index_col = 'Dates')
    df_MsciIndexes.fillna(method='ffill', inplace=True)
    df_MsciIndexes = df_MsciIndexes.loc['2018-12-31 00:00:00':,]
    
    #Load our stock universe closing prices
    df_Universe_PX = pd.read_excel(r'C:\Users\polla\OneDrive\Bureau\ESCP\Thèse\Harmonised_Data\PX_LAST.xlsx', 
                                   index_col = 'Dates')  
    df_Universe_PX.fillna(method='ffill', inplace=True)

    #Load the total number of shares of each of our universe companies
    df_NbShares = pd.read_excel(r'C:\Users\polla\OneDrive\Bureau\ESCP\Thèse\Harmonised_Data\EQY_SH_OUT.xlsx', 
                                index_col = 'Dates')  
    df_NbShares.fillna(method='ffill', inplace=True)
    
    #load the % of free float of our universe companies
    df_EquityFreeFloatPct = pd.read_excel(r'C:\Users\polla\OneDrive\Bureau\ESCP\Thèse\Harmonised_Data\EQY_FREE_FLOAT_PCT.xlsx', 
                                          index_col = 'Dates')
    df_EquityFreeFloatPct.fillna(method='ffill', inplace=True)
    
    #pass the values to %
    df_EquityFreeFloatPct = df_EquityFreeFloatPct/100  
    df_EquityFreeFloatPct = pd.DataFrame(df_EquityFreeFloatPct,index=df_NbShares.index, 
                                         columns=df_EquityFreeFloatPct.columns)
    
    #calcul the floating nb of shares
    df_NbSharesFloat = df_NbShares.copy()
    df_NbSharesFloat = df_NbShares.mul(df_EquityFreeFloatPct, axis='columns')
    
    #load universe dividends history
    df_Dividends = pd.read_excel(r'C:\Users\polla\OneDrive\Bureau\ESCP\Thèse\Harmonised_Data\DVD_HIST_ALL.xlsx', 
                                 index_col = 'Dates')  
    df_Dividends = pd.DataFrame(df_Dividends, index=df_Universe_PX.index, columns=df_Universe_PX.columns)
    df_Dividends.fillna(0, inplace=True)
    
    #load the curr/EUR exchange rates
    df_FX = pd.read_excel(r'C:\Users\polla\OneDrive\Bureau\ESCP\Thèse\Harmonised_Data\Currencies.xlsx', 
                          index_col = 'Dates')  
    df_FX.fillna(method='ffill', inplace=True)
    
    #load our universe Tickers with currency, industry, country intel
    df_TickersCurncy = pd.read_excel(r'C:\Users\polla\OneDrive\Bureau\ESCP\Thèse\Harmonised_Data\Tickers With Currency.xlsx', 
                                     index_col = 'Tickers')
    
    #load the price to book ratio history (for stock picking strat)
    df_PX_To_Book_ratio = pd.read_excel(r'C:\Users\polla\OneDrive\Bureau\ESCP\Thèse\Harmonised_Data\PX_TO_BOOK_RATIO.xlsx', 
                                        index_col = 'Dates')
    df_PX_To_Book_ratio.fillna(method='ffill', inplace=True)
    
    #pass the ratio to Book to price
    df_Book_To_PX_ratio = df_PX_To_Book_ratio**(-1) 
    
    #load the Return on common Equity history
    df_Return_Com_Eqy = pd.read_excel(r'C:\Users\polla\OneDrive\Bureau\ESCP\Thèse\Harmonised_Data\RETURN_COM_EQY.xlsx', 
                                      index_col = 'Dates')
    df_Return_Com_Eqy.fillna(method='ffill', inplace=True)
    
    #calculate the Return on Common Equity history for each sector
    df_Return_Com_Eqy_by_sector = df_Return_Com_Eqy.T
    df_Return_Com_Eqy_by_sector = df_Return_Com_Eqy_by_sector.join(df_TickersCurncy['INDUSTRY_SECTOR'])
    df_Return_Com_Eqy_by_sector = df_Return_Com_Eqy_by_sector.groupby(['INDUSTRY_SECTOR']).mean()
    df_Return_Com_Eqy_by_sector = df_Return_Com_Eqy_by_sector.T
    
    #adjust the Return of Common Equity of each asset by the sector mean (as a ratio)
    for it in df_TickersCurncy.index:
        Industry = df_TickersCurncy.at[it, 'INDUSTRY_SECTOR']
        df_Return_Com_Eqy[it] = df_Return_Com_Eqy[it] / df_Return_Com_Eqy_by_sector[Industry]
    
    df_Return_Com_Eqy = df_Return_Com_Eqy.reindex(df_Universe_PX.index)
    df_Return_Com_Eqy.fillna(method='ffill', inplace=True)
    
    #load the Asset Growth history
    df_Asset_Growth = pd.read_excel(r'C:\Users\polla\OneDrive\Bureau\ESCP\Thèse\Harmonised_Data\ASSET_GROWTH.xlsx', 
                                    index_col = 'Dates')
    df_Asset_Growth.fillna(method='ffill', inplace=True)
    df_Asset_Growth = df_Asset_Growth.reindex(df_Universe_PX.index)
    df_Asset_Growth.fillna(method='ffill', inplace=True)
    
    #convert stocks closing prices to EUR
    df_Universe_EUR_PX = df_Universe_PX.copy()
    for col_name in df_Universe_EUR_PX.columns:
        Crncy = df_TickersCurncy.at[col_name, 'QUOTED_CRNCY']
        if Crncy != 'EUR':
            df_Universe_EUR_PX[col_name] = df_Universe_EUR_PX[col_name] * df_FX[Crncy + '/EUR']
            
    #convert dividends to EUR
    df_EUR_Dividends = df_Dividends.copy()
    for col_name in df_EUR_Dividends.columns:
        Crncy = df_TickersCurncy.at[col_name, 'DVD_CRNCY']
        if Crncy != 'EUR':
            df_EUR_Dividends[col_name] = df_EUR_Dividends[col_name] * df_FX[Crncy + '/EUR']
    #load Eurozone rates to use as risk free rates
    df_risk_free_Rate = pd.read_excel(r'C:\Users\polla\OneDrive\Bureau\ESCP\Thèse\Harmonised_Data\Eurozone.xlsx', 
                                      index_col = 'Dates')
    df_risk_free_Rate.fillna(method='ffill', inplace=True)
    df_risk_free_Rate = df_risk_free_Rate.reindex(df_Universe_PX.index)
    df_risk_free_Rate.fillna(method='ffill', inplace=True)
    df_risk_free_Rate = df_risk_free_Rate / 100
    
    #calculate each stock floating market cap using prices and float nb of shares
    df_Float_Market_Cap = df_Universe_EUR_PX.copy()
    df_Float_Market_Cap = df_Float_Market_Cap.mul(df_NbSharesFloat, axis='columns')
    
    #Load positive, neutral and negative news data  
    df_positive_news = pd.read_excel(r'C:\Users\polla\OneDrive\Bureau\ESCP\Thèse\Harmonised_Data\NEWS_POS_SENTIMENT_COUNT.xlsx', 
                                     index_col = 'Dates')
    df_positive_news.fillna(0, inplace=True)
    df_negative_news = pd.read_excel(r'C:\Users\polla\OneDrive\Bureau\ESCP\Thèse\Harmonised_Data\NEWS_NEG_SENTIMENT_COUNT.xlsx', 
                                     index_col = 'Dates')
    df_negative_news.fillna(0, inplace=True)
    #create a custome sentiment factor summing the past month number of positive news - negative news
    df_news = df_positive_news - df_negative_news
    df_news = df_news.rolling(20).sum()
    df_news = df_news.reindex(df_Universe_PX.index)
    
###############################################################################    
    #build custom index with Value, Growth and Blend funds 
    FundsIndex = BuildFundsIndex(df_ValueFundsPX, df_ValueFundsAuM, df_GrowthFundsPX, df_GrowthFundsAuM, 
                                 df_BlendFundsPX, df_BlendFundsAuM, df_MsciIndexes)
    plt.figure()
    FundsIndex.plot()
    
    #build a price index with our stocks universe
    df_CustomIndexes = PriceIndex(df_Universe_EUR_PX)
    
    #build a capital weighted index with our stocks universe
    df_CustomIndexes= df_CustomIndexes.join(CapiWeightedIndex(df_Universe_EUR_PX, df_NbSharesFloat))
    
    #build a total return index with our stocks universe
    df_CustomIndexes= df_CustomIndexes.join(TotalReturnIndex(df_Universe_EUR_PX, df_EUR_Dividends))
    
    plt.figure()
    df_CustomIndexes.plot()
    
    
    #set portoflio start parameters
    #RebalancingN is the rebalancing interval, 0 means no rebalancing
    #N is the performances interval used for the momentum ranking
    #Cash is the initial portoflio seeding
    #NbAssets is the number of different assets the Ptf needs
    RebalancingN = 126
    N = 252
    Cash = 1000000
    NbAssets = 80
    #set the stock picking strategy
    Selector = 'SixFactors' # Momentum # ThreeFactors # FiveFactors # SixFactors
    #set the allocation strategy
    Allocation = 'Max Sharpe' # Equally Weighted  # Max Sharpe # Min Volatility
    Portfolio = BuildPTF(Cash,N, NbAssets, RebalancingN, df_Universe_EUR_PX, df_EUR_Dividends, df_Float_Market_Cap, 
                           df_Book_To_PX_ratio, df_Return_Com_Eqy, df_Asset_Growth, df_news, df_risk_free_Rate['1Y'], 
                           Selector, Allocation)
    
    plt.figure()
    df_Portfolio = df_CustomIndexes.copy()
    df_Portfolio = df_CustomIndexes.join(Portfolio['Index'])
    df_Portfolio = df_Portfolio.rename(columns = {'Index': Selector})
    df_Portfolio.plot()
    #Display ptf Perf
    print('The Portfolio return is : ',  round(Portfolio['Index'].iloc[-1]-100, 2), '%')
    
    #calculate the portfolio Max Drawdown
    MDD = MaxDD(Portfolio['Index'])
    print('The Max Drawdown is : ',  round(MDD, 2), '%')
    
    #calculate the portfolio Sharpe Ratio
    SR = SharpeRatio(Portfolio['Index'], df_risk_free_Rate.iloc[-1]['1Y'])
    print('The Sharpe Ratio is : ',  round(SR[0], 2))
    print('The Portfolio Volatility is : ',  round(SR[1], 2), '%')
    
    #calculate the portfolio Beta
    (beta, alpha) = Ptf_Beta(Portfolio['Index'], df_CustomIndexes['Price_Index'])
    print('The Portfolio beta is : ',  round(beta, 2))


if __name__ == "__main__":
    main()


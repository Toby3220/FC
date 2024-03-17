#
import xlwings
import mvo_strategies
import numpy as np
import pandas as pd
import FCHW3 as hw3

#-----------------------------------------------------------------
def writefile(filename):
    wb=xlwings.Book(filename)
    ws=wb.sheets['portfolio mvo']
#
    ws.range(7,4).value=w0
    for i in range(0,len(w)):
        ws.range(i+8,4).value=w[i]
#
    return

#-----------------------------------------------------------------
#
interfacefile='mvo.xlsx'
datafile='RawData.csv'

# DebugCompare compares the results between commercial and custom software for Debugging 
# a MOSEK Licence WILL BE NEEDED to run, set to False if it is not installed
# installation instructions can be fonud here URL: https://www.mosek.com/
DebugCompare = False

Data,portmean,riskFree,strategy,lb,ub=hw3.readfile(interfacefile,datafile)

Mu = Data.mean()
Vc = Data.cov()
U = pd.Series(1,index = Mu.index)
n = Mu.shape[0]

#
if(strategy=='long/short'):
    w0,w,lambda1,lambda2=mvo_strategies.mvoLS(n,Mu.tolist(),Vc.values.tolist(),U.tolist(),portmean)
elif(strategy=='long/short with cash'):
    w0,w,lambda1,lambda2=mvo_strategies.mvoLSC(n,Mu.tolist(),Vc.values.tolist(),U.tolist(),portmean,riskFree)
elif(strategy=='long with cash'):
    w0,w=mvo_strategies.mvoLC(n,Mu.tolist(),Vc.values.tolist(),U.tolist(),portmean,riskFree)
elif(strategy=='Bounded long/short with cash'):

    # calls Algorithm 
    opt = hw3.BasicOpt(Mu,Vc,ub,lb,riskFree,portmean)
    flag = opt.Solve()
    w = opt.w
    w0 = opt.w0
    print(w)

    if DebugCompare:
        # calls commercial software 
        optsolution = hw3.MOSEKOpt(Mu,Vc,ub,lb,riskFree,portmean)
        sflag = optsolution.Solve()
        wt = pd.Series(optsolution.w)
        
        # finds solution bounds
        bt = pd.Series(0,index = Mu.index)
        bt[np.isclose(abs(wt-ub),0,atol=1e-5) | (wt>ub)]=2
        bt[np.isclose(abs(wt-lb),0,atol=1e-5) | (wt<lb)]=1
        opt.setbt(bt)

        # print difference
        print(wt-w)

 
#
writefile(interfacefile)
#

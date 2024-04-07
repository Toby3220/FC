#
import xlwings
import matplotlib.pyplot as plt
import math
from tqdm import tqdm 
import numpy as np

class IFDEuropeanKOCall:
    def __init__(self,riskfree,imax, jmax, dprice,sigma_params = {"a":0.05,"b":-0.01,"c":0.25}) -> None:
        self.rf = riskfree
        self.imax = imax
        self.jmax = jmax
        self.dprice = dprice
        self.sigma_params = sigma_params

        self.option_price=np.arange(0,(jmax+1)*dprice,dprice)
        self.underlying_price=self.option_price.copy()
        pass
    
    def _set_params(self):
        self.type = "European"

    def price(self,K,T,L,q):
        self._set_params()

        dtime = T/self.imax
        option = self.option_price
        underlying = self.underlying_price

        option = self.payoff(option,K)
        option = self.boundary_condition(option,K,L,q)
        
        for i in tqdm(range(0,self.imax)):
            t = T - i*dtime
            P,Q = self.constructPQ(t,dtime)
            Q = np.array(Q)
            P = np.array(P)
            option = np.linalg.pinv(P)@Q@option
            option = self.boundary_condition(option,K,L,q)
        
        return underlying, option

    def payoff(self,S,K):
        return np.maximum(S-K,0)
        
    def boundary_condition(self,S,K,L,q):
        intrinsic = self.payoff(self.underlying_price,K)
        option_price = np.where(self.underlying_price<=L,q*intrinsic,S)
        return option_price
    
    def constructPQ(self,t,dtime):
    #  Construct the transformation matrices P and Q under Crank-Nicholson scheme
        from math import exp
        jmax = self.jmax
        riskfree = self.rf
    #
        P=[[0 for k in range(jmax+1)] for j in range(jmax+1)]
        Q=[[0 for k in range(jmax+1)] for j in range(jmax+1)]	
    #
        if(self.type=='American'):
            P[0][0]=1
        elif(self.type=='European'):
            P[0][0]=exp(self.rf*dtime)
    #
        P[jmax][jmax]=1
        Q[0][0]=1
        Q[jmax][jmax]=1
    #
    # insert tridiagonal entries
        for Lr in range(1,jmax-1+1):
            Slr = Lr*self.dprice
            params = self.sigma_params

            # P at time t-1
            sigma = self.curr_sigma(Slr,t-dtime,**params)

            a=0.5*(Lr*riskfree*dtime)-0.5*(Lr**2*sigma**2*dtime)
            d=(riskfree*dtime)+(Lr**2*sigma**2*dtime)
            c=-0.5*(Lr*riskfree*dtime)-0.5*(Lr**2*sigma**2*dtime)
    #
            P[Lr][Lr-1]=0.5*a
            P[Lr][Lr]=1+0.5*d
            P[Lr][Lr+1]=0.5*c
            
            # Q at time t
            sigma = self.curr_sigma(Slr,t,**params)

            a=0.5*(Lr*riskfree*dtime)-0.5*(Lr**2*sigma**2*dtime)
            d=(riskfree*dtime)+(Lr**2*sigma**2*dtime)
            c=-0.5*(Lr*riskfree*dtime)-0.5*(Lr**2*sigma**2*dtime)
    #
            Q[Lr][Lr-1]=-0.5*a
            Q[Lr][Lr]=1-0.5*d
            Q[Lr][Lr+1]=-0.5*c
    #
        return P,Q
    
    def curr_sigma(self,St,t,a,b,c):
        # function to find the current value of volatility (Sigma)
        ## INPUT
        # t:        float   = currnet time 
        # a,b,c:    float   = parameter value of the equation for volatility in the from of
        #                     a*x^2 + b*x + c
        ## RETRUN
        # sigma float   = value of currnet sigma  

        vt = a*(t**2) + b*t + c
        vtdot = 2*a*t + b
        vout = vt**2 + 2*t*vt*vtdot
        sigma = math.sqrt(vout)
        return sigma

class IFDAmericanKOCall (IFDEuropeanKOCall):
    def _set_type(self):
        self.type = "American"
    
    def boundary_condition(self, S, K, L, q):
        intrinsic = self.payoff(self.underlying_price,K)
        maxval = np.maximum(intrinsic,S)
        option_price = np.where(self.underlying_price<=L,q*intrinsic,maxval)
        return option_price

class IFDEuropeanKOCall_VolSurface (IFDEuropeanKOCall):
    def __init__(self, riskfree, imax, jmax, dprice, local_vol_surface) -> None:
        self.prec = 2
        self.VolSurface = local_vol_surface
        super().__init__(riskfree, imax, jmax, dprice, dict())
    def _set_params(self):
        self.type = "European"
    def curr_sigma(self, St, t,):
        t = round(t,self.prec)
        v = max(self.VolSurface[St,t],0) 
        return math.sqrt(v)
 
class IFDEuropeanKOCall_CappedVolSurface(IFDEuropeanKOCall_VolSurface):
    def curr_sigma(self, St, t):
        t = round(t,self.prec)
        v = max(self.VolSurface[St,t],0) 
        v = min(v,3)
        return math.sqrt(v)

#----------------------------------------------------------------------------------
def readfile(filename):
    wb=xlwings.Book(filename)
    ws=wb.sheets['Option Pricing']
# 
    Smax=int(ws["B2"].value)
    strike=int(ws["B3"].value)
    Sko=ws["B4"].value
    maturity=ws["B5"].value
    riskFree=ws["B6"].value
    q=ws["B7"].value
    imax=int(ws["B8"].value)    
    jmax=int(ws["B9"].value)
    type=ws["B10"].value


    dprice = (Smax/jmax)
#
    return jmax,imax,maturity,strike,riskFree,dprice,type,Sko,q
#
#-----------------------------------------------------------------
def writefile(filename,Svec,Fvec,jmax):
    wb=xlwings.Book(filename)
    ws=wb.sheets['Option Pricing']
#
    for i in tqdm(range(0,jmax+1)):
        ws.range(i+2,6).value=Svec[i]
        ws.range(i+2,7).value=Fvec[i]
#
    return
#
#-----------------------------------------------------------------


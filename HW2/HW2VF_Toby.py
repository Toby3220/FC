#
import xlwings
import mop
import matplotlib.pyplot as plt
import math
from tqdm import tqdm 


#-------------------------------------------------------------------
def ifdpricing(maturity,strike,riskFree,imax,jmax,dprice,type,Sko,q):
# Create vector for both underlying and option 
    Fvec=[0 for Lr in range(0,jmax+1)]
    Svec=[0 for Lr in range(0,jmax+1)]
#
    dtime=maturity/imax
    
#
#  Initialize the option vector according to the payoff condition at maturity
    for Lr in range(0,jmax+1):
        Fvec[Lr]=payoff(strike,Lr*dprice) 
        Svec[Lr]=Lr*dprice
    Fvec = boundary_ko(Fvec,jmax,dprice,strike,type,Sko,q)
#
#  Perform the backward iteration
    for i in tqdm(range(1,imax+1)):

        # find sigma according to model, and construct P,Q update matrixes 
        t = maturity - i*dtime
        sigma = currSigma(t,0.05,-0.01,0.25) 
        P,Q=constructPQ(jmax,dtime,riskFree,sigma,type)
        
        # update, and alter Fvec according to the boundary + other conditins
        Fvec=mop.multAbc(Q,Fvec,jmax+1,0,0,0)
        Fvec=mop.solveAxb(P,Fvec,jmax+1,0,0,0)        
        Fvec=boundary_ko(Fvec,jmax,dprice,strike,type,Sko,q)
#
    return Svec,Fvec
#
def currSigma(t, a, b, c):
    # function to find the current value of volatility (Sigma)
    ## INPUT
    # t:        float   = currnet time 
    # a,b,c:    float   = parameter value of the equation for volatility in the from of
    #                     a*x^2 + b*x + c
    ## RETRUN
    # currSigma float   = value of currnet sigma  

    vt = a*(t**2) + b*t + c
    vtdot = 2*a*t + b
    vout = vt**2 + 2*t*vt*vtdot
    currSigma = math.sqrt(vout)
    return currSigma
#-------------------------------------------------------------------
def constructPQ(jmax,dtime,riskfree,sigma,type):
#  Construct the transformation matrices P and Q under Crank-Nicholson scheme
    from math import exp
#
    P=[[0 for k in range(jmax+1)] for j in range(jmax+1)]
    Q=[[0 for k in range(jmax+1)] for j in range(jmax+1)]	
#
    if(type=='American'):
        P[0][0]=1
    elif(type=='European'):
        P[0][0]=exp(riskFree*dtime)
#
    P[jmax][jmax]=1
    Q[0][0]=1
    Q[jmax][jmax]=1
#
# insert tridiagonal entries
    for Lr in range(1,jmax-1+1):
        a=0.5*(Lr*riskfree*dtime)-0.5*(Lr**2*sigma**2*dtime)
        d=(riskfree*dtime)+(Lr**2*sigma**2*dtime)
        c=-0.5*(Lr*riskfree*dtime)-0.5*(Lr**2*sigma**2*dtime)
#
        P[Lr][Lr-1]=0.5*a
        P[Lr][Lr]=1+0.5*d
        P[Lr][Lr+1]=0.5*c
#
        Q[Lr][Lr-1]=-0.5*a
        Q[Lr][Lr]=1-0.5*d
        Q[Lr][Lr+1]=-0.5*c
#
    return P,Q
#
#------------------------------------------------------------------
def payoff(strike,price):
    return max(price-strike,0.0)
#
#------------------------------------------------------------------
def boundary_ko(Fvec,jmax,dprice,strike,type,Sko,q):
    # Payoff function for knockout options

    ## INPUT
    # Fvec :    List    = list of option price at time M - i*dtime, with each value
    #                     corrisponding to a underlying price
    # jmax :    int     = maximum value of the underlying price
    # dprice:   float   = incremental (precision of the) changes in price of the underlying
    # strike:   float   = strike price
    # type:     str     = type of option {European or American}
    # Sko:      float   = knockout price
    # q:        float   = rebate amount (express in decimal form) {i.e. Range : 0 - 1}
    
    ## RETURN
    # Fvec:     List    = udpated Fvec
#   
    for Lr in range(0,jmax+1):
        Slr = Lr*dprice
        intrinsicValue=payoff(strike,Lr*dprice)
        maxval=max(Fvec[Lr],intrinsicValue)

        if(type=='European'):
            if Slr <= Sko: 
                Fvec[Lr]=q*intrinsicValue
            else:
                Fvec[Lr]=Fvec[Lr]
#
        elif(type=='American'):
            if Slr <= Sko: 
                Fvec[Lr] = q*intrinsicValue
            else: 
                Fvec[Lr] = maxval
#
    return Fvec

def boundary(Fvec,jmax,dprice,strike,type,Sko):

#
    if(type=='European'):
        for Lr in range(0,jmax+1):
            Fvec[Lr]=Fvec[Lr]
#
    elif(type=='American'):
        for Lr in range(0,jmax+1):
            intrinsicValue=payoff(strike,Lr*dprice)
            Fvec[Lr] = max(Fvec[Lr],intrinsicValue)

#              
    return Fvec
#
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
def writefile(filename):
    wb=xlwings.Book(filename)
    ws=wb.sheets['Option Pricing']
#
    for i in tqdm(range(0,jmax+1)):
        ws.range(i+2,6).value=Svec[i]
        ws.range(i+2,7).value=Fvec[i]
#
    plt.plot(Svec,Fvec)
    plt.xlabel('Asset Price')
    plt.ylabel('Option Price')
    plt.show(block = True)
#
    return
#
#-----------------------------------------------------------------
# Input parameters
#
jmax,imax,maturity,strike,riskFree,dprice,type,Sko,q=readfile('ifd_call.xlsx')
#
# Perform the matrix calculation

Svec,Fvec=ifdpricing(maturity,strike,riskFree,imax,jmax,dprice,type,Sko,q)

#
# Output option prices
writefile('ifd_call.xlsx')
#

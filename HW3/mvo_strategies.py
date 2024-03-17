#
import mop
import math
#
#-----------------------------------------------------------------
def eps():
    return 1.e-14
#
#-----------------------------------------------------------------
# Mean-variance portfolio optimization with long/short
#
def mvoLS(nasset,mu,vc,u,portmean):
#
    vec1=mop.solveAxb(vc,mu,nasset,0,0,0)
    vec2=mop.solveAxb(vc,u,nasset,0,0,0)
#
    A=mop.multbtc(u,vec1,nasset,0,0)  
    B=mop.multbtc(mu,vec1,nasset,0,0)  
    C=mop.multbtc(u,vec2,nasset,0,0)  
    D=B*C-A*A+eps()
#
    lambda1=(-C*portmean+A)/D
    lambda2=(A*portmean-B)/D
#
    w=[]
    for i in range(nasset):
        w.append(-lambda1*vec1[i]-lambda2*vec2[i])
#
    return 0,w,lambda1,lambda2
#
#-----------------------------------------------------------------
# Mean-variance portfolio optimization with long/short and cash
#
def mvoLSC(nasset,mu,vc,u,portmean,riskFree):
#
    vec1=mop.solveAxb(vc,mu,nasset,0,0,0)
    vec2=mop.solveAxb(vc,u,nasset,0,0,0)
#
    A=mop.multbtc(u,vec1,nasset,0,0)  
    B=mop.multbtc(mu,vec1,nasset,0,0)  
    C=mop.multbtc(u,vec2,nasset,0,0)  
#
    D=C*riskFree*riskFree-2*A*riskFree+B+eps()
    lambda1=-(portmean-riskFree)/D
    lambda2=riskFree*(portmean-riskFree)/D
#
    w=[]
    for i in range(nasset):
        w.append(-lambda1*vec1[i]-lambda2*vec2[i])
#
    w0=1-(portmean-riskFree)*(A-riskFree*C)/D
#
    return w0,w,lambda1,lambda2
#
#-----------------------------------------------------------------
#
#-------------------------------------------------------------------
def binaryComb2(n):
#  generate a size n binary sequence with all combination of 0 and 1
#
    x=[]
    for k in range(n+1):
#        
        if(k==0) :
            outLabel=[[]]
        elif(k==1):
            outLabel=[[L+1] for L in range(n)]
        else:
            y=[]
            for L in range(0,len(outLabel)):
                for j in range(outLabel[L][k-2]+1,n+1):  
                    y.append(outLabel[L]+[j])
            outLabel=y
#        
        for L in range(len(outLabel)):
            y=''
            for i in range(n):
                flag=False
                for j in range(k):
                    if(i==outLabel[L][j]-1):flag=True
                if(flag):
                    y=y+'1'
                else:    
                    y=y+'0'
            x.append(y)
#
    return x
#
#-----------------------------------------------------------------
def binaryComb(n):
#  generate a size n binary sequence with all combination of 0 and 1
#
    x=[]
    for i in range(2**n):
        y=bin(i)[2:]
        for j in range(0,n-len(y)):y='0'+y
        x.append(y)
    return x
#
#-----------------------------------------------------------------
def modifymvc(nasset,mu,vc,u,Iout):
#
# duplicate the arrays
    mum=[]
    um=[]
    vcm=[]   

    for i in range(0,nasset):
        mum.append(mu[i])
        um.append(u[i])
        vec=[]
        for j in range(0,nasset):
            vec.append(vc[i][j])
        vcm.append(vec)       
#
# modify the arrays according to Iout
    for i in range(0,nasset):
        if(Iout[i]!='0') :
            mum[i]=0
            um[i]=0
            for j in range(0,nasset):
                if(i==j):  
                   vcm[i][j]=1
                else:
                   vcm[i][j]=0
                   vcm[j][i]=0
#
    return mum,vcm,um
#
#-----------------------------------------------------------------
def kkt(nasset,mu,vc,u,w,lambda1,lambda2):
    from math import fabs
    
    #Vc@w
    eta=mop.multAbc(vc,w,nasset,0,0,0)

    #LSC = Vc@w + L1*mu + L2*u
    # LSC: lagrangian Stationarity Conditions
    for i in range(0,nasset):
        eta[i]+=lambda1*mu[i]+lambda2*u[i]
#   
    #       if      abs(eta) <= tol AND w > 0 OR 
    #               eta > tol AND abs(w) < 0
    # i.e.  if      LSC = 0 AND w > 0 OR
    #               LSC > 0 AND w = 0 
                         
    for i in range(0,nasset):
        kktflag=(fabs(eta[i])<=eps() and w[i]>=-eps())\
                or (eta[i]>eps() and fabs(w[i])<=eps())
        if(not kktflag): break
#
    return kktflag
#
#-----------------------------------------------------------------
# Mean-variance portfolio optimization with long and cash
# (Markowitz algorithm)
#
def mvoLC(nasset,mu,vc,u,portmean,riskFree):
#
#  generate all possible OUTsubsets
    Icomb=binaryComb(nasset)
    ncomb=len(Icomb)
#
    for k in range(0,ncomb-1):
        
        #  construct the modified arrays according to given OUTsubset 
        Iout=Icomb[k]    
        mum,vcm,um=modifymvc(nasset,mu,vc,u,Iout)

        #  calculate the portfolio content from long/short with cash strategy
        w0,w,lambda1,lambda2=mvoLSC(nasset,mum,vcm,um,portmean,riskFree)

        #  check that the portfolio contents are non-negative
        #  If so, check kkt condition. Otherwise, try another Iout

        kktflag=False
        skipflag=False
        for i in range(0,nasset):
            if(w[i]<-eps()): skipflag=True

        if(not skipflag):
            kktflag=kkt(nasset,mu,vc,u,w,lambda1,lambda2)
            if(kktflag):break

    if(not kktflag):
        w0=1.0
        for i in range(0,nasset):w[i]=0.0
#
    return w0,w 
#
#-------------------------------------------------------------------

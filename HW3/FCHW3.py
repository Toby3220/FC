import numpy as np
import pandas as pd
import math 
import mosek.fusion as mos 
from tqdm import tqdm
import xlwings

class BasicOpt:

    def __init__(
            self,Mu: pd.Series,
            Sigma: pd.Series,
            ub: pd.Series,
            lb: pd.Series,
            mu0: float,
            mut: float,
            ) -> None:
      
        
        self.Mu = Mu
        self.U = pd.Series(1,index = Mu.index)
        self.mut = mut
        self.Sigma = Sigma
        self.ub = ub
        self.lb = lb
        self.mu0 = mu0
        self.n = len(self.U)
        self.tol = 1e-6
        self.btflag = False
        pass

    def GenTCombs (self, n: int, m: int) -> list:
        # return a list of all combinations
        ## INPUT
        # n: int    = dimension of portfolio vector w
        # m: int    = number of options or base
        ## RETURN
        # ans: list = list of all terinary combinations
        
        ans = [np.base_repr(x,m,n-math.floor(math.log(x,m)+1)) for x in range(1,m**n)]
        ans.insert(0,np.base_repr(0,m,n))
        return ans

    def Solve(self):
        # main algorithm to solve the optimsiation problem
        ## INPUT(NONE)
        ## RETURN
        # flag: bool = is True if a feasible solution is found
        #              solution in self.w & self.w0

        Combs = self.GenTCombs(self.n,3)
        for Comb in tqdm(Combs):
            mask = pd.Series([int(x) for x in Comb],index = self.Mu.index)
            if (self.btflag) and (mask==self.bt).all():
                print('test')
            self.trySolve(mask)
            a1, a2, b = self.CheckPrimal()
            if not (abs(b.sum())+abs(a1)+abs(a2)<= self.tol):
                continue 
            elif self.CheckKKT(mask):
                return True
        return False

    def trySolve(self,mask):
        # Try a solution given a partision, calculates trial w, w0
        ## INPUT
        # mask: pd.Series   = defines the out-sets
        ## OUTPUT (NONE)
        ## CALCULATES 
        # w:    pd.Series   = trial solution
        # w0:   float       = trial solution (for cash)
        # L1:   float       = lambda1, lagrangian multiplier
        # L2:   float       = lambda2, lagrangian multiplier

        h = pd.Series(0,index =self.Mu.index)

        # creating a mask
        maskf = mask.copy()
        maskf[mask==0]=1
        maskf[mask!=0]=0
        
        # evaluate h vector
        h[mask==1]=self.lb
        h[mask==2]=self.ub

        beta = self.Sigma@h
        h[mask==0]=-beta
        
        # modify and invert the Sigma Matrix
        Sigmam = self.ElimVar(self.Sigma,mask)
        Sigmainv = np.linalg.pinv(Sigmam)

        # Solve for Am, Bm & Cm
        A =(self.U*maskf).T@Sigmainv@(self.Mu*maskf)
        B =(self.Mu*maskf).T@Sigmainv@(self.Mu*maskf)
        C =(self.U*maskf).T@Sigmainv@(self.U*maskf)
        D = C*(self.mu0**2)-2*A*self.mu0+B
        
        #Lambda 1 & 2
        if abs(D-0)<= 1e-16:
            # set Lambda 1 & 2 to very large number when denominator is 0
            self.L1 = -1e16
            self.L2 = 1e16
        
        else:
            # else calulate as usual        
            self.L1 = -((self.mut - self.mu0 + self.mu0*((self.U).T@Sigmainv@h) - ((self.Mu).T@Sigmainv@h))/D)
            self.L2 = -self.mu0*self.L1

        # evaluate w & w0
        self.w  = pd.Series(-self.L1*Sigmainv@(self.Mu*maskf) -self.L2*Sigmainv@(self.U*maskf) + Sigmainv@h, index = self.Mu.index)
        self.w0 = 1-self.U.T@self.w

    def CheckPrimal(self):
        # Check if trail solution violate the Primal Constaints
        ## INPUT (NONE)
        ## OUTPUT
        #   a1: float       = investment constraint (U@w+w0==1)
        #   a2: float       = target return constraint (Mu@w+mu0*w0=mut)
        #   b:  pd.Series   = weight bound constrains (-b<w<a)

        b = pd.Series(0,index=self.Mu.index)
        b[~np.isclose(self.w,self.ub,atol=self.tol) & (self.w>self.ub)]=2
        b[~np.isclose(self.w,self.lb,atol=self.tol) & (self.w<self.lb)]=1
        a1 = self.U.T@self.w +self.w0 - 1
        a2 = self.w.T@self.Mu+self.w0*self.mu0 - self.mut
        return a1, a2, b
    
    def CheckKKT(self,mask):
        # Check whether trial solution statisfy KKT Conditions
        ## INPUT
        # mask: pd.Series   = defines the out-sets
        ## OUTPUT (NONE)
        # out:  bool        = KKT is statisfied if True

        Vals = self.Sigma@self.w+self.L1*self.Mu+self.L2*self.U
        out = pd.Series(False, index = self.Mu.index)
        out[mask==0]= np.isclose(Vals,0,atol=self.tol)
        out[mask==2]= ~np.isclose(Vals,0,atol=self.tol)&(Vals < 0)
        out[mask==1]= ~np.isclose(Vals,0,atol=self.tol)&(Vals > 0)
        return out.all()
    
    def ElimVar(self,Sigma:pd.DataFrame,mask:pd.Series):
        # sets columns and rows (of out-sets variables) of a matrix to 0, whilst the diagonals (of out-set variables) to 1
        ## INPUT
        #   Sigma:      pd.DataFrame    = full Covariance Matrix
        #   mask:       pd.Series       = information on the out-set
        ## OUTPUT
        #   Sigmaout:   pd.DataFrame    = modified COvariance Matrix (New Object)
        Sigmaout = Sigma.copy()
        for maskval, pos in zip(mask, range(0,mask.shape[0])):
            if maskval != 0:
                Sigmaout.iloc[pos,:] = 0
                Sigmaout.iloc[:,pos] = 0
                Sigmaout.iloc[pos,pos] = 1
        return Sigmaout 
    
    def setbt (self,bt):
        self.bt = bt
        self.btflag = True 
        

class MOSEKOpt (BasicOpt):
    def Solve(self):
        # Calls MOSEK, a commercial conic optimisation software
        # MOSEK Licence WILL BE NEEDED to run, can be obtain for free for academic purposes on their website
        # at URL: https://www.mosek.com/
        
        # Solves for the minimum of gamma
        # where gamma is a slack variable, lower-bounded by the portfolio variance
        # boiler-plate implementation below

        L = np.linalg.cholesky(self.Sigma.values)
        mu0 = self.mu0
        mut = self.mut
        Mu = self.Mu.values

        m= mos.Model("Basic")
        # varibales
        w = m.variable('w',self.n,mos.Domain.inRange(self.lb.values,self.ub.values))
        w0 = m.variable('w0',1,mos.Domain.unbounded())
        gamma = m.variable('gamma', 1,mos.Domain.unbounded())
        
        #objective
        m.objective('obj', mos.ObjectiveSense.Minimize, mos.Expr.add(gamma,0))

        #constraints
        m.constraint('risk', mos.Expr.vstack(gamma,0.2, mos.Expr.mul(L.T,w)),mos.Domain.inRotatedQCone())
        m.constraint('target_return',mos.Expr.add(mos.Expr.dot(Mu,w),mos.Expr.mul(mu0,w0)),mos.Domain.equalsTo(mut))
        m.constraint('investment',mos.Expr.add(mos.Expr.sum(w),w0),mos.Domain.equalsTo(1))
        m.solve()

        solsta = m.getPrimalSolutionStatus()
        if (solsta != mos.SolutionStatus.Optimal):
            # See https://docs.mosek.com/latest/pythonfusion/accessing-solution.html about handling solution statuses.
            raise Exception(f"Unexpected solution status: {solsta}")

        self.w = pd.Series(w.level(),index = self.Mu.index)
        self.w0 = w0.level()
        self.gamma = gamma.level()
        return True

def readfile(interfaceFileName,dataFileName):
    # Modified read file (improved speed & pandas + csv file implementation)
    ## INPUT 
    # interfaceFileName:    str     = name of the interface excel file (need to be in the same directory) 
    # dataFileName:         str     = name of the Datafile (csv) (need to be in the same directory)
    ## OUTPUT
    # rawreturn:    pd.DataFrame    = asset return data
    # portmean:     float           = extracted target return
    # riskFree:     float           = extracted riskfree rate
    # stratergy:    str             = determins which optimisation to call 
    # lb:           pd.Series       = for Bonded Long/Short wt Cash ONLY, lowerbound weights 
    # ub:           pd.Series       = for Bonded Long/Short wt Cash ONLY, upperbound weights
    
    # extracts out data from interface
    wb=xlwings.Book(interfaceFileName)
    ws1=wb.sheets['portfolio mvo']
    
    nasset=int(ws1["B2"].value)
    horizon=int(ws1["E4"].value)
    portmean=ws1["B4"].value
    riskFree=ws1["B5"].value
    strategy=ws1["D5"].value
    # get list of stock tickers
    aptr =ws1.range((8,2),(8+nasset-1,2)).value
    
    #filter out relavent data
    rawdata = pd.read_csv(dataFileName)
    rawdata.set_index(pd.to_datetime(rawdata["Date"].astype('string'),format='%Y%m%d'),inplace=True)
    rawdata.drop(columns=["Date"],inplace=True)
    rawdata = rawdata.loc[:,rawdata.columns.isin(aptr)]

    #calculate returns 
    rawreturn = ((rawdata-rawdata.shift(horizon))/rawdata.shift(horizon)).dropna()
    rawreturn = rawreturn[::horizon]

    lb = pd.Series(ws1.range((8,3),(8+nasset-1,3)).value, index = rawreturn.columns)
    ub = pd.Series(ws1.range((8,5),(8+nasset-1,5)).value, index = rawreturn.columns)
    lb[lb.isnull()]=0
    lb[lb>0]=0

    ub[ub.isnull()]=0
    ub[ub<0]=0
    ub[(ub==0)&(lb==0)]=0.001
    
    return rawreturn,portmean,riskFree,strategy,lb.astype('float64'),ub.astype('float64')
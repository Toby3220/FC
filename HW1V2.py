import math
import pandas as pd 
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import scipy.optimize as sp_op
from tqdm import tqdm 


def raw_convert(data:pd.DataFrame,period:int):
    # convert raw input data into data that samples once every x ticks
    # data      : pd.DataFrame  - Input Raw Data
    # period    : int           - sampling period 
    # return    : pd.DataFrame  - the sampled DataFrame 

    sval = period
    out = []
    for i in tqdm(range(0,data.shape[0])):
        sval -= data.iloc[i,1]
        while sval <=0: 
            out.append(data.iloc[i,0])
            sval+=period
    out = pd.DataFrame(out,columns=["last"])
    fp = Path('nptdata.csv')  
    fp.parent.mkdir(parents=True, exist_ok=True)  
    out.to_csv(fp)
    return out
        
def get_ret(data:pd.DataFrame,period:int=1):
    # get n period retruns 
    # data      : pd.DataFrame  - Input Raw Data
    # period    : int           - sampling period 
    # return    : pd.DataFrame  - the sampled DataFrame 

    ret =  (data-data.shift(period))/data.shift(period)
    return ret

class data:

    
    
    
    # a data object. allow target data and multiple prediective factor set and predictions
    def __init__(self, data: pd.DataFrame, x: list = None, y: str = "Ret") -> None:
        self.datain = data 
        self.y_vec = y
        self.x_vecs = x

        #initialisation
        self.x = dict()
        self.y = dict()
        self.yhat = dict()
        

        # list of models used
        self.models_used = dict()
        self.models_status = dict()

        self.fignum=1


    # split up data seqequence into multiple segments
    def split(self, periods: dict = {"train": 0.6,"test": 0.4}):
        po = 0
        valo = 0
        for period in periods.keys():
            self.x[period] = dict()
            self.yhat[period] = dict()
            pn = po + periods[period]
            valn = round(pn*self.datain.shape[0])
            if pn == 1:
                self.y[period] = (self.datain[self.y_vec].iloc[valo:])
            else:
                self.y[period] = (self.datain[self.y_vec].iloc[valo:valn])
            po = pn
            valo = valn

    def eval_all(self,period:str = "train"):
        for model in self.models_used.keys():
            if self.models_status[model]:
                self.models_used[model].eval(period)
    

            
       
# empty interface class
class Model:
    def __init__(self) -> None:
        pass
    def create_x():
        pass
    
    def train():
       pass

    def crossval():
        pass

    def eval(self,period:str): 
        pass


class EWA_Model_1f(Model):
    def __init__(self,data:data,name:str = None) -> None:
        self.data = data
        
        if name:
            self.name = name
        else: 
            self.name = "EWA_Model_1f"      

        data.models_used[self.name]=self
        data.models_status[self.name]=False
    
    def eval_loss_func(self, sqers, lbd, v0):
        vo = v0
        sqero = sqers.iloc[0]
        lossfunc = self.eval_single_loss(sqers.iloc[0],v0)
        vol = [v0]
        
        for sqer in tqdm(sqers.iloc[1:]) :
            
            # find current vol estimate
            vn = (1-lbd)*sqero+lbd*vo
            # eval loss for vol
            lossfunc += self.eval_single_loss(sqer,vn)
            #record estimated vol
            vol += [vn]
            # update 
            vo=vn
            sqero = sqer

        self.lossfuncval = lossfunc
        self.volhat = pd.DataFrame(vol,index=sqers.index,columns = ["vol"])
        return lossfunc

    def eval_single_loss(self,sqer,v):
        return math.log(v)+sqer/v
    
    def train(self,period:str="train"):
        mu, sqers, v0 = self.create_x(period)
    
        self.opt_res = sp_op.minimize_scalar(
            lambda l: self.eval_loss_func(lbd = l, sqers = sqers, v0 = v0),
            method = "bounded",
            bounds =(0,1),
            options={
                "maxiter" :  5000,
                "xatol" :  1e-12  
            })
        
        self.predict("train")
        print(self.opt_res)

        if self.opt_res["message"]=="Solution found.":
            self.data.models_status[self.name]=True
    
    def create_x(self, period:str):
        tdata = self.data.y[period]

        mu = tdata.mean()
        sqers = (tdata-mu)**2
        v0 = (tdata.iloc[0]-mu)**2
        x  = (mu,sqers,v0)
        
        self.data.x[period][self.name]=x

        return mu,sqers,v0

    def predict(self,period:str = "test"):
        lbest = self.opt_res["x"]
        mu, sqers, v0 = self.create_x(period)
        self.eval_loss_func(lbd=lbest,sqers=sqers,v0=v0)

        self.data.yhat[period][self.name] = self.volhat
    
    def eval(self,period:str = "test"):
        plt.figure(self.data.fignum)
        self.data.fignum+=1

        mu = self.data.x[period][self.name][0]
        vhat = self.data.yhat[period][self.name]
        y = pd.DataFrame(self.data.y[period])
        stdval = (mu+vhat**0.5).rename(columns = {"vol": "Ret"})
        less_than_1std = (abs(y)<=stdval).sum()*100/y.shape[0]

        one_std_upper = plt.plot(mu+vhat**0.5,'b', label = "1-std CI")
        one_std_lower = plt.plot(mu-vhat**0.5,'b')

        actual = plt.plot(y,'g',alpha = 0.5, label = "actual")

        three_std_upper = plt.plot(mu+3*vhat**0.5,'r', label = "3-std CI")
        three_std_lower = plt.plot(mu-3*vhat**0.5,'r')

        plt.xlabel("sample points \n {}% of samples within 1 std".format(round(less_than_1std.values[0],1)))
        plt.ylabel("returns")
        plt.title("predicted return Intervals vs observed returns: {} period".format(period))

        plt.legend()
        plt.show()
         
class GarchPQ (EWA_Model_1f):
    def __init__(self,data:data,name:str = None, P:int = 1, Q: int = 1) -> None:
        self.data = data
        self.P = P
        self.Q = Q

        if name:
            self.name = name
        else: 
            self.name = "Garch{}{}".format      

        data.models_used[self.name]=self
        data.models_status[self.name]=False
    
    def eval_loss_func(self, theta, sqers, v0):
        # Need to Debug #
        pvec = theta[:self.P]
        qvec = theta[self.P:]
        vol = pd.Series(0,range(0,sqers.shape[0]))
        vol.iloc[0:v0.shape[0]]=v0
        lossfunc=0

        sqero = list()
        volo = list()
        if self.P > self.Q:
            sqero[:] = sqers.iloc[:self.P]
        else:
            sqero[:] = sqers.iloc[self.Q-self.P:self.Q]
        volo[:] = v0

        for i in tqdm(range(max(self.P,self.Q),sqers.shape[0])) :
            # find current vol estimate
            vn = 0
            for p in range(1,self.P+1):
                vn+= pvec[p-1]*sqero[self.P-p]
            for q in range(1,self.Q+1):
                vn+= qvec[q-1]*volo[self.Q-q]

            # eval loss for vol
            lossfunc += self.eval_single_loss(sqers.iat[i],vn)
            #record estimated vol
            vol.iat[i] = vn

            # update
            sqero.pop(0)
            sqero.append(sqers.iat[i])

            volo.pop(0)
            volo.append(vn)
            
        vol.index=sqers.index
        self.lossfuncval = lossfunc
        self.volhat = vol
        return lossfunc

    def train(self,period:str="train"):
        # Need to Rework #
        mu, sqers, v0, theta0 = self.create_x(period)
        theta0 = theta0.squeeze()
        # Constraints
        probA = np.ones([1,len(theta0)])
        problb = np.zeros(theta0.shape)
        probub = np.ones(theta0.shape)

        # call optimiser
        self.opt_res=sp_op.minimize(
            fun = lambda theta: self.eval_loss_func(theta,sqers=sqers,v0=v0),
            x0 = theta0,
            bounds = sp_op.Bounds(lb = problb, ub = probub),
            constraints= sp_op.LinearConstraint(A =probA, lb = 0, ub=1),
            options = {
                "disp":True,
                "maxiter":500
            }
            Jac = 
            hess =   
        )

        self.predict("train")
        print(self.opt_res)

        if self.opt_res["message"]=="Solution found.":
            self.data.models_status[self.name]=True
    
    def create_x(self, period:str, a = 0.99):
        # Generate estimations
        tdata = self.data.y[period]

        mu = tdata.mean()
        vol = tdata.var()
        sqers = (tdata-mu)**2

        if self.P > self.Q:
            v0 = sqers.iloc[self.P-self.Q:self.P]
        else:
            v0 = sqers.iloc[:self.Q]

        # finding intial guess (estimating assuming v_t = E(v), and Sum(Q)+Sum(P)=1)
        Mp = pd.DataFrame(index= sqers.index, columns = range(0,self.P))
        for i in range(1,self.P+1): 
            Mp.iloc[:,i-1] = sqers.shift(i)
        Mpones = pd.DataFrame(1,sqers.index,range(0,self.P))
        
        A = (Mp - vol*Mpones).dropna()
        b = pd.DataFrame(1,A.index,[0])
        N = A.T.dot(A)
        N_inv = pd.DataFrame(np.linalg.pinv(N.values),N.index,N.columns)
        p0 = N_inv.dot(A.T.dot(b))
        
        v_ones_Q1 =pd.DataFrame(1,range(0,self.Q),[0])
        M_ones_QQ = pd.DataFrame(1, range(0,self.Q), range(0,self.Q))
        M_ones_QP = pd.DataFrame(1,range(0,self.Q),range(0,self.P))
        M_ones_QQ_inv = pd.DataFrame(np.linalg.pinv(M_ones_QQ.values),M_ones_QQ.index,M_ones_QQ.columns)
        
        q0 = M_ones_QQ_inv.dot(v_ones_Q1)*a- M_ones_QQ_inv.dot(M_ones_QP).dot(p0)
        theta0 = p0.append(q0).values
        x = (mu,sqers,v0,theta0)
        
        self.data.x[period][self.name]=x
        
        return mu,sqers,v0,theta0

    def predict(self,period:str = "test"):
        thetabest = self.opt_res["x"]
        mu, sqers, v0, theta0 = self.create_x(period)
        self.eval_loss_func(theta=thetabest,sqers=sqers,v0=v0)

        self.data.yhat[period][self.name] = self.volhat
    
    def eval(self,period:str = "test"):
        plt.figure(self.data.fignum)
        self.data.fignum+=1

        mu = self.data.x[period][self.name][0]
        vhat = self.data.yhat[period][self.name]
        y = pd.DataFrame(self.data.y[period])
        stdval = (mu+vhat**0.5).rename(columns = {"vol": "Ret"})
        less_than_1std = (abs(y)<=stdval).sum()*100/y.shape[0]

        one_std_upper = plt.plot(mu+vhat**0.5,'b', label = "1-std CI")
        one_std_lower = plt.plot(mu-vhat**0.5,'b')

        actual = plt.plot(y,'g',alpha = 0.5, label = "actual")

        three_std_upper = plt.plot(mu+3*vhat**0.5,'r', label = "3-std CI")
        three_std_lower = plt.plot(mu-3*vhat**0.5,'r')

        plt.xlabel("sample points \n {}% of samples within 1 std".format(round(less_than_1std.values[0],1)))
        plt.ylabel("returns")
        plt.title("predicted return Intervals vs observed returns: {} period".format(period))

        plt.legend()
        plt.show()
    


# begining of main section
rawdata = pd.read_csv("hsiapr_1tick.csv")
rawdata.columns = ["last","number"]

ptdata = pd.read_csv("nptdata.csv")
ptdata = ptdata["last"]

ret = pd.DataFrame()
ret["Ret"] = get_ret(ptdata)
ret.dropna(inplace=True)

retdata = data(ret)
retdata.split()
# vol_model = EWA_Model_1f(retdata,"EWA")
# vol_model.train()
# vol_model.predict("test")

# # evaluate the performance of the model 
# vol_model.eval("test")
# vol_model.eval("train")

Garch_model = GarchPQ(retdata,"Garch",P=2,Q=4)
# mu, sqers, v0, theta0 = Garch_model.create_x("train")
# Garch_model.eval_loss_func(theta0,sqers,v0)
Garch_model.train()
Garch_model.predict("test")
Garch_model.eval("test")

print("end")

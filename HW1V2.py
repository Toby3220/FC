import math
import pandas as pd 
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import scipy.optimize as sp_op
from tqdm import tqdm 


def raw_convert(data:pd.DataFrame,period:int):
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
    # this is the rolling return
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
        self.models_used = set()


    # split up data seqequence into multiple segments
    def split(self, periods: dict = {"train": 0.6,"cv":0.2, "test": 0.2}):
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
    def eval(): 
        pass


class EWA_Model_1f(Model):

    def __init__(self,data:data,name:str = None) -> None:
        self.data = data

        if name:
            self.name = name
        else: 
            self.name = "EWA_Model_1f"      

        data.models_used.add(self.name)
    
    def eval_loss_func(self, sqers, lbd, v0):
        vo = v0
        sqero = sqers.iloc[0]
        lossfunc = self.eval_single_loss(sqers.iloc[0],v0)
        vol = [v0]
        
        for sqer in sqers.iloc[1:] :
            
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
vol_model = EWA_Model_1f(retdata,"EWA")
vol_model.train()
vol_model.predict("test")
plt.figure(1)
vol_model.eval("test")
plt.figure(2)
vol_model.eval("train")


print("end")

import numpy as np
import pandas as pd
from tqdm import tqdm
from newtonRaphson import newtonRaphson
from matplotlib import pyplot as plt


import math

class American_Put_LSMC:
    def __init__(self,dtime,horizon) -> None:
        self.times = np.arange(0,horizon+round(dtime,3),dtime).round(4)
        self.N_steps = len(self.times)
        self.horizon = horizon
        self.dtime = dtime    
        pass
    
    def payoff(self,S,K):
        return np.maximum(K-S,0)

    def sample(self, mu:float, sigma:float, n:int, s0:float, return_data: bool = False):
        """sample n underlying assest paths acoording to mean, and standard deviation of a process
            ### Inputs: 
            mu:             float   = mean \n
            sigma:          float   = standard deviation \n
            n:              int     = number of paths to sample \n
            s0:             float   = inital starting point for underlying \n
            return_data:    bool    = (optional, default: False) return a np.ndarry if True \n
            ### Outputs:
            data:           np.ndarray  = (optional, default: None) return
        """
        LS_data = np.zeros([self.N_steps,n])
        LS_data[0,:] = np.log(s0)

        for i in range(1,self.N_steps):
            dw = np.random.normal(mu,sigma,n)
            
            LS_data[i,:] = LS_data[i-1,:]+dw

        self.S_data = np.exp(LS_data)
        if return_data: 
            return np.exp(LS_data)
        
    def price(self,K,r, N_polynomials, lambda_param = 0.1, S_data=None):
        self.N_polynomails = N_polynomials
        # if no data is provided, use last generated ones 
        if S_data is None:
            S_data = self.S_data

        C_data = np.zeros_like(S_data)
        exercised = np.zeros_like(S_data)

        C_data[-1] = self.payoff(S_data[-1],K)
        exercised[-1] = np.where(self.payoff(S_data[-1],K)>0, 1,0)
        
        # storage of learned functions parameters
        w = np.zeros([self.N_steps,N_polynomials+1])

        mse = np.zeros(self.N_steps)
        mape = np.zeros(self.N_steps)
        mse[:] = np.nan
        mape[:] = np.nan

        for i in range(self.N_steps-2,-1,-1):

            # going from -2 to 0
            # generate 
            X,Y = self.gen_regression_dataset(S_data[i,:],np.exp(-r*self.dtime)*C_data[i+1],K,N_polynomials)
            
            # X,Y = self.gen_regression_dataset(S_data[i,:],C_data[i,:],K,N_polynomials)
            
            w[i,:] = np.linalg.pinv(X.T@X+lambda_param*np.eye(X.shape[1]))@(X.T@Y)

            mse[i] = ((X@w[i,:]-Y)**2).mean()
            mape[i] = (X@w[i,:]-Y).abs().mean()

            X_acc = self.Laguerre_basis(self.N_polynomails,S_data[i,:])
            
            fair_approx = X_acc@w[i,:]
            intrinsic = self.payoff(S_data[i,:],K)

            # exercise if intrinsic is greater than the approximated fair value, and that option is in the money
            C_data[i,:] = np.where((intrinsic>fair_approx)&(intrinsic>0),intrinsic,np.exp(-r*self.dtime)*C_data[i+1,:])
            
            # if exercised, set as True
            exercised[i,:] = np.where((intrinsic>fair_approx)&(intrinsic>0),1,0)
            exercised[i+1:,:] = np.where((intrinsic>fair_approx)&(intrinsic>0),0,exercised[i+1:,:])
        
        self.params = w
        self.training_mse = mse
        self.training_mape = mape

        self.C_data = C_data
        return C_data, exercised
            
    
    def gen_regression_dataset(self,curr_S_data,curr_C_data,K,N_polynomials):
        curr_S_data = pd.Series(curr_S_data.copy())
        curr_C_data = pd.Series(curr_C_data.copy())
    
        # only in the money options
        curr_C_data[self.payoff(curr_S_data,K)==0] = np.nan
        curr_S_data[self.payoff(curr_S_data,K)==0] = np.nan

        x = curr_S_data.dropna()
        Y = curr_C_data.dropna()

        X = self.Laguerre_basis(N_polynomials,x.copy())
       
        return X,Y

    def Laguerre_basis(self,i,x,lb_data = None):
        x = np.array(x)
        # X = np.ones([x.shape[0],self.N_polynomails+1])
        
        lb_data = pd.DataFrame()
        lb_data.loc[:,0] = np.ones_like(x)
        lb_data.loc[:,1] = 1-x 

        for i in range(2,self.N_polynomails+1):
            lb_data.loc[:,i] = self._Lagurre_basis_i(i,x,lb_data.copy())
        return lb_data.to_numpy()


    def _Lagurre_basis_i(self,i,x,lb_data):
        x = np.array(x)
        try:
            return lb_data.iloc[:,i]
        except:
            return ((2*i-1-x)*self._Lagurre_basis_i(i-1,x,lb_data) - (i-1)*self._Lagurre_basis_i(i-2,x,lb_data))/(i)
       
 
    
    def Kfold_cv_train(self,splits,K,r,S_data= None, LPs=None,lambda_params=None,metric="MSE"):

        if S_data is None: 
            S_data = self.S_data
        s0 = S_data[0,0]

        if LPs is None: 
            LPs = [4,5,6]
        
        if lambda_params is None:
            lambda_params = [0,0.05,0.5,1,5]
        
        N_samples = S_data.shape[1]
        S_data = pd.DataFrame(S_data)

        block=N_samples//splits

        metric_average_mse = pd.DataFrame(index = LPs, columns= lambda_params)
        metric_average_mape = pd.DataFrame(index = LPs, columns= lambda_params)

        value_diff = pd.DataFrame(index = LPs, columns= lambda_params)
        
        for Lambda_param in tqdm(lambda_params):
            for Lagurre_param in tqdm(LPs):
                for Split in range(splits):
                    out_mse = np.zeros(splits)
                    out_mape = np.zeros(splits)

                    test_S_split = S_data.iloc[:,(Split*block):((Split+1)*block)]
                    training_S_split =  S_data.loc[:,~S_data.columns.isin(test_S_split.columns)]

                    option_value, exercised = self.price(K,r,Lagurre_param,Lambda_param,training_S_split.to_numpy())
                    mse, mape, option_value_test, exercised_test = self._test(test_S_split.to_numpy(),K,r,Lagurre_param,self.params)

                    out_mse[Split] = pd.Series(mse).mean()
                    out_mape[Split] = pd.Series(mape).mean()

                metric_average_mse.loc[Lagurre_param,Lambda_param] = pd.Series(out_mse).mean()
                metric_average_mape.loc[Lagurre_param,Lambda_param] = pd.Series(out_mape).mean()
                value_diff.loc[Lagurre_param,Lambda_param] = option_value[0].mean() - option_value_test[0].mean()

        return metric_average_mse,metric_average_mape,value_diff
    
    def _test(self,S_data,K,r,Lagurre_param,w):
        
        mse = np.zeros(self.N_steps)
        mape = np.zeros(self.N_steps)
        mse[:] = np.nan
        mape[:] = np.nan

        C_data = np.zeros_like(S_data)
        exercised = np.zeros_like(S_data)
        C_data[-1] = self.payoff(S_data[-1],K)
        exercised[-1] = np.where(self.payoff(S_data[-1],K)>0, 1,0)

        for i in range(S_data.shape[0]-2,-1,-1):

            # generate data
            X,Y = self.gen_regression_dataset(S_data[i,:],np.exp(-r*self.dtime)*C_data[i+1],K,Lagurre_param)
            
            # accuracy on "blind data"
            mse[i] = ((X@w[i,:]-Y)**2).mean()
            mape[i] = (X@w[i,:]-Y).abs().mean()

            X_acc = self.Laguerre_basis(Lagurre_param,S_data[i,:])
            
            # use trained function on blind data
            fair_approx = X_acc@w[i,:]
            intrinsic = self.payoff(S_data[i,:],K)

            # exercise if intrinsic is greater than the approximated fair value, and that option is in the money
            C_data[i,:] = np.where((intrinsic>fair_approx)&(intrinsic>0),intrinsic,np.exp(-r*self.dtime)*C_data[i+1,:])
            
            # if exercised, set as True
            exercised[i,:] = np.where((intrinsic>fair_approx)&(intrinsic>0),1,0)
            exercised[i+1:,:] = np.where((intrinsic>fair_approx)&(intrinsic>0),0,exercised[i+1:,:])
               
        return mse, mape, C_data, exercised

    def evaluate_boundary(self,K):
        exercise_prices = np.zeros(self.N_steps)
        for i in tqdm(range(0,self.N_steps)):
            w = self.params[i,:]
            root , flag, maxdev = newtonRaphson(
                userfunctions = lambda x: self._nr_critical_price_error(x,w,K),
                x = [K],
                prec=1e-6
            )
            exercise_prices[i] = root[0]
        self.exercise_prices = exercise_prices
        return exercise_prices
            

    def _nr_critical_price_error(self,x,w,K):
        
        X = self.Laguerre_basis(self.N_polynomails,x)
        y = X@w - self.payoff(np.array(x),K)
        return list(y)
        
        
        
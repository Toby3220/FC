from scipy.special import ndtr
from scipy.stats import norm as nd 
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt 
import newtonRaphson
from tqdm import tqdm


class OptionPriceFuncs: 

    def __init__(self, risk_free, volatility) -> None:
        self.rf = risk_free
        self.local_vol = volatility
        pass
    
    def _get_rf(self):
        return self.rf
    
    def _get_sigma(self): 
        return np.sqrt(self.local_vol)

    def eval_d1(self, spot, strike, maturity):
        
        rf = self._get_rf()
        sigma = self._get_sigma()

        d1 = (np.log(spot/strike)+maturity*(rf+0.5*sigma**2))/(sigma*np.sqrt(maturity))
        
        return d1

    def eval_d2(self, spot, strike, maturity):

        d1 = self.eval_d1(spot,strike,maturity)
        sigma = self._get_sigma()
        d2 = d1 - sigma*np.sqrt(maturity)
        return d2

class AmericanPut(OptionPriceFuncs):
    """Object for the pricing of American Put Options

    Uses the close form solution of the european put, to iteratively solve for
    the exercise boundary. uses the newton raphson solver in this process
    ### attribute
    - option_prices     : the price of this option, it's corrisponding european price and the option premium at the defined time steps
    - exercise_prices   : the exercise price boundary
    ### Methods
    - price()   : prices an american option given the required information
    """
    def price_european(self,spot,strike,time_to_maturity):
        # price the similar european

        d1 = self.eval_d1(spot,strike,time_to_maturity)
        d2 = self.eval_d2(spot,strike,time_to_maturity)

        rf = self._get_rf()

        price = strike*np.exp(-rf*time_to_maturity)*ndtr(-d2)-spot*ndtr(-d1)
        return price
    
    def find_current_preimum(self,spot,curr_time):
        # calculate for all known B value exp(-rdt_m)N(-d2(x,B_m,dt_m))
        # where B_m is the boundary at time m, & dt_m is (t_m-t) 
        B = self.exercise_prices.dropna()
        times = B.index.to_series() - curr_time

        rf = self._get_rf()

        d2 = self.eval_d2(spot,B,times)
        S = np.exp(-rf*times)*ndtr(-d2)

        # return the sum of such terms
        return S.sum()
          
    def price(self,spot: float,strike: float,times:np.ndarray):
        """ prices the option in question in an iterative process
        ###Inputs
        - spot:     float   = the spot price of the underlying
        - strike:   float   = the option strike price
        - times:    np.ndarry   = a vector of time point for the exercise boundary to be solved
        ###Outputs
        - curr_price:   float   = the current price of the option 
        """
        # initalise
        self.exercise_prices = pd.Series(index=times)
        
        # stores the length of each time period 
        self.dtimes = self.exercise_prices.copy()
        self.dtimes[:] = np.roll(times,-1) - times
        self.dtimes.round(2)

        # stores the option prices information
        self.option_prices = pd.DataFrame(index =times ,columns =["P","p","premium"])
        
        # set maturity
        maturity = times[-1]

        for i in tqdm(range(len(times)-2,-1,-1)):
            # set up time values, backwards iterate from maturity to t = 0
            curr_time = times[i]
            
            # find current american option price at time t            
            curr_price = self.price_at_time(spot,strike, curr_time, maturity, True)
            
            # calls newtonraphson to find exercise price
            B, flag, maxdev = newtonRaphson.newtonRaphson(
                lambda x : self.tangency_err(x,strike = strike, curr_time = curr_time, maturity = maturity),
                x = [spot], 
                prec = 1e-6
            )

            # update exercise price and premiums
            self.exercise_prices.at[curr_time] = max(B[0],0)
            self.option_prices.at[curr_time,"P"] = curr_price

        return curr_price
    
    def price_at_time(self,spot,strike,curr_time,maturity, record = False):
        # find all relavent time information
        rf = self._get_rf()
        dtime = self.dtimes.loc[curr_time.round(2)]
        time_to_maturity = (maturity - curr_time)
        
        # find current european option price & exercise preimum
        p = self.price_european(spot,strike,time_to_maturity)
        S = self.find_current_preimum(spot,curr_time)
        
        if record: 
            self.option_prices.at[curr_time,'p'] = p
            self.option_prices.at[curr_time,'premium'] = rf*strike*dtime*S

        curr_price = p + rf*strike*dtime*S
        return curr_price
        
    def tangency_err(self,spot_B:list,strike:float,curr_time:float,maturity:float)->list:
        # tangency condtion error function for newton raphson
        spot = spot_B[0]
        P = self.price_at_time(spot,strike,curr_time,maturity)
        return [P - strike + spot]
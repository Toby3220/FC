import matplotlib.cm
import cubicSpline
import pandas as pd
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

class CubicSplineSurface:
    def __init__(self,H,params) -> None:
        ## INPUT 
        # H :       pd.DataFrame     = data containing maturities {T_1,T_2,...T_n} (must be sorted from smallest to largest) & h coefficent {h_0, h_1,...}
        # params:   dict()           = all other option parameters (spot, strike, maturities, etc.)
        ## OUTPUT (None)
        
        # store paramters
        self.H = H
        self.params = params
        
        self.maturities = pd.Series(self.H.index.values.copy()) # set of maturites
        self.pointer_lim = self.maturities.shape[0]-2           # max value for the pointer

        self.vol_surface_params = dict()        # stores all parameters needed for the volatility surface
        self.known_maturities_mappings = dict() # stores all information that maps from time to 

        self.imp_vol_surface_flag = False
        self.local_vol_surface_flag = False
        pass
   
    def _get_V(self,price:float)->dict:
        #Private, method to get price
        if price in self.vol_surface_params:
            V = self.vol_surface_params[price]
        else:
            V = self.gen_1d_spline_func(price)
            self.vol_surface_params[price]=V
        return V
    
    def _get_lp(self,maturity:float)->int:
        #Priveate, method to get "left pointer", 
        # a pointer that points to the relavent segment of cubic spline 
        if maturity in self.known_maturities_mappings:
            lp = self.known_maturities_mappings[maturity]
        else:
            # binary search algo
            rp = self.pointer_lim
            lp = self._bucket_binary_search(maturity,0,rp)
            self.known_maturities_mappings[maturity] = lp
        return lp
    
    def _get_Tnn_lp(self,time:float)->int:
        #Priveate, method to get "position pointer", 
        # a pointer that points to the nearest maturity
        lp = self._get_lp(time)
        if lp == -1:
            return 0
        elif lp == -2:
            return self.pointer_lim+1

        lb = self.maturities[lp]
        ub = self.maturities[lp+1]
        if abs(time-lb)<abs(time-ub): 
            Tnn = lb
            pos = lp
        else:
            Tnn = ub
            pos = lp+1
        return pos

    def _bucket_binary_search(self,target,lp,rp):
        #Private, usese binary search to get "left pointer",
        #the pointer that points to the left hand knot (shoter in maturity) of the relavent cubic spline
        bounds = self.maturities
        
        if  (target < bounds[lp]):
            return -1
        elif (bounds[rp+1] < target):
            return -2
            
        if  (bounds[lp] <= target) and (target <=bounds[lp+1]):
            return lp
        
        elif  (bounds[rp] <= target) and (target <=bounds[rp+1]):
            return rp
            
        else:
            trial = (lp+rp)//2
            if target < bounds[trial]:
                return self._bucket_binary_search(target,lp,trial)
            else: 
                return self._bucket_binary_search(target,trial,rp)
    
    def _gen_linear_extrap_params(self,b:list,c:list,d:list,x:list,y:list,pos):
        # Private, find the gradient and constant for the end points
        # gradient matching from cubic function
        m = (3*d[pos]*x[pos]**2)+(2*c[pos]*x[pos])+b[pos]
        # fit x,y on the line
        k = y[pos]-m*x[pos]
        return m, k
    
    def _get_vec_of_lps(self,times,sorted:bool) -> np.ndarray:
        # Private, get a vector "left pointers" from a SORTED list of times
        # by compairing two sorted lists times MUST BE SORTED from SMALLEST TO LARGETS
        out = np.empty([len(times)])
        out[:] = np.nan

        if not sorted:
            print("List MUST be Sorted, from smallest to largest")
            return out
        
        vmax = len(times)-1
        lmax = self.pointer_lim+1
        lp = 0 
        vp = 0
        while vp <= vmax and lp <= lmax:
            if (self.maturities[lp]) > (times[vp]):
                if lp == 0:
                    vp+=1
                    continue
                else:
                    self.known_maturities_mappings[times[vp]]=lp-1
                    out[vp] = lp-1
                    vp+=1
                    continue
            elif (self.maturities[lp]) <= times[vp]:
                    lp+=1
                    continue
        
        return out

    def gen_imp_vol_surface(self,prices,times, sorted_inputs: bool = False, return_dataframe:bool = False):
        # generate a 2d surface of the implied volatility
        ## INPUT 
        # Prices:           vec     = list of prices for the grid
        # times:            vec     = list of times for the grid
        # sorted_inputs:    bool    = True if both time is sorted from smallest to largest
        # return_dataframe: bool    = if True then returns the surface in a dataframe
        ## OUTPUT
        # Surface:  pd.DataFrame    = return the surface in a dataframe, with the index = prices,
        #                           columns = times, values = implied volaitility
        Surface = pd.DataFrame(index = prices, columns = times)
        if sorted_inputs:
            # if sorted, use a faster method to find left pointers
            self._get_vec_of_lps(times,True)
        
        for price in tqdm(prices):
            for time in times:
                    # finds and store informaiton point by point
                    ans = self.eval_imp_vol(price,time)
                    Surface.at[price,time] = ans
        
        # store in the object and set flag
        self.imp_vol_surface = Surface
        self.imp_vol_surface_flag = True

        if return_dataframe:
            return Surface
    
    def gen_local_vol_surface(self, prices, times, sorted_inputs: bool = False, return_dataframe:bool = False, from_imp_vol_surface: bool = False):
        # generate a 2d surface of the local volatility
        ## INPUT 
        # Prices:           vec      = list of prices for the grid
        # times:            vec      = list of times for the grid
        # sorted_inputs:    bool     = True if both time is sorted from smallest to largest
        # return_dataframe: bool     = if True then returns the surface in a dataframe
        # from_imp_vol_surface: bool = if True, uses a previously calculate volatitlity surface (must be of the same prices & times, caution advised)
        ## OUTPUT
        # Surface:  pd.DataFrame    = return the surface in a dataframe, with the index = prices,
        #                           columns = times, values = local volaitility
        
        if not from_imp_vol_surface:
            # calculate a implied volatility surface
            self.gen_imp_vol_surface(prices, times,sorted_inputs)
        
        prices = np.array(prices)
        local_vol = pd.DataFrame(index = prices, columns=times)
        
        for time in tqdm(times): 
            # for each time column, eval local vols for all prices in a vectorised format
            local_vol.loc[:,time] = self.eval_local_vol(prices,time,True)

        # store and set flag
        self.local_vol_surface = local_vol
        self.local_vol_surface_flag = True
        
        if return_dataframe:
            return local_vol
        
    def gen_1d_spline_func(self,price:float)->dict: 
        # generate the cubic splin function for a given query price
        ## INPUT
        # price :   float  = price to be evaluate 
        ## OUPUT
        # V     :   dict() = 1D cublic spline function parameters 

        # generate polynomail moneyness dataset & estimate volatility from H matrix
        X_data = gen_maturity_vol_func_dataset(price,self.maturities,self.params)
        vol_est = (X_data*self.H).sum(1)

        # calls cublic spline
        x= self.maturities.tolist()
        y= vol_est.tolist()
        
        n = len(self.maturities)
        a,b,c,d = cubicSpline.cubicSpline(n,x,y)
        V = dict()
        for i in range(len(a)):
            V[i] = (a[i],b[i],c[i],d[i])
        
        # linearly extrapolate the end points for completeness
        m_min,k_min = self._gen_linear_extrap_params(b,c,d,x,y,0)
        m_max,k_max = self._gen_linear_extrap_params(b,c,d,x,y,-1)
        V[-1] = (k_min,m_min,0,0)
        V[-2] = (k_max,m_max,0,0)
        return V

    def eval_imp_vol(self,price:float,maturity:float)->float:
        # evaluate single point implied volaility
        ## INPUT
        # price :       float = price for implied vol to be evaluated
        # matturity:    float = maturity for implied vol to be evaluated
        ## OUTPUT
        # vol :         float = evaluated implied val

        V = self._get_V(price)
        lp = self._get_lp(maturity)

        vol = 0
        for i, val in zip([0,1,2,3],V[lp]):
            vol += (maturity**i)*(val)

        return vol
    

    def eval_local_vol(self,price:float,time:float, from_imp_vol_surface:bool = False):
        # evaluate single point local volaility, 
        ## INPUT
        # price :       vec or float = price for local vol to be evaluated, optionally a vector (But must have an existing imp vol surface)
        # matturity:    float = maturity for local vol to be evaluated
        ## OUTPUT
        # vol :         float = evaluated local val

        # get nearest maturity
        pos = self._get_Tnn_lp(time)
               
        # evaluate moneyness & extract risk free rate
        x = eval_moneyness(price,time,self.params)
        rf = self.params["rf"]
        spot = self.params["spot"]

        x1 = np.array([0,1,2*x,3*x**2])
        x2 = np.array([0,-1, 2*(1-x), 3*(2*x-x**2)])
        h_vec = self.H.iloc[pos,:]

        # evaluate the K*dv/dK, K^2*d^2v/dK^2 & dv/dT respectively
        Kdiv    = h_vec@x1
        K2div2  = h_vec@x2
        Tdiv = -rf*Kdiv
        
        
        if from_imp_vol_surface:
            # if vector format AND from a imp_vol_surface
            v = self.imp_vol_surface.loc[:,time]
        else:
            v = self.eval_imp_vol(price,time)
        
        # assemblying the vector
        beta = (np.log(spot/price)+(rf+0.5*v**2)*time)/v

        nominator = (v**2) + (2*v*time*Tdiv)+(2*rf*time*v*Kdiv)
        denomintor = (1+beta*Kdiv)**2 + time*v*(K2div2-beta*Kdiv)**2
        local_vol = nominator/denomintor

        return local_vol          

    def check(self)->pd.DataFrame:
        # function to evaluate the goodnessfit of the cublic spline implied volaitlity surface
        ## INPUT (None)
        ## OUTPUT:
        # out: pd.DataFrame = containing the actual implied volatility from data 
        #                     & the estimated implied volatility from the surface
        
        # intialise
        data = self.params["data"]
        out = pd.DataFrame(data.loc[:,["Strike","T","ImpVol"]])
        out.loc[:,"EstImpVol"] = np.nan
        out.loc[:,"EstLocalVol"]=np.nan

        # evaluate and store implied vol at each point
        for indexval in tqdm(data.index):
            price = data.loc[indexval,"Strike"]
            time = data.loc[indexval,"T"]
        
            imp_vol = self.eval_imp_vol(price,time)
            local_val = self.eval_local_vol(price,time)
            out.loc[indexval,"EstImpVol"] = imp_vol
            out.loc[indexval,"EstLocalVol"] = local_val
            
        return out
    
    
    def export_surface(self,surface: str = "local")->dict:
        # export a surface to a dictionary for quick access
        ## INPUT:
        # surface: str      = select local or implied to be exported (take care that times and prices matches with your model)
        ## OUTPUT:
        # OutSurface: dict  = data dictionary 
        if self.local_vol_surface_flag and surface.lower() == "local":
            df = self.local_vol_surface
        elif self.imp_vol_surface_flag and surface.lower() == "implied":
            df = self.imp_vol_surface
        else: 
            print("Error, no Surface")
            return None

        df = df.stack(dropna=False)
        OutSurface = df.to_dict()
        return OutSurface
                    

def plot_surface(X,Y,Z):
    # plot a volatility surface
    ## INPUT:
    # X:    vec  : a vector of prices 
    # Y:    vec  : a vector of times 
    # Z:    pd.dataframe OR Dictionary: containing the corrisponding volatility information
    ## OUTPUT: (None)

    # boiler plate code
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    X.sort()
    Y.sort()

    # Make data.
    if isinstance(Z,pd.DataFrame):
        Z.sort_index(axis=0,inplace=True)
        Z.sort_index(axis=1,inplace=True)
        z = Z.to_numpy(dtype=float)
    else:
        n_x = len(X)
        n_y = len(Y)
        z = np.zeros([n_x,n_y])
        for i in range(n_x): 
            for j in range(n_y):
                z[i,j] = Z[X[i],Y[j]]
    
    x, y = np.meshgrid(X, Y)

    if z.shape != x.shape:
        z = z.T

    # Plot the surface.
    surf = ax.plot_surface(x, y, z, cmap=matplotlib.cm.get_cmap('coolwarm'),
                        linewidth=0, antialiased=False)

    # # Customize the z axis.
    # ax.set_zlim(-1.01, 1.01)
    # ax.zaxis.set_major_locator(LinearLocator(10))
    # # A StrMethodFormatter is used automatically
    # ax.zaxis.set_major_formatter('{x:.02f}')

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()
    
def gen_maturity_vol_func_dataset(price,maturities:np.ndarray,params:dict,n_feilds:int = 3):
    # generate a polynomial moneyness dataset
    ## INPUT
    # prices:       vec OR float = either a vector of prices (in such a case the number MUST match wiht the number of maturities) or single price
    # maturities:   vec          = the vector of maturites to be evaluated
    # params:       dict()       = dictionary containting all option data 
    # n_feilds:     int          = ordder of polynomails 
    ## OUTPUT
    # X_Data:       pd.DataFrame = datarame with maturities as indexes and columns of X0, X1,... Xn_feilds
    #                            if the prices inputted is a vector, each row represents the polynomail 
    #                            moneyness for a (price, maturity) pair

    X_data = pd.DataFrame(index = maturities)
    x = np.array(eval_moneyness(price,maturities,params))

    for i in range(0,n_feilds+1):
        name = "X{}".format(i)
        X_data.loc[:,name] = x**i

    return X_data

def solve_maturity_vol_parms(X_data,imp_vol):
    # evaluates the H matrix, Least Square Trained parameters for a set of polynomail moneyness volatility functions
    ## INPUT:
    # X_Data:   pd.DataFarame           = polynomail monyness data set calculated, each row representing a (price, maturity) pair
    # imp_vol:  pd.Series OR DataFrame  = the associated implied volatility
    ## OUTPUT:
    # H:        pd.DataFrame            = parameteres for the set of polynomail moneyness volatility functions
     
    maturities = set(X_data.index)
    H = pd.DataFrame(index = maturities, columns = X_data.columns)
    for mat in maturities:
        X = X_data.loc[X_data.index==mat]
        Y = np.array(imp_vol.loc[X_data.index==mat])

        # pseudo inverse provide simple least square solution to X@h = Y 
        h = np.linalg.pinv(X)@Y
        H.loc[mat,:] = h.T
    H.sort_index(inplace=True)

    return H
    
def eval_moneyness(price,maturities,params:dict):
    #evaluate the moneyness    
    out = np.log(price/(params["spot"]*np.exp(params["rf"]*maturities)))
    return out
        
    
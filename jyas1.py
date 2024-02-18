import pandas as pd
import numpy as np
import xlwings
def calculate_price_returns(file_path, M):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path, names=['traded_price', 'num_contracts'])
    # Initialize an empty list to store the price returns
    price_returns = []
    contract_sum = 0
    plast = df['traded_price'][0]
    # Iterate through the DataFrame to calculate price returns
    for i in range(1, len(df)):
        contract_sum += df['num_contracts'][i]
        if contract_sum >= M:
            # Calculate price return when M contracts are traded
            contract_sum -= M
            return_value = (df['traded_price'][i] - plast) / plast
            price_returns.append(return_value)
            plast = df['traded_price'][i+1]
    return price_returns

def readfile(filename):
    wb=xlwings.Book(filename)
    ws=wb.sheets['Sheet1']
    M=int(ws["C2"].value)
    prec=ws["C4"].value
    return M,prec

from math import sqrt
from statistics import mean
from scipy.optimize import minimize_scalar
#
def neg_log_likelihood(lambda_,returns):
    sigma2 = np.zeros(n)
    mu = mean(returns[0:len(returns)//2])
    sigma2[0] = 1
    for i in range(1, n):
        sigma2[i] = (1 - lambda_) * (returns[i-1] - mu) ** 2 + lambda_ * sigma2[i-1]
        ll =+ -0.5 * (np.log(sigma2[i]) + (returns[i] - mu) ** 2 / sigma2[i])
    return -ll
#
def ewma_volatility(returns, lambda_, initial_var=None):
    n = len(returns)
    ewma_var = np.zeros(n)
    if initial_var is None:
        ewma_var[0] = np.var(returns)
    else:
        ewma_var[0] = initial_var
    for i in range(1, n):
        ewma_var[i] = lambda_ * ewma_var[i-1] + (1 - lambda_) * (returns[i-1] - np.mean(returns)) ** 2
    return np.sqrt(ewma_var)
#
def writefile(filename):
    wb=xlwings.Book(filename)
    ws=wb.sheets['Sheet1']
    ws["C3"].value=n
    ws["C5"].value=optlmda
    ws["C6"].value = hit_rate
    return
#----Main program----
file_path = 'hsiapr_1tick.csv'
M,prec=readfile('ewma.xlsx')
returns = calculate_price_returns(file_path, M)
#(b)
n = len(returns)//2
lower=0.0
upper=1.0
optlambda = minimize_scalar(neg_log_likelihood, bounds=(lower,upper), args=(returns), method='bounded', options={'xatol':prec})
optlmda = optlambda.x
#(c)
in_sample_returns = returns[:n]
out_of_sample_returns = returns[n:]
initial_var = np.var(in_sample_returns)  # 可以使用样本内的最后一个波动率作为初始值
ewma_vol_out_of_sample = ewma_volatility(out_of_sample_returns, optlmda, initial_var)

# 计算样本内平均回报作为mu
mu = np.mean(in_sample_returns)

# 回测样本外置信区间的准确性
hits = 0
for i, ret in enumerate(out_of_sample_returns):
    lower_bound = mu - ewma_vol_out_of_sample[i]
    upper_bound = mu + ewma_vol_out_of_sample[i]
    if lower_bound <= ret <= upper_bound:
        hits += 1

hit_rate = hits / len(out_of_sample_returns)
writefile("ewma.xlsx")



#
import mop
#
#---------------------------------------------------------------------------
def kroneckerDelta(i,j):
    if(i==j):
        delta=1
    else:
        delta=0
#
    return delta
#
#---------------------------------------------------------------------------
def maxIteration():
    return 1000
#
#---------------------------------------------------------------------------
# Newton-Raphson procedure to determine the zeros of an array of n functions
# with n variables
#
#   userfunctions(x) - user-defined callable functions
#                      Input variables x(0:n-1)
#                      Output function values g(0:n-1)
#   Input:
#   x(0:n-1) - As input, it is the initial guess of the zeros
#   prec - Required precision in the estimation
#
#   Output:
#   x(0:n-1) - As output, it refers to the estimated zero
#   precMet - Flag equals True if the procedure sucessfully determines
#             the zeros under required precision
#             Flag equals False if it is unsuccessful
#   maxDev - If Flag equals False, maxDev represents the maximum deviation
#            of gvec from zero
#
def newtonRaphson(userfunctions,x,prec):
    from math import fabs
#
    n=len(x)
    omega=[[0 for j in range(n)] for i in range(n)]
#
    for nItr in range(maxIteration()):
#
        xold=[]
        for i in range(n):xold.append(x[i])
        gold=userfunctions(xold)
#
#  define the transformation matrix
#
        for j in range(n):
            xshift=[]
            for i in range(n):xshift.append(xold[i]+prec*kroneckerDelta(i,j))
            gshift=userfunctions(xshift)
            for i in range(n):omega[i][j]=(gshift[i]-gold[i])/prec
#
#  iterate and update the array of variables
        Dx=mop.solveAxb(omega,gold,n,0,0,0)
        for i in range(n):x[i]=xold[i]-Dx[i]
#
#  check whether precision limit has been met
        for i in range(n):
            if(fabs(x[i]-xold[i])<=prec):
                precMet=True
            else:
                precMet=False
                break
        if(precMet): break
#
#  determine the maximum deviation at exit
    g=userfunctions(x)
    maxDev=0
    for i in range(n):
        if(fabs(g[i])>maxDev):
            maxDev=fabs(g[i])
#
    return x,precMet,maxDev
#
#---------------------------------------------------------------------------
#



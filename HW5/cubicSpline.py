#
import mop

#
#-------------------------------------------------------------------
# Given n knots x[0:n-1] and y[0:n-1])
# Return spline coefficients a[0:n-2],b[0:n-2],c[0:n-2] and d[0:n-2]
#   
def cubicSpline(n,x,y):
#
    xw=[0 for i in range(n+1)]
    yw=[0 for i in range(n+1)]
    R=[0 for i in range(4*(n-1)+1)]
    M=[[0 for j in range(4*(n-1)+1)] for i in range(4*(n-1)+1)]
#
#  convert into working labels [1 to n]
    for i in range(1,n+1):
        xw[i]=x[i-1]
        yw[i]=y[i-1]     
#
#  define the column vector R
    for i in range(1,n-1+1):
        R[i]=yw[i]
        R[n-1+i]=yw[i+1]
        R[2*(n-1)+i]=0
        R[3*(n-1)+i]=0
#
#  define the entries of M in the first (n-1) rows
    ptr=0
    for i in range(1,n-1+1):
        M[ptr+i][4*(i-1)+1]=1.0
        M[ptr+i][4*(i-1)+2]=xw[i]        
        M[ptr+i][4*(i-1)+3]=xw[i]**2
        M[ptr+i][4*(i-1)+4]=xw[i]**3     
#
#  define the entries of M in the second (n-1) rows
    ptr=n-1
    for i in range(1,n-1+1):
        M[ptr+i][4*(i-1)+1]=1.0
        M[ptr+i][4*(i-1)+2]=xw[i+1]        
        M[ptr+i][4*(i-1)+3]=xw[i+1]**2
        M[ptr+i][4*(i-1)+4]=xw[i+1]**3       
#
#  define the entries of M in the following (n-2) rows
    ptr=2*(n-1)
    for i in range(1,n-2+1):
        M[ptr+i][4*(i-1)+2]=1.0
        M[ptr+i][4*(i-1)+3]=2*xw[i+1]        
        M[ptr+i][4*(i-1)+4]=3*xw[i+1]**2
        M[ptr+i][4*(i-1)+6]=-1.0 
        M[ptr+i][4*(i-1)+7]=-2*xw[i+1]
        M[ptr+i][4*(i-1)+8]=-3*xw[i+1]**2
#
#  define the entries of M in the next (n-2) rows
    ptr=3*(n-1)-1
    for i in range(1,n-2+1):
        M[ptr+i][4*(i-1)+3]=2.0
        M[ptr+i][4*(i-1)+4]=6*xw[i+1]        
        M[ptr+i][4*(i-1)+7]=-2.0
        M[ptr+i][4*(i-1)+8]=-6*xw[i+1] 
#
#  define the entries of M in the last 2 rows
    ptr=4*(n-1)-2
    M[ptr+1][3]=2.0
    M[ptr+1][4]=6*xw[1]
    M[ptr+2][4*(n-1)-1]=2.0
    M[ptr+2][4*(n-1)]=6*xw[n]
#
#  determine the spline coefficients Q by solving the matrix equation
    Q=mop.solveAxb(M,R,4*(n-1),1,1,1)
#
    a=[]
    b=[]
    c=[]
    d=[]
    for i in range(1,n-1+1):
        a.append(Q[4*(i-1)+1])
        b.append(Q[4*(i-1)+2])
        c.append(Q[4*(i-1)+3])
        d.append(Q[4*(i-1)+4])
#
    return a,b,c,d
#
#-------------------------------------------------------------------



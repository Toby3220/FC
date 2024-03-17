#
#-------------------------------------------------------------------
def mvc(rs):
    from statistics import mean
#
    nasset=len(rs)
    ndata=len(rs[1])
#
    vc=[[0 for j in range(nasset)] for i in range(nasset)]
    mu=[0 for i in range(nasset)]
#
    for i in range(nasset):
        xvec=[]
        for k in range(ndata):xvec.append(rs[i][k])
        mu[i]=mean(xvec)
#
        for j in range(i,nasset):
            yvec=[]
            for k in range(ndata):yvec.append(rs[j][k])
#
            vc[i][j]=covariance(xvec,yvec)
            vc[j][i]=vc[i][j]
#
    return mu,vc
#
#------------------------------------------------------------------
def covariance(x,y):
    from statistics import mean
    n=len(x)
    mx=mean(x)
    my=mean(y)
    sum=0
    for i in range(n):
        sum=sum+(x[i]-mx)*(y[i]-my)
#
    return sum/n
#
#----------------------------------------------------------------------------------
#
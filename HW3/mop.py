#
import numpy
#
#------------------------------------------------------------------------
#  Subroutine to perform matrix operation
#    transpose(bvec(iptr:iptr+n-1))*cvec(jptr:jptr+n-1) = scalar
#
def multbtc(bvec,cvec,n,iptr,jptr):
#
    wbvec=[0 for i in range(0,n)]
    wcvec=[0 for i in range(0,n)]
#
    for i in range(0,n):
        wbvec[i]=bvec[iptr+i]
        wcvec[i]=cvec[jptr+i]
#
    product=0.0
    for i in range(0,n):
        product=product+wbvec[i]*wcvec[i]
#
    return product
#
#------------------------------------------------------------------------
#  Subroutine to perform matrix operation
#    Amatrix(iptr:iptr+n-1,jptr:jptr+n-1)*bvec(kptr:kptr+n-1)
#                                      = cvec(kptr:kptr+n-1)
#
def multAbc(Amatrix,bvec,n,iptr,jptr,kptr):
#
    wAmatrix=[[0 for j in range(0,n)] for i in range(0,n)]
    wbvec=[0 for i in range(0,n)]
    cvec=[0 for i in range(kptr,kptr+n-1+1)]
#
    for i in range(0,n):
        wbvec[i]=bvec[kptr+i]
        for j in range(0,n): 
             wAmatrix[i][j]=Amatrix[iptr+i][jptr+j]
#
    b=numpy.asarray(wbvec,dtype=numpy.float64)
    A=numpy.asarray(wAmatrix,dtype=numpy.float64)
    c=A@b
#
    for i in range(kptr,kptr+n-1+1):
        cvec[i]=c[i-kptr]
#
    return cvec
#
#--------------------------------------------------------------------------
#  Subroutine to solve vector xvec from
#    Amatrix(iptr:iptr+n-1,jptr:jptr+n-1)*xvec(kptr:kptr+n-1)
#                                      = bvec(kptr:kptr+n-1)
#
def solveAxb(Amatrix,bvec,n,iptr,jptr,kptr):
#
    wAmatrix=[[0 for j in range(0,n)] for i in range(0,n)]
    wbvec=[0 for i in range(0,n)]
    xvec=[0 for i in range(kptr,kptr+n-1+1)]
#
    for i in range(0,n):
        wbvec[i]=bvec[kptr+i]
        for j in range(0,n): 
             wAmatrix[i][j]=Amatrix[iptr+i][jptr+j]
#
    b=numpy.asarray(wbvec,dtype=numpy.float64)
    A=numpy.asarray(wAmatrix,dtype=numpy.float64)
    inverseA=numpy.linalg.inv(A)
    x=inverseA@b
#
    for i in range(kptr,kptr+n-1+1):
        xvec[i]=x[i-kptr]
#
    return xvec
#
#------------------------------------------------------------------------
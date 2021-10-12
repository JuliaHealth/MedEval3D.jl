using CUDA, LinearAlgebra,Logging

"""
https://stats.stackexchange.com/questions/503058/relationship-between-cholesky-decomposition-and-matrix-inversion
https://www.lume.ufrgs.br/bitstream/handle/10183/151001/001009773.pdf
experiments with cholesky and inverse

code taken from 
https://www.codesansar.com/numerical-methods/python-program-inverse-matrix-using-gauss-jordan.htm

and

https://discourse.julialang.org/t/cholesky-simple-example/69480/7

most important mathemathically
https://stats.stackexchange.com/questions/147210/efficient-fast-mahalanobis-distance-computation/147222#147222

"""



using LinearAlgebra
x = [4 5 9]
A = [ 6 15 55;15 55 225;55 225 979  ]

x* inv(A)* transpose(x)



L = zeros(3,3)
lower =L
matrix = A
n=3

for i in 1:n
    for j in 1:i
        sum1 = 0;
        # summation for diagonals
        if (j == i)
            for k in 1:j
                sum1 += (lower[j,k])^2
            end#for
            lower[j,j] = sqrt(matrix[j,j] - sum1)
        else
            # Evaluating L(i, j)  using L(j, j)
            for k in 1:j
                sum1 += (lower[i,k] * lower[j,k]);
            end#for
            if(lower[j,j] > 0)
                lower[i,j] = (matrix[i,j] - sum1) /lower[j,j]
            end#if
        end#if else
    end#for
end#for

L
cholesky([ 6 15 55;15 55 225;55 225 979  ]).L
# so now according tohttps://stats.stackexchange.com/questions/147210/efficient-fast-mahalanobis-distance-computation/147222#147222
# we need to calculate Ly=x by forward substitution example(https://www.youtube.com/watch?v=8Z5oy80po-A) where y is unknown
# and mahalanobis will be just y times y transpose so we square each entry and add them up

#basic
y=[0.0,0.0,0.0]
x = [4.0,5.0,9.0]
y[1]= x[1]/L[1,1] 
y[2] = (x[2]-L[2,1]*y[1])/L[2,2]
#y[3]= (x[3] -y[2]*L[3,2] - y[3]*L[3,3])/L[3,1]
#y[1]* L[3,1]+ y[2]*L[3,2] + y[3]*L[3,3]= x[3]
y[3]= (x[3]-y[2]*L[3,2]-y[1]* L[3,1])/L[3,3]

L*y
x

######## correct main equation
using LinearAlgebra
x = [4.0,5.0,9.0]
A = [ 6.0 15.0 55.0;
    15.0 55.0 225.0;
    55.0 225.0 979.0  ]

    transpose(x)* inv(A)*x
y[1]^2+y[2]^2+y[3]^2











L2 = zeros(3,3)
n=3
for i in 1:n
    for j in 1:i
        sum1 = 0;
        # summation for diagonals
        if (j == i)
            for k in 1:j
                sum1 += (L2[j,k])^2
                println( """sumAA$(i)$(j) += (L2[$(j),$(k)])^2 """)
            end#for
            L2[j,j] = sqrt(A[j,j] - sum1)
            println(""" L2[$(j),$(j)] = sqrt(A[$(j),$(j)] - sumAA$(i)$(j))   """)

        else
            # Evaluating L(i, j)  using L(j, j)
            for k in 1:j
                sum1 += (L2[i,k] * L2[j,k]);
                println("""sumAA$(i)$(j) += (L2[$(i),$(k)] * L2[$(j),$(k)]) """)
            end#for
            if(L2[j,j] > 0)
                L2[i,j] = (A[i,j] - sum1) /L2[j,j]
                println("""L2[$(i),$(j)] = (A[$(i),$(j)] -sumAA$(i)$(j))/L2[$(j),$(j)] """)

            end#if
        end#if else
    end#for
end#for
L2




#######working unrolled
L2 = zeros(Float32,3,3)
sumAA11=0.0
sumAA21=0.0
sumAA22=0.0
sumAA31=0.0
sumAA32=0.0
sumAA33=0.0


#unrolled 3 by 3 cholesky decomposition# aa 
sumAA11 += (L2[1,1])^2 
 L2[1,1] = sqrt(A[1,1] - sumAA11)
sumAA21 += (L2[2,1] * L2[1,1])
L2[2,1] = (A[2,1] -sumAA21)/L2[1,1]
sumAA22 += (L2[2,1])^2
sumAA22 += (L2[2,2])^2
 L2[2,2] = sqrt(A[2,2] - sumAA22)
sumAA31 += (L2[3,1] * L2[1,1])
L2[3,1] = (A[3,1] -sumAA31)/L2[1,1]
sumAA32 += (L2[3,1] * L2[2,1])
sumAA32 += (L2[3,2] * L2[2,2])
L2[3,2] = (A[3,2] -sumAA32)/L2[2,2]
sumAA33 += (L2[3,1])^2
sumAA33 += (L2[3,2])^2
sumAA33 += (L2[3,3])^2
 L2[3,3] = sqrt(A[3,3] - sumAA33)

#unrolled forward substitiution
y[1]= x[1]/L2[1,1] 
y[2] = (x[2]-L2[2,1]*y[1])/L2[2,2]
y[3]= (x[3]-y[2]*L2[3,2]-y[1]* L2[3,1])/L2[3,3]
#taking square euclidean distance
y[1]^2+y[2]^2+y[3]^2#should be 4.28




#####


L2 = zeros(Float32,3,3)
sumAA11=0.0
sumAA21=0.0
sumAA22=0.0
sumAA31=0.0
sumAA32=0.0
sumAA33=0.0


#unrolled 3 by 3 cholesky decomposition# aa 
sumAA11 += (L2[1,1])^2 
 L2[1,1] = sqrt(A[1,1] - sumAA11)
sumAA21 += (L2[2,1] * L2[1,1])
L2[2,1] = (A[2,1] -sumAA21)/L2[1,1]
sumAA22 += (L2[2,1])^2
sumAA22 += (L2[2,2])^2
 L2[2,2] = sqrt(A[2,2] - sumAA22)
sumAA31 += (L2[3,1] * L2[1,1])
L2[3,1] = (A[3,1] -sumAA31)/L2[1,1]
sumAA32 += (L2[3,1] * L2[2,1])
sumAA32 += (L2[3,2] * L2[2,2])
L2[3,2] = (A[3,2] -sumAA32)/L2[2,2]
sumAA33 += (L2[3,1])^2
sumAA33 += (L2[3,2])^2
sumAA33 += (L2[3,3])^2
 L2[3,3] = sqrt(A[3,3] - sumAA33)

#unrolled forward substitiution
y[1]= x[1]/L2[1,1] 
y[2] = (x[2]-L2[2,1]*y[1])/L2[2,2]
y[3]= (x[3]-y[2]*L2[3,2]-y[1]* L2[3,1])/L2[3,3]
#taking square euclidean distance
y[1]^2+y[2]^2+y[3]^2#should be 4.28


##############



L2 = zeros(Float32,3,3)
sumAA11=0.0
sumAA21=0.0
sumAA22=0.0
sumAA31=0.0
sumAA32=0.0
sumAA33=0.0

varianceX= A[1,1]
covarianceXY= A[2,1]
covarianceXZ = A[3,1]
varianceY = A[2,2]
covarianceYZ = A[3,2]
varianceZ = A[3,3]
#unrolled 3 by 3 cholesky decomposition# aa 
sumAA11 += (L2[1,1])^2 
 L2[1,1] = sqrt(varianceX - sumAA11)
sumAA21 += (L2[2,1] * L2[1,1])
L2[2,1] = (covarianceXY -sumAA21)/L2[1,1]
sumAA22 += (L2[2,1])^2
sumAA22 += (L2[2,2])^2
 L2[2,2] = sqrt(varianceY - sumAA22)
sumAA31 += (L2[3,1] * L2[1,1])
L2[3,1] = (covarianceXZ -sumAA31)/L2[1,1]
sumAA32 += (L2[3,1] * L2[2,1])
sumAA32 += (L2[3,2] * L2[2,2])
L2[3,2] = (covarianceYZ -sumAA32)/L2[2,2]
sumAA33 += (L2[3,1])^2
sumAA33 += (L2[3,2])^2
sumAA33 += (L2[3,3])^2
 L2[3,3] = sqrt(varianceZ - sumAA33)

#unrolled forward substitiution
y[1]= x[1]/L2[1,1] 
y[2] = (x[2]-L2[2,1]*y[1])/L2[2,2]
y[3]= (x[3]-y[2]*L2[3,2]-y[1]* L2[3,1])/L2[3,3]
#taking square euclidean distance
y[1]^2+y[2]^2+y[3]^2#should be 4.28
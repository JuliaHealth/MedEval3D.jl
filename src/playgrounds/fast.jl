using CUDA, LinearAlgebra

"""
https://stats.stackexchange.com/questions/503058/relationship-between-cholesky-decomposition-and-matrix-inversion
https://www.lume.ufrgs.br/bitstream/handle/10183/151001/001009773.pdf
experiments with cholesky and inverse

code taken from 
https://www.codesansar.com/numerical-methods/python-program-inverse-matrix-using-gauss-jordan.htm
"""

cholesky([ 6 15 55;15 55 225;55 225 979  ] )
Asource = [ 6.0 15.0 55.0;15.0 55.0 225.0;55.0 225.0 979.0  ]

invCorrect = inv(Asource)

augmented = hcat(Asource, collect(I(3)))
a= copy(augmented)
size(a)

for i in 1:3
    for j in 1:3
        if i != j
            ratio = a[j,i]/a[i,i]
            for k in 1:6
                a[j,k] = a[j,k] - ratio * a[i,k]
            end#for
        end#if
    end#for
end#for

# Row operation to make principal diagonal element to 1
for i in 1:3
    divisor = a[i,i]
    for j in 1:6
        a[i,j] = a[i,j]/divisor
    end#for
end#for        
a

# @inbounds begin
#     for k = 1:n
#         Akk = realdiag ? real(A[k,k]) : A[k,k]
#         for i = 1:k - 1
#             Akk -= realdiag ? abs2(A[i,k]) : A[i,k]'A[i,k]
#         end
#         A[k,k] = Akk
#         Akk, info = _chol!(Akk, UpperTriangular)
#         if info != 0
#             return UpperTriangular(A), info
#         end
#         A[k,k] = Akk
#         AkkInv = inv(copy(Akk'))
#         for j = k + 1:n
#             for i = 1:k - 1
#                 A[k,j] -= A[i,k]'A[i,j]
#             end
#             A[k,j] = AkkInv*A[k,j]
#         end
#     end
# end
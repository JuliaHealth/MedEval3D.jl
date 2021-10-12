# Python3 program to decompose
# a matrix using Cholesky
# Decomposition
import math
MAX = 100;
 

def Cholesky_Decomposition(matrix, n):

 
    lower = [[0 for x in range(n + 1)]
                for y in range(n + 1)];
 
    # Decomposing a matrix
    # into Lower Triangular
    for i in range(n):
        for j in range(i + 1):
            sum1 = 0;
 
            # summation for diagonals
            if (j == i):
                for k in range(j):
                    sum1 += pow(lower[j][k], 2);
                lower[j][j] = int(math.sqrt(matrix[j][j] - sum1));
            else:
                 
                # Evaluating L(i, j)
                # using L(j, j)
                for k in range(j):
                    sum1 += (lower[i][k] *lower[j][k]);
                if(lower[j][j] > 0):
                    lower[i][j] = int((matrix[i][j] - sum1) /
                                               lower[j][j]);
 
    # Displaying Lower Triangular
    # and its Transpose
    print("Lower Triangular\t\tTranspose");
    for i in range(n):
         
        # Lower Triangular
        for j in range(n):
            print(lower[i][j], end = "\t");
        print("", end = "\t");
         
        # Transpose of
        # Lower Triangular
        for j in range(n):
            print(lower[j][i], end = "\t");
        print("");
 
# Driver Code
n = 3;
matrix = [[4, 12, -16],
          [12, 37, -43],
          [-16, -43, 98]];
Cholesky_Decomposition(matrix, n);
import math
import numpy as np
import copy
from numpy.linalg import inv


# descompunere matrice
def descompunereMatrice(A, size):
    for i in range(0, size):
        computeRow(i, A, size)

# computes elements from a specific row for the upper matrix
def computeRow(row, A, size):
    for j in range(0, size):
        if j < row:
            if A[j][j] == 0 :
                raise ValueError("NU se poate divizia la 0")
            else:

                A[row][j] = (A[row][j] - sum1(j, row)) / A[j][j]
        else:
            A[row][j] -= sum2(row, j)


def sum1(column, row):
    result = 0.0
    for k in range(0, column):
        if row == column:
            result += A[k][column]
        else:
            result += A[row][k] * A[k][column]
    return result

def sum2(row, column):
    result = 0.0
    for k in range(0, row):
        result += A[row][k] * A[k][column]
    return result

def read_matrix():
    with open("input_matrix.txt", "r") as f:
        first_line = f.readline().split(" ")
        size = int(first_line[0])
        epsilon = int(first_line[1])
        the_input = [list(map(float, line.split(" "))) for line in f]
        A = []
        b = []
        for i in range(0, size):
            A.append(the_input[i])
            b.append(the_input[len(the_input)-i-1])
        b = [val for sublist in b for val in sublist] 
        A_init = copy.deepcopy(A)
        return (size, A, A_init, b, epsilon)


def splitLU(A, size):
    L = [[0.0 for i in range(size)] for j in range(size)]
    U = [[0.0 for i in range(size)] for j in range(size)]
    for i in range(0, size):
        for j in range(0, size):
            if i >= j:
                L[i][j] = A[i][j]
            else:
                L[i][j] = 0
    for i in range(0, size):
        for j in range(0, size):
            if i == j:
                U[i][j] = 1.0
            if i < j:
                U[i][j] = A[i][j]
    return (L, U)


def computeDeterminant(matrix, size):
    result = 1.0
    for i in range(0, size):
        result *= matrix[i][i]
    if result == 0 :
        raise ValueError("eroare: det e 0, nu se poate")
    return result

def computeDirectSubstitution(A, X, b, size):
     for i in range(0, size):
        result = 0.0
        for j in range(0, i):
            result += A[i][j] * X[j]
    
        X[i] = b[i] - result

def computeReverseSubstitution(A, X, b, size):
     for i in range(size-1, -1, -1):
        result = 0.0
        for j in range(i+1, size):
            result += A[i][j] * X[j]
        X[i] = (b[i] - result)/A[i][i]

# verify the solution by computing euclidian norm of the error of the vector A_init * X - b
def verify(A_init, X, b, size):
    error = 0.0
    for i in range(0, size):
        aux = 0.0
        for j in range(0, size):
            aux += A_init[i][j] * X[j]
        error += math.pow(aux - b[i], 2.0)
    error = math.sqrt(error)

    return error 

def norm(x,y):
     value = 0.0
     for i in range(0,len(x)):
          value = value + pow(x[i]-y[i],2)
     return math.sqrt(value)

def displayMatrix(m, size):
    for i in range(size):
        for j in range(size):
            print(str(m[i][j]), end=" ")
        print()
    print()

if __name__ == '__main__':
    (size, A, A_init, b, epsilon) = read_matrix()
    print("Matricea A:")
    displayMatrix(A, size)

    # descompunere
    descompunereMatrice(A, size)
    print("Descompunerea matricei A:")
    displayMatrix(A, size)
    (L, U) = splitLU(A, size)
    

    # determinant
    det_A = computeDeterminant(A, size)
    print("det(A) = ", det_A)
    det_L = computeDeterminant(L, size)
    
    det_U = computeDeterminant(U, size)
   

    # metoda substitutiei directe
    X_lu = [None] * size
    print("\nVectorul termenilor liberi: ", b)
    Y = [None] * size
    computeDirectSubstitution(L, Y, b, size)
    # metoda substitutiei inverse
    computeReverseSubstitution(U, X_lu, Y, size)
    print("X = ", X_lu)

   # print("Norm: ", verify(A_init, X_lu, b, size))
    X_lib = np.linalg.solve(A_init, b)
    print("Library solution: \n", X_lib)

    Ainv_Lu = inv(A_init)
    print("Inversa A din librarie:", Ainv_Lu)

    print("||x_lu - x_lib|| : ", norm(X_lu, X_lib))

    prod =  Ainv_Lu.dot(b)
    print("||x_lu - Ainv*b||: ", norm(X_lu, prod))

    #A^-1 = U^-1 * L^-1
    Uinv = inv(U)
    Linv = inv(L)

    Ainv_lib = np.dot(Uinv, Linv)
    print("Ainv folosind LU: ", Ainv_lib)




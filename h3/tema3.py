import numpy as np

def euclidean_norm(vector):
    return np.sqrt(np.sum(vector ** 2))

def sigma_func(A, col, n):
    A = np.asarray(A)
    if col >= n or col < 0:
        raise ValueError("The column index is out of bounds for the specified matrix size.")
    if A.shape[0] < n or A.shape[1] <= col:
        raise ValueError("The matrix 'A' does not match the specified dimensions.")
    
    sigma = np.sum(A[col:n, col]**2)
    return sigma

def compute_k(A, col, n):
    sigma = sigma_func(A, col, n)
    if A[col][col] >= 0:
        k =  np.sqrt(sigma)
    else:
        k = - np.sqrt(sigma)
    return k

def compute_beta(A, col, n):
    sigma = sigma_func(A, col, n)
    k_value = compute_k(A, col, n)
    beta = sigma - k_value * A[col, col]
    return beta

def householder_transform(A):
    n = A.shape[0]
    R = A.copy()  
    Q = np.eye(n)
   
    for col in range(n - 1):
        
        k_value = compute_k(R, col, n)
        beta = compute_beta(R, col, n)

        v = np.zeros(n)
        v[col] = R[col, col] - k_value

        v[col+1:] = R[col+1:, col] 
        v = v.reshape(1,-1)
        P = np.eye(n) -  np.dot(v.T,v) / beta
        
        R = np.dot(P, R)
        Q = np.dot(Q, P)
        
    return Q,R


def back_substitution(R, Q_transpose_b):
    n = R.shape[0]
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (Q_transpose_b[i] - np.dot(R[i,i+1:], x[i+1:])) / R[i,i]
    return x

def inverse_using_householder(A):
    Q, R = householder_transform(A)
    inverse_R = np.linalg.inv(R)
    inverse_Q = Q.T
    return np.dot(inverse_R, inverse_Q)


# Example usage
if __name__ == "__main__":
    

    #Random
    n = 5
    A = np.random.rand(n, n)  
    s = np.random.rand(n, 1)  
    
    b = np.dot(A,s)

    print("Value of b:", b)
    # QR decomposition using library
    Q_lib, R_lib = np.linalg.qr(A)

    # Compute Q_transpose_b
    Q_transpose_b_lib = np.dot(Q_lib.T, b)

    # Solve for x using back substitution
    x_lib = back_substitution(R_lib, Q_transpose_b_lib)

    print("Solution x using library:", x_lib)

    Q, R = householder_transform(A)

    print("Q : ", Q)
    print("R : ", R)

   # Compute Q_transpose_b
    Q_transpose_b = np.dot(Q.T, b).flatten()
    print(Q_transpose_b)
    # Solve for x using back substitution
    x = back_substitution(R, Q_transpose_b)
    print("Solution without using library:", x)

    # Compute the difference vector
    difference_vector = x_lib - x
    # Compute the Euclidean norm of the difference vector
    norm_difference = euclidean_norm(difference_vector)

    print("Euclidean norm of x_lib - x:", norm_difference)

    difference_vector = np.dot(A,x) - b

    norm_difference = euclidean_norm(difference_vector)

    print("Euclidean norm of A*x - b:", norm_difference)

    difference_vector = np.dot(A,x_lib) - b

    norm_difference = euclidean_norm(difference_vector)

    print("Euclidean norm of A*x_lib - b:", norm_difference)

    difference_vector = x - s

    norm = euclidean_norm(difference_vector) / euclidean_norm(s)

    print("Euclidean norm of x-s / b: ", norm)
    
    A_inverse_householder = inverse_using_householder(A)

    print("Inverse of A: ", A_inverse_householder)

    A_inverse_library = np.linalg.inv(A)

    print("Inverse of A using library: ", A_inverse_library)

    norm_difference_inverse = np.linalg.norm(A_inverse_householder - A_inverse_library, ord=2)
    print("Euclidean norm of difference between inverses:", norm_difference_inverse)
    
  
    

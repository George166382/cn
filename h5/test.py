import numpy as np
from numpy.linalg import cholesky, norm, svd, cond, matrix_rank

def input_vectorized_symmetric_matrix(n):
    length = n * (n + 1) // 2
    v = np.zeros(length)
    print(f"Introduceți elementele partea inferioară triunghiulară a matricei simetrice de dimensiune {n}x{n}.")
    for i in range(n):
        for j in range(i + 1):
            v[i * (i + 1) // 2 + j] = float(input(f"Elementul [{i+1}, {j+1}]: "))
    return v

def input_matrix(rows, cols):
    A = np.zeros((rows, cols))
    print(f"Introduceți elementele matricei de dimensiune {rows}x{cols}.")
    for i in range(rows):
        for j in range(cols):
            A[i, j] = float(input(f"Elementul [{i+1}, {j+1}]: "))
    return A

def vector_to_full_matrix(v, n):
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1):
            A[i, j] = A[j, i] = v[i * (i + 1) // 2 + j]
    return A

def jacobi_method_vectorized(v, n, tol=1e-10, max_iterations=100):
    U = np.eye(n)
    for _ in range(max_iterations):
        A = vector_to_full_matrix(v, n)  # Convert to full matrix for the rotation computation
        p, q = max(((i, j) for i in range(n) for j in range(i)), key=lambda index: abs(A[index[0], index[1]]))
        if abs(A[p, q]) < tol:
            break

        app = A[p, p]
        aqq = A[q, q]
        apq = A[p, q]
        theta = 0.5 * np.arctan2(2 * apq, app - aqq)
        c = np.cos(theta)
        s = np.sin(theta)

        # Update only the necessary elements in vector 'v'
        for i in range(n):
            if i != p and i != q:
                ip = min(i, p)
                iq = min(i, q)
                aip = v[max(i, p) * (max(i, p) + 1) // 2 + ip]
                aiq = v[max(i, q) * (max(i, q) + 1) // 2 + iq]
                v[max(i, p) * (max(i, p) + 1) // 2 + ip] = c * aip + s * aiq
                v[max(i, q) * (max(i, q) + 1) // 2 + iq] = c * aiq - s * aip

        v[p * (p + 1) // 2 + p] = c**2 * app + s**2 * aqq + 2 * c * s * apq
        v[q * (q + 1) // 2 + q] = s**2 * app + c**2 * aqq - 2 * c * s * apq
        v[p * (p + 1) // 2 + q] = 0  # clear off-diagonal

        A_prev = A
        A = vector_to_full_matrix(v, n)
        norm_difference = norm(A - A_prev, 1)
        print("Norma diferenței (1-norm) între iterații:", norm_difference)

        U[:, [p, q]] = U[:, [p, q]].dot(np.array([[c, -s], [s, c]]))

    eigenvalues = np.array([v[i * (i + 1) // 2 + i] for i in range(n)])
    return np.round(eigenvalues), U


def jacobi_method_general(A, tol=1e-10, max_iterations=100):
    n = len(A)
    U = np.eye(n)
    A_prev = np.copy(A)
    for _ in range(max_iterations):
        p, q = max(((i, j) for i in range(n) for j in range(i)), key=lambda index: abs(A[index[0], index[1]]))
        if abs(A[p, q]) < tol:
            break

        app = A[p, p]
        aqq = A[q, q]
        apq = A[p, q]
        theta = 0.5 * np.arctan2(2 * apq, app - aqq)
        c = np.cos(theta)
        s = np.sin(theta)

        new_ap = c * A[:, p] + s * A[:, q]
        new_aq = c * A[:, q] - s * A[:, p]
        A[:, p], A[:, q] = new_ap, new_aq
        A[p, :], A[q, :] = A[:, p], A[:, q]
        A[p, p] = c**2 * app + s**2 * aqq + 2 * c * s * apq
        A[q, q] = s**2 * app + c**2 * aqq - 2 * c * s * apq
        A[p, q] = A[q, p] = 0

        norm_difference = norm(A - A_prev, 1)
        A_prev = np.copy(A)
        print("Norma diferenței (1-norm) între iterații:", norm_difference)

        U[:, [p, q]] = U[:, [p, q]].dot(np.array([[c, -s], [s, c]]))

    eigenvalues = np.diag(A)
    return np.round(eigenvalues), U

def calculate_matrix_series(A, tol=1e-6, kmax=100):
    k = 0
    L = None
    while k < kmax:
        try:
            L = cholesky(A)
        except np.linalg.LinAlgError:
            print("Factorizarea Cholesky a eșuat. Matricea nu este pozitiv definită.")
            break

        A_next = L.T @ L
        if norm(A - A_next, 'fro') < tol:
            break

        A = A_next
        k += 1

    return A, k, L

def main_menu():
    print("Selectați tipul de matrice cu care doriți să lucrați:")
    print("1. Matrice simetrică")
    print("2. Matrice pătratică generală")
    print("3. Matrice rectangulară pentru SVD și pseudoinversă")
    choice = int(input("Introduceți opțiunea (1, 2 sau 3): "))

    if choice == 1:
        n = int(input("Introduceți dimensiunea matricei n: "))
        v = input_vectorized_symmetric_matrix(n)
        A = vector_to_full_matrix(v, n)
        eigenvalues, eigenvectors = jacobi_method_vectorized(v, n)
        print("\nMatricea simetrică introdusă este:")
        print(A)
        print("Valorile proprii (rotunjite):", eigenvalues)
        print("Vectorii proprii:\n", eigenvectors)
        A_final, iterations, L = calculate_matrix_series(A)
        print("\nUltima matrice calculată A_final este:")
        print(A_final)
        print(f"Numărul de iterații efectuate: {iterations}")

        if L is not None:
            print("\nForma finală a matricei și valorile proprii ale acesteia sunt:")
            print(np.diag(A_final))
    elif choice == 2:
        n = int(input("Introduceți dimensiunea matricei n: "))
        A = input_matrix(n, n)
        eigenvalues, eigenvectors = jacobi_method_general(A)
        print("\nMatricea pătratică generală introdusă este:")
        print(A)
        print("Valorile proprii (rotunjite):", eigenvalues)
        print("Vectorii proprii:\n", eigenvectors)
        A_final, iterations, L = calculate_matrix_series(A)
        print("\nUltima matrice calculată A_final este:")
        print(A_final)
        print(f"Numărul de iterații efectuate: {iterations}")

        if L is not None:
            print("\nForma finală a matricei și valorile proprii ale acesteia sunt:")
            print(np.diag(A_final))
    elif choice == 3:
        p = int(input("Introduceți dimensiunea matricei p (numărul de rânduri): "))
        n = int(input("Introduceți dimensiunea matricei n (numărul de coloane): "))
        A = input_matrix(p, n)
        svd_and_pseudoinverse(A, p, n)

def svd_and_pseudoinverse(A, p, n):
    U, s, VT = svd(A)
    print("Valorile singulare ale matricei A:", s)
    rank = matrix_rank(A)
    print("Rangul matricei A:", rank)
    condition_number = cond(A)
    print("Numărul de condiționare al matricei A:", condition_number)
    S_inv = np.zeros((n, p))
    for i in range(rank):
        S_inv[i, i] = 1 / s[i]
    A_pseudo = VT.T @ S_inv @ U.T
    print("Pseudoinversa Moore-Penrose a matricei A (A'):\n", A_pseudo)
    A_pseudo_least_squares = np.linalg.pinv(A)
    print("Matricea pseudo-inversă în sensul celor mai mici pătrate (A''):\n", A_pseudo_least_squares)
    norm_difference = norm(A_pseudo - A_pseudo_least_squares, 1)
    print("Norma diferenței între cele două pseudoinverse ||A' - A''||1:", norm_difference)

main_menu()
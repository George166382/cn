import numpy as np
import math

epsilon = 0.0000000001

def norma(x):
    return np.linalg.norm(x, ord=math.inf)

def is_zero(x):
    return math.fabs(x) < epsilon

def read_a(file_name):
    matrix = dict()
    with open(file_name, "rt") as f:
        ls = f.readlines()
        n = int(ls[0])
        for l in ls[1:]:
            l = l.split(',')
            if len(l) == 3:
                val, lin, col = float(l[0]), int(l[1]), int(l[2])
                if val != 0:
                    if lin not in matrix:
                        matrix[lin] = []
                    line = matrix[lin]
                    idd = len(line)
                    for idx, i in enumerate(line):
                        if i[1] == col:
                            i[0] += val
                            if i[0] == 0:
                                line.pop(idx)
                            return
                        elif i[1] > col and (idd == len(line) or i[1] < line[idd][1]):
                            idd = idx
                    line.insert(idd, [val, col])
    return matrix

def read_b(file_name):
    b = None
    with open(file_name, "rt") as f:
        ls = f.readlines()
        n = int(ls[0])
        b = np.zeros((n,))
        for idx in range(n):
            b[idx] = float(ls[idx+1])
    return b



def get_diag(a, size):
    diag = np.zeros((size,))
    for l in range(size):
        if l in a:
            for c in a[l]:
                if c[1] == l:
                    diag[l] = c[0]
                    break
        else:
            diag[l] = 0
    return diag

def diag_nul(a):
    n = len(a)
    return not np.any(get_diag(a, n))


def gauss_streidel(a, b):
    n = len(a)
    xc = np.arange(n, dtype=float) + 1
    kmax = 10_000
    dx = 100
    k = 0
    diag = get_diag(a, n)  # Pass size of the matrix to get_diag
    if diag_nul(a):
        return "No solution, diag is null"
    while not is_zero(dx) and k <= kmax and dx <= 1e8:
        dx = 0
        for i in range(n):
            column = 0
            if i in a:
                column = [(l[0] * xc[l[1]]) for l in a[i] if l[1] != i]
            temp = (b[i] - sum(column)) / diag[i]
            dx = max(dx, xc[i] - temp)
            xc[i] = temp
        k += 1
    if is_zero(dx):
        return xc, k
    else:
        return 'divergenta'



import os

# Construct absolute path to the files directory
files_directory = os.path.join(os.getcwd(), "h4/files")


for i in range(1, 6):  # Start from 1 since your files are named from a_1.txt to a_5.txt
    a = read_a(os.path.join(files_directory, f"a_{i}.txt"))
    b = read_b(os.path.join(files_directory, f"b_{i}.txt"))
    rez = gauss_streidel(a, b)
    if isinstance(rez, str):
        print(i, rez)
    else:
        x, steps = rez
        print(i, x, steps, "iteratii")
        verif = np.zeros_like(b)
        for l in a:
            verif[l] = sum([(c[0] * x[c[1]]) for c in a[l]])
        print("norma:", norma(verif - b))



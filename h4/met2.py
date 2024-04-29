import numpy as np, math

EPS = 1e-10
# matrar [0] = val, [1] = col

norma = lambda x: np.linalg.norm(x, ord=math.inf)

#vector(vector_rar)
class MatRara:
	def __init__(self, n):
		self.n = n
		self.m = [[] for _ in range(n)]

	def add_elem(self, val, lin, col):
		if val == 0:
			return
		line = self.m[lin]
		idd = len(line)
		for idx, i in enumerate(line):
			if i[1] == col:
				i[0] += val
				if i[0] == 0:
					line.remove(i)
					print("Removed 0 elem after sum")
				return
			elif i[1] > col and (idd == len(line) or i[1] < line[idd][1]):
				idd = idx
		line.insert(idd, [val, col])

	def __str__(self):
		s = "-*-\n"
		for l in self.m[:5] + ["..."] +  self.m[-5:]:
			s += l.__str__() + '\n'
		s += "-*-"
		return s

	def __eq__(self, a):
		print("Using custom eq")
		if a.n != self.n:
			return False
		for i in range(self.n):
			if len(a.m[i]) != len(self.m[i]):
				print("Equality: line lenghts not equal")
				print(len(a.m[i]), len(self.m[i]))
				print(a.m[i])
				print(self.m[i])
				return False
			for aa, bb in zip(a.m[i], self.m[i]):
				if not (aa[0] == bb[0] and aa[1] == bb[1]):
					return False
		return True

def is_zero(x: float) -> bool:
	return math.fabs(x) < EPS

def read(file_name):
	mat = None
	with open(file_name, "rt") as f:
		ls = f.readlines()
		n = int(ls[0])
		mat = MatRara(n)
		for l in ls[1:]:
			# print(l)
			l = l.split(',')
			if len(l) == 3:
				val, lin, col = float(l[0]), int(l[1]), int(l[2])
				mat.add_elem(val, lin, col)
	return mat
def read_b(file_name):
	b = None
	with open(file_name, "rt") as f:
		ls = f.readlines()
		n = int(ls[0])
		b = np.zeros((n,))
		for idx in range(n):
			b[idx] = float(ls[idx+1])
	return b

def bonus_sum(a, b):
	if a.n != b.n:
		return

	c = MatRara(a.n)
	for id, l in enumerate(a.m):
		for co in l:
			c.add_elem(co[0], id, co[1])
	for id, l in enumerate(b.m):
		for co in l:
			c.add_elem(co[0], id, co[1])
	return c

def get_diag(a):
	diag = np.zeros((a.n,))
	for idx, l in enumerate(a.m):
		for c in l:
			if c[1] == idx:
				diag[idx] = c[0]
				break
		else:
			diag[idx] = 0
	return diag

def diag_nul(a):
	return not np.any(get_diag(a))

norma = lambda x: np.linalg.norm(x, ord=2)

def gauss_streidel(a, b):
	xc = np.arange(a.n, dtype=float) + 1
	kmax = 10_000
	dx = 100
	k = 0
	diag = get_diag(a)
	if diag_nul(a):
		return "No solution, diag is null"
	while not is_zero(dx) and k <= kmax and dx <= 1e8:
		dx = 0
		for i in range(a.n):
			column = [(l[0] * xc[l[1]]) for l in a.m[i] if l[1] != i]
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
files_directory = "D:\\eu\\calcul numeric\\h4\\files"

print("Files directory:", files_directory)

for i in range(1, 6):
    a_file_path = os.path.join(files_directory, f"a_{i}.txt")
    b_file_path = os.path.join(files_directory, f"b_{i}.txt")
  
    if not os.path.exists(a_file_path):
        print("File not found:", a_file_path)
    if not os.path.exists(b_file_path):
        print("File not found:", b_file_path)

    a = read(a_file_path)
    b = read_b(b_file_path)
    rez = gauss_streidel(a, b)
    if isinstance(rez, str):
            print(i, rez)
    else:
            x, steps = rez
            print(i, x, steps, "iteratii")
            verif = np.zeros_like(b)
            for i in range(a.n):
                verif[i] = sum([l[0] * x[l[1]] for l in a.m[i]])
            print("norma:", norma(verif - b))

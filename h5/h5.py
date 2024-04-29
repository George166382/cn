import numpy as np
import os
import tkinter as tk
from tkinter import scrolledtext, messagebox

max_number_of_iterations = 100000
EPS = 1e-8


# Define the Matrix class
class Matrix:
    def __init__(self):
        self.p = 0
        self.n = 0
        self.a = []

    def init(self, file_name):
        try:
            with open(file_name, "r", encoding="utf-8") as file_in:
                self.p, self.n = map(int, file_in.readline().strip().split())
                self.a = []
                for _ in range(self.p):
                    row = list(map(float, file_in.readline().strip().split()))
                    if len(row) != self.n:
                        err("Invalid number of elements in a row!")
                        return False
                    self.a.append(row)
            return True
        except FileNotFoundError:
            err("Failed to open '{}' file!".format(file_name))
            return False
        except ValueError:
            err("Invalid file format! ('{}') Expected two numbers on the first line and matrix elements in subsequent lines.".format(file_name))
            return False




    def compute_pq(self):
        maxx = -99999999.0
        p = -1
        q = -1
        print("Matrix a:", self.a)
        print("Matrix dimensions:", len(self.a), "x", len(self.a[0]))  # Add this line
        if len(self.a) < 2:
            return -1, -1  # If the matrix has fewer than two rows, return invalid indices
        for i in range(len(self.a)):
            for j in range(i):
                print("i:", i, "j:", j)
                if i < len(self.a) and j < len(self.a[0]):  # Add this condition to ensure indices are within bounds
                    print("self.a[i][j]:", self.a[i][j])
                    if abs(self.a[i][j]) > maxx and abs(self.a[i][j]) > EPS:
                        maxx = abs(self.a[i][j])
                        p = i
                        q = j
        print("Computed p, q:", p, q)
        return p, q





    def compute_tcs(self, pq):
        if pq[0] == -1:
            return 0.0, (0.0, 0.0)
        p, q = pq
        alpha = (1.0 * (self.a[p][p] - self.a[q][q])) / (2.0 * self.a[p][q])
        sign = 1.0 if alpha >= 0 else -1.0
        t = -alpha + sign * np.sqrt((alpha * alpha) + 1)
        c = 1.0 / np.sqrt(1 + t * t)
        s = t / np.sqrt(1.0 + t * t)
        print("Computed t, c, s:", t, c, s)
        return t, (c, s)


    def generate_in(self):
        eye_matrix = np.eye(self.n)
        print("Generated identity matrix:")
        print(eye_matrix)
        return eye_matrix


    def task1(self, out):
        a_init = np.array(self.a)
        iter_count = max_number_of_iterations
        pq = self.compute_pq()
        tcs = self.compute_tcs(pq)
        u = self.generate_in()
        while iter_count > 0 and pq[0] != -1:
            t, (c, s) = tcs
            new_a = np.array(self.a)
            for j in range(self.n):
                if j != pq[0] and j != pq[1]:
                    new_a[pq[0]][j] = c * self.a[pq[0]][j] + s * self.a[pq[1]][j]
                    new_a[pq[1]][j] = -s * self.a[pq[0]][j] + c * self.a[pq[1]][j]
                    new_a[j][pq[1]] = -s * self.a[j][pq[0]] + c * self.a[j][pq[1]]
                    new_a[j][pq[0]] = self.a[j][pq[0]]
            new_a[pq[0]][pq[0]] = self.a[pq[0]][pq[0]] + t * self.a[pq[0]][pq[1]]
            new_a[pq[1]][pq[1]] = self.a[pq[1]][pq[1]] - t * self.a[pq[0]][pq[1]]
            new_a[pq[0]][pq[1]] = new_a[pq[1]][pq[0]] = 0.0
            for i in range(self.n):
                old = u[i][pq[0]]
                u[i][pq[0]] = c * u[i][pq[0]] + s * u[i][pq[1]]
                u[i][pq[1]] = -s * old + c * u[i][pq[1]]
            self.a = new_a
            pq = self.compute_pq()
            tcs = self.compute_tcs(pq)
            iter_count -= 1

        out.insert(tk.END, "-----------------------------------\n")
        out.insert(tk.END, "Valorile proprii:\n\n")
        for i in range(self.n):
            out.insert(tk.END, "{}, ".format(self.a[i][i]))
        out.insert(tk.END, "\n\n-----------------------------------\n")
        out.insert(tk.END, "Vectorii proprii:\n\n")
        for j in range(self.n):
            for i in range(self.n):
                out.insert(tk.END, "{}, ".format(u[i][j]))
            out.insert(tk.END, '\n')
        U = np.zeros((self.p, self.n))
        Lambda = np.zeros((self.p, self.n))
        for i in range(self.p):
            for j in range(self.n):
                U[i, j] = u[i][j]
                if i == j:
                    Lambda[i, j] = self.a[i][i]
                else:
                    Lambda[i, j] = 0.0
        prod1 = np.dot(a_init, U)
        prod2 = np.dot(U, Lambda)
        out.insert(tk.END, "\n-----------------------------------\n")
        out.insert(tk.END, "A^(init) * U:\n{}\n".format(prod1))
        out.insert(tk.END, "U * Λ:\n{}\n".format(prod2))
        out.insert(tk.END, "|| A^(init) * U - U * Λ ||:\n{}\n".format(np.linalg.norm(prod1 - prod2)))


    def task2(self, out):
        last_eigen_A = np.array(self.a)
        diff = 100.0
        iter_count = max_number_of_iterations
        while iter_count > 0 and abs(diff) >= EPS:
            last_L = np.linalg.cholesky(last_eigen_A)
            eigen_A = np.dot(last_L.T, last_L)
            diff = np.linalg.norm(eigen_A - last_eigen_A)
            last_eigen_A = eigen_A
            iter_count -= 1
        out.insert(tk.END, "\n-----------------------------------\n")
        out.insert(tk.END, "A^k:\n{}\n".format(last_eigen_A))

    def task3(self, out):
        eigen_A = np.array(self.a)
        out.insert(tk.END, "----------------------------\n")
        out.insert(tk.END, "Task 3 - using NumPy library:\n")
        u, singular_values, vt = np.linalg.svd(eigen_A, full_matrices=True)
        out.insert(tk.END, "\nSingular values:\n{}\n".format(singular_values))
        out.insert(tk.END, "\nMatrix U:\n{}\n".format(u))
        out.insert(tk.END, "\nMatrix V^T:\n{}\n".format(vt))
        matrix_rank = np.sum(singular_values > EPS)
        sigma_max = np.max(singular_values)
        sigma_min = np.min(singular_values[singular_values > EPS])
        condition_number = sigma_max / sigma_min if sigma_min > 0 else np.inf
        out.insert(tk.END, "\nMatrix rank:\n{}\n".format(matrix_rank))
        out.insert(tk.END, "\nCondition number of matrix:\n{}\n".format(condition_number))
        A_inv = np.linalg.pinv(eigen_A)
        A_pseudo_inv = np.linalg.pinv(eigen_A)
        diff = np.linalg.norm(A_inv - A_pseudo_inv)
        out.insert(tk.END, "\nMoore-Penrose pseudoinverse of matrix:\n{}\n".format(A_inv))
        out.insert(tk.END, "\nPseudoinverse least squares of matrix:\n{}\n".format(A_pseudo_inv))
        out.insert(tk.END, "||A^I - A^J|| = {}\n".format(diff))


def err(message):
    messagebox.showerror("Error", message)

import os

# Update the directory where input files are located
input_directory = r"D:\eu\calcul numeric\h5"

# Update the directory where output files will be located
output_directory = r"D:\eu\calcul numeric\h5"

def solve_for_matrix(test_number, file_suff, output_text):
    a_file_name = os.path.join(input_directory, "matrix{}.txt".format(test_number))
    m = Matrix()
    if not m.init(a_file_name):
        messagebox.showerror("Error", error_message)
        return False

    output_text.insert(tk.END, "---------------------------------\n")
    output_text.insert(tk.END, "Testul {}:\n".format(test_number))

    try:
        output_text.insert(tk.END, str(m) + "\n")
        output_text.insert(tk.END, "---------------------------------\n")
        output_text.insert(tk.END, "{:.6f}\n".format(EPS))
        m.task1(output_text)
        m.task2(output_text)
        m.task3(output_text)
    except Exception as e:
        messagebox.showerror("Error", "Error: {}".format(e))
        return False

    return True




error_message = ""


# Define the function to run the tests
def run_tests():
    # Clear the output text area
    output_text.delete(1.0, tk.END)

    
    error_message = ""
    i = 1
    if not solve_for_matrix(i, "_sol", output_text):
            messagebox.showerror("Error", "Error - matrix {}".format(i))
            output_text.insert(tk.END, error_message + "\n")


# Create the main window
window = tk.Tk()
window.title("Matrix Solver")

# Create a label for the output
output_label = tk.Label(window, text="Output:")
output_label.pack()

# Create a text area for displaying the output
output_text = scrolledtext.ScrolledText(window, width=60, height=20)
output_text.pack()

# Create a button to run the tests
run_button = tk.Button(window, text="Run Tests", command=run_tests)
run_button.pack()

# Run the Tkinter event loop
window.mainloop()
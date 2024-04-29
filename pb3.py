import numpy as np
import math

def T4(a):
    return ((105 * a) - (10 * a**3)) / (105 - 45 * a**2 + a**4)

def T5(a):
    return ((945 * a) - (105 * a**3) + a**5) / (945 - (420 * a**2) + (15*a**4))

def T6(a):
    return ((10395 * a) - (1260 * a**3) + 21*(a**5)) / (10395 - (4725 * a**2) + (210*a**4)- a**6)

def T7(a):
    return ((135135 * a) - (17325 * a**3) + 378*(a**5) - a**7) / (135135 - (62370 * a**2) + 3150*(a**4) - (630*a**6) + a**8)

def T8(a):
    return ((2027025 * a) - (270270 * a**3) + 6930*(a**5) - (36*a**7)) / (2027025  - (945945 * a**2) + 51975*(a**4) - (36*a**7))

def T9(a):
    return ((34459425 * a) - (4729725 * a**3) + 135135*(a**5) - (990*a**7) + a**9) / (34459425  - (16216200 * a**2) + 945945*(a**4) - (13860*a**6) + 45*a**8)

# Generate 10,000 random numbers in the interval [-pi/2, pi/2]
random_numbers = np.random.uniform(low=-np.pi/2, high=np.pi/2, size=10000)

# Compute exact tangent values using math.tan
exact_tan_values = [math.tan(a) for a in random_numbers]

# Compute errors for each approximation
errors_T4 = [abs(T4(a) - math.tan(a)) for a in random_numbers]
errors_T5 = [abs(T5(a) - math.tan(a)) for a in random_numbers]
errors_T6 = [abs(T6(a) - math.tan(a)) for a in random_numbers]
errors_T7 = [abs(T7(a) - math.tan(a)) for a in random_numbers]
errors_T8 = [abs(T8(a) - math.tan(a)) for a in random_numbers]
errors_T9 = [abs(T9(a) - math.tan(a)) for a in random_numbers]

# Find the indices of the three functions with the smallest errors for each random number
min_errors_indices = np.argsort([errors_T4[i] + errors_T5[i] + errors_T6[i] + errors_T7[i] + errors_T8[i] + errors_T9[i] for i in range(len(random_numbers))])

# Display the three functions with the smallest errors for the first 10 random numbers
for i in range(10000):
    idx = min_errors_indices[i]
    best_approximations = [np.argmin([errors_T4[idx], errors_T5[idx], errors_T6[idx], errors_T7[idx], errors_T8[idx], errors_T9[idx]]),
                           np.argsort([errors_T4[idx], errors_T5[idx], errors_T6[idx], errors_T7[idx], errors_T8[idx], errors_T9[idx]])[1],
                           np.argsort([errors_T4[idx], errors_T5[idx], errors_T6[idx], errors_T7[idx], errors_T8[idx], errors_T9[idx]])[2]]
    print(f"Random number: {random_numbers[idx]}, Exact tan: {exact_tan_values[idx]}, "
          f"Best approximations: {best_approximations }")

# Initialize lists to store the errors and corresponding functions
errors = [[] for _ in range(6)]

# Compute errors for each function
for a in random_numbers:
    exact_tan = math.tan(a)
    errors[0].append(abs(T4(a) - exact_tan))
    errors[1].append(abs(T5(a) - exact_tan))
    errors[2].append(abs(T6(a) - exact_tan))
    errors[3].append(abs(T7(a) - exact_tan))
    errors[4].append(abs(T8(a) - exact_tan))
    errors[5].append(abs(T9(a) - exact_tan))

# Find the indices of the three functions with the smallest errors for each random number
best_approximations = np.argsort(errors, axis=0)[:3, :]

    

# Count the frequency of each function being among the top three for the best approximations
function_counts = np.zeros(6)
for i in range(6):
    function_counts[i] = np.sum(best_approximations == i)

# Display the hierarchy based on the frequency of being among the top three approximations
hierarchy = np.argsort(-function_counts)
print("Hierarchy of functions based on frequency:")
for i, func_index in enumerate(hierarchy):
    print(f"Rank {i+1}: T{func_index+4}")



def find_machine_precision():
    eps = 1.0

    while eps + 1.0 > 1.0:
        eps /= 10.0
    
    return eps


u = find_machine_precision()


x = 1.0
y = u / 10.0
z = u / 10.0

# Non-associativity of addition
left_side_addition = (x + y) + z
right_side_addition = x + (y + z)

# Non-associativity of multiplication
left_side_multiplication = (x * y) * z
right_side_multiplication = x * (y * z)

# Print the results
print("For addition:")
print(f"(x + y) + z = {left_side_addition}")
print(f"x + (y + z) = {right_side_addition}")
print(f"The result is {left_side_addition != right_side_addition}")

print("\nFor multiplication:")
print(f"(x * y) * z = {left_side_multiplication}")
print(f"x * (y * z) = {right_side_multiplication}")
print(f"The result is {left_side_multiplication != right_side_multiplication}")

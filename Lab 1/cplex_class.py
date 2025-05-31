from docplex.mp.model import Model

# Create a model
mdl = Model(name='DemoDocplex')

# Variables
x = mdl.continuous_var(name='x', lb=0)
y = mdl.continuous_var(name='y', lb=0)

# Objective: Maximize x + y
mdl.maximize(x + y)

# Constraints
mdl.add_constraint(x + 2*y <= 4, 'c1')
mdl.add_constraint(x*4 + 3*y <= 12, 'c2')

# Solve
sol = mdl.solve(log_output=True)

# Output
if sol:
    print(f"Objective Value: {mdl.objective_value}")
    mdl.print_solution()
else:
    print("No solution found")
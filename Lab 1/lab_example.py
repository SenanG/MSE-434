from gurobipy import Model, GRB, quicksum

# -----------------------------
# Data
facilities = ['F1', 'F2', 'F3']
customers = ['C1', 'C2', 'C3', 'C4']

fixed_cost = {'F1': 1000, 'F2': 1200, 'F3': 1100}
capacity = {'F1': 50, 'F2': 60, 'F3': 40}
demand = {'C1': 10, 'C2': 20, 'C3': 15, 'C4': 25}

distance = {
    ('F1', 'C1'): 4, ('F1', 'C2'): 6, ('F1', 'C3'): 9, ('F1', 'C4'): 5,
    ('F2', 'C1'): 5, ('F2', 'C2'): 4, ('F2', 'C3'): 7, ('F2', 'C4'): 6,
    ('F3', 'C1'): 6, ('F3', 'C2'): 3, ('F3', 'C3'): 4, ('F3', 'C4'): 7,
}

# -----------------------------
# Model
m = Model("Capacitated Facility Location")

# Variables
y = m.addVars(facilities, vtype=GRB.BINARY, name="Open")
x = m.addVars(facilities, customers, vtype=GRB.CONTINUOUS, name="Ship")

# Objective: fixed cost + transport cost
m.setObjective(
    quicksum(fixed_cost[i] * y[i] for i in facilities) +
    quicksum(distance[i, j] * x[i, j] for i in facilities for j in customers),
    GRB.MINIMIZE
)

# Constraints

# 1. Demand must be satisfied
for j in customers:
    m.addConstr(quicksum(x[i, j] for i in facilities) == demand[j], name=f"Demand_{j}")

# 2. Facility capacity
for i in facilities:
    m.addConstr(quicksum(x[i, j] for j in customers) <= capacity[i] * y[i], name=f"Capacity_{i}")

# -----------------------------
# Solve
m.optimize()

# -----------------------------
# Output
if m.status == GRB.OPTIMAL:
    print(f"Total cost: ${m.objVal:.2f}")
    for i in facilities:
        if y[i].x > 0.5:
            print(f"Open facility {i}")
            for j in customers:
                if x[i, j].x > 0:
                    print(f"  Ship {x[i, j].x} units to customer {j}")
else:
    print("No optimal solution found.")

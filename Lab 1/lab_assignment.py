import pandas as pd
from gurobipy import Model, GRB, quicksum

# --- Load CSV ---
df = pd.read_csv("LabData.csv")

# --- Extract facility and customer data ---
facility_data = df[['Facility', 'Fixed Cost ($)', 'Capacity (units)']].dropna().reset_index(drop=True)
facilities = facility_data['Facility'].tolist()
fixed_cost = dict(zip(facility_data['Facility'], facility_data['Fixed Cost ($)']))
capacity = dict(zip(facility_data['Facility'], facility_data['Capacity (units)']))

customer_cols = df.columns[df.columns.get_loc('Facility.1')+1:]
customers = customer_cols.tolist()

customer_data = df[['Customer', 'Demand (units)']].dropna().reset_index(drop=True)
demand = dict(zip(customer_data['Customer'], customer_data['Demand (units)']))

distance = {}
for i, row in facility_data.iterrows():
    facility = row['Facility']
    for customer in customers:
        distance[(facility, customer)] = df.loc[i, customer]

# --- Gurobi Model ---
m = Model("FacilityLocation_Lab1")

# Parameters
revenue_per_unit = 1000 # Assumed constant profit per unit delivered

# Decision variables
y = m.addVars(facilities, vtype=GRB.BINARY, name="Open")  # Open facility
x = m.addVars(facilities, customers, vtype=GRB.CONTINUOUS, name="Ship")  # Quantity shipped
z = m.addVars(facilities, customers, vtype=GRB.BINARY, name="Assign")  # Assignment

# --- Objective: Maximize profit ---
# Revenue from shipped goods - Fixed costs - Transport cost
m.setObjective(
    quicksum(revenue_per_unit * x[i, j] for i in facilities for j in customers)
    - quicksum(fixed_cost[i] * y[i] for i in facilities)
    - quicksum(distance[i, j] * x[i, j] for i in facilities for j in customers),
    GRB.MAXIMIZE
)

# --- Constraints ---

# 1. Each customer assigned to at most one facility
for j in customers:
    m.addConstr(quicksum(z[i, j] for i in facilities) <= 1, name=f"AssignOnce_{j}")

# 2. Shipping only allowed if assigned
for i in facilities:
    for j in customers:
        m.addConstr(x[i, j] <= demand[j] * z[i, j], name=f"ShipIfAssigned_{i}_{j}")

# 3. Facility capacity not exceeded
for i in facilities:
    m.addConstr(quicksum(x[i, j] for j in customers) <= capacity[i] * y[i], name=f"Capacity_{i}")

# 4. Facility must be open to serve
for i in facilities:
    for j in customers:
        m.addConstr(z[i, j] <= y[i], name=f"ServeIfOpen_{i}_{j}")

# 5. At most 3 facilities can be open
m.addConstr(quicksum(y[i] for i in facilities) <= 3, name="Max3Facilities")

# --- Solve ---
m.optimize()

# --- Print results ---
if m.status == GRB.OPTIMAL:
    print(f"\nTotal Profit: {m.objVal:.2f}")
    for i in facilities:
        if y[i].x > 0.5:
            print(f"\nFacility {i} is OPEN")
            for j in customers:
                if x[i, j].x > 0:
                    print(f"  Ships {x[i, j].x:.1f} units to Customer {j}")
else:
    print("No optimal solution found.")
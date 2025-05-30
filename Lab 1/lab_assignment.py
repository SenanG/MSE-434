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
m = Model("CapacitatedFacilityLocation")

y = m.addVars(facilities, vtype=GRB.BINARY, name="Open")
x = m.addVars(facilities, customers, vtype=GRB.CONTINUOUS, name="Ship")

m.setObjective(
    quicksum(fixed_cost[i] * y[i] for i in facilities) +
    quicksum(distance[i, j] * x[i, j] for i in facilities for j in customers),
    GRB.MINIMIZE
)

for j in customers:
    m.addConstr(quicksum(x[i, j] for i in facilities) == demand[j], name=f"Demand_{j}")

for i in facilities:
    m.addConstr(quicksum(x[i, j] for j in customers) <= capacity[i] * y[i], name=f"Capacity_{i}")

m.optimize()

# --- Print results ---
if m.status == GRB.OPTIMAL:
    print(f"Total Cost: {m.objVal:.2f}")
    for i in facilities:
        if y[i].x > 0.5:
            print(f"\nFacility {i} is OPEN")
            for j in customers:
                if x[i, j].x > 0:
                    print(f"  Ships {x[i, j].x} units to Customer {j}")
else:
    print("No optimal solution found.")

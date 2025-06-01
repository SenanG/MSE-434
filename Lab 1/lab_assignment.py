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

# --- Calculate transport costs from distances ---
# Assumes distances are in the main df, in rows indexed by 'Facility' and columns named by customers.
# The 3.5 is the shipping cost per unit of distance.
shipping_rate = 3.5
transport_costs_matrix = {}

# Prepare a DataFrame view for easier lookup of distances.
# It's assumed that the rows in `df` that define facilities (used for `facility_data`)
# are the same rows that contain the distance data under customer columns.
facility_rows_for_distances_df = df[df['Facility'].isin(facilities)].copy()
facility_rows_for_distances_df.set_index('Facility', inplace=True)

for i in facilities:  # i is facility name, e.g., 'F1'
    transport_costs_matrix[i] = {}
    for j in customers:  # j is customer name, e.g., 'C1'
        try:
            distance = facility_rows_for_distances_df.loc[i, j]
            if pd.isna(distance) or str(distance).strip() == "":
                print(f"Warning: Missing or invalid distance for Facility {i} to Customer {j}. Found '{distance}'. Using a high placeholder cost.")
                # Using a very high cost to effectively prevent shipping if distance is missing/invalid
                transport_costs_matrix[i][j] = 999999 * shipping_rate
            else:
                transport_costs_matrix[i][j] = float(distance) * shipping_rate
        except KeyError:
            print(f"Critical Error: Data inconsistency. Could not find distance entry for Facility '{i}' / Customer '{j}'.")
            print(f"Ensure Facility '{i}' (from facilities list) has a corresponding row in the CSV's distance matrix section,")
            print(f"and Customer '{j}' (from customer list derived from CSV headers) is a column in that section.")
            transport_costs_matrix[i][j] = 9999999 * shipping_rate # Prohibitive cost
        except ValueError:
            print(f"Error: Non-numeric distance value for Facility {i}, Customer {j}: '{facility_rows_for_distances_df.loc[i, j]}'. Using high placeholder cost.")
            transport_costs_matrix[i][j] = 9999999 * shipping_rate # Prohibitive cost


# --- Gurobi Model ---
m = Model("FacilityLocation_Lab1")

# Parameters
revenue_per_unit = 1000  # Assumed constant revenue per unit shipped
# transport_cost_per_unit = 3.5  # Fixed transportation cost per unit shipped (REPLACED)

# Decision variables
y = m.addVars(facilities, vtype=GRB.BINARY, name="Open")           # Open facility
x = m.addVars(facilities, customers, vtype=GRB.CONTINUOUS, name="Ship")  # Quantity shipped
z = m.addVars(facilities, customers, vtype=GRB.BINARY, name="Assign")    # Assignment

# --- Objective: Maximize profit ---
# Profit = revenue - fixed cost - transport cost
m.setObjective(
    quicksum(revenue_per_unit * x[i, j] for i in facilities for j in customers)
    - (quicksum(fixed_cost[i] * y[i] for i in facilities)
    + quicksum(transport_costs_matrix[i][j] * x[i, j] for i in facilities for j in customers)),
    GRB.MAXIMIZE
)

# --- Constraints ---

# 1. Each customer must be assigned to exactly one facility
for j in customers:
    m.addConstr(quicksum(z[i, j] for i in facilities) == 1, name=f"AssignOnce_{j}")

# 2. Shipping only allowed if assigned
for i in facilities:
    for j in customers:
        m.addConstr(x[i, j] <= demand[j] * z[i, j], name=f"ShipIfAssigned_{i}_{j}")

# 2b. If assigned, must ship at least 1 unit (if demand allows - see note in code)
# This ensures that if z[i,j] is 1, then x[i,j] must be at least 1.
# This may cause infeasibility if demand[j] < 1 for any customer j.
for i in facilities:
    for j in customers:
        m.addConstr(x[i, j] >= z[i, j], name=f"MinShipmentIfAssigned_{i}_{j}")

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
    
    open_facilities_details = {}
    # First, identify open facilities and prepare for detailed output
    for i in facilities:
        if y[i].x > 0.5:
            open_facilities_details[i] = {"name": i, "ships_to": [], "total_shipped_from_facility": 0}

    print("\n--- Customer Assignments and Shipments ---")
    all_customers_fully_served = True
    customers_not_fully_served_details = []

    for j in customers: # For each customer
        assigned_facility_name = None
        shipped_quantity_to_j = 0
        customer_j_is_assigned = False

        for i in facilities: # Check all facilities for assignment to customer j
            if z[i, j].x > 0.5: # If customer j is assigned to facility i
                customer_j_is_assigned = True
                assigned_facility_name = i
                shipped_quantity_to_j = x[i, j].x
                
                if assigned_facility_name in open_facilities_details: # This facility i must be open
                    open_facilities_details[assigned_facility_name]["ships_to"].append(
                        {"customer": j, "quantity": shipped_quantity_to_j}
                    )
                    open_facilities_details[assigned_facility_name]["total_shipped_from_facility"] += shipped_quantity_to_j
                else:
                    # This case should ideally not happen if z[i,j]=1 implies y[i]=1 (Constraint 4)
                    print(f"WARNING: Customer {j} assigned to facility {i} (z[{i},{j}].x = {z[i,j].x:.2f}), but facility {i} is marked closed (y[{i}].x = {y[i].x:.2f}). Check ServeIfOpen constraint.")
                break # Customer j found its one assigned facility

        customer_demand_j = demand.get(j, 0) # Get demand, default to 0 if not found
        if customer_j_is_assigned:
            print(f"Customer {j}: Assigned to Facility {assigned_facility_name}, Shipped: {shipped_quantity_to_j:.1f}, Demand: {customer_demand_j:.1f}")
            if abs(shipped_quantity_to_j - customer_demand_j) > 0.001 and shipped_quantity_to_j < customer_demand_j:
                print(f"  WARNING: Customer {j} demand NOT fully met (Shipped: {shipped_quantity_to_j:.1f} / Demand: {customer_demand_j:.1f}).")
                all_customers_fully_served = False
                customers_not_fully_served_details.append(f"Customer {j} (Demand: {customer_demand_j:.1f}, Shipped: {shipped_quantity_to_j:.1f})")
            if shipped_quantity_to_j < 0.001 and customer_demand_j > 0:
                print(f"  INFO: Customer {j} assigned but received no significant shipment. Possible reasons: high transport cost, facility capacity reached.")
                # This customer contributes to the "left out" feeling if they don't get goods
                if all_customers_fully_served: # Avoid double flag if already not fully served
                     all_customers_fully_served = False # Technically assigned, but effectively left out if no shipment
                if f"Customer {j}" not in str(customers_not_fully_served_details):
                    customers_not_fully_served_details.append(f"Customer {j} (Demand: {customer_demand_j:.1f}, Shipped: {shipped_quantity_to_j:.1f} - effectively no shipment)")


    print("\n--- Facility Shipment Summary ---")
    for fac_id in facilities:
        if y[fac_id].x > 0.5: # If facility is open
            details = open_facilities_details[fac_id]
            print(f"\nFacility {details['name']} is OPEN. Capacity: {capacity.get(fac_id, 'N/A')}. Total Shipped from here: {details['total_shipped_from_facility']:.1f}")
            if details["ships_to"]:
                for shipment_info in details["ships_to"]:
                    if shipment_info['quantity'] > 0.001: # Only detail actual shipments
                        print(f"  Ships {shipment_info['quantity']:.1f} units to Customer {shipment_info['customer']}")
            else:
                print(f"  Ships to no customers (or all shipment quantities were zero).")

    if not all_customers_fully_served:
        print("\n--- Summary of Service Issues ---")
        print("One or more customers were not fully served or had issues with their assignment:")
        for detail in customers_not_fully_served_details:
            print(f"- {detail}")
            
elif m.status == GRB.INFEASIBLE:
    print("\nModel is INFEASIBLE.")
    

else:
    print(f"\nOptimization stopped with status code {m.status}. Solution may not be optimal or feasible.")
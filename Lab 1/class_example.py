import gurobipy as gp
from gurobipy import GRB

m = gp.Model("DemoExample")

x = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name="x")
y = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name="y")

m.setObjective(x + y, GRB.MAXIMIZE)

m.addConstr(x+2*y <=4, "c1")
m.addConstr(x*4+3*y <= 12, "c2")

m.optimize()

if m.status == GRB.OPTIMAL:
    if m.status == GRB.OPTIMAL:
        print(f"Objective Value: {m.objVal}")
        print(f"x: {x.X}, y: {y.X}")
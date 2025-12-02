import pulp
import pandas as pd
from geopy.distance import geodesic

class FleetOptimizer:
    def __init__(self, stations_df):
        self.stations = stations_df.set_index("station_id")
        self.station_ids = self.stations.index.tolist()
        self.cost_matrix = self._calculate_cost_matrix()

    def _calculate_cost_matrix(self):
        """
        Calculates distance matrix between all stations.
        Returns dict: (i, j) -> distance_km
        """
        costs = {}
        for i in self.station_ids:
            for j in self.station_ids:
                if i == j:
                    costs[(i, j)] = 0
                else:
                    loc_i = (self.stations.loc[i, "lat"], self.stations.loc[i, "lon"])
                    loc_j = (self.stations.loc[j, "lat"], self.stations.loc[j, "lon"])
                    # Simple geodesic distance
                    dist = geodesic(loc_i, loc_j).km
                    costs[(i, j)] = dist
        return costs

    def solve_relocation(self, current_fleet, target_demand, lambda_penalty=1.0):
        """
        Solves the fleet relocation problem.
        
        Args:
            current_fleet (dict): station_id -> current cars
            target_demand (dict): station_id -> target cars needed
            lambda_penalty (float): Cost per unit of shortfall (unmet demand)
            
        Returns:
            moves (list): List of dicts {'from', 'to', 'amount', 'cost'}
            results (df): DataFrame with final state per station
        """
        prob = pulp.LpProblem("Fleet_Relocation", pulp.LpMinimize)
        
        # Sets
        S = self.station_ids
        
        # Parameters
        A = current_fleet # Available
        C = self.stations["capacity"].to_dict() # Capacity
        T = target_demand # Target
        D = self.cost_matrix # Distance/Cost
        
        # Variables
        # f_ij: cars moved from i to j
        f = pulp.LpVariable.dicts("flow", (S, S), lowBound=0, cat="Integer")
        # x_i: final cars at station i
        x = pulp.LpVariable.dicts("final_cars", S, lowBound=0, cat="Integer")
        # s_i: shortfall at station i
        s = pulp.LpVariable.dicts("shortfall", S, lowBound=0, cat="Integer")
        
        # Objective Function
        # Minimize: Transport Cost + Penalty for Shortfall
        # Transport cost = sum(distance * cars_moved)
        # We add a small cost for moving 0 distance to avoid loops, though not strictly needed if D[i,i]=0
        prob += (
            pulp.lpSum(D[i, j] * f[i][j] for i in S for j in S if i != j) +
            lambda_penalty * pulp.lpSum(s[i] for i in S)
        )
        
        # Constraints
        for i in S:
            # 1. Flow Balance: Final = Initial + In - Out
            prob += (
                x[i] == A.get(i, 0) + 
                        pulp.lpSum(f[j][i] for j in S if j != i) - 
                        pulp.lpSum(f[i][j] for j in S if j != i)
            )
            
            # 2. Capacity Constraint
            prob += x[i] <= C[i]
            
            # 3. Shortfall Definition: s_i >= Target - Final
            prob += s[i] >= T.get(i, 0) - x[i]
            
        # 4. Conservation of Fleet
        total_initial = sum(A.values())
        prob += pulp.lpSum(x[i] for i in S) == total_initial
        
        # Solve
        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        
        # Extract Results
        moves = []
        for i in S:
            for j in S:
                if i != j:
                    val = f[i][j].value()
                    if val and val > 0:
                        moves.append({
                            "from_id": i,
                            "from_name": self.stations.loc[i, "station_name"],
                            "to_id": j,
                            "to_name": self.stations.loc[j, "station_name"],
                            "amount": int(val),
                            "cost": val * D[i, j]
                        })
                        
        results_data = []
        for i in S:
            results_data.append({
                "station_id": i,
                "station_name": self.stations.loc[i, "station_name"],
                "initial": A.get(i, 0),
                "final": int(x[i].value()),
                "target": T.get(i, 0),
                "shortfall": int(s[i].value()),
                "capacity": C[i]
            })
            
        return moves, pd.DataFrame(results_data)

    def solve_relocation_longterm(self, current_fleet, future_demand_df, lambda_penalty=1.0):
        """
        Solves the fleet relocation problem based on AGGREGATE future demand.
        
        This is a strategic optimization: redistribute fleet to match expected demand
        over the next 3 months, moving cars from low-demand to high-demand zones.
        
        Args:
            current_fleet (dict): station_id -> current cars
            future_demand_df (DataFrame): columns=['station_id', 'date', 'prediction']
                                          Contains predictions for next 90 days
            lambda_penalty (float): Cost per unit of aggregate shortfall
            
        Returns:
            moves (list): List of dicts {'from', 'to', 'amount', 'cost'}
            results (df): DataFrame with final state per station
        """
        # Aggregate demand by station (sum over 3 months)
        agg_demand = future_demand_df.groupby('station_id')['prediction'].sum().to_dict()
        
        # Use the same optimization logic but with aggregate targets
        prob = pulp.LpProblem("Fleet_Relocation_LongTerm", pulp.LpMinimize)
        
        # Sets
        S = self.station_ids
        
        # Parameters
        A = current_fleet # Available
        C = self.stations["capacity"].to_dict() # Capacity (daily)
        T = agg_demand # Target (aggregate over 3 months)
        D = self.cost_matrix # Distance/Cost
        
        # Variables
        f = pulp.LpVariable.dicts("flow", (S, S), lowBound=0, cat="Integer")
        x = pulp.LpVariable.dicts("final_cars", S, lowBound=0, cat="Integer")
        s = pulp.LpVariable.dicts("shortfall", S, lowBound=0, cat="Integer")
        
        # Objective Function
        # Minimize: Transport Cost + Penalty for Aggregate Shortfall
        prob += (
            pulp.lpSum(D[i, j] * f[i][j] for i in S for j in S if i != j) +
            lambda_penalty * pulp.lpSum(s[i] for i in S)
        )
        
        # Constraints
        for i in S:
            # 1. Flow Balance
            prob += (
                x[i] == A.get(i, 0) + 
                        pulp.lpSum(f[j][i] for j in S if j != i) - 
                        pulp.lpSum(f[i][j] for j in S if j != i)
            )
            
            # 2. Capacity Constraint (we still respect daily capacity)
            prob += x[i] <= C[i]
            
            # 3. Shortfall Definition: s_i >= Aggregate_Target - (Final * 90 days)
            # The idea: if we have x[i] cars daily for 90 days, we can serve x[i]*90 total
            prob += s[i] >= T.get(i, 0) - (x[i] * 90)
            
        # 4. Conservation of Fleet
        total_initial = sum(A.values())
        prob += pulp.lpSum(x[i] for i in S) == total_initial
        
        # Solve
        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        
        # Extract Results
        moves = []
        for i in S:
            for j in S:
                if i != j:
                    val = f[i][j].value()
                    if val and val > 0:
                        moves.append({
                            "from_id": i,
                            "from_name": self.stations.loc[i, "station_name"],
                            "to_id": j,
                            "to_name": self.stations.loc[j, "station_name"],
                            "amount": int(val),
                            "cost": val * D[i, j]
                        })
                        
        results_data = []
        for i in S:
            results_data.append({
                "station_id": i,
                "station_name": self.stations.loc[i, "station_name"],
                "initial": A.get(i, 0),
                "final": int(x[i].value()) if x[i].value() else 0,
                "target_aggregate": int(T.get(i, 0)),
                "capacity_aggregate": C[i] * 90,  # 90 days
                "shortfall": int(s[i].value()) if s[i].value() else 0,
            })
            
        return moves, pd.DataFrame(results_data)


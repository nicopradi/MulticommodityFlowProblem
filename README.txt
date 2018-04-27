Multicommodity Flow Problem : 

Lastest version : Medium_instance/Updated_data/medium_instance.py (Take about 105sec to solve the problem entirely)

The sub-algorithms used to find the initial feasible set/solve the pricing problem  are explained in each solver python file. Here is a summary :

Small_instance : 1/ Initial feasible set : For each commodity take all the paths with distance < (shortest_path_distance)*constant
		         		   Make the constant grows until a feasible set is found
		 2/ Return ONLY ONE path, corresponding to the most negative reduced cost value

Medium_instance : Old_data : Check methods comments (TODO)
		  Updated_data : 1/ Check getInitSet() comment
			     2/ Check pricingProblem() comment


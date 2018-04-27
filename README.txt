Multicommodity Flow Problem : 

The sub-algorithms used to find the initial feasible set are explained in each solver python file. Here is a summary :

Small_instance : 1/ Initial feasible set : For each commodity take all the paths with distance < (shortest_path_distance)*constant
		         Make the constant grows until a feasible set is found
		     2/ Return ONLY ONE path, corresponding to the most negative reduced cost value

Medium_instance : Old_data : Check methods comments (TODO)
		          New_data : 1/ Check getInitSet() comment
				        2/ Check pricingProblem() comment


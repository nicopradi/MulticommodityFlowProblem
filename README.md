# Multicommodity-Flow-Problem

1/ Initialization : Run Dijkstra for all nodes, 
                    For each commodity : Select paths with dist < limit * shortest_path
                    
   Pricing : Add the path corresponding to the MOST violated constraint
   
2/ Initialization : Run Dijkstra for all nodes, 
                    For each commodity : Select paths with dist < limit * shortest_path                   
   Pricing : Add the path corresponding to the FIRST violated constraint

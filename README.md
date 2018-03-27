# Multicommodity-Flow-Problem

1/ Initialization : Run Dijkstra for all nodes, 
                    For each commodity : Select paths with dist < limit * shortest_path
                    
   Pricing : Add the path corresponding to the most violated constraint

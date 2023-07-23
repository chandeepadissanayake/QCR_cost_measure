import networkx as nx
import matplotlib.pyplot as plt

# Create an empty graph
G = nx.Graph()

# Define the set of vertices
vertices = ['a', 'b', 'c', 'd', 'e', 'f']

# Add vertices to the graph
G.add_nodes_from(vertices)

# Define the set of edges
edges = [('a', 'b'), ('a', 'e'), ('b', 'c'), ('b', 'd'), ('b', 'f'), ('c', 'd'), ('d', 'e'), ('e', 'f')]

# Add edges to the graph
G.add_edges_from(edges)

# Draw the graph
nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray', font_weight='bold')

# Show the graph
plt.show()

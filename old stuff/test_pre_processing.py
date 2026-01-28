import networkx as nx

import networkx as nx
import lzma

#Loads a graph file in .gr or .gr.xz format and returns an undirected graph of type NetworkX.
def load_gr_file(filepath):

    G = nx.DiGraph()

    # Opening a file by type
    if filepath.endswith('.xz'):
        open_func = lambda f: lzma.open(f, mode='rt')
    else:
        open_func = open

    with open_func(filepath) as file:
        for line in file:
            if line.startswith('c') or line.startswith('p'):
                # Comment or general definition lines - ignored
                continue
            parts = line.strip().split()
            if len(parts) == 2:
                try:
                    u = int(parts[0])
                    v = int(parts[1])
                    G.add_edge(u, v)
                except ValueError:
                    pass  # Skip invalid lines

    return G

#Comprehensive check of graph before work
def analyze_graph(G):

    print("=== Graph analysis ===")
    print(f"type: {type(G).__name__}")
    print(f"nodes: {G.number_of_nodes()}")
    print(f"edges: {G.number_of_edges()}")
    print(f"Density: {nx.density(G):.4f}")

    if G.is_directed():
        print(f"Strongly connected: {nx.is_strongly_connected(G)}")
        print(f"Weakly connected: {nx.is_weakly_connected(G)}")
    else:
        print(f"connected: {nx.is_connected(G)}")
        print(f"Components: {nx.number_connected_components(G)}")

    isolated = list(nx.isolates(G))
    if isolated:
        print(f"⚠️  Isolated vertices: {len(isolated)}")

    self_loops = nx.number_of_selfloops(G)
    if self_loops > 0:
        print(f"⚠️  Self-loops: {self_loops}")

    degrees = [d for n, d in G.degree()]
    print(f"Degree - Min: {min(degrees)}, Max: {max(degrees)}, Avg: {sum(degrees) / len(degrees):.2f}")

    print("=================")



if __name__ == '__main__':

    #PACE 2017 instances
    # "C:\\Users\\Osnat\\Dropbox\\Treewidth-PACE-2017-instances\\gr\\exact"

    #named graphs
    # "C:\\Users\\Osnat\\Dropbox\\named-graphs-master\\gr"

    graph_path = "C:\\Users\\Osnat\\Dropbox\\Treewidth-PACE-2017-instances\\gr\\exact"
    file_name = "ex070.gr.xz"
    ##0. load graph
    g = load_gr_file(graph_path+"\\"+file_name)
    analyze_graph(g)


    ##1. get causal vertices
    #c_v = find_causal_vertices_sets_optimized()

    ##2. create proper backdoor graph
    #pbd_graph = create_proper_backdoor_graph()

    ##3. get the induce graph of pbd_graph
    #create_induced_subgraph()

    ##4. get the moral graph
    #create_moral_graph()

    ##5. forbidden nodes
    #create_clique_on_neighbors(graph, nodes)

    ##6. forced inclusion set (I)
    #connect_I_to_XY(graph, I, X, Y)

    ##7. calculate the diversity


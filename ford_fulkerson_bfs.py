import networkx as nx
import random
import numpy as np
import array
from networkx.algorithms.connectivity import minimum_st_node_cut
from utils import visualize_g
import json
from networkx.algorithms.connectivity.cuts import minimum_st_edge_cut
from graph_alg_c_lib import fordfulkerson_clib

# Visualize the graph that is represented in the glist format
# We first convert the graph to Networkx format and then visualize it
def visualize_glist(glist, s, t):
    # return to the networkx graph format
    g = nx.DiGraph()
    g.graph['st'] = (s, t)
    node_index = 1
    while node_index < len(glist):
        node = glist[node_index]
        g.add_node(node)
        node_index += glist[node_index + 1] * 2 + glist[node_index + glist[node_index + 1] * 2 + 2] * 2 + 3
    node_index = 1
    while node_index < len(glist):
        node = glist[node_index]
        for k in range(glist[node_index + 1]):
            g.add_edge(node, glist[node_index + 2 + k])
        node_index += glist[node_index + 1] * 2 + glist[node_index + glist[node_index + 1] * 2 + 2] * 2 + 3
    visualize_g(g)
    return


# This is the pythonic version of the bfs_find_path function that is implemented in C.
def bfs_find_path(g, s, t):
    nodes = list(g.nodes())
    parent_dict = dict.fromkeys(nodes, 0)  # a node label can't be 0
    visited_dict = dict.fromkeys(nodes, 0)  # a node label can't be 0
    queue = [s]
    visited_dict[s] = 1

    while queue:
        u = queue.pop(0)
        for v in g.succ[u]:
            if visited_dict[v]:
                continue
            if (g[u][v]['dir'] == 1 and (g[u][v]['flow'] < 1)) or (g[u][v]['dir'] == -1 and (g[u][v]['flow'] < 0)):
                queue.append(v)
                visited_dict[v] = 1
                parent_dict[v] = u
                if v == t:
                    return True, parent_dict
    return False, parent_dict

def bfs_find_reachable(g, s):
    nodes = list(g.nodes())

    parent_dict = dict.fromkeys(nodes, 0)  # a node label can't be 0
    visited_dict = dict.fromkeys(nodes, 0)  # a node label can't be 0
    queue = [s]
    visited_dict[s] = 1
    while queue:
        u = queue.pop(0)
        for v in g.succ[u]:
            if visited_dict[v]:
                continue
            if g[u][v]['dir'] == 1:
                capacity = 1 if v == -u else np.inf
            else:
                capacity = 0
            if g[u][v]['flow'] < capacity:
                queue.append(v)
                visited_dict[v] = 1
                parent_dict[v] = u

    true_visited = [v for v in visited_dict if visited_dict[v]]
    return true_visited

def path_dict_to_list(parent_dict, s, t):
    v = t
    path = []
    while v != s:
        u = parent_dict[v]
        path.append((u, v))
        v = u
    return path

def bfs_find_path_clib_local(glist, s, t, nodes):
    visited_dict = dict.fromkeys(nodes, 0)
    parent_dict = dict.fromkeys(nodes, 0)
    q = []
    q.append(s)
    visited_dict[s] = 1

    while q:
        u = q.pop(0)
        u_index = 1
        while u_index < len(glist):
            v = glist[u_index]
            if v == u:
                break
            u_index += glist[u_index + 1] * 2 + 2
        adg_len = glist[u_index + 1] # get the number of successor nodes of u
        # iterate over the successor nodes of u. A successor node can be reached only if it has not been visited and
        # the flow over the edge is less than 1 for a forward edge or less than 0 for a reverse edge.
        # For edge (u,v) the direction will be forward if u and v are a node pair (same magnitude) and v is the duplicate (negative)
        # Or if u and v have different magnitudes and u is negative while v is positive.
        # The direction will be reverse if u and v are a node pair (same magnitude) and u is the duplicate (negative)
        # Or if u and v have different magnitudes and u is positive while v is negative.
        for i in range(adg_len):
            val = glist[u_index + 2 + i * 2]
            if visited_dict[val]:
                continue
            is_forward = (u == -val) and (val < 0) or (u != -val) and (u < 0)
            capacity = 1 if is_forward else 0
            if glist[u_index + 2 + i * 2 + 1] < capacity:
                q.append(val)
                visited_dict[val] = 1
                parent_dict[val] = u
                if val == t:
                    return 1, parent_dict
    return 0, parent_dict


def bfs_find_reachable_clib_local(glist, s, nodes):
    num_nodes = glist[0]

    visited_dict = dict.fromkeys(nodes, 0)
    q = []
    q.append(s)
    visited_dict[s] = 1
    while q:
        u = q.pop(0)
        # find the index of u in the glist
        u_index = 1
        while u_index < len(glist):
            v = glist[u_index]
            if v == u:
                break
            u_index += glist[u_index + 1] * 2 + 2
        adg_len = glist[u_index + 1]
        for i in range(adg_len):
            val = glist[u_index + 2 + i * 2]
            if visited_dict[val] == 1:
                continue
            # This is a special case for the vertex cut case. We are only interested in cuts through the edges
            # that are between two node pairs. A node pair is a node that originates from duplicating the original
            # node set. By our convention, node pairs only have one edge directed from v to -v. all other edges are
            # assumed to have infinite capacity and are not considered in the cut. This will eventually lead to the
            # cut being the set of edges between the original nodes and the duplicated nodes, which will be mapped to
            # the original nodes in the final vertex cut set.
            is_forward = (u == -val) and (val < 0) or (u != -val) and (u < 0)
            if is_forward:
                capacity = 1 if val == -u else np.inf # if the edge is between a node pair we set the capacity to 1
            else:
                capacity = 0
            if glist[u_index + 2 + i * 2 + 1] < capacity:  # check if not saturated
                q.append(val)
                visited_dict[val] = 1

    true_visited = array.array('i', [0] * num_nodes)
    i = 0
    for u in range(num_nodes):
        if visited_dict[nodes[u]] == 1:
            true_visited[i] = nodes[u]
            i += 1
    visited = array.array('i', true_visited[:i])
    return visited


def debug_print_flows_nx(g, filename):
    with open(filename, 'w') as f:
        for u in g.nodes():
            f.write(f"{u}\n")
            for v in g.succ[u]:
                f.write(f"  {v}: {g[u][v]['flow']}, {g[u][v]['dir']}\n")


def debug_print_flows(glist, filename):
    with open(filename, 'w') as f:
        i = 1
        while i < len(glist):
            node = glist[i]
            f.write(f"{node}\n")
            for j in range(glist[i + 1]):
                v = glist[i + 2 + j * 2]
                is_forward = 1 if ((node == -v) and (v < 0)) or ((node != -v) and (node < 0)) else -1
                f.write(f"  {v}: {glist[i + 2 + j * 2 + 1]}, {is_forward}\n")
            i += glist[i + 1] * 2  + 2

def fordfulkerson_clib_local(r_graph, s, t, K):

    num_nodes = r_graph[0]
    nodes = array.array('i', [0] * num_nodes)
    node_index = 1
    i = 0
    while node_index < len(r_graph):
        node = r_graph[node_index]
        if node == 0:
            return None  # Error- the node label can't be 0
        nodes[i] = node
        node_index += r_graph[node_index + 1] * 2 + 2
        i += 1
    # calculate degrees
    if s not in nodes:
        print("*** Error: source node not in the graph")
    num_paths = 0
    while 1:
        num_paths += 1
        path_found, parent_dict = bfs_find_path_clib_local(r_graph, s, t, nodes)
        if not path_found:
            break
        if num_paths > K:
            return None
        v = t
        while v != s:
            u = parent_dict[v]
            #update flow in the residual graph
            node_index = 1
            flows_updated = 0 # we need to update the flow in both directions
            while node_index < len(r_graph):
                node = r_graph[node_index]
                if node == u: # update forward edge flow
                    for i in range(r_graph[node_index + 1]): # scan in forward edges
                        if r_graph[node_index + 2 + i*2] == v:
                            r_graph[node_index + 2 + i*2 + 1] += 1 # increase the flow over the edge
                            flows_updated += 1
                            break
                if node == v: # update reverse edge flow
                    for i in range(r_graph[node_index + 1]):  # scan in forward edges
                        if r_graph[node_index + 2 + i * 2] == u:
                            r_graph[node_index + 2 + i * 2 + 1] -= 1  # decrease the flow over the edge
                            flows_updated += 1
                            break
                if flows_updated == 2: # we updated both edges
                    break
                node_index += r_graph[node_index + 1] * 2 + 2
            v = u
    reachables = bfs_find_reachable_clib_local(r_graph, s, nodes)
    cut = set()
    for i in range(len(reachables)):
        u = reachables[i]
        node_index = 1
        while node_index < len(r_graph):
            node = r_graph[node_index]
            if node == u:
                for j in range(r_graph[node_index + 1]):
                    v = r_graph[node_index + 2 + j*2]
                    if v == -u and u > 0: # we are only interested in the edges from the original nodes to the duplicated nodes
                        if v not in reachables:
                            cut.add((-1 - v)) # we subtract 1 to map the duplicated nodes to the original nodes
                        break
                break
            node_index += r_graph[node_index + 1] * 2 + 2
    return cut


# This is the pythonic version of the ford-fulkerson algorithm that is implemented in C.
# It should be the same except that the graph represenatation is different.
# The graph is represented as a networkx graph in the python version, and as an array in the C version.
def fordfulkerson_nx(g0, s, t, K):
    g = g0.copy()
    s = -s
    #print(f"s: {s}, t: {t}")
    edge_list = list(g.edges())
    reverse_edges = [(v, u) for (u, v) in edge_list]
    g.add_edges_from(reverse_edges)
    dir_values = {edge: 1 for edge in edge_list} # 1 is for forward direction
    nx.set_edge_attributes(g, dir_values, 'dir')
    dir_values = {edge: -1 for edge in reverse_edges}  # -1 is for reverse direction
    nx.set_edge_attributes(g, dir_values, 'dir')
    flow_values = {edge: 0 for edge in edge_list + reverse_edges} # set flow to 0 for all edges and reverse edges
    nx.set_edge_attributes(g, flow_values, 'flow')
    num_paths = 0
    while 1:
        num_paths += 1
        path_found, parent_dict = bfs_find_path(g, s, t)
        if not path_found:
            break
        if num_paths > K:
            return None
        v = t
        while v != s:
            u = parent_dict[v]
            g[u][v]['flow'] += 1
            g[v][u]['flow'] -= 1
            v = u
    reachables = bfs_find_reachable(g, s)
    cut = set()
    for u in reachables:
        for v in g0.succ[u]:
            if v == -u and v not in reachables:
                cut.add((-1 - v))
    return cut


# r_graph is an array of integer that represent the graph g
# The structure of the array is as follows:
# The first element is the number of nodes in the graph
# Then follows segments, where each segment represents a node and the adjacency list of that node.
# In order to represent a resudual graph, reverse edges are added to the graph.
# Reverse edjes can be identified by the following rule:
# For edge (u,v) the direction will be forward if u and v are a node pair (same magnitude) and v is the duplicate (negative)
# Or if u and v have different magnitudes and u is negative while v is positive.
# The direction will be reverse if u and v are a node pair (same magnitude) and u is the duplicate (negative)
# Or if u and v have different magnitudes and u is positive while v is negative.
# Need to update the bfs_find_path_clib_local accordingly
# Need to update the fordfulkerson_clib_local accordingly
# Need to update the bfs_find_reachable_clib_local accordingly

def fordfulkerson(g0, s, t, K):
    s = -s
    #print(f"s: {s}, t: {t}")
    g = g0.copy()
    edge_list = list(g.edges())
    reverse_edges = [(v, u) for (u, v) in edge_list]
    g.add_edges_from(reverse_edges)
    g_nodes = list(g.nodes())
    out_degrees = [deg for node, deg in g.out_degree()]
    arraysize = 2 * g.number_of_nodes() + 2 * np.sum(out_degrees) + 1
    r_graph = array.array('i', [0] * arraysize)
    r_graph[0] = g.number_of_nodes()
    i = 1
    for k in range(len(g_nodes)):
        node = g_nodes[k]
        r_graph[i] = node
        r_graph[i + 1] = g.out_degree(node)
        i += 2
        for x in g.succ[g_nodes[k]]: # add information of outgoing edges (successors)
            r_graph[i] = x
            r_graph[i + 1] = 0
            i += 2
    # r_graph is the residual graph in array format
    #return fordfulkerson_clib_local(r_graph, s, t)  # call the C function
    return fordfulkerson_clib_local(r_graph, s, t, K)
    #return fordfulkerson_clib(r_graph, s, t)  # call the C function

def min_st_v_cut(g, s, t, K):
    # Generate the duplicate graph which is a graph that has a duplicate of each node v, called -v.
    # The duplicate graph is directed. Each node v in the original graph is connected to its duplicate -v,
    # with an edge (v,-v) with weight 1. Each edge (u,v) in the original graph is replaced by two edges (u,v) and (-v,u)
    # this means that a path u-v-w in the original graph will become u->v->-v->w in the duplicate graph.
    # Then the edges (x,-x) of the min edge cut in the duplicate graph represent the nodes x in the min vertex cut.
    # Since g may have a note 0, we add 1 to make all nodes positive integers.
    if (s, t) in g.edges():
        return {}
    nodes = list(g.nodes())
    dup_nodes = set([-node - 1 for node in nodes]) | set([node + 1 for node in nodes])
    g_dup = nx.DiGraph()
    g_dup.add_nodes_from(dup_nodes)
    for v in g_dup.nodes():
        if v > 0:
            g_dup.add_edge(v, -v, weight=1)
    #g_dup.remove_node(s+1) # remove the source node since we start on the duplicate
    #g_dup.remove_node(-t-1) # remove the target node since we don't reach the duplicate
    for (u, v) in g.edges():
        g_dup.add_edge(-u-1, v+1, weight=1)
        g_dup.add_edge(-v-1, u+1, weight=1)

    s = s + 1
    t = t + 1
    g_dup.graph['st'] = (s, t)

    cut0 = fordfulkerson(g_dup, s, t, K)
    cut =  fordfulkerson_nx(g_dup, t, s)
    return set(cut)

if __name__ == '__main__':
    print("Starting")
    use_saved = 1
    n = 10  # number of nodes in the graph
    p = 0.3  # probability of edge creation
    K = 100
    numiter = 1000
    if use_saved:
        g = nx.read_gml("ford_fulkerson_failed.gml")
        g = nx.relabel_nodes(g, lambda x: int(x))
        s = g.graph['st'][0]
        t = g.graph['st'][1]
        #g.graph['st'] = (16, 17)
        # s = 16
        # t = 17
        numiter = 1
        visualize_g(g)

    for iter in range(numiter):
        if use_saved == 0:
            g = nx.gnp_random_graph(n, p, directed=False)
            for (u, v) in g.edges():
                g.edges[u, v]['weight'] = 1
            random_nodes = list(g.nodes())
            random.shuffle(random_nodes)
            t = None
            for start_index in range(len(random_nodes)):
                if t != None:
                    break
                s = random_nodes[start_index]
                for k in random_nodes:
                    if k == s:
                        continue
                    if nx.has_path(g, s, k):
                        t = k
                        break
                if t == None:
                    continue
            if t == None:
                print("Skipping- Graph is not connected")
                continue
            g.graph['st'] = (s, t)
            # nx.write_gml(g, "ford_fulkerson_failed.gml")  # save for next time

        #visualize_g(g)
        vcut1 = set(min_st_v_cut(g, s, t, K))
        vcut2 = set(minimum_st_node_cut(g, s, t))


        x1 = len(vcut1)
        x2 = len(vcut2)

        if vcut1 != vcut2:
            print("Error")
            print(f"Max flow: {x1}")
            print(f"MX Max flow: {x2}")
            print(f"Cut1: {vcut1}")
            print(f"Cut2: {vcut2}")
            print(f"s: {g.graph['st'][0]}")
            print(f"t: {g.graph['st'][1]}")
            print(f"s: {s}")
            print(f"t: {t}")
            nx.write_gml(g, "ford_fulkerson_failed1.gml")  # save for next time
            exit(0)

    print("OK")

    exit(0)
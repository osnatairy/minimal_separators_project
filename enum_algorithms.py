import networkx as nx
from networkx.algorithms.connectivity import minimum_st_node_cut
import heapq
from utils import sum_weights, visualize_g, bail
import utils
from itertools import combinations, product
from utils import g_queue, g1_queue
import numpy as np
import time
from graph_alg_c_lib import fordfulkerson_clib
import array
import ford_fulkerson_bfs

# Saturating G over u == adding edges as to make all neighbors of u to a clique
def saturate(G, u):
    N = nx.neighbors(G, u)
    H = G.copy()
    edges = combinations(N, 2)  # all pairs in N
    H.add_edges_from(list(edges))
    return H


# contract a set of nodes into one node incident on all edges that terminates on a node from the set
def contract_nodes(G, nodes, new_name):
    Ng = set()
    for x in nodes:
        Ng |= set(nx.neighbors(G, x))  # set union
    Ng = Ng - nodes
    H = nx.Graph(nx.subgraph(G, G.nodes - nodes))  # the external function nx.Graph() forces a copy
    H.add_node(new_name)  # the contracted node can be named anything
    edges = list(product(Ng, [new_name]))
    H.add_edges_from(edges)
    return H

# enumerate separators by cardinality
# Note that minimality is not guaranteed.
start_time = 0
timeline3 = []
def RankedEnumSeps(G, event, minimal_only=0):
    global timeline3
    global start_time3
    start_time3 = time.time()
    s = G.graph['st'][0]
    t = G.graph['st'][1]
    S = minimum_st_node_cut(G, s, t)
    min_weight = len(S)
    result = []
    timeline3 = []
    Q = g_queue()
    Q.push(G, S, set(), 1)

    while not Q.isempty():
        (w, (G, S, I, valid)) = Q.pop()
        #if minimal_only: # filter out the non-minimal separators
        #    notmin = 0
        #    for u in S:
        #        h2 = g.copy()
        #        h2.remove_nodes_from(S - {u})
        #        if not nx.has_path(h2, s, t):
        #            notmin = 1
        #            break
        #    if valid == notmin:
        #        print("Valid check failed", S)
        #        exit(1)
        #    if notmin == 0:
        #        result.append(S)
        if valid == 1:
            result.append(S)
        else: # save all including the non-minimal separators
            result.append(S)
        mid = time.time()
        # print(f"Found at time {mid-start_time} [sec]: \n{S}")
        timeline3.append(mid - start_time)
        v = list(set(S) - I)
        for i in range(len(v)):
            if event.is_set():
                print("Event is set, stopping enumeration")
                Q.clear() # clear the queue to stop the enumeration
                break
            Ii = I | set(v[:i])
            if G.has_edge(v[i], s) and G.has_edge(v[i], t):
                continue  # this case will lead to an non s-t separable saturated graph
            else:
                H = saturate(G, v[i])
                T = minimum_st_node_cut(H.subgraph(H.nodes() - Ii), s, t)
                valid = 1
                if minimal_only:
                    if len(T) + len(Ii) != nx.node_connectivity(H):
                        valid = 0  # T is not a minimal separator
                S_temp = set(T) | Ii
                Q.push(H, S_temp, Ii, valid)
    total_time = time.time() - start_time3
    print(f"Return 3 after {total_time} sec")
    return result, timeline3, total_time


# This is an exhaustive enumeration algorithm just for comparison.
# It is useful for small graphs with up to 15 nodes. Set K to some number smaller than
# the number of nodes to limit the size of the returned separators. Setting K=0(default)
# will cause all separators to be returned
def enumeration_exhaustive(g, s, t, K=0):
    S_sep = set()  # set of all separators
    nodes = set(g.nodes()) - {s, t}
    if K <= 0:
        max_set_size = len(nodes)
    else:
        max_set_size = min(K, len(nodes))
    for k in range(max_set_size):
        for subset in combinations(nodes, k + 1):
            includes_a_separator = 0
            for sep in S_sep:
                if set(sep).issubset(subset):  # subset includes a separator found before
                    includes_a_separator = 1
                    break  # since subset size is non-decreasing, it is not possible that it is included in s
            if not includes_a_separator:
                h_nodes = (nodes - set(subset)) | {s, t}
                h = nx.induced_subgraph(g, h_nodes)
                if not nx.has_path(h, s, t):
                    S_sep.add(subset)  # set union
    Q = []
    pc = 0  # push counter (see explanation in utils module)
    for x in S_sep:
        pc += 1
        w = sum_weights(g, x)  # calculate weight
        heapq.heappush(Q, (w, pc, x))  # push sorted by weight and pc

    result_list = []
    while Q != []:
        (w, pc, subset) = heapq.heappop(Q)
        result_list.append(set(subset))
    return result_list

def all_seps_exhaustive(g, s, t):
    S_sep = set()  # set of all separators
    nodes = set(g.nodes()) - {s, t}
    max_set_size = len(nodes)
    for k in range(max_set_size):
        for subset in combinations(nodes, k + 1):
            h_nodes = (nodes - set(subset)) | {s, t}
            h = nx.induced_subgraph(g, h_nodes)
            if not nx.has_path(h, s, t):
                S_sep.add(subset)  # set union
    return S_sep

# This is a version for a minimal s=t separator closest to a set A which contains s
def MinimalstSepCloseToA(G, A):
    s = G.graph['st'][0]
    t = G.graph['st'][1]
    if s not in A: # A must contain s for this case
        return set()
    Na = set()
    for x in A:
        Na |= set(nx.neighbors(G, x))  # set union
    Na = Na - A
    S1 = set(nx.nodes(G))-Na
    Ct = nx.node_connected_component(nx.subgraph(G, S1), t)
    Nc = set()
    for x in Ct:
        Nc |= set(nx.neighbors(G, x))  # set union
    Nc = Nc - Ct
    return Nc


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
    #return ford_fulkerson_bfs.fordfulkerson_clib_local(r_graph, s, t, K)
    return fordfulkerson_clib(r_graph, s, t, K)  # call the C function

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
    for (u, v) in g.edges():
        g_dup.add_edge(-u-1, v+1, weight=1)
        g_dup.add_edge(-v-1, u+1, weight=1)

    s = s + 1
    t = t + 1
    cut = fordfulkerson(g_dup, s, t, K)
    return cut

# Find the important smallest st-separator in G
# Based on the proof of lemma 10 in Michal Wlodarczyk Lecture 9 (7.12.2012)
# Note that the role of s and t is exchanged due to the different way importance is defined in the paper
# NOte that finding the important smallest separator is already acheived by finding the minimum cut using the Ford
# Fulkerson algorithm with BFS for path finding
def important_smallest(G, s, t, K):
    if (s,t) in G.edges() or (t,s) in G.edges():
        return None
    S = min_st_v_cut(G, s, t, K)  # Find the min vertex cut that is closer to s
    return S # Using the assumption that the Ford Fulkerson algorithm already finds the important smallest cut


# This function is based on the recursive pseudocode from Michal Wlodarczyk Lecture 9 (7.12.2012)
# the algorithm returns all important separator but also non-important separators with sizes similar to important separators.
# Note that the roll of s and t is exchanged due to the different way importance is defined in the paper
gen_cuts_recursive_result = []
def gen_cuts_recursive(G, S, s, t, Z, K):
    # Note that S is the important smallest separator if this is a first branch (Branch 1)
    # visualize_g(G)
    if S == []: # is S is empty it is the first call or right branch (Branch 2)
        S = important_smallest(G, s, t, K) # The we need to find the important smallest separator
    if S == None:
        return
    elif K == 0 or S == set():
        gen_cuts_recursive_result.append(frozenset(Z))
    else:
        Ct = nx.node_connected_component(nx.subgraph(G, set(nx.nodes(G)) - S), t)
        v = list(S)[0]  # arbitrary node from S
        G1 = contract_nodes(G, Ct, t)  # Branch 1: contract Ct(S) into t (this may create non-minimal separators)
        G2 = nx.contracted_edge(G1, (t, v), self_loops=False)  # Branch 2: contract {t,v} into t
        G1.remove_node(v)
        gen_cuts_recursive(G1, S-{v}, s, t, Z | {v}, K - 1) # Branch 1- S-{v} is still going to be the important smallest separator
        gen_cuts_recursive(G2, [], s, t, Z, K) # Branch 2- empty S means that we should look for the important smallest separator again


# Enumerates all important st-separators which size does not exceed K. Uses
# the function gen_cuts_recursive which is based on an adaptation of the recursive pseudocode
# from Michal Wlodarczyk Lecture 9 (7.12.2012).
def gen_important_small_seps(G, s, t, K):
    global gen_cuts_recursive_result
    gen_cuts_recursive_result = []
    gen_cuts_recursive(G, [], s, t, set(), K)
    # Filter out the non-minimal separators. This is nessesary since branch 1 in the recursive algorithm may create them.
    non_minimal_sep = utils.find_non_minimal_st_sep(G, gen_cuts_recursive_result)
    for S in non_minimal_sep[0]:  # This may happen, non minimals should be removed
         gen_cuts_recursive_result.remove(S)
    # Filter out the non-important separators
    return gen_cuts_recursive_result


# enumerate all minimal separators (not ranked) with cardinality smaller or equal to K
# it is assumed that s and t are not in E(G).
# The function can be used with multiprocessing by providing a Pipe connection as output_conn. In this case,
# each found separator S is sent via output_conn.send(S) and a sentinel None is sent at the end. In multiprocessing mode
# all prints are disabled to avoid mixing logs with the parent process. The default use
# is to run without multiprocessing, and return the list of separators.
start_time = 0
timeline1 = []
def SmallMinimalSeps(G, K, event, output_conn=None):
    global timeline1
    global start_time
    timeline1 = []
    result = []
    Q = g1_queue(G)
    ClosestTos_separators = gen_important_small_seps(G, G.graph['st'][0], G.graph['st'][1] , K)
    if ClosestTos_separators == None:
        return None, None
    for S in ClosestTos_separators:
        Q.push(S)
    while not Q.isempty():
        x, S = Q.pop()
        # Before adding the result check for duplications
        if S in result:
            continue
        if output_conn is not None: # this is the case where the algorithm is used with multiprocessing and each S is sent via a Pipe
            try:
                output_conn.send(S)
            except Exception:
                pass # not falling back to appending to result or loging for now
        else:
            result.append(S)

        mid = time.time()
        timeline1.append(mid-start_time)
        Hs = G.copy()
        Cs = nx.node_connected_component(nx.subgraph(Hs, set(nx.nodes(Hs)) - S), G.graph['st'][0])
        Hs.remove_nodes_from(Cs - {G.graph['st'][0]})
        edges = list(product(S, {G.graph['st'][0]}))  # add edges connecting nodes in S to s
        Hs.add_edges_from(edges)
        for v in S:
            Hs_v = Hs.copy()
            edges_v = list(product(set(nx.neighbors(Hs, v)) - {G.graph['st'][0]}, {G.graph['st'][0]}))  # add edges connecting nodes in NH(v) to s
            Hs_v.add_edges_from(edges_v)
            ClosestTos_separators_v = gen_important_small_seps(Hs_v, Hs_v.graph['st'][0], Hs_v.graph['st'][1], K)
            if ClosestTos_separators_v == None:
                continue
            for T in ClosestTos_separators_v:
                Q.push(T, 1) # second argument = 1 means don't push if Sv is in already the queue
            if event.is_set():
                if output_conn is None: # only print if not using multiprocessing
                    print("Event is set, stopping enumeration")
                Q.clear() # clear the queue to stop the enumeration
                break

    total_time = time.time() - start_time
    if output_conn is None: # only print if not using multiprocessing
        print(f"Return 1 after {total_time} sec")
    else: # close the connection (child end) if it has close()
        try:
            output_conn.send(None) # send sentinel to indicate completion
        except Exception:
            pass
        try:
            output_conn.close() # close the connection (child end) if it has close()
        except Exception:
            pass
    return result, timeline1, total_time

# ListMinSepRecursive: Given a graph G, nodes s and t and sets A and U, find all s-t separators S such that
# A is a subset of Cs(S) (the connected component of G-S that includes s) and there is no node in Cs(S) whcih is also in U
# (the intersection of U and Cs(S) is empty).
# As a result of this definition, when calling ListMinSepRecurs(A, U) with A = {s} and U = Neighbors(t), the result will be
# all s-t separators of G. The algorithm is from Ken Takata- "Spaec-optimal, backtracking algorithms to list the minimal vertex seoarators of a graph"
timeline2 = []
over = 0
ListMinSepRecursive_result = []

def ListMinSepRecursive(G, A, U, event):
    global timeline2
    global over
    global start_time2
    if over or event.is_set():
        over = 1
        return
    SA = MinimalstSepCloseToA(G, A)
    A_star = set(nx.node_connected_component(nx.subgraph(G,set(nx.nodes(G))-SA), G.graph['st'][0]))
    if A_star.intersection(U) == set():
        A = A_star.copy()
        NAlessU = set()
        for x in A:
            NAlessU |= set(nx.neighbors(G, x)) - U
        NAlessU = NAlessU - A
        if len(NAlessU) > 0:
            v = NAlessU.pop()
            ListMinSepRecursive(G, A|{v}, U, event)
            ListMinSepRecursive(G, A, U|{v}, event)
        else:
            t = time.time()-start_time2
            timeline2.append(t)
            ListMinSepRecursive_result.append(SA) # output to list

def ListMinSepTakata(G, event):
    global ListMinSepRecursive_result
    global start_time2
    start_time2 = time.time()
    s = G.graph['st'][0]
    t = G.graph['st'][1]
    Nt = set(nx.neighbors(G, t))
    ListMinSepRecursive_result = []
    ListMinSepRecursive(G, {s}, Nt, event)
    total_time = time.time() - start_time2
    print(f"Return 2 after {total_time} sec")
    return ListMinSepRecursive_result, timeline2, total_time
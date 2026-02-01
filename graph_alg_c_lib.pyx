# To compile run in terminal:
# python setup.py build_ext --inplace
from libcpp.unordered_map cimport unordered_map
from libcpp.queue cimport queue
from cpython cimport array
from libc.limits cimport INT_MAX


def debug_print_nodes_clib(int[:] glist):
    cdef int num_nodes
    cdef int u_index
    cdef int v
    num_nodes = glist[0]
    u_index = 1
    nodes = []
    while u_index < len(glist):
        v = glist[u_index]
        nodes.append(v)
        u_index += glist[u_index + 1] * 2 + 2
    print(nodes)


# find an augmenting path in the residual graph using BFS
cdef bfs_find_path_clib(int[:] glist, int s, int t, unordered_map[int, int]& parent_dict, array.array nodes):
    #cdef int node
    cdef int node_index = 0
    cdef int adg_len
    cdef int i
    cdef int u
    cdef int u_index
    cdef int val
    cdef unordered_map[int, int] visited_dict = dict.fromkeys(nodes, 0)
    cdef queue[int] q
    q.push(s)
    visited_dict[s] = 1

    while not q.empty():
        u = q.front()
        q.pop()
        u_index = 1
        #print(u)
        while u_index < len(glist):
            v = glist[u_index]
            #print(u_index, v)
            if v == u:
                break
            u_index += glist[u_index + 1] * 2 + 2
        if u_index + 1 >= len(glist):
            print("Error: node not found") # this should not happen
            print(u)
        adg_len = glist[u_index + 1]  # get the number of successor nodes of u
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
            if glist[u_index + 2 + i * 2 + 1] < capacity: # check if the flow is less than the capacity
                q.push(val)
                visited_dict[val] = 1
                parent_dict[val] = u
                if val == t:
                    return 1
    return 0


cdef bfs_find_reachable_clib(int[:] glist, int s, array.array nodes):
    #cdef int node
    cdef int node_index = 0
    cdef int adg_len
    cdef int i
    cdef int u
    cdef int u_index
    cdef int val
    cdef int num_nodes
    num_nodes = glist[0]

    cdef unordered_map[int, int] visited_dict = dict.fromkeys(nodes, 0)
    #cdef unordered_map[int, int] parent_dict = dict.fromkeys(nodes, 0) # a node label can't be 0
    cdef queue[int] q
    q.push(s)
    visited_dict[s] = 1
    while not q.empty():
        u = q.front()
        q.pop()
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
                capacity = 1 if val == -u else INT_MAX  # if the edge is between a node pair we set the capacity to 1
            else:
                capacity = 0
            if glist[u_index + 2 + i * 2 + 1] < capacity:  # check if not saturated
                q.push(val)
                visited_dict[val] = 1
    cdef array.array true_visited = array.array('i', [0] * num_nodes)
    i = 0
    for u in range(num_nodes):
        if visited_dict[nodes[u]] == 1:
            true_visited[i] = nodes[u]
            i += 1
    cdef array.array visited = array.array('i', true_visited[:i])
    return visited

def path_dict_to_list_clib(unordered_map[int, int] parent_dict, s, t):
    cdef int v = t
    cdef list path = []
    while v != s:
        u = parent_dict[v]
        path.append((u, v))
        v = u
    return path

# Ford-Fulkerson algorithm- k is the maximum number of augmenting paths. if the number of augmenting paths found is
# less than or equal to k, the algorithm stops and returns the vertex cut. Otherwise, it returns None.
def fordfulkerson_clib(int [:] r_graph, int s, int t, int K):

    cdef int node
    cdef int node_index
    cdef int i
    cdef int j
    cdef int v
    cdef int num_nodes
    cdef int num_paths = 0

    num_nodes = r_graph[0]
    cdef array.array nodes = array.array('i', [0] * num_nodes)
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
    cdef array.array degrees = array.array('i', [0] * num_nodes)
    cdef unordered_map[int, int] parent_dict = dict.fromkeys(nodes, 0) # a node label can't be 0
    while 1:
        num_paths += 1
        path_found = bfs_find_path_clib(r_graph, s, t, parent_dict, nodes)
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
                if node == u:  # update forward edge flow
                    for i in range(r_graph[node_index + 1]):  # scan in forward edges
                        if r_graph[node_index + 2 + i * 2] == v:
                            r_graph[node_index + 2 + i * 2 + 1] += 1  # increase the flow over the edge
                            flows_updated += 1
                            break
                if node == v:  # update reverse edge flow
                    for i in range(r_graph[node_index + 1]):  # scan in forward edges
                        if r_graph[node_index + 2 + i * 2] == u:
                            r_graph[node_index + 2 + i * 2 + 1] -= 1  # decrease the flow over the edge
                            flows_updated += 1
                            break
                if flows_updated == 2:  # we updated both edges
                    break
                node_index += r_graph[node_index + 1] * 2 + 2
            v = u
    reachables = bfs_find_reachable_clib(r_graph, s, nodes)
    cut = set()
    for i in range(len(reachables)):
        u = reachables[i]
        node_index = 1
        while node_index < len(r_graph):
            node = r_graph[node_index]
            if node == u:
                for j in range(r_graph[node_index + 1]):
                    v = r_graph[node_index + 2 + j * 2]
                    if v == -u and u > 0:  # we are only interested in the edges from original nodes to duplicated nodes
                        if v not in reachables:
                            cut.add((-1 - v))  # we subtract 1 to map the duplicated nodes to the original nodes
                        break
                break
            node_index += r_graph[node_index + 1] * 2 + 2
    return cut
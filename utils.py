import random
import networkx as nx
import matplotlib.pyplot as plt
import heapq
from itertools import combinations
import time
import lzma


def reset_random_seed(seed):
    global random
    random.seed(123+seed)
# for reproducibility


time_is_up = 0
def is_timeout(start_time, maxtime):
    if time.time() - start_time > maxtime:
        return 1
    else:
        return 0


# Stop the timeout manager by setting the event from another process
def timeout_manager_stop(event):
    event.set()


# Check if timeout has occurred by checking the event- this can be used from another process
def check_timeout(event):
    return event.is_set()


def timeout_manager(event, start_time, maxtime):
    while is_timeout(start_time, maxtime) == 0:
        if event.is_set(): # check if event is already set by another process usign timeout_manager_stop
            return
        time.sleep(1)
    print("Event set")
    event.set()


def import_graph_from_gr(path):
    if path.endswith('.xz'):
        f1 = lzma.open(path, 'rt')
    elif path.endswith('.gr'):
        f1 =  open(path, 'r')
    else:
        return -1
    g=nx.Graph()
    e = []
    data_started = 0
    for line in f1:
        if line.startswith('c'):
            continue
        if data_started:
            s = line.split(' ')
            if len(s) == 2:
                if s[0].strip().isdigit() and s[1].strip().isdigit():
                    e.append((int(s[0].strip()), int(s[1].strip())))
        else:
            if line.startswith('p'):
                data_started = 1

    f1.close()
    g.add_edges_from(e)
    return g


def visualize_g(g):
    node_color = ['white'] * g.number_of_nodes()
    node_color[list(g.nodes()).index(g.graph['st'][0])] = 'magenta'
    node_color[list(g.nodes()).index(g.graph['st'][1])] = 'green'
    pos = nx.spring_layout(g, seed=5)
    nx.draw_networkx(g, pos=pos, node_color=node_color)
    plt.show()

def sum_weights(g, nodes):
    weighted = g.graph['weighted']
    if weighted == 0:
        sum_w = len(nodes)
    else:
        sum_w = 0
        for n in nodes:
            sum_w += g.subgraph(nodes).nodes.data()[n]['weight']
    return sum_w

first_reset = True
def randomize_graph(num_nodes, edge_prob, seed):
    seed_internal = seed+123
    global first_reset
    if first_reset:
        g = nx.fast_gnp_random_graph(num_nodes, edge_prob, seed=seed_internal)
        if nx.is_connected(g):
            print("Graph is connected")
        else:
            print("Graph is not connected")
        print(f"Resetting graph seed to {seed_internal}")
        first_reset = False
        while not (nx.is_connected(g) and (g.number_of_edges() < num_nodes * (num_nodes - 1) / 2)):  # make sure graph is connected but not complete
            g = nx.fast_gnp_random_graph(num_nodes, edge_prob)
    else:
        while not (nx.is_connected(g) and (g.number_of_edges() < num_nodes * (num_nodes - 1) / 2)):  # make sure graph is connected but not complete
            g = nx.fast_gnp_random_graph(num_nodes, edge_prob)
    if not nx.is_connected(g):
        print("Error: Graph is not connected")
        exit(-1)
    return g

def add_weights(g, method):
    if method == 'random':
        for k in range(g.number_of_nodes()):
            g.nodes[k]['weight'] = random.random()
        return g

def randomize_st(g):
    list_of_nodes = list(nx.nodes(g))
    while 1:
        [si, ti] = random.sample(range(len(list_of_nodes)), 2)
        if not g.has_edge(list_of_nodes[si], list_of_nodes[ti]):
            return [list_of_nodes[si], list_of_nodes[ti]]

def gen_all_possible_st(g):
    if not nx.is_connected(g):
        return []
    possible_st = []
    for st in combinations(g.nodes, 2):
        if not g.has_edge(st[0], st[1]):
            possible_st.append(st)
    return possible_st

# This is an implementation of the sorted queue, which is based on the heapq library
# priority queue. The original heapq requires a secondary value to be used in case of
# a tie in the queue values.
# To make this transparent to the user, we add a push counter as a secondary value.
# Since the push counter (pc) advances on every push (and only then) it will always resolve
# to pop the element that was pushed earlier (fifo).
# Note the there is no limit on the number of push operations, therefore, the value of
# pc is unlimited. Python 3 does not require that an integer will have a limited value.

# This version is for the separator enumeration algorithm RankedEnumSeps
class g_queue():
    def __init__(self):
        self.pc = int(0)
        self.q = []
    def clear(self):
        # reinitialize internal state to empty
        self.__init__()

    def push(self, G, S, I, valid):
        w = sum_weights(G, S)
        self.pc += 1
        heapq.heappush(self.q, (w, self.pc, (G, S, I, valid)))

    def pop(self):
        # the element with the lowest w will be popped, in case of tie pc will be used to resolve
        # The lowest pc will be popped which means the element that was pushed earlier will be popped
        (w, pc, (G, S, I, valid)) = heapq.heappop(self.q)
        return (w, (G, S, I, valid))

    def isempty(self):
        return self.q == []

# This version is for the small minimal separator enumeration algorithm SmallMinimalSeps
# In this type of queue G is not part of the element but a property of the queue
class g1_queue():
    def __init__(self, G = None):
        self.pc = int(0)
        self.q = []
        self.graph = G

    def clear(self):
        # reinitialize internal state to empty, with no graph (useful for flushing the queue before exiting)
        self.__init__()

    def push(self, S, ignore_existing = 0): # set ignore_existing = 1 to avoid pushing a set that is already in the queue
        # The order used here is a partial order where S < T iff Cs(S) is included in Cs(t)
        # However since we are only interested in cases where S and T are comparable and not equal, if we
        # make sure that the priority of S will be smaller than the priority of T whenever |S| < |T|
        # the priority will hold for any comparable S and T. When |S|=|T| they must be incomparable
        # (otherwise they are equal) and therefore any secondary priority (such as order of arrival) will do.
        if ignore_existing:
            for item in self.q:
            #for k in range(len(self.q)):
                #item = self.q[k]
                if S == item[2]:
                    return # S is already in the queue
        H = nx.subgraph(self.graph, self.graph.nodes()-S)
        Cs = nx.neighbors(H, self.graph.graph['st'][0])
        w = len(list(Cs)) + 1 # s is always in Cs
        self.pc += 1
        heapq.heappush(self.q, (w, self.pc, S))

    def pop(self):
        # the element with the lowest w will be popped, in case of tie pc will be used to resolve
        # The lowest pc will be popped which means the element that was pushed earlier will be popped
        (w, pc, S) = heapq.heappop(self.q)
        return (w, S)

    def isempty(self):
        return self.q == []

# if check_not_sep = 1 then the function will return a list of sets that are not separators
# Otherwize it will return a list of sets that are separators but not minimal
def find_non_minimal_st_sep(g, separators, check_not_sep = 0):
    #result = {'NOT_MIN': [], 'NOT_SEP': []}
    result = []
    result_not_sep = []
    nonmin_index = []
    n = 0
    for sep in separators:
        h1 = g.copy()
        h1.remove_nodes_from(sep)
        if check_not_sep == 1:
            if nx.has_path(h1, g.graph['st'][0], g.graph['st'][1]):
                 result_not_sep.append(sep)  # not a separator
        for v in sep:
            h2 = g.copy()
            h2.remove_nodes_from(sep-{v})
            if not nx.has_path(h2, g.graph['st'][0], g.graph['st'][1]):
                #result['NOT_MIN'].append(sep) # not minimal (sep-v is a separator)
                result.append(sep) # not minimal
                nonmin_index.append(n)
                break
        n += 1
    return [result, result_not_sep, nonmin_index]


def bail(g, tstart):
    nx.write_gml(g, "last_g.gml")  # save for next time
    print("graph saved to file: last_g.gml")
    end = time.time()
    print(f'Run time: {end - tstart} seconds')
    exit(-1)
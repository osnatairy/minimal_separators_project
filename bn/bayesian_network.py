import networkx as nx
from itertools import product
from copy import deepcopy

class BN:
    def __init__(self):
        self.g = nx.DiGraph()
        self.domains = {}         # var -> [values]
        self.cpts = {}            # var -> { parent_tuple -> { value: prob } }
        self.parent_order = {}    # var -> [parents in order]

    def add_var(self, name, domain):
        """Add a variable node and its finite domain."""
        self.g.add_node(name)
        self.domains[name] = list(domain)
        self.cpts[name] = {}
        self.parent_order[name] = []

    def add_edge(self, parent, child):
        """Add a directed edge parent -> child; ensure we stay a DAG."""
        if parent not in self.g or child not in self.g:
            raise ValueError("Add variables before adding edges.")
        self.g.add_edge(parent, child)

        # ensure acyclic
        if not nx.is_directed_acyclic_graph(self.g):
            self.g.remove_edge(parent, child)
            raise ValueError(f"Adding {parent}->{child} creates a cycle.")

    def set_parent_order(self, var, order):
        """Fix the order of parents used to index CPT rows for `var`."""
        preds = list(self.g.predecessors(var))
        if set(order) != set(preds):
            raise ValueError(f"Parent order for {var} must be a permutation of predecessors {preds}.")
        self.parent_order[var] = list(order)

    def set_cpt(self, var, cpt, strict=True):
        """
        cpt: { parent_tuple -> { value: prob } }, keyed by parent_order[var].
        - Checks: keys match domains, rows sum to 1.
        - If strict=True: require a row for every combination of parent values.
        """
        if var not in self.g:
            raise ValueError(f"Unknown variable {var}.")
        # default a stable parent order if user forgot (optional)
        if not self.parent_order[var]:
            self.parent_order[var] = sorted(self.g.predecessors(var))

        # optional strict coverage check: ensure a row for every parent combo
        parents = self.parent_order[var]
        if strict:
            all_parent_keys = list(product(*[self.domains[p] for p in parents])) if parents else [()]
            missing = [pt for pt in all_parent_keys if pt not in cpt]
            if missing:
                raise ValueError(f"CPT for {var} missing rows for parent assignments: {missing}")

        # per-row checks
        for pkey, row in cpt.items():
            if len(pkey) != len(parents):
                raise ValueError(f"CPT key length mismatch for {var}: {pkey} vs parents {parents}")
            # parent values in-domain
            for i, val in enumerate(pkey):
                p = parents[i]
                if val not in self.domains[p]:
                    raise ValueError(f"CPT for {var}: parent value {val} not in domain of {p}")
            # row covers child domain & normalized
            if set(row.keys()) != set(self.domains[var]):
                raise ValueError(f"CPT row for {var} must define probs for all values in {self.domains[var]}")
            s = sum(row.values())
            if abs(s - 1.0) > 1e-9:
                raise ValueError(f"CPT row for {var} {pkey} not normalized (sum={s})")

        self.cpts[var] = cpt

    def parents(self, var):
        return list(self.parent_order[var]) if self.parent_order[var] else list(self.g.predecessors(var))

    def _parent_tuple(self, var, assignment):
        """Build the tuple of parent values in the fixed order for CPT lookup."""
        return tuple(assignment[p] for p in self.parents(var))

    def _enumerate_assignments(self, vars_subset):
        """Yield dicts over the Cartesian product of domains for `vars_subset`."""
        vars_subset = list(vars_subset)
        for values in product(*[self.domains[v] for v in vars_subset]):
            yield dict(zip(vars_subset, values))

    def conditional_prob(self, var, value, assignment):
        """P(var=value | parents(var) = assignment[parents]) via CPT lookup."""
        pkey = self._parent_tuple(var, assignment)
        return self.cpts[var][pkey][value]

    def joint_prob(self, full_assignment):
        """P(assignment) = ∏_v P(v | pa(v)) for a COMPLETE assignment dict."""
        # Iterate in topological order (not required, but nice)
        import networkx as nx
        prob = 1.0
        for v in nx.topological_sort(self.g):
            prob *= self.conditional_prob(v, full_assignment[v], full_assignment)
        return prob

    def marginal_prob(self, fixed):
        """Sum joint over all other variables: P(fixed)."""
        others = [v for v in self.g.nodes() if v not in fixed]
        total = 0.0
        for assign in self._enumerate_assignments(others):
            total += self.joint_prob({**fixed, **assign})
        return total

    def conditional(self, target, given):
        """P(target | given) = P(target, given) / P(given). dicts must be disjoint."""
        if set(target) & set(given):
            raise ValueError("target and given must be disjoint.")
        num = self.marginal_prob({**target, **given})
        den = self.marginal_prob(given)
        if den == 0.0:
            raise ZeroDivisionError("P(given)=0; conditional undefined.")
        return num / den

    def domain(self, var: str):
        """Return a copy of the domain list for `var`."""
        if var not in self.domains:
            raise ValueError(f"Unknown variable '{var}'.")
        return list(self.domains[var])

    def remove_incoming_edges_do(self, target_node: str, init_value=None) -> "bn":
        """
        מחזירה bn חדש לאחר 'התערבות הכנה' על target_node:
        - מסירה את כל הקשתות הנכנסות ל-target_node (סמנטיקת do).
        - מגדירה ל-target_node CPT דטרמיניסטי (דלתא) ללא הורים:
            ערך בודד מקבל 1 וכל השאר 0.
          ברירת מחדל (אם init_value=None): הערך הראשון ב-domain של הצומת.

        למה לא לשים אפסים בלבד?
        - set_cpt שלך דורשת סכום=1 בכל שורה. לכן צריך דלתא חוקית, לא 'אפסים'.

        פרמטרים:
            target_node : שם הצומת (למשל "X")
            init_value  : הערך שיקבל 1 בעת ההכנה; אם None -> הערך הראשון בדומיין.

        החזרה:
            bn חדש (עותק) עם גרף מעודכן ו-CPT דלתא על target_node.
        """
        if target_node not in self.g:
            raise ValueError(f"Unknown node '{target_node}'.")

        # 1) בונים bn חדש ומעתיקים משתנים/דומיינים
        bn_new = BN()
        for v in self.g.nodes():
            bn_new.add_var(v, self.domains[v])

        # 2) מעתיקים כל קשת, מלבד כאלה שנכנסות ל-target_node
        for u, v in self.g.edges():
            if v != target_node:
                bn_new.add_edge(u, v)

        # 3) מעתיקים סדר הורים + CPTs לצמתים שאינם היעד
        for v in self.g.nodes():
            if v == target_node:
                continue
            pa_order = self.parents(v)
            bn_new.set_parent_order(v, pa_order)
            bn_new.set_cpt(v, deepcopy(self.cpts[v]), strict=False)

        # 4) יעד: אין הורים -> CPT ללא הורים (דלתא)
        bn_new.set_parent_order(target_node, [])
        vals = self.domains[target_node]
        if not vals:
            raise ValueError(f"Node '{target_node}' has empty domain.")

        if init_value is None:
            init_value = vals[0]  # ברירת מחדל: הערך הראשון בדומיין

        if init_value not in vals:
            raise ValueError(f"init_value {init_value!r} not in domain of {target_node}: {vals}")

        delta_row = {val: (1.0 if val == init_value else 0.0) for val in vals}
        bn_new.set_cpt(target_node, {(): delta_row}, strict=False)

        # 5) בדיקת DAG
        if not nx.is_directed_acyclic_graph(bn_new.g):
            raise RuntimeError("Resulting bn is not a DAG (unexpected).")

        return bn_new

    def set_do_value(self, target_node: str, value, inplace: bool = True) -> "bn":
        """
        מעדכנת את ה-CPT של target_node לדלתא: P(target_node=value)=1 וכל השאר 0.
        מצופה שהצומת כבר *ללא הורים* (למשל לאחר remove_incoming_edges_do).

        פרמטרים:
            target_node : שם הצומת (למשל "X")
            value       : הערך שיקבל הסתברות 1
            inplace     : אם True (ברירת מחדל) – מעדכן את העצם הנוכחי.
                          אם False – מחזיר עותק bn חדש עם השינוי.

        החזרה:
            אם inplace=True -> מחזיר self (נוח לשרשור).
            אם inplace=False -> מחזיר bn חדש עם השינוי.
        """
        if target_node not in self.g:
                raise ValueError(f"Unknown node '{target_node}'.")

        vals = self.domains[target_node]
        if value not in vals:
            raise ValueError(f"value {value!r} not in domain of {target_node}: {vals}")

        target_bn = self if inplace else deepcopy(self)

        # ודאו שאין הורים (לא חובה מתמטית לקבוע דלתא, אבל כך עקבי עם do)
        if list(target_bn.g.predecessors(target_node)):
            # לא חובה לשבור כאן, אבל זה סימן שה-do לא הוכן
            # אפשר להסיר אוטומטית:
            for u in list(target_bn.g.predecessors(target_node)):
                target_bn.g.remove_edge(u, target_node)
            target_bn.set_parent_order(target_node, [])

        delta_row = {val: (1.0 if val == value else 0.0) for val in vals}
        target_bn.set_cpt(target_node, {(): delta_row}, strict=False)

        return target_bn

    def graph_without_incoming(self, target_node: str) -> nx.DiGraph:
        """
        מחזיר גרף חדש (DiGraph) עם כל הצמתים והקשתות,
        פרט להסרת כל הקשתות הנכנסות ל-target_node.
        (לא משנה CPTs; רק גרף.)
        """
        if target_node not in self.g:
            raise ValueError(f"Unknown node '{target_node}'.")
        G = nx.DiGraph()
        G.add_nodes_from(self.g.nodes())
        G.add_edges_from([(u, v) for (u, v) in self.g.edges() if v != target_node])
        return G

'''
# build a tiny bn: Z -> X -> Y and Z -> Y (all binary)
bn = bn()
bn.add_var("Z", [0,1])
bn.add_var("X", [0,1])
bn.add_var("Y", [0,1])

bn.add_edge("Z","X")
bn.add_edge("X","Y")
bn.add_edge("Z","Y")

bn.set_parent_order("Z", [])
bn.set_parent_order("X", ["Z"])
bn.set_parent_order("Y", ["X","Z"])

bn.set_cpt("Z", {(): {0:0.7, 1:0.3}})
bn.set_cpt("X", {(0,): {0:0.8, 1:0.2},
                 (1,): {0:0.2, 1:0.8}})
bn.set_cpt("Y", {(0,0): {0:0.9, 1:0.1},
                 (0,1): {0:0.3, 1:0.7},
                 (1,0): {0:0.4, 1:0.6},
                 (1,1): {0:0.1, 1:0.9}})

assert nx.is_directed_acyclic_graph(bn.g)

# Check a couple of probabilities
print("P(Z=1) =", bn.marginal_prob({"Z":1}))
print("P(Y=1 | X=1, Z=1) =", bn.conditional({"Y":1}, {"X":1, "Z":1}))
print("P(Y=1 | X=1) =", bn.conditional({"Y":1}, {"X":1}))
'''
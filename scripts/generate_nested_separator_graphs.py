import matplotlib.pyplot as plt
from pathlib import Path
import networkx as nx
import random
import os

from graph.separators_bruteforce import get_minimal_separators
from analysis.adjustment_hasse import cy_component

if __name__ == "__main__":
    print("Searching for graphs with nested separator property...")

    attempts = 0
    found_count = 0

    output_dir = "outputs/nested_graphs"
    os.makedirs(output_dir, exist_ok=True)

    while found_count < 3 and attempts < 10000:
        attempts += 1
        n = random.randint(6, 9)
        p = 0.4
        G = nx.fast_gnp_random_graph(n, p)

        if not nx.is_connected(G):
            continue

        # Iterate all non-adjacent pairs
        pairs = []
        for u in G.nodes():
            for v in G.nodes():
                if u < v and not G.has_edge(u, v):
                    pairs.append((u, v))

        graph_found = False
        for u, v in pairs:
            seps = get_minimal_separators(G, u, v)
            if len(seps) < 2:
                continue

            # Check nested condition
            for i in range(len(seps)):
                for j in range(len(seps)):
                    if i == j: continue
                    S1 = seps[i]
                    S2 = seps[j]

                    C1 = cy_component(G, u, S1)
                    C2 = cy_component(G, u, S2)

                    if C1 < C2:  # Strict subset: C1 is strictly contained in C2
                        print(f"\n[FOUND] Match at attempt {attempts}")
                        print(f"Graph Edges: {list(G.edges())}")
                        print(f"Pair: ({u}, {v})")
                        print(f"Separator S1: {S1}, Component(u): {C1}")
                        print(f"Separator S2: {S2}, Component(u): {C2}")
                        print(f"Is C1 < C2? {C1 < C2}")
                        found_count += 1

                        # Visualization
                        try:
                            pos = nx.spring_layout(G, seed=42)
                            fig = plt.figure(figsize=(12, 6))

                            # Plot 1: The Graph with S1 highlighted
                            ax1 = fig.add_subplot(1, 2, 1)
                            node_colors = []
                            for node in G.nodes():
                                if node in S1:
                                    node_colors.append('red')
                                elif node == u or node == v:
                                    node_colors.append('green')
                                elif node in C1:
                                    node_colors.append('orange')
                                else:
                                    node_colors.append('lightblue')

                            nx.draw(G, pos, ax=ax1, with_labels=True, node_color=node_colors)
                            ax1.set_title(f"S1={S1}, C1(u) Orange")

                            # Plot 2: The Graph with S2 highlighted
                            ax2 = fig.add_subplot(1, 2, 2)
                            node_colors = []
                            for node in G.nodes():
                                if node in S2:
                                    node_colors.append('red')
                                elif node == u or node == v:
                                    node_colors.append('green')
                                elif node in C2:
                                    node_colors.append('orange')
                                else:
                                    node_colors.append('lightblue')

                            nx.draw(G, pos, ax=ax2, with_labels=True, node_color=node_colors)
                            ax2.set_title(f"S2={S2}, C2(u) Orange (Should contain C1)")

                            # project_root = התיקייה מעל scripts/ (בהנחה שהסקריפט נמצא ב-minimal_separators_project/scripts/)
                            PROJECT_ROOT = Path(__file__).resolve().parents[1]

                            output_dir = PROJECT_ROOT / "output" / "nested_graphs"
                            output_dir.mkdir(parents=True, exist_ok=True)

                            filename = output_dir / f"nested_separator_example_{found_count}.png"

                            #filename = os.path.join(output_dir, f"nested_separator_example_{found_count}.png")
                            plt.savefig(filename)
                            print(f"Saved plot to {filename}")
                            plt.close(fig)
                        except Exception as e:
                            print("Plotting failed:", e)
                            import traceback
                            traceback.print_exc()

                        graph_found = True
                        break
                if graph_found: break
            if graph_found: break

    if found_count == 0:
        print("No graphs found in max attempts.")





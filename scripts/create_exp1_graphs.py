from experiments.create_buckets_graphs import make_bucket_boxplots


if __name__ == '__main__':

    nodes = [15]  # [20,30,40, 50]# different sizes of nodes in  a tree
    prob_nodes = [0.07, 0.1, 0.15, 0.2]  # the probability of an edge
    betas = [0.7]
    types= ["sem"]#,"sem"]

    for type in types:
        for node in nodes:
            for prob_node in prob_nodes:
                for k_roots in [node, int(node*0.3), 3, 1]:
                    variance = f"_main_{node}_{prob_node}_{k_roots}"#_beta07"

                    bucket_file = f"outputs_{type}/2026_04_13_bucket_statistics_{variance}.csv"
                    output_path = f"outputs_{type}/"+bucket_file.split("/")[-1].replace(".csv", "")
                    # make_bucket_boxplots(
                    #     bucket_file,
                    #     variance=variance,
                    #     #output_dir=f"bucket_statistics_SEM{variance}",
                    #     mode="global",
                    # )
                    make_bucket_boxplots(
                        bucket_file,
                        variance=variance,
                        output_dir=output_path,
                        mode="per_run",
                    )
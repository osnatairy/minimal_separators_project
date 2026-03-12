from experiments.create_buckets_graphs import make_bucket_boxplots


if __name__ == '__main__':


    nodes = [25, 50, 100]
    prob_nodes = [0.15,0.20, 0.25]#, 0.5, 0.7]
    betas = [0.7]

    for node in nodes:
        for prob_node in prob_nodes:
            for beta in betas:
                variance = f"SEM_{node}_{prob_node}_beta07"
                bucket_file = f"outputs/11_3_23_bucket_statistics_{variance}.csv"
                # make_bucket_boxplots(
                #     bucket_file,
                #     variance=variance,
                #     #output_dir=f"bucket_statistics_SEM{variance}",
                #     mode="global",
                # )
                make_bucket_boxplots(
                    bucket_file,
                    variance=variance,
                    # output_dir=f"bucket_statistics_SEM{variance}",
                    mode="per_run",
                )
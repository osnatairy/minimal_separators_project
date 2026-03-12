from experiments.create_buckets_graphs import make_bucket_boxplots, make_bucket_bar_plots,parse_bucket_stats_csv


nodes = [25, 50]
prob_nodes = ["025", "05", "07"]
betas = [0.7]


for node in nodes:
    for prob_node in prob_nodes:

        variance = f"SEM_{node}_{prob_node}_beta07"
        bucket_file = f"outputs/26_2_23_bucket_statistics_{variance}.csv"

        make_bucket_boxplots(
            bucket_file,
            variance,
            mode="global",
        )
        # Example:
        parse_bucket_stats_csv(
             f"outputs/26_2_23_bucket_statistics_{variance}.csv",
            variance,
            # agg="mean",
        )

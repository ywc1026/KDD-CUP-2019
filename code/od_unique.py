import pandas as pd


def get_data():
    train_queries = pd.read_csv("../data/data_set_phase2/train_queries.csv", parse_dates=['req_time'])
    test_queries = pd.read_csv("../data/data_set_phase2/test_queries.csv", parse_dates=['req_time'])

    data = pd.concat([train_queries, test_queries], ignore_index=True)
    return data


def get_unique(data, grouped_col, target_col):
    unique = data.groupby(grouped_col)[target_col].nunique().reset_index().rename(
        columns={target_col: "unique_{}".format(target_col)})
    result_sum = data.groupby(grouped_col, as_index=False)[target_col].agg({"{}_count".format(target_col): "count"})
    unique = unique.merge(result_sum, on=grouped_col, how="left")
    unique["{}_ratio".format(target_col)] = unique["unique_{}".format(target_col)].div(
        unique["{}_count".format(target_col)])
    return unique


def get_unique_feature():
    queries = get_data()
    o_unique = get_unique(queries, "o", "d")
    o_unique.to_csv("../output/o_unique.csv", index=False)
    d_unique = get_unique(queries, "d", "o")
    d_unique.to_csv("../output/d_unique.csv", index=False)
    queries["pid"] = queries["pid"].fillna(-1)
    o_pid_unique = get_unique(queries, "o", "pid")
    o_pid_unique.to_csv("../output/o_pid_unique.csv", index=False)
    d_pid_unique = get_unique(queries, "d", "pid")
    d_pid_unique.to_csv("../output/d_pid_unique.csv", index=False)
    queries["od"] = queries["o"] + queries["d"]
    od_pid_unique = get_unique(queries, "od", "pid")
    od_pid_unique.to_csv("../output/od_pid_unique.csv", index=False)


if __name__ == "__main__":
    get_unique_feature()


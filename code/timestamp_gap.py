import pandas as pd


def get_data():
    train_queries = pd.read_csv("../data/data_set_phase2/train_queries.csv", parse_dates=['req_time'])
    test_queries = pd.read_csv("../data/data_set_phase2/test_queries.csv", parse_dates=['req_time'])

    data = pd.concat([train_queries, test_queries], ignore_index=True)
    return data


def get_last_diff(data, cols, col_name):
    last_diff = data.groupby(cols, as_index=False)["req_time"].shift(1).rename(
        columns={"req_time": col_name})
    data = pd.concat([data, last_diff], axis=1)
    data[col_name] = (data[col_name] - data['req_time']).dt.seconds // 60
    data[col_name] = data[col_name].fillna(-1)
    return data


def get_next_diff(data, cols, col_name):
    next_diff = data.groupby(cols, as_index=False)["req_time"].shift(-1).rename(
        columns={"req_time": col_name})
    data = pd.concat([data, next_diff], axis=1)
    data[col_name] = (data[col_name] - data['req_time']).dt.seconds // 60
    data[col_name] = data[col_name].fillna(-1)
    return data


def get_diff_feature():
    queries = get_data()
    queries = get_last_diff(queries, ['pid'], 'last_diff')
    queries = get_last_diff(queries, ['pid'], 'next_diff')
    queries = get_last_diff(queries, ["pid", "o"], 'last_o_diff')
    queries = get_last_diff(queries, ["pid", "o"], 'next_o_diff')
    queries = get_last_diff(queries, ["pid", "d"], 'last_d_diff')
    queries = get_last_diff(queries, ["pid", "d"], 'next_d_diff')
    queries = get_last_diff(queries, ["pid", "o", "d"], 'last_od_diff')
    queries = get_last_diff(queries, ["pid", "o", "d"], 'next_od_diff')
    return queries


if __name__ == "-__main__":
    result = get_diff_feature()
    features = ['next_diff', 'last_diff', 'next_o_diff', 'last_o_diff',
                'next_d_diff', 'last_d_diff', 'next_od_diff', 'last_od_diff']
    result[['sid'] + features].to_csv('../output/time_gap.csv', index=False)


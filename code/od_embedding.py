import pandas as pd
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
import warnings
warnings.filterwarnings("ignore")


def get_data():
    queries = pd.read_csv("../data/data_set_phase2/train_queries.csv", parse_dates=['req_time'])
    test_queries = pd.read_csv("../data/data_set_phase2/test_queries.csv", parse_dates=['req_time'])

    queries_data = pd.concat([queries, test_queries])
    queries_data["time_bins"] = queries_data["req_time"].dt.hour // 2
    return queries_data


def encode_data(data, primary_col, used_cols):
    rows = data[primary_col].nunique()
    result = np.zeros((rows, 12))
    result = pd.DataFrame(result)
    result[primary_col] = data[primary_col].unique().tolist()

    for i, nums in enumerate(data[used_cols].values):
        result.loc[result[primary_col] == nums[0], nums[1]] = nums[2]

    cols = result.columns.tolist()
    cols.remove(primary_col)
    sum_temp = result[cols].sum(axis=1)
    result[cols] = result[cols].div(sum_temp + 0.01, axis="rows")
    return result


def get_embedding(col):
    """
    Get the functional embedding of OD
    :param col: O or D
    :return: embedding
    """
    data = get_data()
    o_time_nums = data.groupby([col, "time_bins"], as_index=False)[col].agg({"{}_nums".format(col): "count"})
    rows = o_time_nums[col].nunique()
    o_time_features = np.zeros((rows, 12))
    o_time_features = pd.DataFrame(o_time_features)
    o_time_features[col] = o_time_nums[col].unique().tolist()

    for i, nums in enumerate(o_time_nums.values):
        o_time_features.loc[o_time_features[col] == nums[0], nums[1]] = nums[2]

    cols = o_time_features.columns.tolist()
    cols.remove(col)
    sum_temp = o_time_features[cols].sum(axis=1)
    o_time_features[cols] = o_time_features[cols].div(sum_temp, axis="rows")
    cols = o_time_features.columns.tolist()
    cols.remove(col)
    print("Topic model starting...")
    svd_enc = LatentDirichletAllocation(n_components=5, max_iter=50)
    mode_svd = svd_enc.fit_transform(o_time_features[cols].values)
    mode_svd = pd.DataFrame(mode_svd)
    mode_svd.columns = ['{}_time_{}'.format(col, i) for i in range(5)]
    result = pd.concat([o_time_features[col], mode_svd], axis=1)
    print("{} embedding done.".format(col))
    return result


if __name__ == "__main__":
    o_time = get_embedding("o")
    o_time.to_csv("../output/o_time_embedding.csv", index=False)
    d_time = get_embedding("d")
    d_time.to_csv("../output/d_time_embedding.csv", index=False)

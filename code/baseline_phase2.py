import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import recall_score, precision_score
from baidu.code.cal_dist import manhattan, bearing_array
from tqdm import tqdm
import json
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import gc

import warnings
warnings.filterwarnings("ignore")


def read_profile_data():
    profile_data = pd.read_csv('../data/data_set_phase2/profiles.csv')
    profile_na = np.zeros(67)
    profile_na[0] = -1
    profile_na = pd.DataFrame(profile_na.reshape(1, -1))
    profile_na.columns = profile_data.columns
    profile_data = profile_data.append(profile_na)
    return profile_data


def gen_profile_feas(data):
    profile_data = read_profile_data()
    x = profile_data.drop(['pid'], axis=1).values
    svd = TruncatedSVD(n_components=20, n_iter=20, random_state=2019)
    svd_x = svd.fit_transform(x)
    svd_feas = pd.DataFrame(svd_x)
    svd_feas.columns = ['svd_fea_{}'.format(i) for i in range(20)]
    svd_feas['pid'] = profile_data['pid'].values
    data['pid'] = data['pid'].fillna(-1)
    data = data.merge(svd_feas, on='pid', how='left')
    return data


train_queries = pd.read_csv('../data/data_set_phase2/train_queries.csv', parse_dates=['req_time'])
train_plans = pd.read_csv('../data/data_set_phase2/train_plans.csv', parse_dates=['plan_time'])
train_clicks = pd.read_csv('../data/data_set_phase2/train_clicks.csv')
profiles = read_profile_data()
test_queries = pd.read_csv('../data/data_set_phase2/test_queries.csv', parse_dates=['req_time'])
test_plans = pd.read_csv('../data/data_set_phase2/test_plans.csv', parse_dates=['plan_time'])

train_p_sid = pd.read_csv("../output/train_p_sid.csv")
test_p_sid = pd.read_csv("../output/test_p_sid.csv")
train_queries = pd.concat([train_queries, train_p_sid], axis=1)
test_queries = pd.concat([test_queries, test_p_sid], axis=1)

del train_p_sid
del test_p_sid
gc.collect()


train = train_queries.merge(train_plans, 'left', ['sid'])
test = test_queries.merge(test_plans, 'left', ['sid'])
train = train.merge(train_clicks, 'left', ['sid'])
train['click_mode'] = train['click_mode'].fillna(0).astype(int)
data = pd.concat([train, test], ignore_index=True)


data = gen_profile_feas(data)

data['o_lng'] = data['o'].apply(lambda x: float(x.split(',')[0]))
data['o_lat'] = data['o'].apply(lambda x: float(x.split(',')[1]))
data['d_lng'] = data['d'].apply(lambda x: float(x.split(',')[0]))
data['d_lat'] = data['d'].apply(lambda x: float(x.split(',')[1]))


encoding = data.groupby("o").size()
encoding = encoding / len(data)
data["o_encode"] = data["o"].map(encoding)

encoding = data.groupby("d").size()
encoding = encoding / len(data)
data["d_encode"] = data["d"].map(encoding)

data["od"] = data["o"] + data["d"]
encoding = data.groupby("od").size()
encoding = encoding / len(data)
data["od_encode"] = data["od"].map(encoding)
map_feature = ["o_encode", "d_encode", "od_encode"]


manhattan_distance = []
angle = []
tmp = data[['o', 'd']]
for i in tmp.values:
    lat1, lon1 = eval(i[0])[1], eval(i[0])[0]
    lat2, lon2 = eval(i[1])[1], eval(i[1])[0]
    manhattan_distance.append(manhattan(float(lat1), float(lon1), float(lat2), float(lon2)))
    angle.append(bearing_array(float(lat1), float(lon1), float(lat2), float(lon2)))
data.loc[:, 'manhattan'] = manhattan_distance
data.loc[:, 'angle'] = angle
o_time_embedding = pd.read_csv("../output/o_time_embedding.csv")
data = data.merge(o_time_embedding, on="o", how="left")
d_time_embedding = pd.read_csv("../output/d_time_embedding.csv")
data = data.merge(d_time_embedding, on="d", how="left")

time_gap = pd.read_csv("../output/time_gap.csv")
data = data.merge(time_gap, on="sid", how="left")
time_gap_features = ['next_diff', 'last_diff', 'next_o_diff', 'last_o_diff',
                     'next_d_diff', 'last_d_diff', 'next_od_diff', 'last_od_diff']


pid_dist = data.groupby("pid", as_index=False)["manhattan"].agg({"pid_dist_max": "max", "pid_dist_min": "min",
                                                                 "pid_dist_mean": "mean", "pid_dist_std": "std"})
data = data.merge(pid_dist, on="pid", how="left")

pid_o_dist = data.groupby(["pid", "o"], as_index=False)["manhattan"].agg({"pid_o_dist_max": "max",
                                                                          "pid_o_dist_min": "min",
                                                                          "pid_o_dist_mean": "mean",
                                                                          "pid_o_dist_std": "std"})
data = data.merge(pid_o_dist, on=["pid", "o"], how="left")

o_unique = pd.read_csv("../output/o_unique.csv")
data = data.merge(o_unique, on="o", how="left")
d_unique = pd.read_csv("../output/d_unique.csv")
data = data.merge(d_unique, on="d", how="left")
unique_feature = ["d_ratio", "unique_d", "d_count", "o_ratio", "unique_o", "o_count",
                  "o_hour_count", "d_hour_count"]

o_unique = pd.read_csv("../output/o_pid_unique.csv")
data = data.merge(o_unique, on="o", how="left")
d_unique = pd.read_csv("../output/d_pid_unique.csv")
data = data.merge(d_unique, on="d", how="left")

od_unique = pd.read_csv("../output/od_pid_unique.csv")
data = data.merge(od_unique, on="od", how="left")


def dist_part(x):
    if x == 0:
        return 0
    elif 0 < x <= 1:
        return 1
    elif 1 < x <= 3:
        return 2
    elif 3 < x <= 5:
        return 3
    elif 5 < x <= 8:
        return 4
    elif 8 < x <= 10:
        return 5
    elif 10 < x <= 15:
        return 6
    elif 15 < x <= 25:
        return 7
    elif 25 < x <= 50:
        return 8
    elif x > 50:
        return 9


def eta_part(x):
    if 0 <= x <= 200:
        return 1
    elif 200 < x <= 500:
        return 2
    elif 500 < x <= 1000:
        return 3
    elif 1000 < x <= 1200:
        return 4
    elif 1200 < x <= 1500:
        return 5
    elif 1500 < x <= 2000:
        return 6
    elif 2000 < x <= 3000:
        return 7
    elif 3000 < x <= 4000:
        return 8
    elif x > 5000:
        return 9


def price_part(x):
    if x == 0:
        return 0
    elif 0 < x <= 300:
        return 1
    elif 300 < x <= 400:
        return 2
    elif 400 < x <= 500:
        return 3
    elif 500 < x <= 600:
        return 4
    elif 600 < x <= 1000:
        return 5
    elif 1000 < x <= 1500:
        return 6
    elif 1500 < x <= 2500:
        return 7
    elif x > 2500:
        return 8


data['manhattan_box'] = data['manhattan'].map(dist_part)
data['manhattan_log'] = np.log1p(data['manhattan'])

time_feature = []
for i in ['req_time']:
    data[i + '_hour'] = data[i].dt.hour
    data[i + '_weekday'] = data[i].dt.weekday
    data[i + 'period'] = data[i].dt.hour // 2
    data['elapsed_time'] = data[i].dt.hour * 60 + data[i].dt.minute
    data["minute"] = data.req_time.dt.minute
    data["hour_num"] = data[i + '_hour'] + data["minute"] / 60
    data["month"] = data.req_time.dt.month
    data["day"] = data.req_time.dt.day
    time_feature.append(i + '_hour')
    time_feature.append(i + '_weekday')
    time_feature.append(i + 'period')
    time_feature.append('elapsed_time')
    time_feature.append('hour_num')


o_hour_count = data.groupby(["o", "req_time_hour"], as_index=False)["req_time_hour"].agg({"o_hour_count": "count"})
data = data.merge(o_hour_count, on=["o", "req_time_hour"], how="left")
d_hour_count = data.groupby(["d", "req_time_hour"], as_index=False)["req_time_hour"].agg({"d_hour_count": "count"})
data = data.merge(d_hour_count, on=["d", "req_time_hour"], how="left")

o_day_hour_count = data.groupby(["o", "month", "day", "req_time_hour"], as_index=False)["req_time_hour"].agg(
    {"o_day_hour_count": "count"})
o_day_count = data.groupby(["o", "month", "day"], as_index=False)["req_time_hour"].agg(
    {"o_day_count": "count"})
o_day_hour_count = o_day_hour_count.merge(o_day_count, on=["o", "month", "day"], how="left")
o_day_hour_count['o_day_ratio'] = o_day_hour_count['o_day_hour_count'].div(o_day_hour_count['o_day_count'])
data = data.merge(o_day_hour_count, on=["o", "month", "day", "req_time_hour"], how="left")

d_day_hour_count = data.groupby(["d", "month", "day", "req_time_hour"], as_index=False)["req_time_hour"].agg(
    {"d_day_hour_count": "count"})
d_day_count = data.groupby(["d", "month", "day"], as_index=False)["req_time_hour"].agg(
    {"d_day_count": "count"})
d_day_hour_count = d_day_hour_count.merge(d_day_count, on=["d", "month", "day"], how="left")
d_day_hour_count['d_day_ratio'] = d_day_hour_count['d_day_hour_count'].div(d_day_hour_count['d_day_count'])
data = data.merge(d_day_hour_count, on=["d", "month", "day", "req_time_hour"], how="left")


o_hour_stat = data.groupby("o", as_index=False)["req_time_hour"].agg({"o_hour_max": "max",
                                                                      "o_hour_min": "min",
                                                                      "o_hour_mean": "mean",
                                                                      "o_hour_std": "std"})
data = data.merge(o_hour_stat, on="o", how="left")
o_elapsed_stat = data.groupby("o", as_index=False)["elapsed_time"].agg({"o_elapsed_max": "max",
                                                                        "o_elapsed_min": "min",
                                                                        "o_elapsed_mean": "mean",
                                                                        "o_elapsed_std": "std"})
data = data.merge(o_elapsed_stat, on="o", how="left")
d_hour_stat = data.groupby("d", as_index=False)["req_time_hour"].agg({"d_hour_max": "max",
                                                                      "d_hour_min": "min",
                                                                      "d_hour_mean": "mean",
                                                                      "d_hour_std": "std"})
data = data.merge(d_hour_stat, on="d", how="left")
d_elapsed_stat = data.groupby("d", as_index=False)["elapsed_time"].agg({"d_elapsed_max": "max",
                                                                        "d_elapsed_min": "min",
                                                                        "d_elapsed_mean": "mean",
                                                                        "d_elapsed_std": "std"})
data = data.merge(d_elapsed_stat, on="d", how="left")

od_time_feature = ["o_hour_max", "o_hour_min", "o_hour_mean", "o_hour_std",
                   "d_hour_max", "d_hour_min", "d_hour_mean", "d_hour_std",
                   "o_elapsed_max", "o_elapsed_min", "o_elapsed_mean", "o_elapsed_std",
                   "d_elapsed_max", "d_elapsed_min", "d_elapsed_mean", "d_elapsed_std",
                   "o_day_hour_count", "o_day_count", "o_day_ratio", "d_day_hour_count", "d_day_count", "d_day_ratio"]


data['plans_json'] = data['plans'].fillna('[]').apply(lambda x: json.loads(x))


def gen_plan_feas(data):
    n = data.shape[0]
    mode_list_feas = np.zeros((n, 12))
    speed_list_feas = np.zeros((n, 12))
    max_dist, min_dist, mean_dist, std_dist = np.zeros((n,)), np.zeros((n,)), np.zeros((n,)), np.zeros((n,))

    max_price, min_price, mean_price, std_price = np.zeros((n,)), np.zeros((n,)), np.zeros((n,)), np.zeros((n,))

    max_eta, min_eta, mean_eta, std_eta = np.zeros((n,)), np.zeros((n,)), np.zeros((n,)), np.zeros((n,))

    min_dist_mode, max_dist_mode, min_price_mode, max_price_mode, min_eta_mode, max_eta_mode, first_mode = \
        np.zeros((n,)), np.zeros((n,)), np.zeros((n,)), np.zeros((n,)), np.zeros((n,)), np.zeros((n,)), np.zeros((n,))

    mode_texts = []
    for i, plan in tqdm(enumerate(data['plans_json'].values)):
        if len(plan) == 0:
            cur_plan_list = []
        else:
            cur_plan_list = plan
        if len(cur_plan_list) == 0:
            mode_list_feas[i, 0] = 1
            first_mode[i] = 0
            max_dist[i] = -1
            min_dist[i] = -1
            mean_dist[i] = -1
            std_dist[i] = -1
            max_price[i] = -1
            min_price[i] = -1
            mean_price[i] = -1
            std_price[i] = -1
            max_eta[i] = -1
            min_eta[i] = -1
            mean_eta[i] = -1
            std_eta[i] = -1
            min_dist_mode[i] = -1
            max_dist_mode[i] = -1
            min_price_mode[i] = -1
            max_price_mode[i] = -1
            min_eta_mode[i] = -1
            max_eta_mode[i] = -1
            mode_texts.append('word_null')
        else:
            distance_list = []
            price_list = []
            eta_list = []
            mode_list = []
            speed_list = []
            for tmp_dit in cur_plan_list:
                distance_list.append(int(tmp_dit['distance']))
                if tmp_dit['price'] == '':
                    price_list.append(0)
                else:
                    price_list.append(int(tmp_dit['price']))
                eta_list.append(int(tmp_dit['eta']))
                mode_list.append(int(tmp_dit['transport_mode']))
                speed = int(tmp_dit['distance']) / int(tmp_dit['eta'])
                speed_list.append(speed)
            mode_texts.append(
                ' '.join(['word_{}'.format(mode) for mode in mode_list]))
            distance_list = np.array(distance_list)
            price_list = np.array(price_list)
            eta_list = np.array(eta_list)
            mode_list = np.array(mode_list, dtype='int')
            mode_list_feas[i, mode_list] = 1
            speed_list_feas[i, mode_list] = speed_list
            distance_sort_idx = np.argsort(distance_list)
            price_sort_idx = np.argsort(price_list)
            eta_sort_idx = np.argsort(eta_list)
            max_dist[i] = distance_list[distance_sort_idx[-1]]
            min_dist[i] = distance_list[distance_sort_idx[0]]
            mean_dist[i] = np.mean(distance_list)
            std_dist[i] = np.std(distance_list)
            max_price[i] = price_list[price_sort_idx[-1]]
            min_price[i] = price_list[price_sort_idx[0]]
            mean_price[i] = np.mean(price_list)
            std_price[i] = np.std(price_list)
            max_eta[i] = eta_list[eta_sort_idx[-1]]
            min_eta[i] = eta_list[eta_sort_idx[0]]
            mean_eta[i] = np.mean(eta_list)
            std_eta[i] = np.std(eta_list)
            first_mode[i] = mode_list[0]
            max_dist_mode[i] = mode_list[distance_sort_idx[-1]]
            min_dist_mode[i] = mode_list[distance_sort_idx[0]]
            max_price_mode[i] = mode_list[price_sort_idx[-1]]
            min_price_mode[i] = mode_list[price_sort_idx[0]]
            max_eta_mode[i] = mode_list[eta_sort_idx[-1]]
            min_eta_mode[i] = mode_list[eta_sort_idx[0]]
    feature_data = pd.DataFrame(mode_list_feas)
    feature_data.columns = ['mode_feas_{}'.format(i) for i in range(12)]
    feature_data['max_dist'] = max_dist
    feature_data['min_dist'] = min_dist
    feature_data['mean_dist'] = mean_dist
    feature_data['std_dist'] = std_dist
    feature_data['max_price'] = max_price
    feature_data['min_price'] = min_price
    feature_data['mean_price'] = mean_price
    feature_data['std_price'] = std_price
    feature_data['max_eta'] = max_eta
    feature_data['min_eta'] = min_eta
    feature_data['mean_eta'] = mean_eta
    feature_data['std_eta'] = std_eta
    feature_data['max_dist_mode'] = max_dist_mode
    feature_data['min_dist_mode'] = min_dist_mode
    feature_data['max_price_mode'] = max_price_mode
    feature_data['min_price_mode'] = min_price_mode
    feature_data['max_eta_mode'] = max_eta_mode
    feature_data['min_eta_mode'] = min_eta_mode
    feature_data['first_mode'] = first_mode
    speed_feature = pd.DataFrame(speed_list_feas)
    speed_feature.columns = ['speed_feas_{}'.format(i) for i in range(12)]
    feature_data = pd.concat([feature_data, speed_feature], axis=1)
    print('mode tfidf...')
    tfidf_enc = TfidfVectorizer(ngram_range=(1, 2))
    tfidf_vec = tfidf_enc.fit_transform(mode_texts)
    svd_enc = TruncatedSVD(n_components=10, n_iter=20, random_state=2019)
    mode_svd = svd_enc.fit_transform(tfidf_vec)
    mode_svd = pd.DataFrame(mode_svd)
    mode_svd.columns = ['svd_mode_{}'.format(i) for i in range(10)]
    plan_fea = pd.concat([feature_data, mode_svd], axis=1)
    plan_fea['sid'] = data['sid'].values

    return plan_fea


data_plans = gen_plan_feas(data)
plan_features = [col for col in data_plans.columns if col not in ['sid']]
data = data.merge(data_plans, on='sid', how='left')

time_dist = data.groupby(['o', 'req_time_hour', 'manhattan_box'], as_index=False)[
    ['mode_feas_{}'.format(i) for i in range(12)]].mean()
time_dist = time_dist.rename(columns=lambda x: 'time_dist_{}_mean'.format(x) if 5 < len(x) < 13 else x)
data = data.merge(time_dist, on=['o', 'req_time_hour', 'manhattan_box'], how="left")

mode_mean = data.groupby('o', as_index=False)[['mode_feas_{}'.format(i) for i in range(12)]].mean()
mode_mean = mode_mean.rename(columns=lambda x: 'o_{}_mean'.format(x) if len(x) > 5 else x)
data = data.merge(mode_mean, on="o", how="left")
mode_sum = data.groupby('o', as_index=False)[['mode_feas_{}'.format(i) for i in range(12)]].sum()
mode_sum = mode_sum.rename(columns=lambda x: 'o_{}_sum'.format(x) if len(x) > 5 else x)
data = data.merge(mode_sum, on="o", how="left")

mode_mean = data.groupby('d', as_index=False)[['mode_feas_{}'.format(i) for i in range(12)]].mean()
mode_mean = mode_mean.rename(columns=lambda x: 'd_{}_mean'.format(x) if len(x) > 5 else x)
data = data.merge(mode_mean, on="d", how="left")
mode_sum = data.groupby('d', as_index=False)[['mode_feas_{}'.format(i) for i in range(12)]].sum()
mode_sum = mode_sum.rename(columns=lambda x: 'd_{}_sum'.format(x) if len(x) > 5 else x)
data = data.merge(mode_sum, on="d", how="left")

mode_mean = data.groupby('od', as_index=False)[['mode_feas_{}'.format(i) for i in range(12)]].mean()
mode_mean = mode_mean.rename(columns=lambda x: 'od_{}_mean'.format(x) if len(x) > 5 else x)
data = data.merge(mode_mean, on="od", how="left")
mode_sum = data.groupby('od', as_index=False)[['mode_feas_{}'.format(i) for i in range(12)]].sum()
mode_sum = mode_sum.rename(columns=lambda x: 'od_{}_sum'.format(x) if len(x) > 5 else x)
data = data.merge(mode_sum, on="od", how="left")

od_mode_feature = ['o_mode_feas_{}_mean'.format(i) for i in range(12)] +\
                  ['o_mode_feas_{}_sum'.format(i) for i in range(12)] +\
                  ['d_mode_feas_{}_mean'.format(i) for i in range(12)] +\
                  ['d_mode_feas_{}_sum'.format(i) for i in range(12)] +\
                  ['od_mode_feas_{}_mean'.format(i) for i in range(12)] +\
                  ['od_mode_feas_{}_sum'.format(i) for i in range(12)] +\
                  ['time_dist_mode_feas_{}_mean'.format(i) for i in range(12)]


mode_mean = data.groupby(['pid', 'o'], as_index=False)[['mode_feas_{}'.format(i) for i in range(12)]].mean()
mode_mean = mode_mean.rename(columns=lambda x: 'pido_{}_mean'.format(x) if len(x) > 5 else x)
data = data.merge(mode_mean, on=['pid', 'o'], how="left")
mode_sum = data.groupby(['pid', 'o'], as_index=False)[['mode_feas_{}'.format(i) for i in range(12)]].sum()
mode_sum = mode_sum.rename(columns=lambda x: 'pido_{}_sum'.format(x) if len(x) > 5 else x)
data = data.merge(mode_sum, on=['pid', 'o'], how="left")

mode_mean = data.groupby(['pid', 'd'], as_index=False)[['mode_feas_{}'.format(i) for i in range(12)]].mean()
mode_mean = mode_mean.rename(columns=lambda x: 'pidd_{}_mean'.format(x) if len(x) > 5 else x)
data = data.merge(mode_mean, on=['pid', 'd'], how="left")
mode_sum = data.groupby(['pid', 'd'], as_index=False)[['mode_feas_{}'.format(i) for i in range(12)]].sum()
mode_sum = mode_sum.rename(columns=lambda x: 'pidd_{}_sum'.format(x) if len(x) > 5 else x)
data = data.merge(mode_sum, on=['pid', 'd'], how="left")

mode_mean = data.groupby(['pid', 'od'], as_index=False)[['mode_feas_{}'.format(i) for i in range(12)]].mean()
mode_mean = mode_mean.rename(columns=lambda x: 'pidod_{}_mean'.format(x) if len(x) > 5 else x)
data = data.merge(mode_mean, on=['pid', 'od'], how="left")
mode_sum = data.groupby(['pid', 'od'], as_index=False)[['mode_feas_{}'.format(i) for i in range(12)]].sum()
mode_sum = mode_sum.rename(columns=lambda x: 'pidod_{}_sum'.format(x) if len(x) > 5 else x)
data = data.merge(mode_sum, on=['pid', 'od'], how="left")
del data["od"]

mode_mean = data.groupby('pid', as_index=False)[['mode_feas_{}'.format(i) for i in range(12)]].mean()
mode_mean = mode_mean.rename(columns=lambda x: 'pid_{}_mean'.format(x) if len(x) > 5 else x)
data = data.merge(mode_mean, on='pid', how="left")
mode_sum = data.groupby('pid', as_index=False)[['mode_feas_{}'.format(i) for i in range(12)]].sum()
mode_sum = mode_sum.rename(columns=lambda x: 'pid_{}_sum'.format(x) if len(x) > 5 else x)
data = data.merge(mode_sum, on='pid', how="left")

pid_od_mode_feature = ['pido_mode_feas_{}_mean'.format(i) for i in range(12)] +\
                  ['pido_mode_feas_{}_sum'.format(i) for i in range(12)] +\
                  ['pidd_mode_feas_{}_mean'.format(i) for i in range(12)] +\
                  ['pidd_mode_feas_{}_sum'.format(i) for i in range(12)] +\
                      ['pidod_mode_feas_{}_mean'.format(i) for i in range(12)] +\
                      ['pidod_mode_feas_{}_sum'.format(i) for i in range(12)] +\
                      ['pid_mode_feas_{}_mean'.format(i) for i in range(12)] +\
                      ['pid_mode_feas_{}_sum'.format(i) for i in range(12)]

mode_mean = data.groupby('pid', as_index=False)[['o_time_{}'.format(i) for i in range(5)]].mean()
mode_mean = mode_mean.rename(columns=lambda x: 'pid_o_time_{}_mean'.format(x) if len(x) > 5 else x)
data = data.merge(mode_mean, on='pid', how="left")
mode_mean = data.groupby('pid', as_index=False)[['d_time_{}'.format(i) for i in range(5)]].mean()
mode_mean = mode_mean.rename(columns=lambda x: 'pid_d_time_{}_mean'.format(x) if len(x) > 5 else x)
data = data.merge(mode_mean, on='pid', how="left")


data['max_dist_box'] = data['max_dist'].apply(lambda x: dist_part(x))
data['min_dist_box'] = data['min_dist'].apply(lambda x: dist_part(x))
data['mean_dist_box'] = data['mean_dist'].apply(lambda x: dist_part(x))
data['max_dist_log'] = data['max_dist'].map(np.log1p)
data['min_dist_log'] = data['min_dist'].map(np.log1p)
data['mean_dist_log'] = data['mean_dist'].map(np.log1p)

data['max_eta_box'] = data['max_eta'].apply(lambda x: eta_part(x))
data['min_eta_box'] = data['min_eta'].apply(lambda x: eta_part(x))
data['mean_eta_box'] = data['mean_eta'].apply(lambda x: eta_part(x))
data['max_eta_log'] = data['max_eta'].map(np.log1p)
data['min_eta_log'] = data['min_eta'].map(np.log1p)
data['mean_eta_log'] = data['mean_eta'].map(np.log1p)

data['max_price_box'] = data['max_eta'].apply(lambda x: price_part(x))
data['min_price_box'] = data['min_eta'].apply(lambda x: price_part(x))
data['mean_price_box'] = data['mean_eta'].apply(lambda x: price_part(x))
data['max_price_log'] = data['max_price'].map(np.log1p)
data['min_price_log'] = data['min_price'].map(np.log1p)
data['mean_price_log'] = data['mean_price'].map(np.log1p)


def f1_weighted(labels, preds):
    preds = np.argmax(preds.reshape(12, -1), axis=0)
    score = f1_score(y_true=labels, y_pred=preds, average='weighted')
    return 'f1_weighted', score, True


profile_feature = ['svd_fea_{}'.format(i) for i in range(20)]
origin_num_feature = ['o_lng', 'o_lat', 'd_lng', 'd_lat', 'manhattan', "angle",
                      "pid_dist_max", "pid_dist_min", "pid_dist_mean", "pid_dist_std",
                      "pid_o_dist_max", "pid_o_dist_min", "pid_o_dist_mean", "pid_o_dist_std",
                      "pid_o_ratio", "unique_o_pid", "pid_o_count", "pid_d_ratio",
                      "unique_d_pid", "pid_d_count", "pid_od_ratio", "unique_od_pid", "pid_od_count"] + profile_feature
box_feature = ['manhattan_box', 'max_dist_box', 'min_dist_box', 'mean_dist_box', 'max_eta_box',
               'min_eta_box', 'mean_eta_box', 'max_price_box', 'min_price_box', 'mean_price_box']
external_feature = ["o_subway", "d_subway", "od_subway"]
cate_feature = ['max_dist_mode', 'min_dist_mode', 'max_price_mode', 'min_price_mode', 'max_eta_mode',
                'min_eta_mode', 'first_mode', 'req_time_weekday']
embedding_feature = ['o_time_{}'.format(i) for i in range(5)] + ['d_time_{}'.format(i) for i in range(5)]
display_distance_feature = ['distance_mode_{}'.format(i) for i in range(1, 12)]
display_eta_feature = ['eta_mode_{}'.format(i) for i in range(1, 12)]
display_price_feature = ['price_mode_{}'.format(i) for i in range(1, 12)]
display_mode_feature = ['transport_mode_mode_{}'.format(i) for i in range(1, 12)]
display_feature = display_distance_feature + display_eta_feature + display_price_feature + display_mode_feature
binary_feature = ['binary_prob_{}'.format(i) for i in range(1, 12)]
mode_rank_feature = ["mode_{}_rank".format(i) for i in range(1, 12)]

data[display_feature] = data[display_feature].fillna(0)
data[display_distance_feature] = data[display_distance_feature] / 1000

log_feature = ['manhattan_log', 'max_dist_log', 'min_dist_log', 'mean_dist_log', 'max_eta_log',
               'min_eta_log', 'mean_eta_log', 'max_price_log', 'mean_price_log']
for dist in display_distance_feature:
    data["{}_box".format(dist)] = data[dist].map(dist_part)
    box_feature.append("{}_box".format(dist))
    data["{}_log".format(dist)] = data[dist].map(np.log1p)
    log_feature.append("{}_log".format(dist))

for eta in display_eta_feature:
    data["{}_box".format(eta)] = data[eta].map(eta_part)
    box_feature.append("{}_box".format(eta))
    data["{}_log".format(eta)] = data[eta].map(np.log1p)
    log_feature.append("{}_log".format(eta))

for price in display_price_feature:
    data["{}_box".format(price)] = data[price].map(price_part)
    box_feature.append("{}_box".format(price))
    data["{}_log".format(price)] = data[price].map(np.log1p)
    log_feature.append("{}_log".format(price))

feature = origin_num_feature + plan_features + time_feature + embedding_feature +\
          display_eta_feature + display_price_feature + display_distance_feature +\
          map_feature + log_feature + unique_feature + od_mode_feature +\
          pid_od_mode_feature + od_time_feature + time_gap_features

train_index = (data.req_time < '2018-11-23')
train_x = data[train_index][feature].reset_index(drop=True)
train_y = data[train_index].click_mode.reset_index(drop=True)

valid_index = (data.req_time > '2018-11-23') & (data.req_time < '2018-12-01')
valid_x = data[valid_index][feature].reset_index(drop=True)
valid_y = data[valid_index].click_mode.reset_index(drop=True)

test_index = (data.req_time > '2018-12-01')
test_x = data[test_index][feature].reset_index(drop=True)

print(len(feature), feature)
lgb_model = lgb.LGBMClassifier(boosting_type="gbdt", num_leaves=60, reg_alpha=0, reg_lambda=1,
                               max_depth=-1, n_estimators=2000, objective='multiclass',
                               subsample=0.8, colsample_bytree=0.8, subsample_freq=1, min_child_samples=50,
                               learning_rate=0.05, random_state=2019, metric="None", n_jobs=-1)
eval_set = [(valid_x, valid_y)]
lgb_model.fit(train_x, train_y, eval_set=eval_set, eval_metric=f1_weighted,
              categorical_feature=cate_feature, verbose=10, early_stopping_rounds=100)

val_score = lgb_model.predict_proba(valid_x)
val_df = pd.DataFrame(val_score)
val_df["sid"] = data[valid_index]['sid'].values
val_df['label'] = valid_y.values
val_df['pred'] = np.argmax(val_score, axis=1)
val_df.to_csv("../output/baseline_valid_phase_2_score.csv", index=False)

weights = [1, 0.80, 0.80, 2.3, 3.06, 0.70, 1.79, 0.81, 1.44, 0.85, 1.11, 1.09]
weight_score = weights * val_score
pred = np.argmax(weight_score, axis=1)

df_analysis = pd.DataFrame()
df_analysis['sid'] = data[valid_index]['sid']
df_analysis['label'] = valid_y.values
df_analysis['pred'] = pred
df_analysis['label'] = df_analysis['label'].astype(int)

dic_ = df_analysis['label'].value_counts(normalize=True)


def get_weighted_fscore(y_pred, y_true):
    f_score = 0
    for i in range(12):
        yt = y_true == i
        yp = y_pred == i
        f_score += dic_[i] * f1_score(y_true=yt, y_pred=yp)
        print(i, dic_[i], f1_score(y_true=yt, y_pred=yp), precision_score(y_true=yt, y_pred=yp), recall_score(y_true=yt,
                                                                                                             y_pred=yp))
    print(f_score)
    return f_score


scores = get_weighted_fscore(y_true=df_analysis['label'], y_pred=df_analysis['pred'])

del train_x
del train_y
del valid_x
del valid_y
gc.collect()

all_train_x = data[data.req_time < '2018-12-01'][feature].reset_index(drop=True)
all_train_y = data[data.req_time < '2018-12-01'].click_mode.reset_index(drop=True)
print(lgb_model.best_iteration_)
lgb_model.n_estimators = lgb_model.best_iteration_
lgb_model.fit(all_train_x, all_train_y, categorical_feature=cate_feature)
print('fit over')
result = pd.DataFrame()
result['sid'] = data[test_index]['sid']
result_prob = lgb_model.predict_proba(test_x)
weight_prob = weights * result_prob

result_df = pd.DataFrame(result_prob)
result_df["sid"] = test['sid'].values
result_df.to_csv("../output/result_score.csv", index=False)

feature_importance = pd.DataFrame()
feature_importance["feature"] = feature
feature_importance["importance"] = lgb_model.feature_importances_
feature_importance.sort_values(by='importance', ascending=False, inplace=True)
feature_importance.to_csv('../output/feature_importance.csv', index=False)

result_pred = np.argmax(weight_prob, axis=1)
result['recommend_mode'] = result_pred
result['recommend_mode'] = result['recommend_mode'].astype(int)
print(len(result))
print(result['recommend_mode'].value_counts())
result[['sid', 'recommend_mode']].to_csv('../submit/baseline_phase2_{:.6f}.csv'.format(scores), index=False)


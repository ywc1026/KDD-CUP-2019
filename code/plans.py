import pandas as pd
import numpy as np
from tqdm import tqdm
import json
import time


def get_data():
    train_queries = pd.read_csv('../data/data_set_phase2/train_queries.csv', parse_dates=['req_time'])
    train_plans = pd.read_csv('../data/data_set_phase2/train_plans.csv')
    train = train_queries.merge(train_plans, 'left', ['sid'])
    test_queries = pd.read_csv('../data/data_set_phase2/test_queries.csv', parse_dates=['req_time'])
    test_plans = pd.read_csv('../data/data_set_phase2/test_plans.csv')
    test = test_queries.merge(test_plans, 'left', ['sid'])
    return train, test


def get_plans_feature(data):
    print(data.shape)
    display_distance_feature = ['distance_mode_{}'.format(i) for i in range(1, 12)]
    display_eta_feature = ['eta_mode_{}'.format(i) for i in range(1, 12)]
    display_price_feature = ['price_mode_{}'.format(i) for i in range(1, 12)]
    display_mode_feature = ['transport_mode_mode_{}'.format(i) for i in range(1, 12)]
    display_feature = display_distance_feature + display_eta_feature + display_price_feature + display_mode_feature

    display = np.zeros((data.shape[0], 44))
    display = pd.DataFrame(display)
    display.columns = display_feature

    start = time.time()
    for i, plan in tqdm(enumerate(data['plans'].values)):
        try:
            cur_plan_list = json.loads(plan)
        except:
            cur_plan_list = []
        if len(cur_plan_list) == 0:
            continue
        else:
            for tmp in cur_plan_list:
                mode = int(tmp["transport_mode"])
                display.loc[i, 'transport_mode_mode_{}'.format(mode)] = 1
                display.loc[i, 'distance_mode_{}'.format(mode)] = int(tmp["distance"])
                display.loc[i, 'eta_mode_{}'.format(mode)] = int(tmp["eta"])
                if tmp["price"] == '':
                    display.loc[i, 'price_mode_{}'.format(mode)] = 0
                else:
                    display.loc[i, 'price_mode_{}'.format(mode)] = int(tmp["price"])
    end = time.time()
    print("Time is {} minutes".format(round((end - start) / 60, 1)))
    return display


if __name__ == "__main__":
    train, test = get_data()
    train_display = get_plans_feature(train)
    train_display.to_csv("../output/train_p_sid.csv", index=False)
    test_display = get_plans_feature(test)
    test_display.to_csv("../output/test_p_sid.csv", index=False)



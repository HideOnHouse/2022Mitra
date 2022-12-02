import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

GLOBAL_SEED = 755


class Dataset:
    def __init__(self, data_path='./data', train=True, test=False) -> None:
        self.initialized = False
        self.df = None
        self.except_cols = {'src_ip', 'dst_ip', 'timestamp', 'label'}
        self.feature_names = None
        self.test = test
        if test == False:
            self.outer_mal = pd.read_pickle(f"{data_path}{os.sep}outer_mal_IP.pkl")
            self.outer_ben = pd.read_pickle(
                f"{data_path}{os.sep}outer_benign_IP.pkl")
            self.outer = self.outer_ben | self.outer_mal
            df_path = f"{data_path}{os.sep}{'train' if train else 'valid'}"
        else:
            self.outer = pd.read_pickle('./data/outer_ip_set.pkl')
            df_path = './data/project2_test'
        print(f"reading {df_path}")
        if os.path.exists(df_path + os.extsep + 'pkl'):
            self.df = pd.read_pickle(df_path + os.extsep + 'pkl')
        else:
            self.df = pd.read_csv(df_path + os.extsep + 'csv')
            self.df.to_pickle(df_path + os.extsep + 'pkl')
        self.__preprocess()

    def __swap_src_dst(self):
        df = self.df
        df.columns = [i.replace(" ", "_").lower() for i in df.columns]
        idx = df['src_ip'].apply(lambda x: x in self.outer)
        swap_src = ['src_ip', 'src_port', 'dst_ip', 'dst_port']
        swap_dst = ['dst_ip', 'dst_port', 'src_ip', 'src_port']
        df.loc[idx, swap_src] = df.loc[idx, swap_dst].values

    def __process_port(self):
        df = self.df
        df['src_port'] = df['src_port'].apply(lambda x: 0 if x <= 1024 else 1)
        df['dst_port'] = df['src_port'].apply(lambda x: 0 if x <= 1024 else 1)
        self.df = df

    def __encode_label(self):
        df = self.df
        try:
            int(df['label'][0])
        except ValueError:
            df['label'] = df['label'].apply(
                lambda x: 0 if x == 'Benign_IP' else 1)

    def __process_ordinal(self):
        df = self.df
        ordinal_cols = ['protocol', 'flags']
        df[ordinal_cols] = OrdinalEncoder().fit_transform(df[ordinal_cols])

    def __preprocess(self):
        self.__swap_src_dst()
        # self.__process_port()
        if not self.test:
            self.__encode_label()
        self.__process_ordinal()
        self.initialized = True

    def get_feature_names(self):
        return self.feature_names

    def get_xy(self):
        assert self.initialized
        raise NotImplementedError


class FlowDataset(Dataset):
    def __init__(self, data_path='./data', train=True, test=False) -> None:
        super().__init__(data_path, train, test)
        self.x = None
        self.y = None

    def get_xy(self, inference=False):
        assert self.initialized
        if self.x is None or self.y is None:
            df = self.df.copy()
            feature_cols = [i for i in df.columns if i not in self.except_cols]
            if inference:
                x, y = df[feature_cols].values, None
            else:
                x, y = df[feature_cols].values, df['label'].values
            self.x = x
            self.y = y
        self.feature_names = feature_cols
        return self.x, self.y


class GroupDataset(Dataset):
    def __init__(self, data_path='./data', train=True, test=False) -> None:
        super().__init__(data_path, train, test)
        self.x = None
        self.y = None
        self.k = None
        self.feature_names = []
        self.agg_names = ['std', 'min', 'max',
                          'median', 'mean', 'kurt', 'skew']

    def __fmt(self, arg):
        h, m, s = map(int, arg.split(":"))
        return h * 3600 + m * 60 + s

    def __get_interval_feature(self, x):
        ret = []
        x = sorted(x)
        for i in range(1, len(x)):
            left = self.__fmt(x[i - 1])
            right = self.__fmt(x[i])
            ret.append(right - left)
        ret = pd.Series(ret, dtype=np.float32)
        features = [ret.std(), ret.min(), ret.max(), ret.median(),
                    ret.mean(), ret.kurt(), ret.skew()]
        return np.array(features)

    def __get_agg_feature(self):
        self.feature_names.clear()
        df = self.df.copy()
        feature_cols = [i for i in df.columns if i not in self.except_cols]
        raw = df.groupby('dst_ip')
        ret = []

        # interval features
        temp = raw['timestamp'].apply(self.__get_interval_feature)
        temp = np.array(temp.tolist())
        ret.append(temp)
        self.feature_names.extend(['interval_std',
                                   'interval_min',
                                   'interval_max',
                                   'interval_median',
                                   'interval_mean',
                                   'interval_kurt',
                                   'interval_skew'
                                   ])
        # end interval features

        key = raw['dst_ip'].count().index.to_numpy()
        raw = raw[feature_cols]
        for agg in self.agg_names:
            if agg == 'sum':
                temp = raw.sum().values
                self.feature_names.extend([f"{i}_sum" for i in feature_cols])
            elif agg == 'std':
                temp = raw.std().values
                self.feature_names.extend([f"{i}_std" for i in feature_cols])
            elif agg == 'min':
                temp = raw.min().values
                self.feature_names.extend([f"{i}_min" for i in feature_cols])
            elif agg == 'max':
                temp = raw.max().values
                self.feature_names.extend([f"{i}_max" for i in feature_cols])
            elif agg == 'median':
                temp = raw.median().values
                self.feature_names.extend(
                    [f"{i}_median" for i in feature_cols])
            elif agg == 'mean':
                temp = raw.mean().values
                self.feature_names.extend([f"{i}_mean" for i in feature_cols])
            elif agg == 'kurt':
                temp = raw.apply(pd.DataFrame.kurt).values
                self.feature_names.extend([f"{i}_kurt" for i in feature_cols])
            elif agg == 'skew':
                temp = raw.apply(pd.DataFrame.skew).values
                self.feature_names.extend([f"{i}_skew" for i in feature_cols])
            elif agg == 'mode':
                temp = raw.apply(pd.DataFrame.mode).values
                self.feature_names.extend([f"{i}_mode" for i in feature_cols])
            else:
                raise NotImplementedError(
                    f"aggregation {agg} not implemented yet")
            ret.append(temp)

        agg = np.concatenate(ret, axis=1)
        agg = np.nan_to_num(agg, nan=0)
        ind = np.argsort(self.feature_names)
        agg = agg[:, ind]
        self.feature_names = [self.feature_names[i] for i in ind]

        return agg, key

    def __get_group_label(self, key):
        temp = self.df.groupby('dst_ip')['label'].value_counts()
        temp = temp.reset_index(name='count')
        temp = temp[temp['label'] == 1]
        mal = set(temp[temp['count'] > 0]['dst_ip'])
        ret = []
        for ip in key:
            if ip in mal:
                ret.append(1)
            else:
                ret.append(0)
        return np.array(ret)

    def get_xy(self, inference=False):
        assert self.initialized
        if self.x is None or self.y is None:
            if self.test:
                x, k = self.__get_agg_feature()
                y = None
            else:
                if inference:
                    x, k = self.__get_agg_feature()
                    y = None
                else:
                    x, k = self.__get_agg_feature()
                    y = self.__get_group_label(k)
                assert x.shape[0] == y.shape[0]
            self.x = x
            self.y = y
            self.k = k
        return self.x, self.y

    def get_k(self):
        return self.k


class FrequencyRankingEncoder:
    def __init__(self):
        self.encoding_tables = []
        self.min_values = []

    def fit(self, x: np.array):
        self.encoding_tables.clear()
        self.min_values.clear()
        for col_idx in range(x.shape[1]):
            col = x[:, col_idx]
            # build frequency table
            freq_table = {}
            x_list, freq_list = np.unique(col, return_counts=True)
            for i, freq in enumerate(freq_list):
                if freq not in freq_table:
                    freq_table[freq] = []
                freq_table[freq].append(x_list[i])

            # make encoding table
            self.encoding_tables.append(dict())

            # Cumulative Density Function
            # total_cnt = sum(freq_list)
            # current_cum = 0
            # for i, freq in enumerate(sorted(freq_table)):
            #     current_cum += freq * len(freq_table[freq])
            #     for data in freq_table[freq]:
            #         self.encoding_table[data] = current_cum / total_cnt

            encoding_table = self.encoding_tables[-1]
            no_of_ranking = len(list(freq_table.keys()))
            for ranking, freq in enumerate(sorted(freq_table)):
                for i in freq_table[freq]:
                    encoding_table[i] = (ranking + 1) / no_of_ranking
            self.min_values.append(min(encoding_table.values()))

    def transform(self, x: np.array):
        ret = []
        for col_idx in range(x.shape[1]):
            col = x[:, col_idx]
            encoding_table = self.encoding_tables[col_idx]
            temp = np.empty_like(col, dtype=float)
            for idx in range(temp.shape[0]):
                temp[idx] = encoding_table.get(col[idx], 0)
            ret.append(temp)
        ret = np.array(ret).T
        return ret

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

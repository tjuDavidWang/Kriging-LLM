import os
import sys
import numpy as np
import pandas as pd
from torch.utils.data import Dataset,DataLoader

from .pd_dataset import PandasDataset

sys.path.append('/home/zgjgroup/wwd/Time-LLM/')  # Add the path to the root of the repository

from utils.timefeatures import time_features
from sklearn.preprocessing import StandardScaler


datasets_path = {'la':"/home/zgjgroup/wwd/Time-LLM/dataset/metr_la"}

def sample_mask(shape, p=0.002, p_noise=0., mode="random"):
    # shape:(time,node)
    # random则是对其中某些时间点mask，road是将整个node都mask
    assert mode in ["random", "road", "mix"], "The missing mode must be 'random' or 'road' or 'mix'."
    rand = np.random.random
    mask = np.zeros(shape).astype(np.bool_)
    if mode == "random" or mode == "mix":
        mask = mask | (rand(mask.shape) < p)
    if mode == "road" or mode == "mix":
        road_shape = mask.shape[1] # nodes
        rand_mask = rand(road_shape) < p_noise # node mask
        road_mask = np.zeros(shape).astype(np.bool_)
        road_mask[:, rand_mask] = True
        mask |= road_mask
    return mask.astype('uint8')

class MetrLA(PandasDataset):
    """
    METR-LA dataset
    207
    with 0, no nan
    """
    def __init__(self, impute_zeros=False, freq='5T'):

        df, raw_mask = self.load(impute_zeros=impute_zeros)
        self.raw_mask = raw_mask
        self.observed_mask = self.get_observed_idxs(df)
        self.df = df
        self.description = \
        """The METR-LA dataset is an essential resource in urban traffic management and research, comprising average vehicle speeds recorded by 207 detectors strategically placed across the Los Angeles County Highway system. The data covers a period from March 1, 2012, to June 27, 2012, with a sampling interval of every 5 minutes. This dataset is instrumental for analyzing traffic flow dynamics and optimizing traffic control strategies to enhance road safety and efficiency."""
        
        # PandasDataset中可以采用不同的方式对TS进行重采样
        super().__init__(dataframe=df, u=None, mask=raw_mask, name='la', freq=freq, aggr='nearest')

    def load(self, impute_zeros=True):
        path = os.path.join(datasets_path['la'], 'metr_la.h5')
        df = pd.read_hdf(path)
        
        # 移除其中的freq属性，并加上新的'5T'
        if hasattr(df.index, 'freq'):
            df.index.freq = None
        datetime_idx = sorted(df.index)
        date_range = pd.date_range(datetime_idx[0], datetime_idx[-1], freq='5T')
        df = df.reindex(index=date_range)
        
        # 将值为空和0的 mask=False
        raw_mask = ~np.isnan(df.values)
        if impute_zeros:
            raw_mask = raw_mask * (df.values != 0.).astype('uint8')
            # 使用前向填充（forward fill）方法将0值替换为前一个非0值
            df = df.replace(to_replace=0., method='ffill')
        else:
            raw_mask = None
            
        self.dist = self.load_distance_matrix()
        self.adj = self.get_similarity()
        
        return df, raw_mask

    def load_distance_matrix(self):
        # dist.shape=(207,207) with 'inf' for none
        path = os.path.join(datasets_path['la'], 'metr_la_dist.npy')
        try:
            dist = np.load(path)
        except:
            distances = pd.read_csv(os.path.join(datasets_path['la'], 'distances_la.csv'))
            with open(os.path.join(datasets_path['la'], 'sensor_ids_la.txt')) as f:
                ids = f.read().strip().split(',')
            num_sensors = len(ids)
            dist = np.ones((num_sensors, num_sensors), dtype=np.float32) * np.inf
            sensor_id_to_ind = {int(sensor_id): i for i, sensor_id in enumerate(ids)}
            for row in distances.values:
                if row[0] not in sensor_id_to_ind or row[1] not in sensor_id_to_ind:
                    continue
                dist[sensor_id_to_ind[row[0]], sensor_id_to_ind[row[1]]] = row[2]
            np.save(path, dist)
        return dist

    def get_similarity(self, thr=0.1, force_symmetric=False, sparse=False):
        # 将dist转化成可以使用的相似矩阵
        finite_dist = self.dist.reshape(-1)
        finite_dist = finite_dist[~np.isinf(finite_dist)] # 移除无穷大值
        sigma = finite_dist.std() 
        # with guassian kernel
        adj = np.exp(-np.square(self.dist / sigma)) 
        adj[adj < thr] = 0.
        if force_symmetric:
            adj = np.maximum.reduce([adj, adj.T])
        if sparse:
            import scipy.sparse as sps
            # save as sparse matrix
            adj = sps.coo_matrix(adj)
        return adj

    def get_observed_idxs(self,df):
        path = os.path.join(datasets_path['la'], 'known_set.npy')
        idxs = np.load(path)
        observed_mask = np.zeros(df.shape[1],dtype=bool)
        observed_mask[idxs]=1
        return observed_mask
        
def splitter(dataset, val_len=0.1, test_len=0.2, window=24):
    idx = np.arange(len(dataset))
    if test_len < 1:
        test_len = int(test_len * len(idx))
    if val_len < 1:
        val_len = int(val_len * (len(idx) - test_len))
    test_start = len(idx) - test_len
    val_start = test_start - val_len
    return idx[:val_start - window], idx[val_start:test_start - window], idx[test_start:]

class Dataset_MetrLA(Dataset):
    def __init__(self, flag='train', features='M', window_size=24,
                 scale=True, timeenc=1, freq='5T', percent=100):

        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.flag = flag

        self.percent = percent # few-shot实验的时候会用到吧
        self.features = features
        self.scale = scale
        self.timeenc = timeenc # time_embedding=='timeF'  [timeF, fixed, learned]
        self.freq = freq
        self.window_size = window_size
        
        self.__read_data__()

        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.window_size + 1

    def __read_data__(self):
        self.scaler = StandardScaler()
        dataset = MetrLA()
        
        # 从时间维度划分数据集
        # 其中有封装numpy，pytorch将dataframe转化出来
        train_idx, val_idx, test_idx=splitter(dataset)
        
        if self.flag == 'train':
            self.raw_mask = dataset.raw_mask
            self.observed_mask = dataset.observed_mask
            self.adj_matrix = dataset.adj
            self.description = dataset.description
            
        border1s = [train_idx.min(), val_idx.min(), test_idx.min()]
        border2s = [train_idx.max(), val_idx.max(), test_idx.max()] 

        border1 = border1s[self.set_type] # get the start of ['train', 'test', 'val']
        border2 = border2s[self.set_type] # # get the end of ['train', 'test', 'val']

        # if self.set_type == 0:
        #     border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len # adjust end with percent of train set

        # 选取所有数据一同作为输入(without DatetimeIndex)
        if self.features == 'M' or self.features == 'MS':
            cols_data = dataset.df.columns
            df_data = dataset.df[cols_data]


        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]] # 通过训练集获得scaler
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = dataset.df.index[border1:border2]
        # if self.timeenc == 0:
        #     df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
        #     df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
        #     df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
        #     df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
        #     data_stamp = df_stamp.drop(['date'], 1).values
        # elif self.timeenc == 1:
        data_stamp = time_features(pd.to_datetime(df_stamp.values), freq=self.freq) # time embedding [4,8640]
        data_stamp = data_stamp.transpose(1, 0) #  [8640,4]

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp


    def __getitem__(self, index):
        # 应该要输出的是用0-mask和没有mask的
        seq_x = self.data_x[index:index+self.window_size]
        seq_y = self.data_y[index:index+self.window_size]
        seq_x_mark = self.data_stamp[index:index+self.window_size]
        seq_y_mark = self.data_stamp[index:index+self.window_size]
        
        observed_mask = self.observed_mask
        if observed_mask.shape[0] != seq_x.shape[1]:
            observed_mask = observed_mask.T
        seq_x = seq_x * observed_mask.T
        
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return (len(self.data_x) - self.window_size + 1)



if __name__ == '__main__':
    print("test data loading......")
    # dataset = MetrLA()
    
    # # 从时间维度划分数据集
    # # 其中有封装numpy，pytorch将dataframe转化出来
    # train_idx, val_idx,test_idx=splitter(dataset)
    # train_set = dataset.pytorch()[train_idx] 
    # val_set = dataset.pytorch()[val_idx]
    # test_set = dataset.pytorch()[test_idx]
    # print(train_idx.max(),train_idx.min())
    # raw_mask = dataset.raw_mask
    # observed_mask = dataset.observed_mask
    # adj_matrix = dataset.adj
    # description = dataset.description
    dataset = Dataset_MetrLA(flag='train', features='M', window_size=24, scale=True, timeenc=1, freq='5T', percent=100)
    print(dataset.description)
    print(dataset.adj_matrix)
    # 创建 DataLoader 实例
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # 遍历 DataLoader
    for i, (seq_x, seq_y, seq_x_mark, seq_y_mark) in enumerate(dataloader):
        print(f"Batch {i}")
        print("seq_x:", seq_x)
        
        # 为了测试，我们只打印一个 batch 的数据
        if i == 0:
            break
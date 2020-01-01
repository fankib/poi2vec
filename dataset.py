import numpy as np
import random

class Data():
    def __init__(self):

        self.id2route = None
        self.id2lr = None
        self.id2prob = None

        self.user_train = None 
        self.context_train = None
        self.target_train = None
        self.user_valid = None 
        self.context_valid = None
        self.target_valid = None
        self.user_test = None 
        self.context_test = None
        self.target_test = None
        self.maxlen_context = 32

    def load(self):
        print("Loading data...")
        poi_list = np.load("./npy/id2poi.npy", allow_pickle=True)
        user_list = np.load("./npy/id2user.npy", allow_pickle=True)
        self.id2route = np.load("./npy/id2route.npy", allow_pickle=True)
        self.id2lr = np.load("./npy/id2lr.npy", allow_pickle=True)
        self.id2prob = np.load("./npy/id2prob.npy", allow_pickle=True)

        self.user_train = np.load("./npy/train_user.npy", allow_pickle=True)
        self.context_train = np.load("./npy/train_context.npy", allow_pickle=True)
        self.target_train = np.load("./npy/train_target.npy", allow_pickle=True)
        self.user_valid = np.load("./npy/valid_user.npy", allow_pickle=True)
        self.context_valid = np.load("./npy/valid_context.npy", allow_pickle=True)
        self.target_valid = np.load("./npy/valid_target.npy", allow_pickle=True)
        self.user_test = np.load("./npy/test_user.npy", allow_pickle=True)
        self.context_test = np.load("./npy/test_context.npy", allow_pickle=True)
        self.target_test = np.load("./npy/test_target.npy", allow_pickle=True)
        print("Train/Valid/Test/POI/User: {:d}/{:d}/{:d}/{:d}/{:d}".format(len(self.user_train), len(self.user_valid), len(self.user_test), len(poi_list), len(user_list)))
        print("==================================================================================")

        return len(poi_list), len(user_list)

    def train_batch_iter(self, batch_size):
        data = list(zip(self.user_train, self.context_train, self.target_train))
        random.shuffle(data)
        return self.batch_iter(data, batch_size)

    def valid_batch_iter(self, batch_size):
        data = list(zip(self.user_valid, self.context_valid, self.target_valid))
        return self.batch_iter(data, batch_size)

    def test_batch_iter(self, batch_size):
        data = list(zip(self.user_test, self.context_test, self.target_test))
        return self.batch_iter(data, batch_size)

    def batch_iter(self, data, batch_size):
        data_size = float(len(data))
        num_batches = int(np.ceil(data_size / batch_size))
        for batch_num in range(num_batches):
            start_index = int(batch_num * batch_size)
            end_index = min(int((batch_num + 1) * batch_size), int(data_size))
            yield data[start_index:end_index]

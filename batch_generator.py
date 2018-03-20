# zhangshulin
# 2018-3-20
# e-mail: zhangslwork@yeah.net


from keras.utils import Sequence
import numpy as np


class Batch_generator(Sequence):

    def __init__(self, dataset, batch_size, classes, n_a):
        self._X, self._Y = self._create_X_Y(dataset)
        self._batch_size = batch_size
        self._classes = classes
        self._n_a = n_a 
        self._counts = len(dataset)
        self._eye = np.eye(classes)


    def __len__(self):
        return np.ceil(self._counts / self._batch_size).astype(np.int32)


    def __getitem__(self, idx):
        last_full_batch = self._counts // self._batch_size
        batch_index = idx % len(self)

        if batch_index == last_full_batch:
            batch = (self._X[last_full_batch * self._batch_size:],
                     self._Y[last_full_batch * self._batch_size:])
        else:
            batch = (self._X[batch_index * self._batch_size: (1 + batch_index) * self._batch_size], 
                     self._Y[batch_index * self._batch_size: (1 + batch_index) * self._batch_size]) 

        a0 = c0 = np.zeros((len(batch[0]), self._n_a))

        return ([self._create_onehot(batch[0]), a0, c0],
               list(np.transpose(self._create_onehot(batch[1]), [1, 0, 2])))


    def _create_onehot(self, input):
        return self._eye[input]


    def _create_X_Y(self, data_set):
        m = data_set.shape[0]
        zeros_start = np.zeros((m, 1), dtype=np.int32)
    
        X = np.concatenate([zeros_start, data_set], axis=1)
        X = X[:, :-1]
        Y = data_set

        return X, Y


# TEST
if __name__ == '__main__':
    print("START")
    FILF_PATH = './datasets/all_couplets.txt'

    from datasets_creator import Datasets_creator

    creator = Datasets_creator(FILF_PATH, vocabs_size=100)
    sample = creator.load_sample()

    generator = Batch_generator(sample, batch_size=35, classes=100, n_a=16)

    num = 0
    for x, y in generator:
        print(x[0].shape, len(y), y[0].shape, x[1].shape)
        num += 1
        if num == 10:
            break

    print('END')
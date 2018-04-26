# Author: zhangshulin 
# Email: zhangslwork@yeah.net 
# Date: 2018-04-18 10:41:10 
# Last Modified by: zhangshulin
# Last Modified Time: 2018-04-18 10:41:10 


import tensorflow as tf
import numpy as np
import helper


class CoupletsDataGenerator:

    def __init__(self, set_array, shuffle=True, buffer_size=10000):
        self._data = tf.data.Dataset.from_tensor_slices(set_array)
        
        if shuffle:
            self._data = self._data.shuffle(buffer_size=buffer_size)


    def get_batch(self, session, batch_size, epochs):
        batch_data = self._data.batch(batch_size)
        repeat_data = batch_data.repeat(epochs)
        
        iterator = repeat_data.make_one_shot_iterator()
        next = iterator.get_next()

        while True:
            try:
                data_set = session.run(next)
                Y = data_set
                fore_zeros = np.zeros((data_set.shape[0], 1), dtype=np.int32)
                X = np.concatenate((fore_zeros, Y[:, : -1]), axis=1)

                yield X, Y

            except tf.errors.OutOfRangeError:
                break


def test():
    import helper 

    _, _, _, train_set, _, test_set = helper.process_dataset()
    data_g = CoupletsDataGenerator(train_set)
    sess = tf.Session()
    batch_g = data_g.get_batch(sess, 128, 1)

    for X, Y in batch_g:
        print('X: ', X.shape)
        print('Y: ', Y.shape)


if __name__ == '__main__':
    test()

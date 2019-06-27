import tensorflow as tf
from math import ceil
from numpy.random import choice
from os.path import exists

def get_data_iterator(filenames, batch, sess, max_t = 40):
    
    def _map_func(example):
        features = {
            'len': tf.FixedLenSequenceFeature(shape = [], dtype = tf.int64, allow_missing = True),
            'index': tf.FixedLenSequenceFeature(shape = (max_t,), dtype = tf.int64, allow_missing = True)
        }
        
        parsed_features = tf.parse_single_example(example, features)
        parsed_features['index'] = tf.reshape(parsed_features['index'], (max_t,))
        return parsed_features
    
    _filenames = tf.placeholder(tf.string, shape = [None])
    
    _dataset = tf.data.TFRecordDataset(filenames)
    
    _dataset.shuffle(40000)
    _dataset.shuffle(40000)
    
    _dataset = _dataset.map(_map_func)
    _dataset = _dataset.repeat()
    _dataset = _dataset.batch(batch)
    iterator = _dataset.make_initializable_iterator()
    
    sess.run(iterator.initializer, feed_dict={_filenames: filenames})
    return iterator

def get_train_test_set(template, p):
    train_files = []
    test_files = []
    n_files = 0
    
    for i in range(1000):
        if exists(template.format(str(i).zfill(3))):
            n_files += 1
        else:
            break
    
    n_test = ceil(n_files * (1 - p))
    test_idx = choice(n_files, n_test)
    
    for i in range(n_files):
        if i in test_idx:
            test_files.append(template.format(str(i).zfill(3)))
        else:
            train_files.append(template.format(str(i).zfill(3)))
    return train_files, test_files
   
def get_train_test_iter(template, p, batch, sess):
    train_files, test_files = get_train_test_set(template, p)
    # return get_data_iterator(train_files, batch, sess), get_data_iterator(test_files, batch, sess)
    return get_data_iterator(train_files, batch, sess)

if __name__ == '__main__':
    sess = tf.Session()
    filenames = [f'../data/data_t=40.tfrecord']
    iterator = get_data_iterator(filenames, 2, sess, 40)
    for i in range(10):
        x = sess.run(iterator.get_next())
        print (f"index shape: {x['index'].shape}")
        print (f"index: {x['index']}")
        print (f"len: {x['len']}")
        
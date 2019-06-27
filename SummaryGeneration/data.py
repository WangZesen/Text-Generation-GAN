import pickle
import numpy as np
import random
import json

class Data():
    def __init__(self, **args):
        """
            Load pre-built vocabulary and document

            Args: 
                vocab_file: path to vocabulary file
                doc_file: path to document file (including both source text and summary)
                max_src_time: maximal time step for source text
                max_sum_time: maximal time step for summary
        """

        # Parse Arguments
        self.vocab_file = args.get('vocab_file')
        self.doc_file = args.get('doc_file')
        self.max_src_time = args.get('src_time')
        self.max_sum_time = args.get('sum_time')
        self.max_oov_bucket = args.get('max_oov_bucket', -1)
        self.batch_size = args.get('batch_size')
        self.train_ratio = args.get('train_ratio')
        self.seed = args.get('seed', 888)
        self.batch_seed = 999 # random.randint(1, 1000000) # 999 
        random.seed(self.seed)

        # Load Files
        self.vocab = pickle.load(open(self.vocab_file, 'rb'))
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.doc = pickle.load(open(self.doc_file, 'rb'))
        
        # Shuffle
        random.shuffle(self.doc)
        random.seed(self.batch_seed)
        self.train_range = int(len(self.doc) * self.train_ratio)
        
        if self.max_oov_bucket == -1:
            self._detect_max_oov_bucket()

        # parameters
        self.vocab_size = len(self.vocab)
        self.ptr = 0

    def _detect_max_oov_bucket(self):
        """
            Find the maximal number of out-of-vocabulary words in one paired data
        """

        invalid_cnt = 0
        for i in range(len(self.doc)):
            oov = {}
            for j in range(len(self.doc[i][0])): # Source text
                if not (self.doc[i][0][j] in self.vocab):
                    oov[self.doc[i][0][j]] = True
            invalid = False
            for j in range(len(self.doc[i][1])): # Summary
                if not (self.doc[i][1][j] in self.vocab):
                    oov[self.doc[i][1][j]] = True
            if invalid:
                invalid_cnt += 1                    
            self.max_oov_bucket = max(self.max_oov_bucket, len(oov))
        print (f'Maximum oov: {self.max_oov_bucket}')

    def word2id(self, _context, _summary):
        """
            Convert context (List[str]) and summary (List[str]) to index for embedding 
            input (List[int]) and index for pointer mapping (List[int]) and index for 
            output 

            Args:
                _context: a list of token (may include <START>)
                _summary: a list of token (may include <START>)
            Rets:
                src2idx: mapping from input source text to embedding index (shape = (self.max_src_time, ))
                atten2final: mapping from attention distribution to final distribution (shape = (self.max_src_time, ))
                sum2final: mapping from summary to final distribution (shape = (self.max_sum_time, ))
                src_len: length of input source text (scalar, int)
                sum_len: length of summary (scalar, int)
        """

        context = _context[1:] if _context[0] == '<START>' else _context
        summary = _summary[1:] if _summary[0] == '<START>' else _summary

        src2idx = np.zeros((self.max_src_time, ), dtype = np.int32)
        atten2final = np.zeros((self.max_src_time, ), dtype = np.int32)
        sum2final = np.zeros((self.max_sum_time, ), dtype = np.int32)
        src_len = len(context)
        sum_len = len(summary)

        oov = {}
        oov_idx = self.vocab_size

        for i in range(len(context)):
            if (not (context[i] in self.vocab)) and (not (context[i] in oov)):
                oov[context[i]] = oov_idx
                src2idx[i] = self.vocab['<UNKNOWN>']
                atten2final[i] = oov_idx
                oov_idx += 1
            elif not (context[i] in self.vocab):
                src2idx[i] = self.vocab['<UNKNOWN>']
                atten2final[i] = oov[context[i]]
            else:
                src2idx[i] = self.vocab[context[i]]
                atten2final[i] = self.vocab[context[i]]

        for i in range(len(summary)):
            if summary[i] in oov:
                sum2final[i] = oov[summary[i]]
            else:
                sum2final[i] = self.vocab[summary[i]]

        return src2idx, atten2final, sum2final, src_len, sum_len, json.dumps(oov)
    
    def id2word(self, feed_dict, gen_seq):
        batch_size = feed_dict['src_len:0'].shape[0]
        
        src = [[] for i in range(batch_size)]
        ref = [[] for i in range(batch_size)]
        gen = [[] for i in range(batch_size)]
        
        for i in range(batch_size):
            oov = json.loads(feed_dict['oov:0'][i])
            inv_oov = {v: k for k, v in oov.items()}
            for j in range(len(gen_seq[i])):
                if gen_seq[i][j] in self.inv_vocab:
                    gen[i].append(self.inv_vocab[gen_seq[i][j]])
                    if gen[i][-1] == '<END>':
                        break
                else:
                    try:
                        gen[i].append(inv_oov[gen_seq[i][j]] + '(OOV)')
                    except:
                        gen[i].append('<KEY ERROR>')
            for j in range(self.max_src_time):
                if feed_dict['atten2final:0'][i][j][1] == 0:
                    src[i].append('<END>')
                    break
                if feed_dict['atten2final:0'][i][j][1] in self.inv_vocab:
                    src[i].append(self.inv_vocab[feed_dict['atten2final:0'][i][j][1]])
                else:
                    src[i].append(inv_oov[feed_dict['atten2final:0'][i][j][1]] + '(OOV)')
            for j in range(self.max_sum_time):
                if feed_dict['sum2final:0'][i][j][2] == 0:
                    ref[i].append('<END>')
                    break
                if feed_dict['sum2final:0'][i][j][2] in self.inv_vocab:
                    ref[i].append(self.inv_vocab[feed_dict['sum2final:0'][i][j][2]])
                else:
                    ref[i].append(inv_oov[feed_dict['sum2final:0'][i][j][2]] + '(OOV)')
        return src, ref, gen

    @property
    def n_train_batch(self):
        """
            Return the number of batches in training data
        """
        return self.train_range // self.batch_size

    @property
    def n_test_batch(self):
        """
            Return the number of batches in training data
        """
        return (len(self.doc) - self.train_range) // self.batch_size
        
    def get_next_epoch_test(self):
        """
            Return generator of the next epoch of test data. End if running out of data

            Rets:
                src2idx: mapping from input source text to embedding index (shape = (self.batch_size, self.max_src_time, ))
                atten2final: mapping from attention distribution to final distribution (shape = (self.batch_size, self.max_src_time, 2))
                sum2final: mapping from summary to final distribution (shape = (self.batch_size, self.max_sum_time, 3))
                src_len: length of input source text (shape = (self.batch_size, ))
                sum_len: length of summary (shape = (self.batch_size, ))
        """
        
        src2idx = np.zeros((self.batch_size, self.max_src_time), dtype = np.int32)
        atten2final = np.zeros((self.batch_size, self.max_src_time, 2), dtype = np.int32)
        sum2final = np.zeros((self.batch_size, self.max_sum_time, 3), dtype = np.int32)
        sum2finalidx = np.zeros((self.batch_size, self.max_sum_time), dtype = np.int32)
        src_len = np.zeros((self.batch_size, ), dtype = np.int32)
        sum_len = np.zeros((self.batch_size, ), dtype = np.int32)
        oov = ["" for i in range(self.batch_size)]

        # atten2final[:, :, 0] = np.arange(self.batch_size)

        index = random.sample(range(self.train_range, len(self.doc)), ((len(self.doc) - self.train_range) // self.batch_size) * self.batch_size)
        for i in range((len(self.doc) - self.train_range) // self.batch_size):
            offset = i * self.batch_size
            for j in range(self.batch_size):
                src2idx[j], atten2final[j, :, 1], sum2final[j, :, 2], src_len[j], sum_len[j], oov[j] = self.word2id(self.doc[index[offset + j]][0], self.doc[index[offset + j]][1])
                atten2final[j, :, 0] = j
                sum2final[j, :, 0] = j
                sum2final[j, :, 1] = np.arange(self.max_sum_time)
            sum2finalidx = sum2final[:, :, 2]
            feed_dict = {
                'src2idx:0': src2idx,
                'atten2final:0': atten2final,
                'sum2final:0': sum2final,
                'src_len:0': src_len,
                'sum_len:0': sum_len,
                'sum2finalidx:0': sum2finalidx,
                'coverage_on:0': False,
                'oov:0': oov
            }
            yield feed_dict
    
    def get_next_sample(self):
        src2idx = np.zeros((1, self.max_src_time), dtype = np.int32)
        atten2final = np.zeros((1, self.max_src_time, 2), dtype = np.int32)
        sum2final = np.zeros((1, self.max_sum_time, 3), dtype = np.int32)
        sum2finalidx = np.zeros((1, self.max_sum_time), dtype = np.int32)
        src_len = np.zeros((1, ), dtype = np.int32)
        sum_len = np.zeros((1, ), dtype = np.int32)
        oov = [""]
        while True:
            index = random.randint(0, self.train_range - 1)
            src2idx[0], atten2final[0, :, 1], sum2final[0, :, 2], src_len[0], sum_len[0], oov[0] = self.word2id(self.doc[index][0], self.doc[index][1])
            atten2final[0, :, 0] = 0
            sum2final[0, :, 0] = 0
            sum2final[0, :, 1] = np.arange(self.max_sum_time)
            sum2finalidx = sum2final[:, :, 2]
            feed_dict = {
                'src2idx:0': src2idx,
                'atten2final:0': atten2final,
                'sum2final:0': sum2final,
                'src_len:0': src_len,
                'sum_len:0': sum_len,
                'sum2finalidx:0': sum2finalidx,
                'coverage_on:0': False,
                'oov:0': oov
            }
            yield feed_dict

    def get_next_epoch(self):
        """
            Return generator of the next epoch of train data. End if running out of data

            Rets:
                src2idx: mapping from input source text to embedding index (shape = (self.batch_size, self.max_src_time, ))
                atten2final: mapping from attention distribution to final distribution (shape = (self.batch_size, self.max_src_time, 2))
                sum2final: mapping from summary to final distribution (shape = (self.batch_size, self.max_sum_time, 3))
                src_len: length of input source text (shape = (self.batch_size, ))
                sum_len: length of summary (shape = (self.batch_size, ))
        """
        
        src2idx = np.zeros((self.batch_size, self.max_src_time), dtype = np.int32)
        atten2final = np.zeros((self.batch_size, self.max_src_time, 2), dtype = np.int32)
        sum2final = np.zeros((self.batch_size, self.max_sum_time, 3), dtype = np.int32)
        sum2finalidx = np.zeros((self.batch_size, self.max_sum_time), dtype = np.int32)
        src_len = np.zeros((self.batch_size, ), dtype = np.int32)
        sum_len = np.zeros((self.batch_size, ), dtype = np.int32)
        oov = ["" for i in range(self.batch_size)]

        index = random.sample(range(self.train_range), (self.train_range // self.batch_size) * self.batch_size)
        for i in range(self.train_range // self.batch_size):
            offset = i * self.batch_size
            for j in range(self.batch_size):
                src2idx[j], atten2final[j, :, 1], sum2final[j, :, 2], src_len[j], sum_len[j], oov[j] = self.word2id(self.doc[index[offset + j]][0], self.doc[index[offset + j]][1])
                atten2final[j, :, 0] = j
                sum2final[j, :, 0] = j
                sum2final[j, :, 1] = np.arange(self.max_sum_time)
            sum2finalidx = sum2final[:, :, 2]
            feed_dict = {
                'src2idx:0': src2idx,
                'atten2final:0': atten2final,
                'sum2final:0': sum2final,
                'src_len:0': src_len,
                'sum_len:0': sum_len,
                'sum2finalidx:0': sum2finalidx,
                'coverage_on:0': False,
                'oov:0': oov
            }
            yield feed_dict


if __name__ == '__main__':
    test_args = {
        'vocab_file': './data/vocab.p',
        'doc_file': './data/doc.p',
        'src_time': 1000,
        'sum_time': 120,
        'max_oov_bucket': 280,
        'batch_size': 10
    }
    data = Data(**test_args)

    generator = data.get_next_epoch()

    import time
    t = time.time()
    batch_cnt = 0
    for feed_dict in generator:
        batch_cnt += 1
    elapsed = time.time() - t

    print (f"# of batches: {batch_cnt} with batch size {test_args['batch_size']}")
    print (f'time taken: {elapsed} sec')

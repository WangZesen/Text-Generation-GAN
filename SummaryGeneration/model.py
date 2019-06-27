import tensorflow as tf
import numpy as np
import pickle
from datetime import date
from tqdm import tqdm_notebook
import copy
from numba import cuda

# auxiliary functions

def split_to_list(x, t):
    '''
        Split the tensor to a list on the time axis

        Args:
            x: 3D tensor (batch, time, h_dim)
            t: # of time steps
        Rets:
            one list containing all slices
    '''
    results = tf.split(x, t, 1)
    for i in range(t):
        results[i] = tf.squeeze(results[i], 1)
    return results


# Class for SummaryModel

class SummaryModel:
    def __init__(self, **args):
        '''
            Constructor for Pointer-Generator Model
        '''
        
        ## Initial Tensorflow Session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        
        self.sess = tf.Session(config = config)
        
        ## Fetch Arguments
        
        # File Directories
        self.emb_file = args['emb_file']
        self.model_prefix = 'Pointer'
        self.model_save_dir = './model/'
        self.model_save_interval = args['save_interval']
        self._load_data()
        
        # Model Structure
        self.num_unit = 256
        self.dim_v = 16
        self.vocab_dim = self.emb_np.shape[0]
        self.final_dim = self.emb_np.shape[0] + args['max_oov_bucket']
        self.emb_np = np.concatenate((self.emb_np, np.zeros((self.final_dim - self.vocab_dim, self.emb_dim))), axis = 0)
        self.max_src_time = args['src_time']
        self.max_sum_time = args['sum_time']
        self.beam = 16

        # Hyperparamter
        self.batch_size = args['batch_size']
        self.gen_lr = args['gen_lr']
        self.dis_lr = args['dis_lr']
        self.cov_weight = args['cov_weight']

        self._add_placeholder_variable()
        self._construct_encoder()
        self._construct_decoder()
        
        self.fake_sum = tf.nn.embedding_lookup(self.emb, self.tokens)
        self.real_sum = tf.nn.embedding_lookup(self.emb, self.sum2finalidx)
        
        self.real_reward = self._construct_discriminator(self.real_sum, self.sum_len)
        self.fake_reward = self._construct_discriminator(self.fake_sum, tf.cast(self.tokens_len, tf.int32))
        
        self.real_bas = self._construct_baseline(self.real_sum, self.sum_len)
        self.fake_bas = self._construct_baseline(self.fake_sum, tf.cast(self.tokens_len, tf.int32))
        
        self._build_loss_opt()
        self._build_summary()
        self._beam_search_tf()
        
        
        
        # Initialize Parameters
        self.sess.run(tf.initializers.global_variables())
        
        # Saver
        # all_variable = tf.all_variables()
        # all_variable = tf.contrib.framework.filter_variables(all_variable, exclude_patterns = ['Adam'])
        # self.saver = tf.train.Saver(allow_empty = True, name = f'Pointer_{date.today().strftime("_%m_%d")}', var_list = all_variable)
        self.saver = tf.train.Saver(allow_empty = True, name = f'Pointer_{date.today().strftime("_%m_%d")}')
        
        if args['load_pretrain']:
            
            all_variable = tf.all_variables()
            all_variable = tf.contrib.framework.filter_variables(all_variable, exclude_patterns = ['discriminator', 'baseline'])
            self.saver = tf.train.Saver(name = f'Pointer_{date.today().strftime("_%m_%d")}', var_list = all_variable)
            if ('checkpoint' in args) and (not (args['checkpoint'] == None)):
                print (f"Restore Model from {args['checkpoint']}")
                self.saver.restore(self.sess, args['checkpoint'])
            self.saver = tf.train.Saver(allow_empty = True, name = f'Pointer_{date.today().strftime("_%m_%d")}')
        else:
            if ('checkpoint' in args) and (not (args['checkpoint'] == None)):
                
                print (f"Restore Model from {args['checkpoint']}")
                self.saver.restore(self.sess, args['checkpoint'])
        self.saver = tf.train.Saver(allow_empty = True, name = f'Pointer_{date.today().strftime("_%m_%d")}')

    def _load_data(self):
        '''
            Load embedding matrix and store as tensor to self.emb
        '''
        self.emb_np = pickle.load(open(self.emb_file, 'rb'))
        self.emb_np = (self.emb_np - np.min(self.emb_np)) / (np.max(self.emb_np) - np.min(self.emb_np)) * 2 - 1
        self.emb_dim = self.emb_np.shape[1]
        

    def _add_placeholder_variable(self):
        '''
            Set placeholders for inputs and global steps for optimizers
        '''
        self.src2idx = tf.placeholder(shape = [self.batch_size, self.max_src_time], dtype = tf.int32, name = 'src2idx')
        self.atten2final = tf.placeholder(shape = [self.batch_size, self.max_src_time, 2], dtype = tf.int32, name = 'atten2final')
        self.sum2final = tf.placeholder(shape = [self.batch_size, self.max_sum_time, 3], dtype = tf.int32, name = 'sum2final')
        self.sum2finalidx = tf.placeholder(shape = [self.batch_size, self.max_sum_time], dtype = tf.int32, name = 'sum2finalidx')
        self.src_len = tf.placeholder(shape = [self.batch_size], dtype = tf.int32, name = 'src_len')
        self.sum_len = tf.placeholder(shape = [self.batch_size], dtype = tf.int32, name = 'sum_len')
        self.oov = tf.placeholder(shape = [self.batch_size], dtype = tf.string, name = 'oov')
        self.gen_global_step = tf.Variable(0, trainable = False, name = 'gen_global_step')
        self.dis_global_step = tf.Variable(0, trainable = False, name = 'dis_global_step')
        self.coverage_on = tf.placeholder(shape = [], dtype = tf.bool, name = 'coverage_on')
        self.emb = tf.Variable(self.emb_np, dtype = tf.float32, name = 'encoder/embedding_matrix')

    def _construct_encoder(self):
        '''
            Construct Encoder
            
            Args (from self):
                self.emb_input: index of source text, shape = (batch_size, max_src_time)
                self.emb: embedding matrix, shape = (vocab_size, emb_dim)
            
            Rets (to self):
                self.enc_output: tensor, shape = (batch_size, max_src_time, num_unit * 2)
                self.enc_final_state: LSTMStateTuple, shape = (batch_size, num_unit * 2), (batch_size, num_unit * 2)
        '''
        with tf.variable_scope('encoder', reuse = tf.AUTO_REUSE):
            self.emb_input = tf.nn.embedding_lookup(self.emb, self.src2idx)
            self.enc_fw_unit = tf.nn.rnn_cell.LSTMCell(self.num_unit, name = 'encoder_forward_cell')
            self.enc_bw_unit = tf.nn.rnn_cell.LSTMCell(self.num_unit, name = 'encoder_backward_cell')
            (self.enc_fw_output, self.enc_bw_output), (self.enc_fw_state, self.enc_bw_state) = tf.nn.bidirectional_dynamic_rnn(self.enc_fw_unit, self.enc_bw_unit, self.emb_input, self.src_len, dtype = tf.float32)

            self.enc_output = tf.concat([self.enc_fw_output, self.enc_bw_output], axis = 2)

            self.enc_c_state = tf.concat([self.enc_fw_state.c, self.enc_bw_state.c], axis = 1)
            self.enc_h_state = tf.concat([self.enc_fw_state.h, self.enc_bw_state.h], axis = 1)
            
            self.enc_dense_c = tf.layers.Dense(self.num_unit, activation = tf.nn.relu, kernel_initializer = tf.contrib.layers.variance_scaling_initializer(), name = 'encoder_reduce_c_dense')
            self.enc_dense_h = tf.layers.Dense(self.num_unit, activation = tf.nn.relu, kernel_initializer = tf.contrib.layers.variance_scaling_initializer(), name = 'encoder_reduce_h_dense')
            
            self.enc_final_state = tf.concat([self.enc_dense_c(self.enc_c_state), self.enc_dense_h(self.enc_h_state)], axis = 1)

    def _construct_decoder(self):
        '''
            Construct Decoder

            Args (from self):
                self.enc_final_state: state of the last time step of the encoder, type = LSTMStateTuple, shape = (batch_size, num_unit * 2)
                self.enc_output: outputs from the time steps in encoder, shape = (batch_size, max_src_time, num_unit * 2)

            Rets (to self):
                self.final_dist: word probability distribution for the summary
        '''
        with tf.variable_scope('decoder', reuse = tf.AUTO_REUSE):
                   
                   
            
            self.dec_unit = tf.nn.rnn_cell.LSTMCell(self.num_unit, state_is_tuple = False, name = 'decoder_cell')

            # For Input
            self.dec_emb_input = tf.TensorArray(tf.float32, size = self.max_sum_time, element_shape = [self.batch_size, self.emb_dim])
            self.dec_emb_input = self.dec_emb_input.unstack(tf.transpose(tf.nn.embedding_lookup(self.emb, self.sum2finalidx), perm = [1, 0, 2]))
            self.dec_input_single = tf.Variable(tf.zeros([1, self.emb_dim + self.num_unit * 2], dtype = tf.float32))
            self.dec_input = tf.tile(self.dec_input_single, [self.batch_size, 1], name = 'dec_input')

            # For Attention Distribution
            self.linear_enc_output = tf.layers.Dense(self.dim_v, use_bias = True, name = 'linear_enc_output')
            self.linear_dec_state = tf.layers.Dense(self.dim_v, use_bias = False, name = 'linear_dec_state')
            self.linear_coverage = tf.layers.Dense(self.dim_v, use_bias = False, name = 'linear_coverage')
            self.v = tf.layers.Dense(1, use_bias = False, name = 'atten_dense_v')
            self.atten_mask = tf.sequence_mask(self.src_len, self.max_src_time, dtype = tf.float32)
            '''
            self._coverage_filter = tf.Variable(tf.random.uniform([1, self.batch_size], -1, 1, dtype = tf.float32), name = 'single_cov_filter')
            self.coverage_fitler = tf.tile(self._coverage_filter, [self.atten_mask.get_shape()[0], 1])
            '''
            # For Vocabulary Distribution
            self.linear_vocab_1 = tf.layers.Dense(self.num_unit * 2, activation = tf.nn.leaky_relu, name = 'linear_vocab_1')
            self.linear_vocab_2 = tf.layers.Dense(self.vocab_dim, activation = tf.nn.leaky_relu, name = 'linear_vocab_2')
            
            # For Final Distribution
            self.linear_context = tf.layers.Dense(1, name = 'linear_context')
            self.linear_state = tf.layers.Dense(1, name = 'linear_state')
            self.linear_input = tf.layers.Dense(1, name = 'linear_input')

            # Recurrent Neural Network (Supervised)
            def next_time_step(step, _state, _input, emb_input, coverage, cov_loss, atten_dist, vocab_dist, final_dist, tokens, p_gen_dist):
                next_output, next_state = self.dec_unit(_input, _state)

                # Build Attention Distribution
                linear_1 = self.linear_enc_output(self.enc_output) # batch_size * max_src_time * v_dim
                linear_2 = self.linear_dec_state(next_state) # batch_size * v_dim
                linear_2 = tf.tile(tf.expand_dims(linear_2, axis = 1), [1, self.max_src_time, 1]) # batch_size * max_src_time * v_dim
                linear_3 = self.linear_coverage(tf.expand_dims(coverage, axis = 2))
                e = tf.cond(self.coverage_on, 
                            lambda: tf.squeeze(self.v(tf.tanh(linear_1 + linear_2 + linear_3))),
                            lambda: tf.squeeze(self.v(tf.tanh(linear_1 + linear_2))) )
                atten = tf.nn.softmax(e) * self.atten_mask # batch_size * max_src_time [0, 1]
                normalize = tf.reshape(tf.reduce_sum(atten, axis = 1), [-1, 1])
                atten = atten / normalize
                atten_dist = atten_dist.write(step, atten)
                   
                # Calculate Coverage Loss
                cov_loss = cov_loss.write(step, tf.reduce_sum(tf.math.minimum(coverage, atten), axis = 1))
                coverage += atten

                # Build Vocabulary Distribution
                context_vec = tf.reduce_sum(self.enc_output * tf.expand_dims(atten, 2), axis = 1) # batch_size * (num_unit * 2)
                context_state = tf.concat([context_vec, next_state], axis = 1) # batch_size * (num_unit * 2 + num_unit * 2)
                vocab = tf.nn.softmax(self.linear_vocab_2(self.linear_vocab_1(context_state)))
                vocab_dist = vocab_dist.write(step, vocab)

                # Build Final Distribution
                ## Calculate p_{gen}
                p_gen = tf.sigmoid(self.linear_context(context_vec) + self.linear_state(next_state) + self.linear_input(_input))
                p_gen_dist = p_gen_dist.write(step, p_gen)

                ## Update attention to final distribution
                vocab_reshape = tf.concat([p_gen * vocab, tf.zeros([self.batch_size, self.final_dim - self.vocab_dim])], axis = 1)
                atten_reshape = tf.scatter_nd(self.atten2final, (1 - p_gen) * atten, vocab_reshape.get_shape())
                final = vocab_reshape + atten_reshape
                final_dist = final_dist.write(step, final)
                
                # Get Next Token
                next_token = tf.argmax(final, axis = 1, output_type = tf.int32)
                # next_input = tf.nn.embedding_lookup(self.emb, next_token)
                next_input = emb_input.read(step)
                next_input = tf.concat([next_input, context_vec], axis = 1)
                tokens = tokens.write(step, next_token)
                
                return step + 1, next_state, next_input, emb_input, coverage, cov_loss, atten_dist, vocab_dist, final_dist, tokens, p_gen_dist         
                   
            step = tf.constant(0, dtype = tf.int32)
            cond = lambda step, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10: step < self.max_sum_time
            last_step, final_state, final_input, emb_input, self.coverage, self.cov_loss, self.atten_dist, self.vocab_dist, self.final_dist, _, self.p_gen = tf.while_loop(
                cond = cond,
                body = next_time_step,
                loop_vars = (
                    step,
                    self.enc_final_state,
                    self.dec_input,
                    self.dec_emb_input,
                    tf.zeros([self.batch_size, self.max_src_time], dtype = tf.float32),
                    tf.TensorArray(tf.float32, size = self.max_sum_time, element_shape = [self.batch_size]),
                    tf.TensorArray(tf.float32, size = self.max_sum_time, element_shape = [self.batch_size, self.max_src_time]),
                    tf.TensorArray(tf.float32, size = self.max_sum_time, element_shape = [self.batch_size, self.vocab_dim]),
                    tf.TensorArray(tf.float32, size = self.max_sum_time, element_shape = [self.batch_size, self.final_dim]),
                    tf.TensorArray(tf.int32, size = self.max_sum_time, element_shape = [self.batch_size]),
                    tf.TensorArray(tf.float32, size = self.max_sum_time, element_shape = [self.batch_size, 1])
                    ),
                swap_memory = True,
                return_same_structure = True
            )
            
            self.atten_dist = tf.transpose(self.atten_dist.stack(name = 'attention_distribution'), perm = [1, 0, 2])
            self.vocab_dist = tf.transpose(self.vocab_dist.stack(name = 'vocabulary_distribution'), perm = [1, 0, 2])
            self.final_dist = tf.transpose(self.final_dist.stack(name = 'final_distribution'), perm = [1, 0, 2])
            self.cov_loss = tf.transpose(self.cov_loss.stack(name = 'covarge_loss'), perm = [1, 0])
            # self.tokens = tf.transpose(self.tokens.stack(name = 'generated_sequence'), perm = [1, 0])
            self.p_gen = tf.transpose(tf.squeeze(self.p_gen.stack(name = 'p_gen')), perm = [1, 0])
                   
            # Unsupervisedly Generate token       
            def next_time_step_unsup(step, _state, _input, emb_input, coverage, cov_loss, tokens, p_gen_dist, gen_prob_dist, is_end):
                next_output, next_state = self.dec_unit(_input, _state)

                # Build Attention Distribution
                linear_1 = self.linear_enc_output(self.enc_output) # batch_size * max_src_time * v_dim
                linear_2 = self.linear_dec_state(next_state) # batch_size * v_dim
                linear_2 = tf.tile(tf.expand_dims(linear_2, axis = 1), [1, self.max_src_time, 1]) # batch_size * max_src_time * v_dim
                linear_3 = self.linear_coverage(tf.expand_dims(coverage, axis = 2))
                e = tf.cond(self.coverage_on, 
                            lambda: tf.squeeze(self.v(tf.tanh(linear_1 + linear_2 + linear_3))),
                            lambda: tf.squeeze(self.v(tf.tanh(linear_1 + linear_2))) )
                atten = tf.nn.softmax(e) * self.atten_mask # batch_size * max_src_time [0, 1]
                normalize = tf.reshape(tf.reduce_sum(atten, axis = 1), [-1, 1])
                atten = atten / normalize
                
                # Calculate Coverage Loss
                cov_loss = cov_loss.write(step, tf.reduce_sum(tf.math.minimum(coverage, atten), axis = 1))
                coverage += atten
                
                # Build Vocabulary Distribution
                context_vec = tf.reduce_sum(self.enc_output * tf.expand_dims(atten, 2), axis = 1) # batch_size * (num_unit * 2)
                context_state = tf.concat([context_vec, next_state], axis = 1) # batch_size * (num_unit * 2 + num_unit * 2)
                vocab = tf.nn.softmax(self.linear_vocab_2(self.linear_vocab_1(context_state)))
                
                
                # Build Final Distribution
                ## Calculate p_{gen}
                p_gen = tf.sigmoid(self.linear_context(context_vec) + self.linear_state(next_state) + self.linear_input(_input))
                p_gen_dist = p_gen_dist.write(step, p_gen)
                
                ## Update attention to final distribution
                vocab_reshape = tf.concat([p_gen * vocab, tf.zeros([self.batch_size, self.final_dim - self.vocab_dim])], axis = 1)
                atten_reshape = tf.scatter_nd(self.atten2final, (1 - p_gen) * atten, vocab_reshape.get_shape())
                final = vocab_reshape + atten_reshape
                
                
                # Get Next Token
                log_prob = tf.log(tf.clip_by_value(final, 1e-30, 1.0))
                #next_token = tf.cond(tf.random.uniform([], 0, 1) > 0.1, 
                #                    lambda: tf.squeeze(tf.random.multinomial(log_prob, 1, output_dtype = tf.int32)), 
                #                    lambda: tf.random.uniform([self.batch_size], 0, self.final_dim + 1, dtype = tf.int32) )
                next_token = tf.squeeze(tf.random.multinomial(log_prob, 1, output_dtype = tf.int32))
                next_token = tf.where(tf.equal(is_end, 0), tf.zeros(next_token.get_shape(), dtype = tf.int32), next_token)
                       
                is_end = tf.where(tf.equal(next_token, 0), tf.zeros(next_token.get_shape()), is_end)
                       
                next_input = tf.nn.embedding_lookup(self.emb, next_token)
                next_input = tf.concat([next_input, context_vec], axis = 1)
                tokens = tokens.write(step, next_token)
                   
                
                next_token_mask = tf.one_hot(next_token, self.final_dim, dtype = tf.float32)
                
                step_log_prob = tf.reduce_sum(tf.multiply(log_prob, next_token_mask), axis = 1)
                gen_prob_dist = gen_prob_dist.write(step, step_log_prob)
                
                return step + 1, next_state, next_input, emb_input, coverage, cov_loss, tokens, p_gen_dist, gen_prob_dist, is_end
                   
            step = tf.constant(0, dtype = tf.int32)
            cond = lambda step, _1, _2, _3, _4, _5, _6, _10, _11, _12: step < self.max_sum_time
            _, _, _, _, _, _, self.tokens, _, self.gen_prob, _ = tf.while_loop(
                cond = cond,
                body = next_time_step_unsup,
                loop_vars = (
                    step,
                    self.enc_final_state,
                    self.dec_input,
                    self.dec_emb_input,
                    tf.zeros([self.batch_size, self.max_src_time], dtype = tf.float32),
                    tf.TensorArray(tf.float32, size = self.max_sum_time, element_shape = [self.batch_size]),
                    tf.TensorArray(tf.int32, size = self.max_sum_time, element_shape = [self.batch_size]),
                    tf.TensorArray(tf.float32, size = self.max_sum_time, element_shape = [self.batch_size, 1]),
                    tf.TensorArray(tf.float32, size = self.max_sum_time, element_shape = [self.batch_size]),
                    tf.ones([self.batch_size], dtype = tf.float32)
                    ),
                swap_memory = True,
                return_same_structure = True
            )
            
            self.tokens = tf.transpose(self.tokens.stack(name = 'generated_sequence'), perm = [1, 0])
            self.tokens_len = tf.stop_gradient(tf.clip_by_value(self.max_sum_time - tf.reduce_sum(tf.cast(tf.equal(self.tokens, 0), dtype = tf.float32), axis = 1) + 1, 0., self.max_sum_time))
            self.gen_prob = tf.transpose(self.gen_prob.stack(name = 'generated_sequence_prob'), perm = [1, 0])
                   
    def _construct_discriminator(self, sum_emb, sum_len):
        '''
            Construct discriminator
            
            Idea: 
                1. Readablility => CNN
                2. Content Correlation => LSTM
        '''
        
        with tf.variable_scope('discriminator', reuse = tf.AUTO_REUSE):
            # LSTM Classifier
            with tf.variable_scope('lstm', reuse = tf.AUTO_REUSE):
                
                self.dis_enc_unit = tf.nn.rnn_cell.LSTMCell(self.num_unit, name = 'dis_enc_unit')
                # self.dis_enc_bw_unit = tf.nn.rnn_cell.LSTMCell(self.num_unit, name = 'dis_enc_bw_unit')
                self.dis_dec_unit = tf.nn.rnn_cell.LSTMCell(self.num_unit, name = 'dis_dec_unit')
                  
                                    
                _, state = tf.nn.dynamic_rnn(self.dis_enc_unit, self.emb_input, sequence_length = self.src_len, swap_memory = False, dtype = tf.float32)
                # (_, _), (final_fs, final_bs) = tf.nn.bidirectional_dynamic_rnn(self.dis_enc_unit, self.dis_enc_bw_unit, self.emb_input, sequence_length = self.src_len, swap_memory = False, dtype = tf.float32)
                
                                    
                # self.dis_state_dense = tf.layers.Dense(self.num_unit * 2, name = 'dis_state_layer')
                # self.dis_final_state = self.dis_state_dense(tf.concat([final_fs,final_bs], axis = 1))
                # enc_c_state = tf.concat([final_fs.c, final_bs.c], axis = 1)
                # enc_h_state = tf.concat([final_fs.h, final_bs.h], axis = 1)
                # print (enc_c_state.get_shape())
                                    
                output, state = tf.nn.dynamic_rnn(self.dis_dec_unit, sum_emb, initial_state = state, sequence_length = sum_len, swap_memory = False)
                
                
                score_layer = tf.layers.Dense(1, name = 'dis_score_layer')
                reward = tf.squeeze(tf.sigmoid(score_layer(output)), 2)
                
                reward = tf.pad(reward, [[0, 0], [0, self.max_sum_time - tf.shape(reward)[1]]], 'CONSTANT', constant_values = 0)
                
                return reward
    
    def _construct_baseline(self, sum_emb, sum_len):
        
        
        with tf.variable_scope('baseline', reuse = tf.AUTO_REUSE):
            # LSTM Classifier
            with tf.variable_scope('lstm', reuse = tf.AUTO_REUSE):
                
                _sum_emb = tf.slice(sum_emb, [0, 0, 0], [self.batch_size, self.max_sum_time - 1, self.emb_dim])
                _sum_emb = tf.concat([tf.zeros([self.batch_size, 1, self.emb_dim]), _sum_emb], axis = 1)
                       
                self.bas_enc_unit = tf.nn.rnn_cell.LSTMCell(self.num_unit, name = 'bas_enc_unit')
                # self.bas_enc_bw_unit = tf.nn.rnn_cell.LSTMCell(self.num_unit, name = 'bas_enc_bw_unit')
                self.bas_dec_unit = tf.nn.rnn_cell.LSTMCell(self.num_unit, name = 'bas_dec_unit')
                
                # (_, _), (final_fs, final_bs) = tf.nn.bidirectional_dynamic_rnn(self.bas_enc_unit, self.bas_enc_bw_unit, self.emb_input, sequence_length = self.src_len, swap_memory = True, dtype = tf.float32)
                _, state = tf.nn.dynamic_rnn(self.bas_enc_unit, self.emb_input, sequence_length = self.src_len, swap_memory = True, dtype = tf.float32)
                
                # self.dis_state_dense = tf.layers.Dense(self.num_unit * 2, name = 'bas_state_layer')
                # self.bas_final_state = self.dis_state_dense(tf.concat([final_fs,final_bs], axis = 1))
                
                output, state = tf.nn.dynamic_rnn(self.bas_dec_unit, _sum_emb, initial_state = state, sequence_length = sum_len, swap_memory = True)
                
                score_layer = tf.layers.Dense(1, name = 'bas_score_layer')
                reward = tf.squeeze(tf.sigmoid(score_layer(output)), 2)
                
                reward = tf.pad(reward, [[0, 0], [0, self.max_sum_time - tf.shape(reward)[1]]], 'CONSTANT', constant_values = 0)
                
                return reward
    
                   
    def _build_loss_opt(self):

        # Gather Variables
        var_dec = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'decoder')
        var_enc = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'encoder')
        var_dis = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'discriminator')
        var_bas = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'baseline')

        # self.decay_weight = tf.tile(tf.expand_dims(tf.range(self.max_sum_time, 0, -1, dtype = tf.float32), 0), [self.batch_size, 1])
                                    
        # Supervised Training Part
        self.seq_mask = tf.sequence_mask(self.sum_len, self.max_sum_time, dtype = tf.float32)
        
        self.fake_seq_mask = tf.stop_gradient(tf.sequence_mask(tf.cast(self.tokens_len, dtype = tf.int32), self.max_sum_time, dtype = tf.float32))
        self.gather_log_prob = - tf.log(tf.clip_by_value(tf.gather_nd(self.final_dist, self.sum2final), 1e-30, 1.))
        
        self.final_cov_loss = tf.reduce_mean((tf.reduce_sum(tf.multiply(self.seq_mask, self.cov_loss), axis = 1) * self.cov_weight) / tf.cast(self.sum_len, dtype = tf.float32))
        #self.final_pri_loss = tf.reduce_sum((tf.reduce_sum(tf.multiply(self.seq_mask, self.gather_log_prob), axis = 1)) / tf.reshape(tf.cast(self.sum_len, dtype = tf.float32), [-1, 1])) / self.batch_size
        self.final_pri_loss = tf.reduce_mean((tf.reduce_sum(tf.multiply(self.seq_mask, self.gather_log_prob), axis = 1)) / tf.cast(self.sum_len, dtype = tf.float32))
                   
        self.target_loss = tf.cond(self.coverage_on, lambda: self.final_cov_loss + self.final_pri_loss, lambda: self.final_pri_loss)

        self.supervised_opt = tf.train.AdamOptimizer(self.gen_lr).minimize(self.target_loss, var_list = var_dec + var_dec, global_step = self.gen_global_step)
        
        # GAN Part
        self.real_loss = tf.log(tf.reduce_sum(tf.multiply(self.seq_mask, self.real_reward), axis = 1) / tf.cast(self.sum_len, dtype = tf.float32))
        # self.real_loss = tf.reduce_sum(tf.multiply(self.seq_mask, self.real_reward), axis = 1) / tf.cast(self.sum_len, dtype = tf.float32)
        self.fake_loss = tf.log(1 - tf.reduce_sum(tf.multiply(self.fake_seq_mask, self.fake_reward), axis = 1) / tf.cast(self.tokens_len, dtype = tf.float32))
        # self.fake_loss = - tf.reduce_sum(tf.multiply(self.fake_seq_mask, self.fake_reward), axis = 1) / tf.cast(self.tokens_len, dtype = tf.float32)
        self.dis_loss = tf.reduce_mean(- self.real_loss - self.fake_loss)
        
        # self.gen_loss = - tf.reduce_mean(tf.reduce_sum(tf.multiply(tf.multiply(self.fake_reward - self.fake_bas, self.gen_prob), self.fake_seq_mask) , 1) / tf.cast(self.tokens_len, dtype = tf.float32) )
        self.gen_loss = - tf.reduce_mean( \
                            tf.reduce_sum( \
                                
                                    tf.multiply( \
                                        tf.multiply(self.fake_reward - self.fake_bas, self.gen_prob) \
                                        , self.fake_seq_mask) \

                                , 1) \
                            )
        
        self.bas_loss = tf.reduce_sum(tf.multiply(tf.square(self.fake_reward - self.fake_bas), self.fake_seq_mask))
                       
        with tf.variable_scope('discriminator', reuse = tf.AUTO_REUSE):
            self.gen_global_step_2 = tf.Variable(0, trainable = False, name = 'gen_global_step_2')
            self.dis_global_step_2 = tf.Variable(0, trainable = False, name = 'dis_global_step_2')
            self.dis_opt = tf.train.AdamOptimizer(self.dis_lr, name = 'disAdam').minimize(self.dis_loss, var_list = tf.contrib.framework.filter_variables(var_dis, exclude_patterns = ['embedding_matrix']), global_step = self.dis_global_step_2)
            self.gen_opt = tf.train.AdamOptimizer(self.gen_lr * 0.1, name = 'genAdam').minimize(self.gen_loss, var_list = tf.contrib.framework.filter_variables(var_dec, exclude_patterns = ['embedding_matrix']), global_step = self.gen_global_step_2)
            self.bas_opt = tf.train.AdamOptimizer(self.dis_lr, name = 'basAdam').minimize(self.bas_loss, var_list = var_bas)

    def _build_summary(self):
        self.log_suffix = date.today().strftime('_%m_%d')
        summary_target_loss = tf.summary.scalar('Target_Loss', self.target_loss)
        summary_cov_loss = tf.summary.scalar('Coverage_Loss', self.final_cov_loss) 
        summary_prime_loss = tf.summary.scalar('Prime_Loss', self.final_pri_loss)
        summary_dis_loss = tf.summary.scalar('Dis_Loss', self.dis_loss)
        summary_gen_loss = tf.summary.scalar('Gen_Loss', self.gen_loss)
        summary_bas_loss = tf.summary.scalar('Bas_Loss', self.bas_loss)
        self.supervised_summary = tf.summary.merge([summary_target_loss, summary_cov_loss, summary_prime_loss])
        self.unsupervised_summary = tf.summary.merge([summary_dis_loss, summary_gen_loss, summary_bas_loss])
        self.log_writer = tf.summary.FileWriter('./log/', filename_suffix = self.log_suffix)

    def _beam_search_tf(self):
        '''
            Function for constructing the results of beam search (Need auxiliary function to decode: self.beam_search)
        '''
        
        _state_split = tf.TensorArray(tf.float32, size = self.batch_size, element_shape = [1, self.num_unit * 2])
        self.state_split = _state_split.unstack(tf.expand_dims(self.enc_final_state, axis = 1))
        _atten2final_split = tf.TensorArray(tf.int32, size = self.batch_size, element_shape = [1, self.max_src_time, 1])
        self.atten2final_split = _atten2final_split.unstack(tf.expand_dims(tf.slice(self.atten2final, [0, 0, 1], [self.batch_size, self.max_src_time, 1]), axis = 1))
        _src_len_split = tf.TensorArray(tf.int32, size = self.batch_size, element_shape = [1])
        self.src_len_split = _src_len_split.unstack(tf.expand_dims(self.src_len, axis = 1))
        _enc_output_split = tf.TensorArray(tf.float32, size = self.batch_size, element_shape = [1, self.max_src_time, self.num_unit * 2])
        self.enc_output_split = _enc_output_split.unstack(tf.expand_dims(self.enc_output, axis = 1))


        def next_sample(step, state_array, src_len_array, atten2final_array, enc_output_array, gen_token, gen_prefix, gen_value, gen_atten):
            initial_state = state_array.read(step)
            src_len = src_len_array.read(step)
            atten_mask = tf.sequence_mask(src_len, self.max_src_time, dtype = tf.float32)
            atten2final = atten2final_array.read(step)
            enc_output = enc_output_array.read(step)


            initial_token_np = np.zeros([self.beam, 1], dtype = np.int32)
            initial_token_np[0, 0] = 1
            initial_token = tf.convert_to_tensor(initial_token_np)
            initial_prefix = tf.zeros([self.beam, 1], dtype = tf.int32)
            initial_value = tf.zeros([self.beam, 1], dtype = tf.float32)

            def next_step(inner_step, _input, _state, coverage, _atten_mask, _atten2final, _enc_output, tokens, prefix, value, value_array, atten_array):

                # tile before start
                atten_mask = tf.tile(_atten_mask, [_input.get_shape()[0], 1])

                atten2final_dim0 = tf.tile(tf.reshape(tf.range(0, self.beam, delta = 1, dtype = tf.int32), [self.beam, 1, 1]), [1, self.max_src_time, 1])
                atten2final = tf.concat([atten2final_dim0, tf.tile(_atten2final, [_input.get_shape()[0], 1, 1])], axis = 2)

                enc_output = tf.tile(_enc_output, [_input.get_shape()[0], 1, 1])

                _, next_state = self.dec_unit(_input, _state)

                # Build Attention Distribution
                linear_1 = self.linear_enc_output(enc_output) # beam * max_src_time * v_dim
                linear_2 = self.linear_dec_state(next_state) # beam * v_dim
                linear_2 = tf.tile(tf.expand_dims(linear_2, axis = 1), [1, self.max_src_time, 1]) # beam * max_src_time * v_dim
                linear_3 = self.linear_coverage(tf.expand_dims(coverage, axis = 2))
                e = tf.cond(self.coverage_on, 
                            lambda: tf.squeeze(self.v(tf.tanh(linear_1 + linear_2 + linear_3))),
                            lambda: tf.squeeze(self.v(tf.tanh(linear_1 + linear_2))) )
                atten = tf.nn.softmax(e) * atten_mask # beam * max_src_time [0, 1]
                normalize = tf.reshape(tf.reduce_sum(atten, axis = 1), [-1, 1])
                atten = atten / normalize
                
                atten_array = atten_array.write(inner_step, atten)
                   
                # Calculate Coverage
                coverage += atten

                # Build Vocabulary Distribution
                context_vec = tf.reduce_sum(enc_output * tf.expand_dims(atten, 2), axis = 1) # beam * (num_unit * 2)
                context_state = tf.concat([context_vec, next_state], axis = 1) # beam * (num_unit * 2 + num_unit * 2)
                vocab = tf.nn.softmax(self.linear_vocab_2(self.linear_vocab_1(context_state)))

                # Build Final Distribution
                ## Calculate p_{gen}
                p_gen = tf.sigmoid(self.linear_context(context_vec) + self.linear_state(next_state) + self.linear_input(_input))

                ## Update attention to final distribution
                vocab_reshape = tf.concat([p_gen * vocab, tf.zeros([_input.get_shape()[0], self.final_dim - self.vocab_dim])], axis = 1)
                # _atten_reshape = tf.zeros(vocab_reshape.get_shape(), dtype = tf.float32)
                atten_reshape = tf.scatter_nd(atten2final, (1 - p_gen) * atten, vocab_reshape.get_shape())
                final = tf.log(vocab_reshape + atten_reshape)

                is_end = tf.cast(tf.equal(tf.slice(tokens, [0, inner_step], [self.beam, 1]), 0), dtype = tf.float32)
                is_end = tf.tile(is_end, [1, self.final_dim]) * (- 1e10)

                final = final + is_end + tf.tile(value, [1, self.final_dim])

                new_value, indices = tf.math.top_k(tf.reshape(final, [1, -1]), k = self.beam)

                beam_ids = tf.reshape(indices // self.final_dim, [self.beam])
                ids = tf.reshape(indices % self.final_dim, [self.beam])

                prefix = tf.concat([prefix, tf.expand_dims(beam_ids, axis = 1)], axis = 1)
                tokens = tf.concat([tokens, tf.expand_dims(ids, axis = 1)], axis = 1)
                new_value = tf.reshape(new_value, [self.beam, 1])
                value_array = value_array.write(inner_step, new_value)

                next_state = tf.gather(next_state, beam_ids)
                
                next_input = tf.concat([tf.nn.embedding_lookup(self.emb, ids), tf.gather(context_vec, beam_ids)], axis = 1)

                next_coverage = tf.gather(coverage, beam_ids)


                return inner_step + 1, next_input, next_state, next_coverage, _atten_mask, _atten2final, _enc_output, tokens, prefix, new_value, value_array, atten_array

            cond = lambda step, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11: step < self.max_sum_time # ) and (not (all_end(tokens) == 0))
            _, _, _, _, _, _, _, gen_token_array, gen_prefix_array, _, gen_value_array, gen_atten_array = tf.while_loop(cond = cond,
                            body = next_step,
                            shape_invariants = (
                                tf.TensorShape([]),
                                tf.TensorShape([self.beam, self.emb_dim + self.num_unit * 2]),
                                tf.TensorShape([self.beam, self.num_unit * 2]),
                                tf.TensorShape([self.beam, self.max_src_time]),
                                atten_mask.get_shape(),
                                atten2final.get_shape(),
                                enc_output.get_shape(),
                                tf.TensorShape([self.beam, None]),
                                tf.TensorShape([self.beam, None]),
                                tf.TensorShape([self.beam, 1]),
                                tf.TensorShape([]),
                                tf.TensorShape([])
                            ),
                            loop_vars = (
                                tf.constant(0, dtype = tf.int32),
                                tf.tile(self.dec_input_single, [self.beam, 1]),
                                tf.tile(initial_state, [self.beam, 1]),
                                tf.zeros([self.beam, self.max_src_time], dtype = tf.float32),
                                atten_mask,
                                atten2final,
                                enc_output,
                                initial_token,
                                initial_prefix,
                                initial_value,
                                tf.TensorArray(tf.float32, size = self.max_sum_time, element_shape = [self.beam, 1]),
                                tf.TensorArray(tf.float32, size = self.max_sum_time, element_shape = [self.beam, self.max_src_time])
                            ))

            gen_token = gen_token.write(step, gen_token_array)
            gen_prefix = gen_prefix.write(step, gen_prefix_array)
            gen_value = gen_value.write(step, tf.squeeze(tf.transpose(gen_value_array.stack(), perm = [1, 0, 2])))
            gen_atten = gen_atten.write(step, tf.transpose(gen_atten_array.stack(), perm = [1, 0, 2]))

            return step + 1, state_array, src_len_array, atten2final_array, enc_output_array, gen_token, gen_prefix, gen_value, gen_atten


        cond = lambda step, _1, _2, _3, _4, _5, _6, _7, _8: step < self.batch_size
        _, _, _, _, _, _gen_token, _gen_prefix, _gen_value, _gen_atten = tf.while_loop(cond = cond,
                                                body = next_sample,
                                                loop_vars = (
                                                    tf.constant(0, dtype = tf.int32),
                                                    self.state_split,
                                                    self.src_len_split,
                                                    self.atten2final_split,
                                                    self.enc_output_split,
                                                    tf.TensorArray(tf.int32, size = self.batch_size, element_shape = [self.beam, self.max_sum_time + 1]),
                                                    tf.TensorArray(tf.int32, size = self.batch_size, element_shape = [self.beam, self.max_sum_time + 1]),
                                                    tf.TensorArray(tf.float32, size = self.batch_size, element_shape = [self.beam, self.max_sum_time]),
                                                    tf.TensorArray(tf.float32, size = self.batch_size, element_shape = [self.beam, self.max_sum_time, self.max_src_time])
                                                ))
        self.gen_token = _gen_token.stack()
        self.gen_prefix = _gen_prefix.stack()
        self.gen_value = _gen_value.stack()
        self.gen_atten = _gen_atten.stack()
                   
    def beam_search(self, feed_dict, coverage_on = False, top_k = 1):
        '''
            Beam search for best generated sequence
            Rets:
                sequences: (batch_size, max_sum_time)
        '''
        feed_dict['coverage_on:0'] = coverage_on
        seq, pre, val, att = self.sess.run([self.gen_token, self.gen_prefix, self.gen_value, self.gen_atten], feed_dict = feed_dict)
        if top_k == 1:
            all_tokens = []
            all_scores = []
            all_attens = []
            for i in range(self.batch_size):
                best_val = -1e10
                index = [-1, -1]
                for j in range(self.beam):
                    for k in range(self.max_sum_time - 1):
                        if (val[i, j, k] / (k + 1) > best_val) and (seq[i, j, k + 1] == 0):
                            best_val = val[i, j, k] / (k + 1)
                            index = [j, k + 1]
                tokens = []
                attens = []
                while index[1] > 0:
                    tokens.append(seq[i, index[0], index[1]])
                    attens.append(att[i, index[0], index[1]])
                    index[0], index[1] = pre[i, index[0], index[1]], index[1] - 1
                all_scores.append(best_val)
                tokens.reverse()
                all_tokens.append(tokens)
                attens.reverse()
                all_attens.append(attens)
            return all_tokens, all_scores, all_attens
        else:
            all_tokens = [[] for col in range(top_k)]
            all_scores = [[] for col in range(top_k)]
            all_attens = [[] for col in range(top_k)]
            for i in range(self.batch_size):
                last_best = 1e10
                for _top_k in range(top_k):
                    best_val = -1e10
                    index = [-1, -1]
                    for j in range(self.beam):
                        for k in range(self.max_sum_time - 1):
                            if (val[i, j, k] / (k + 1) > best_val) and (seq[i, j, k + 1] == 0) and (val[i, j, k] / (k + 1) < last_best - 1e-10):
                                best_val = val[i, j, k] / (k + 1)
                                index = [j, k + 1]
                    tokens = []
                    attens = []
                    while index[1] > 0:
                        tokens.append(seq[i, index[0], index[1]])
                        attens.append(att[i, index[0], index[1]])
                        index[0], index[1] = pre[i, index[0], index[1]], index[1] - 1
                    all_scores[_top_k].append(best_val)
                    tokens.reverse()
                    all_tokens[_top_k].append(tokens)
                    attens.reverse()
                    all_attens[_top_k].append(attens)
                    last_best = best_val
            return all_tokens, all_scores, all_attens
        
                   
    def train_one_epoch(self, generator, total_n_batch, coverage_on = False, model_name = 'pointer_cov_supervised'):
        '''
            Train for one epoch
            
            Args:
                generator of the train data
            Rets:
                None
        '''
        step = self.sess.run(self.gen_global_step)
        for feed_dict in tqdm_notebook(generator, total = total_n_batch):
            feed_dict['coverage_on:0'] = coverage_on
            self.sess.run(self.supervised_opt, feed_dict = feed_dict)
            step += 1
            if step % 10 == 0:
                log = self.sess.run(self.supervised_summary, feed_dict = feed_dict)
                self.log_writer.add_summary(log, step)
            if (step + 1) % self.model_save_interval == 0:
                self._save_model(step + 1, model_name = model_name)
    
    def train_one_epoch_pre_dis(self, generator, total_n_batch, coverage_on = False, model_name = 'pointer_unsupervised_dis_pre'):
        '''
            Train for one epoch
            
            Args:
                generator of the train data
            Rets:
                None
        '''
        self.n_dis = 3
        step = self.sess.run(self.dis_global_step_2)
        
        for feed_dict in tqdm_notebook(generator, total = total_n_batch // (self.n_dis + 1) * (self.n_dis + 1)):
            feed_dict['coverage_on:0'] = coverage_on
            self.sess.run([self.dis_opt, self.bas_opt], feed_dict = feed_dict)
            step += 1
            
            if step % 10 == 0:
                log = self.sess.run(self.unsupervised_summary, feed_dict = feed_dict)
                self.log_writer.add_summary(log, step)
            if (step + 1) % self.model_save_interval == 0:
                self._save_model(step + 1, model_name = model_name)
                       
    def train_one_epoch_unsup(self, generator, total_n_batch, coverage_on = False, model_name = 'pointer_unsupervised'):
        '''
            Train for one epoch
            
            Args:
                generator of the train data
            Rets:
                None
        '''
        self.n_dis = 3
        step = self.sess.run(self.gen_global_step_2) * (1 + self.n_dis)
        
        cnt = 0
                                    
        for feed_dict in tqdm_notebook(generator, total = total_n_batch // (self.n_dis + 1) * (self.n_dis + 1)):
            feed_dict['coverage_on:0'] = coverage_on
            if step % (self.n_dis + 1) == 0:
                self.sess.run([self.bas_opt, self.gen_opt, tf.assign(self.dis_global_step_2, self.gen_global_step_2)], feed_dict = feed_dict)
            else:
                self.sess.run([self.dis_opt], feed_dict = feed_dict)
                
            
            
            if cnt % 250 == 0:
                self._save_model(step + 1, model_name = f'inter')
                self.sess.close()
                # cuda.select_device(0)
                # cuda.close()
                
                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True
                self.sess = tf.Session(config = config)
                
                self.saver.restore(self.sess, f'./model/inter-{step + 1}')
                
                print ("Restart Done!")
            
            cnt += 1
                                    
            step += 1
            if step % 15 == 0:
                log = self.sess.run(self.unsupervised_summary, feed_dict = feed_dict)
                self.log_writer.add_summary(log, step // (1 + self.n_dis))
            #if (step + 1) % self.model_save_interval == 0:
            #    self._save_model(step + 1, model_name = model_name)
    
    def test_one_batch(self, batch):
        '''
            Return the explicit result (in index) for one batch
            
            Args:
                batch: one batch of data (in feed_dict)
            Rets:
                result: indexes of the generated summary
        '''
        return self.sess.run(self.tokens, feed_dict = batch)
    
    def test_one_epoch(self, generator, total_n_batch):
        '''
            Return the loss for the test data
            
            Args:
                generator: generator of the test data
            Rets:
                loss: average loss of the results
        '''
        
        loss = 0.
        cnt = 0
        for feed_dict in tqdm_notebook(generator, total = total_n_batch):
            loss += self.sess.run(self.target_loss, feed_dict = feed_dict)
            cnt += 1
        return loss / cnt
                   
    def _save_model(self, step, model_name):
        self.saver.save(self.sess, self.model_save_dir + model_name, global_step = step)

if __name__ == '__main__':
    pass
    
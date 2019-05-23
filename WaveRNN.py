# -*- encoding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from hparam import hparams as hp


class WaveRNN_Alternative(object):
    def __init__(self, is_training=True):
        self.n_classes = 2 ** hp.bits
        self.num_mels = hp.num_mels
        self.rnn_dims = hp.rnn_dims
        self._is_training = is_training
        self.batch_size = hp.batch_size

        # tranposed upsampling
        initializer = tf.contrib.layers.xavier_initializer_conv2d()
        bias_initializer = tf.constant_initializer(value=0.0, dtype=tf.float32)
        with tf.variable_scope('WaveRNN', reuse=tf.AUTO_REUSE):
            self.upsample_conv_in = tf.get_variable('upsample_conv_in', [5, 80, 128], initializer=initializer)
            self.upsample_filter1 = tf.get_variable('upsampling_filter1', [16 * 3, 128, 128], initializer=initializer)
            self.upsample_filter2 = tf.get_variable('upsampling_filter2', [16 * 3, 128, 128], initializer=initializer)

            # auxiliary local condition
            self.aux_conv_in = tf.get_variable('aux_conv_in', [5, 80, 128], initializer=initializer)
            self.aux_conv_out = tf.get_variable('aux_conv_out', [1, 128, 128], initializer=initializer)
            self.aux_conv_out_bias = tf.get_variable('aux_conv_out_bias', [128], initializer=bias_initializer)
            self.aux_in_bn = tf.layers.BatchNormalization(name='aux_in_bn')
            self.aux_res = {}
            for i in range(hp.res_blocks):
                res_blocks = []
                conv1_params = tf.get_variable('res_{}_conv1'.format(i), [1, 128, 128], initializer=initializer)
                batch_norm1 = tf.layers.BatchNormalization(name='res_{}_bn1'.format(i))
                conv2_params = tf.get_variable('res_{}_conv2'.format(i), [1, 128, 128], initializer=initializer)
                batch_norm2 = tf.layers.BatchNormalization(name='res_{}_bn2'.format(i))
                res_blocks.append(conv1_params)
                res_blocks.append(batch_norm1)
                res_blocks.append(conv2_params)
                res_blocks.append(batch_norm2)
                self.aux_res[i] = res_blocks

            self.gru1 = tf.contrib.cudnn_rnn.CudnnGRU(1, hp.rnn_dims, name='gru1')
            self.gru2 = tf.contrib.cudnn_rnn.CudnnGRU(1, hp.rnn_dims, name='gru2')

            # linear layers
            self.fc_1 = tf.get_variable('fc_1', [1 + 128 + 32, hp.rnn_dims], initializer=initializer)
            self.bias_fc_1 = tf.get_variable('bias_fc_1', [hp.rnn_dims], initializer=bias_initializer)

            self.fc_2 = tf.get_variable('fc_2', [hp.rnn_dims + 32, hp.fc_dims], initializer=initializer)
            self.bias_fc_2 = tf.get_variable('bias_fc_2', [hp.fc_dims], initializer=bias_initializer)

            self.fc_3 = tf.get_variable('fc_3', [hp.fc_dims + 32, hp.fc_dims], initializer=initializer)
            self.bias_fc_3 = tf.get_variable('bias_fc_3', [hp.fc_dims], initializer=bias_initializer)

            self.fc_4 = tf.get_variable('fc_4', [hp.fc_dims, self.n_classes], initializer=initializer)
            self.bias_fc_4 = tf.get_variable('bias_fc_4', [self.n_classes], initializer=bias_initializer)

        ## inference states
        self.states_gru1 = self.gru1._zero_state(batch_size=1)
        self.states_gru2 = self.gru2._zero_state(batch_size=1)

    def build_network(self, x, y, mel):
        """
        build alternative model
        :param x: B*1800*1
        :param y: B*1800
        :param mel: B*5*80
        :return:
        """
        # encode y by one-hot
        y = tf.one_hot(tf.cast(y, tf.int32), depth=2**hp.bits, dtype=tf.float32)

        # 1. upsample network
        aux, mel = self.upsample_network(mel)
        a1, a2, a3, a4 = tf.split(aux, num_or_size_splits=4, axis=-1)        # B*1800*32

        # 2. linear
        input = tf.concat([x, mel, a1], axis=-1)                    # B*1800*(1+128+32)
        input = tf.reshape(input, [-1, 1+128+32])
        x = tf.matmul(input, self.fc_1)
        x = tf.nn.bias_add(x, self.bias_fc_1)                       # B*1800*512
        x = tf.reshape(x, [self.batch_size, -1, hp.rnn_dims])

        # 3. GRU
        # make input time major
        res = x
        x = tf.transpose(x, [1, 0, 2])                              # T*B*512
        outputs, output_states = self.gru1(x, training=True)
        outputs = tf.transpose(outputs, [1, 0, 2])                  # B*T*512
        x = outputs + res

        # 4. GRU
        res = x                                                     # B*1800*512
        x = tf.concat([x, a2], axis=-1)                             # B*1800*(512+32)

        x = tf.transpose(x, [1, 0, 2])                              # T*B*(512+32)
        outputs, states = self.gru2(x, training=True)
        outputs = tf.transpose(outputs, [1, 0, 2])                  # B*T*512
        x = outputs + res

        # 5. linear
        x = tf.concat([x, a3], axis=-1)
        x = tf.reshape(x, [-1, hp.rnn_dims + 32])
        x = tf.matmul(x, self.fc_2)
        x = tf.nn.bias_add(x, self.bias_fc_2)
        x = tf.nn.relu(x)
        x = tf.reshape(x, [self.batch_size, -1, hp.fc_dims])

        # 6. linear
        x = tf.concat([x, a4], axis=-1)
        x = tf.reshape(x, [-1, hp.fc_dims + 32])
        x = tf.matmul(x, self.fc_3)
        x = tf.nn.bias_add(x, self.bias_fc_3)
        x = tf.nn.relu(x)

        # 7. linear to softmax: 2**hp.bits (512)
        output = tf.matmul(x, self.fc_4)
        output = tf.nn.bias_add(output, self.bias_fc_4)

        # compute loss
        output = tf.reshape(output, [-1, self.n_classes])
        y = tf.reshape(y, [-1, self.n_classes])
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y)
        loss = tf.reduce_mean(loss)
        return loss

    def upsample_network(self, mel):
        """
        upsample the input local condition: mel spectrum
        :param mel: B*9*80, 9 frames
        :return: mel, aux      B*5*128, B*5*128
        """
        local_condition = mel
        lc_shape = tf.shape(mel)
        batch_size, lc_length, lc_dim = lc_shape[0], lc_shape[1], lc_shape[2]

        # part1: res net for auxiliary local condition
        mel = tf.nn.conv1d(mel, self.aux_conv_in, stride=1, padding='VALID')        # B*5*128
        mel = self.aux_in_bn(mel, training=self._is_training)                       # batch norm
        mel = tf.nn.relu(mel)

        for i in range(hp.res_blocks):
            input = mel
            params = self.aux_res[i]
            output = tf.nn.conv1d(input, params[0], stride=1, padding='SAME')
            output = params[1](output, training=self._is_training)                  # batch norm
            output = tf.nn.relu(output)

            output = tf.nn.conv1d(output, params[2], stride=1, padding='SAME')
            output = params[3](output, training=self._is_training)                  # batch norm
            mel = output + input

        mel = tf.nn.conv1d(mel, self.aux_conv_out, stride=1, padding='SAME')
        mel = tf.nn.bias_add(mel, self.aux_conv_out_bias)

        # upsampling by repeat
        mel = tf.tile(mel, [1, 1, hp.upsampling_rate])
        aux = tf.reshape(mel, [batch_size, -1, 128])             # B*1800*128

        # part 2: upsampling local condition with tranoposed conv1d
        input = local_condition                                     # B*9*80
        mel = tf.nn.conv1d(input, self.upsample_conv_in, stride=1, padding='VALID')  # B*5*128

        # tranposed upsampling 1
        lc_shape = tf.shape(mel)
        batch_size, lc_length, lc_dim = lc_shape[0], lc_shape[1], lc_shape[2]
        stride1 = 16
        output_shape = [batch_size, lc_length * stride1, 128]
        mel = tf.contrib.nn.conv1d_transpose(mel, self.upsample_filter1, output_shape, stride=stride1)
        mel = tf.nn.relu(mel)

        # tranposed upsampling 2
        lc_shape = tf.shape(mel)
        batch_size, lc_length, lc_dim = lc_shape[0], lc_shape[1], lc_shape[2]
        stride2 = 16
        output_shape = [batch_size, lc_length * stride2, 128]
        mel = tf.contrib.nn.conv1d_transpose(mel, self.upsample_filter2, output_shape, stride=stride2)
        mel = tf.nn.relu(mel)

        return aux, mel

    def inference(self, x, mel, aux):
        """
        inference
        :param x: B*1 [-1, 1]
        :param mel: B*128
        :param aux: B*128
        :return:
        """

        # 1. local condition upsampling
        a1, a2, a3, a4 = tf.split(aux, num_or_size_splits=4, axis=-1)

        # 2. linear
        input = tf.concat([x, mel, a1], axis=-1)                    # B*(1+128+32)
        x = tf.matmul(input, self.fc_1)
        x = tf.nn.bias_add(x, self.bias_fc_1)                       # B*512

        ## make it to [T, B, d] tensors
        # 3. GRU
        x = tf.expand_dims(x, axis=1)                               # B*1*512
        res = x
        x = tf.transpose(x, [1, 0, 2])                              # 1*B*512
        outputs, states_1 = self.gru1(x, initial_state=self.states_gru1, training=False)
        outputs = tf.transpose(outputs, [1, 0, 2])                  # B*1*512
        x = outputs + res                                           # B*1*512
        x = tf.squeeze(x, axis=1)

        # 4. GRU
        res = x                                                     # B*512
        x = tf.concat([x, a2], axis=-1)                             # B*(512+32)
        x = tf.expand_dims(x, axis=1)                               # B*1*(512+32)
        x = tf.transpose(x, [1, 0, 2])                              # 1*B*(512+32)
        outputs, states_2 = self.gru2(x, initial_state=self.states_gru2, training=False)
        outputs = tf.transpose(outputs, [1, 0, 2])                  # B*1*512
        outputs = tf.squeeze(outputs, axis=1)                       # B*512
        x = outputs + res                                           # B*512

        # 5. linear
        x = tf.concat([x, a3], axis=-1)                             # B*(512+32)
        x = tf.matmul(x, self.fc_2)
        x = tf.nn.bias_add(x, self.bias_fc_2)
        x = tf.nn.relu(x)

        # 6. linear
        x = tf.concat([x, a4], axis=-1)
        x = tf.reshape(x, [-1, hp.fc_dims + 32])
        x = tf.matmul(x, self.fc_3)
        x = tf.nn.bias_add(x, self.bias_fc_3)
        x = tf.nn.relu(x)

        # 7. linear to softmax: 2**hp.bits (512)
        output = tf.matmul(x, self.fc_4)
        output = tf.nn.bias_add(output, self.bias_fc_4)             # B*512

        output = tf.nn.softmax(output, axis=-1)
        output_dist = tf.distributions.Categorical(probs=output)
        sample = output_dist.sample(sample_shape=(1,))

        return sample, states_1, states_2

import tensorflow as tf
from scipy.io import wavfile
from WaveRNN import WaveRNN_Alternative
import argparse
import numpy as np
from tqdm import tqdm
from data_reader import read_binary_lc
from hparam import hparams


def encode_16bits(x):
    return np.clip(x * 2**15, -2**15, 2**15 - 1).astype(np.int16)


def split_signal(x):
    encoded = encode_16bits(x)
    unsigned = encoded + 2**15
    coarse = unsigned // 256
    fine = unsigned % 256
    return coarse, fine


def combine_signal(coarse, fine):
    signal = coarse * 256 + fine
    signal -= 2**15
    return signal.astype(np.int16)


def generate_seq(args, wave_rnn, sess, mel, aux):
    x_placeholder = tf.placeholder(dtype=tf.float32, shape=[1, 1], name='x')
    mel_placeholder = tf.placeholder(dtype=tf.float32, shape=[1, 128], name='mel')
    aux_placeholder = tf.placeholder(dtype=tf.float32, shape=[1, 128], name='aux')

    infer_ops = wave_rnn.inference(x_placeholder, mel_placeholder, aux_placeholder)

    output = np.array([0.0], dtype=np.float32)

    hidden1 = sess.run(wave_rnn.states_gru1)
    hidden2 = sess.run(wave_rnn.states_gru2)

    seqlen = len(mel)
    x = np.array([[0.0]], dtype=np.float32)
    for i in tqdm(range(seqlen)):
        # get current lc
        aux_current = aux[i:i+1, :]
        aux_current = np.reshape(aux_current, [-1, 128]).astype(np.float32)  # 1*128
        mel_current = mel[i:i+1, :]
        mel_current = np.reshape(mel_current, [-1, 128]).astype(np.float32) # 1*128

        x, hidden1, hidden2 = sess.run(infer_ops, feed_dict={
            x_placeholder: x,
            mel_placeholder: mel_current,
            aux_placeholder: aux_current,
            wave_rnn.states_gru1: hidden1,
            wave_rnn.states_gru2: hidden2
        })

        # convert x to [-1, 1]
        x = 2 * x / (2 ** hparams.bits - 1.) - 1.
        output = np.append(output, x.flatten())

    # convert float [-1,1] to 16bit
    output = encode_16bits(output.flatten())
    wavfile.write(args.wave_name, hparams.sample_rate, output)
    print("generate done!")


def get_arguments():
    def _str_to_bool(s):
        """Convert string to bool (in argparse context)."""
        if s.lower() not in ['true', 'false']:
            raise ValueError('Argument needs to be a '
                             'boolean, got {}'.format(s))
        return {'true': True, 'false': False}[s.lower()]

    parser = argparse.ArgumentParser(description='Parallel WaveNet Network')
    parser.add_argument('--lc', type=str, default=None, required=True,
                        help='local condition file')
    parser.add_argument('--wave_name', type=str, default='wavernn.wav')
    parser.add_argument('--restore_from', type=str, default=None,
                        help='restore model from checkpoint')
    parser.add_argument('--wave', type=str, default=None,
                        help='ground truth wave')
    parser.add_argument('--gta', type=_str_to_bool, default='false',
                        help='gta mode')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_arguments()
    if args.restore_from is None:
        raise Exception('restore_from could not be None')

    seq_len = hparams.upsampling_rate * 5
    batch_size = hparams.batch_size
    lc_dim = hparams.num_mels

    with tf.Graph().as_default():

        x_placeholder = tf.placeholder(dtype=tf.float32, shape=[batch_size, seq_len, 1])      # B*1800*1
        y_placeholder = tf.placeholder(dtype=tf.float32, shape=[batch_size, seq_len])         # B*1800
        lc_placeholder = tf.placeholder(dtype=tf.float32, shape=[batch_size, 9, lc_dim])      # B*9*80

        # currently need to first build the network to keep tensorflow variable scope the same as training procedure
        wave_rnn = WaveRNN_Alternative(is_training=False)
        loss = wave_rnn.build_network(x_placeholder, y_placeholder, lc_placeholder)

        lc_placeholder_2 = tf.placeholder(dtype=tf.float32, shape=[1, None, lc_dim])      # B*9*80
        lc_upsample_net = wave_rnn.upsample_network(lc_placeholder_2)

        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)

        saver = tf.train.Saver()
        saver.restore(sess, save_path=args.restore_from)

        # load local condition
        lc = read_binary_lc(args.lc, hparams.num_mels)

        # pad local condition before and after to make sure same length after the conv1d opertiona with filter_width=3
        lc = np.pad(lc, ((2, 2), (0, 0)), mode='constant')      # pad 2 frames in both start & end
        lc = np.reshape(lc, [1, -1, hparams.num_mels])

        # lc go through the lc_net
        aux, mel = sess.run(lc_upsample_net, feed_dict={
            lc_placeholder_2: lc
        })

        mel = np.reshape(mel, [-1, 128])
        aux = np.reshape(aux, [-1, 128])

        # create generation model
        print('generating samples')
        generate_seq(args, wave_rnn, sess, mel, aux)

#! -*- encoding: utf-8 -*-
from __future__ import print_function
from data_reader import DataReader
from hparam import hparams
import tensorflow as tf
import time
import argparse
import os
import sys
import numpy as np
from scipy.io import wavfile
from datetime import datetime
from WaveRNN import WaveRNN_Alternative
from tqdm import tqdm


STARTED_DATESTRING = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())


def get_arguments():
    def _str_to_bool(s):
        """Convert string to bool (in argparse context)."""
        if s.lower() not in ['true', 'false']:
            raise ValueError('Argument needs to be a '
                             'boolean, got {}'.format(s))
        return {'true': True, 'false': False}[s.lower()]

    parser = argparse.ArgumentParser(description='WaveRNN Network')
    parser.add_argument('--filelist', type=str, default=None, required=True,
                        help='filelist path for training data.')
    parser.add_argument('--wave_dir', type=str, default=None, required=True,
                        help='wave data directory for training data.')
    parser.add_argument('--lc_dir', type=str, default=None, required=True,
                        help='local condition directory for training data.')
    parser.add_argument('--ngpu', type=int, default=1, help='gpu numbers')
    parser.add_argument('--run_name', type=str, default='wavernn',
                        help='run name for log saving')
    parser.add_argument('--restore_from', type=str, default=None,
                        help='restore model from checkpoint')
    parser.add_argument('--store_metadata', type=_str_to_bool, default=False,
                        help='Whether to store advanced debugging information')
    return parser.parse_args()


def write_wav(waveform, sample_rate, filename):
    """
    :param waveform: [-1,1]
    :param sample_rate:
    :param filename:
    :return:
    """
    # TODO: write wave to 16bit PCM, don't use librosa to write wave
    y = np.array(waveform, dtype=np.float32)
    y *= 32767
    wavfile.write(filename, sample_rate, y.astype(np.int16))
    print('Updated wav file at {}'.format(filename))


def save(saver, sess, logdir, step, write_meta_graph=False):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)
    print('Storing checkpoint to {} ...'.format(logdir), end="")
    sys.stdout.flush()

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    saver.save(sess, checkpoint_path, global_step=step, write_meta_graph=write_meta_graph)
    print(' Done.')


def load(saver, sess, logdir):
    print("Trying to restore saved checkpoints from {} ...".format(logdir),
          end="")

    ckpt = tf.train.get_checkpoint_state(logdir)
    if ckpt:
        print("  Checkpoint found: {}".format(ckpt.model_checkpoint_path))
        global_step = int(ckpt.model_checkpoint_path
                          .split('/')[-1]
                          .split('-')[-1])
        print("  Global step was: {}".format(global_step))
        print("  Restoring...", end="")
        saver.restore(sess, ckpt.model_checkpoint_path)
        print(" Done.")
        return global_step
    else:
        print(" No checkpoint found.")
        return None


def read_binary_lc(file_path, dimension):
    f = open(file_path, 'rb')
    features = np.fromfile(f, dtype=np.float32)
    f.close()
    assert features.size % float(dimension) == 0.0,\
        'specified dimension %s not compatible with data' % (dimension,)
    features = features.reshape((-1, dimension))
    return features


def encode_16bits(x):
    return np.clip(x * 2**15, -2**15, 2**15 - 1).astype(np.int16)


def generate_seq(wave_save_name, wave_rnn, sess, mel, aux, infer_ops, x_placeholder, mel_placeholder, aux_placeholder):
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
    wavfile.write(wave_save_name, hparams.sample_rate, output)
    print("generate done!")


def main():
    args = get_arguments()
    args.logdir = os.path.join(hparams.logdir_root, args.run_name)
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)

    assert hparams.upsampling_rate == hparams.hop_length, 'upsamling rate should be same as hop_length'

    # Create coordinator.
    coord = tf.train.Coordinator()

    with tf.device('/cpu:0'):
        with tf.name_scope('inputs'):
            reader = DataReader(coord, args.filelist, args.wave_dir, args.lc_dir)

    with tf.Graph().as_default():
        global_step = tf.get_variable("global_step", [], initializer=tf.constant_initializer(0), trainable=False)
        learning_rate = tf.train.exponential_decay(hparams.lr, global_step, hparams.decay_steps, 0.95, staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        sess = tf.Session(config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True))
        reader.start_threads()

        seq_len = hparams.seqlen
        batch_size = hparams.batch_size
        lc_dim = hparams.num_mels

        assert hparams.upsampling_rate == hparams.hop_length
        assert hparams.seqlen // hparams.upsampling_rate == 0
        assert np.cumprod(hparams.upsample_factors)[-1] == hparams.upsampling_rate

        lc_frames = hparams.seqlen // hparams.upsampling_rate + 2 * hparams.lc_pad

        x_placeholder = tf.placeholder(dtype=tf.float32, shape=[batch_size, seq_len, 1])          # B*1800*1
        y_placeholder = tf.placeholder(dtype=tf.float32, shape=[batch_size, seq_len])             # B*1800
        lc_placeholder = tf.placeholder(dtype=tf.float32, shape=[batch_size, lc_frames, lc_dim])  # B*9*80

        wave_rnn = WaveRNN_Alternative()
        loss = wave_rnn.build_network(x_placeholder, y_placeholder, lc_placeholder)

        grads = optimizer.compute_gradients(loss)
        grads = [(tf.clip_by_norm(grad, 2), var) for grad, var in grads]
        train_ops = optimizer.apply_gradients(grads, global_step=global_step)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)     # batch norm update
        train_ops = tf.group([train_ops, update_ops])

        tf.summary.scalar('loss', loss)

        # Set up logging for TensorBoard.
        writer = tf.summary.FileWriter(args.logdir)
        writer.add_graph(tf.get_default_graph())
        summaries = tf.summary.merge_all()

        # create test model
        test_lcnet_placeholder = tf.placeholder(dtype=tf.float32, shape=[1, None, hparams.num_mels])
        # build lc network for transposed upsampling
        test_lc_upsample_net = wave_rnn.upsample_network(test_lcnet_placeholder)

        test_x_placeholder = tf.placeholder(dtype=tf.float32, shape=[1, 1], name='x')
        test_mel_placeholder = tf.placeholder(dtype=tf.float32, shape=[1, hparams.lc_dims], name='mel')
        test_aux_placeholder = tf.placeholder(dtype=tf.float32, shape=[1, hparams.lc_dims], name='aux')

        infer_ops = wave_rnn.inference(test_x_placeholder, test_mel_placeholder, test_aux_placeholder)

        # Set up session
        init = tf.global_variables_initializer()
        sess.run(init)
        print('parameters initialization finished')

        saver = tf.train.Saver(max_to_keep=30)

        wave_dir = os.path.join(args.logdir, 'test_wave')
        if not os.path.exists(wave_dir):
            os.makedirs(wave_dir)

        # load local condition
        test_lc = read_binary_lc(reader.test_lc, hparams.num_mels)

        saved_global_step = 0
        if args.restore_from is not None:
            try:
                saved_global_step = load(saver, sess, args.restore_from)
                if saved_global_step is None:
                    # The first training step will be saved_global_step + 1,
                    # therefore we put -1 here for new or overwritten trainings.
                    saved_global_step = 0
            except Exception:
                print("Something went wrong while restoring checkpoint. "
                      "We will terminate training to avoid accidentally overwriting "
                      "the previous model.")
                raise

            print("restore model successfully!")

        print('start training.')
        last_saved_step = saved_global_step
        try:
            for step in range(saved_global_step + 1, hparams.train_steps):
                x, y, lc = reader.dequeue(num_elements=hparams.batch_size * args.ngpu)

                y = np.reshape(y, [hparams.batch_size, seq_len])

                start_time = time.time()
                summary, loss_value, _, lr = sess.run([summaries, loss, train_ops, learning_rate],
                                                      feed_dict={
                                                          x_placeholder: x,
                                                          y_placeholder: y,
                                                          lc_placeholder: lc
                                                      })
                writer.add_summary(summary, step)

                duration = time.time() - start_time
                step_log = 'step {:d} - loss = {:.3f}, lr={:.8f}, time cost={:4f}'\
                    .format(step, loss_value, lr, duration)
                print(step_log)

                if step % hparams.save_model_every == 0:
                    save(saver, sess, args.logdir, step)
                    last_saved_step = step

                    # pad local condition before and after to make sure same length after the conv1d opertiona with filter_width=3
                    lc = np.pad(test_lc, ((hparams.lc_pad, hparams.lc_pad), (0, 0)), mode='constant')  # pad 2 frames in both start & end
                    lc = np.reshape(lc, [1, -1, hparams.num_mels])

                    # lc go through the lc_net
                    aux, mel = sess.run(test_lc_upsample_net, feed_dict={
                        test_lcnet_placeholder: lc
                    })

                    mel = np.reshape(mel, [-1, hparams.lc_dims])
                    aux = np.reshape(aux, [-1, hparams.lc_dims])

                    # create generation model
                    print('generating samples')
                    wave_save_name = os.path.join(wave_dir, 'wavernn_test_model_{}.wav'.format(str(step).zfill(7)))
                    generate_seq(wave_save_name, wave_rnn, sess, mel, aux, infer_ops, test_x_placeholder,
                                 test_mel_placeholder, test_aux_placeholder)

        except KeyboardInterrupt:
            # Introduce a line break after ^C is displayed so save message
            # is on its own line.
            print()
        finally:
            if step > last_saved_step:
                save(saver, sess, args.logdir, step)
            coord.request_stop()
            coord.join()


if __name__ == '__main__':
    main()

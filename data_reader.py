import os
import random
import threading
import codecs
import queue
import librosa
import numpy as np
from hparam import hparams as hp


def read_binary_lc(file_path, dimension):
    f = open(file_path, 'rb')
    features = np.fromfile(f, dtype=np.float32)
    f.close()
    assert features.size % float(dimension) == 0.0,\
        'specified dimension %s not compatible with data' % (dimension,)
    features = features.reshape((-1, dimension))
    return features


def read_wave_and_lc_features(filelist_scpfile, wave_dir, lc_dir):
    filelist = []
    with codecs.open(filelist_scpfile, 'r', 'utf-8') as f:
        for line in f:
            line = line.strip()
            file_id = line
            filelist.append(file_id)

    random.shuffle(filelist)
    for file_id in filelist:
        wave_path = os.path.join(wave_dir, file_id + '.wav')
        lc_path = os.path.join(lc_dir, file_id + '.mel')

        # read wave
        audio, _ = librosa.load(wave_path, sr=hp.sample_rate, mono=True)
        audio = audio.reshape(-1, 1)

        # read local condition
        lc_features = read_binary_lc(lc_path, hp.num_mels)

        yield audio, lc_features, file_id


def encode_16bits(x):
    return np.clip(x * 2**15, -2**15, 2**15 - 1).astype(np.int16)


def split_signal(x):
    encoded = encode_16bits(x)
    unsigned = encoded + 2**15
    coarse = unsigned // 256
    fine = unsigned % 256
    return coarse, fine


def float_2_label(x, bits):
    assert abs(x).max() <= 1.0
    x = (x + 1.) * (2**bits - 1) / 2
    x = x.clip(0, 2**bits - 1)
    return np.array(x, dtype=np.int32)


def label_2_float(x, bits):
    return 2 * x / (2**bits - 1.) - 1.


class DataReader(object):
    '''Generic background audio reader that preprocesses audio files
    and enqueues them into a TensorFlow queue.'''

    def __init__(self,
                 coord,
                 filelist,
                 wave_dir,
                 lc_dir,
                 queue_size=128):
        self.coord = coord
        self.filelist = filelist
        self.wave_dir = wave_dir
        self.lc_dir = lc_dir
        self.lc_dim = hp.num_mels
        # recompute a sample size
        self.sample_size = hp.seqlen                 # 256 * 5
        self.upsample_rate = hp.upsampling_rate      # 256
        self.threads = []
        self.queue = queue.Queue(maxsize=queue_size)

        self.lc_frames = hp.seqlen // hp.upsampling_rate + 2 * hp.lc_pad    # 9 frames
        self.test_lc = os.path.join(lc_dir, hp.test_lc)

    def dequeue(self, num_elements):
        x = np.empty([0, self.sample_size, 1])
        y = np.empty([0, self.sample_size, 1])
        mel = np.empty([0, self.lc_frames, self.lc_dim])
        for i in range(num_elements):
            _x, _y, _lc = self.queue.get(block=True)
            _x = np.reshape(_x, [1, self.sample_size, 1])
            _y = np.reshape(_y, [1, self.sample_size, 1])
            _lc = np.reshape(_lc, [1, self.lc_frames, self.lc_dim])

            x = np.concatenate([x, _x], axis=0)
            y = np.concatenate([y, _y], axis=0)
            mel = np.concatenate([mel, _lc], axis=0)

        return x, y, mel

    def thread_main(self):
        stop = False
        # Go through the dataset multiple times
        while not stop:
            iterator = read_wave_and_lc_features(self.filelist,
                                                 self.wave_dir,
                                                 self.lc_dir)
            for audio, lc_features, file_id in iterator:
                if self.coord.should_stop():
                    stop = True
                    break

                # force align wave & local condition
                if len(audio) > len(lc_features) * self.upsample_rate:
                    # clip audio
                    audio = audio[:len(lc_features) * self.upsample_rate, :]
                elif len(audio) < len(lc_features) * self.upsample_rate:
                    # clip local condition and audio
                    audio_frames = len(audio) // self.upsample_rate
                    frames = min(audio_frames, len(lc_features))
                    audio = audio[:frames*self.upsample_rate, :]
                    lc_features = lc_features[:frames, :]
                else:
                    pass

                mel_offsets = [i for i in range(len(lc_features) - (self.lc_frames +  + hp.lc_pad * 2))]  # 9 frames
                audio_offsets = [(i + hp.lc_pad) * hp.upsampling_rate for i in mel_offsets]

                for audio_index, mel_index in zip(audio_offsets, mel_offsets):
                    audio_piece = audio[audio_index:audio_index + self.sample_size + 1, :]     # 5 frames
                    lc_piece = lc_features[mel_index:mel_index + self.lc_frames, :]            # 9 frames

                    # encode
                    x = audio_piece[:self.sample_size, :]
                    y = audio_piece[1:, :]
                    x = float_2_label(x, hp.bits)
                    y = float_2_label(y, hp.bits)
                    x = label_2_float(x, hp.bits)  # convert back quantized x to float
                    self.queue.put([x, y, lc_piece])

    def start_threads(self, n_threads=1):
        for _ in range(n_threads):
            thread = threading.Thread(target=self.thread_main, args=())
            thread.daemon = True  # Thread will close when parent quits.
            thread.start()
            self.threads.append(thread)

        return self.threads

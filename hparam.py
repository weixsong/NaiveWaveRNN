import tensorflow as tf


hparams = tf.contrib.training.HParams(
    # Audio:
    num_mels=80,
    n_fft=2048,
    sample_rate=22050,
    win_length=1024,
    hop_length=256,
    preemphasis=0.97,
    min_level_db=-100,
    ref_level_db=20,

    # train
    lr=0.0001,
    train_steps=1000000,
    save_model_every=4000,
    logdir_root='./logdir',
    decay_steps=50000,

    # network
    bits=9,
    lc_pad=2,
    rnn_dims=512,
    seqlen=256 * 5,                     # sequence length for RNN
    batch_size=32,                      # batch size
    upsampling_rate=256,                # same as hop_length

    # transposed upsampling
    res_blocks=10,
    fc_dims=512,
    lc_dims=128,
    conv_in_width=5,
    upsample_factors=(16, 16),
    upsample_widths=(3, 3),

    test_lc='LJ001-0002.mel',
)

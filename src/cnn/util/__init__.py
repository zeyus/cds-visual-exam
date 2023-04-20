import os
import tensorflow as tf

def set_tf_optim(num_threads):
    os.environ["OMP_NUM_THREADS"] = f"{num_threads}"
    os.environ["TF_NUM_INTRAOP_THREADS"] = f"{num_threads}"
    os.environ["TF_NUM_INTEROP_THREADS"] = f"{num_threads}"

    tf.config.threading.set_inter_op_parallelism_threads(
        num_threads
    )
    tf.config.threading.set_intra_op_parallelism_threads(
        num_threads
    )
    tf.config.set_soft_device_placement(True)
""" Loads an already trained model and evaluates it on new examples """

from os.path import join, isfile
import tensorflow as tf
import numpy as np
from parse_reg_args import get_parser
from build_tf_model import build_graph_from_args_dict
from tqdm import tqdm


def check_equal(iterator):
    """ returns true if all items in iterator are equal, otherwise returns false """
    iterator = iter(iterator)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    return all(first == rest for rest in iterator)


def batch_generator(data_arrays, batch_size, skip_last_batch=True, num_epochs=-1, shuffle=True):
    """ generates batches from given data and labels """
    if len(data_arrays) == 0:
        raise ValueError("No data arrays.")

    data_lens = [len(data_array) for data_array in data_arrays]
    if not check_equal(data_lens):
        raise ValueError("Data arrays are different lengths.")

    data_len = data_lens[0]

    batching_data_arrays = data_arrays

    epoch = 0
    while True:
        # shuffle the input data for this batch
        if shuffle:
            idxs = np.arange(0, data_len)
            np.random.shuffle(idxs)

            batching_data_arrays = []
            for data_array in data_arrays:
                batching_data_arrays.append(data_array[idxs])

        for batch_idx in range(0, data_len, batch_size):
            # skip last batch if it is the wrong size
            if skip_last_batch and (batch_idx + batch_size > data_len):
                continue

            data_batches = []
            for batching_data_array in batching_data_arrays:
                data_batches.append(batching_data_array[batch_idx:(batch_idx + batch_size)])

            yield data_batches

        epoch += 1
        if epoch == num_epochs:
            break


def load_args_dict(args_fn):
    """ loads the args used to train a model in the form of a dictionary """
    if not isfile(args_fn):
        raise FileNotFoundError("couldn't find args.txt: {}".format(args_fn))

    # load the arguments used to train this model
    parser = get_parser()
    args_dict = vars(parser.parse_args(["@{}".format(args_fn)]))  # also convert to a dict
    return args_dict


def restore_sess(ckpt_prefix_fn):
    """ create a TensorFlow session with restored parameters """

    # create a fresh graph for use with this session
    graph = tf.Graph()
    with graph.as_default():
        # reload graph from saved meta file
        saver = tf.compat.v1.train.import_meta_graph(ckpt_prefix_fn + ".meta", clear_devices=True)

    # set up the session and restore the parameters
    sess = tf.compat.v1.Session(graph=graph)
    saver.restore(sess, ckpt_prefix_fn)

    return sess


def restore_sess_with_dropout(ckpt_prefix_fn, args_fn, dropout_rate=0.2, rseed=0):
    graph = tf.Graph()
    with graph.as_default():
        tf.compat.v1.set_random_seed(rseed)
        # re-build graph from args used to train the model
        args = load_args_dict(args_fn)
        inf_graph, train_graph = build_graph_from_args_dict(args, (1, 56, 40), dropout_rate=dropout_rate, reset_graph=False)
        saver = tf.compat.v1.train.Saver(tf.compat.v1.trainable_variables())

    # set up the session and restore the parameters
    sess = tf.compat.v1.Session(graph=graph)
    saver.restore(sess, ckpt_prefix_fn)

    return sess


def restore_sess_from_pb(pb_fn):
    optimized_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(pb_fn, "rb") as f:
        optimized_graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(optimized_graph_def, name="")

    # set up the session from the graph (which should already contain all the necessary weights)
    sess = tf.compat.v1.Session(graph=graph)

    return sess


def run_inference_lr(encoded_data, ckpt_prefix_fn=None, sess=None, output_tensor_name="output/BiasAdd:0",
                  batch_size=256, enable_dropout=False):
    """ run inference on new data """

    if sess is None and ckpt_prefix_fn is None:
        raise ValueError("must provide restored session or checkpoint from which to restore")

    if enable_dropout and sess is None:
        raise ValueError("must provide a restored session from restore_sess_with_dropout() in order to use dropout")

    created_sess = False
    if sess is None:
        sess = restore_sess(ckpt_prefix_fn)
        created_sess = True

    # get operations from the TensorFlow graph that are needed to run new examples through the network
    # placeholder for the input sequences
    input_seqs_ph = sess.graph.get_tensor_by_name("raw_seqs_placeholder:0")
    #training_ph = sess.graph.get_tensor_by_name("training_ph:0")

    # predictions come from "output/BiasAdd:0" which refers to the final fully connected layer of the network
    # after the bias has been added. all my network_specs will have this same "output" layer. the extra squeeze
    # operation is added for data formatting
    predictions = sess.graph.get_tensor_by_name(output_tensor_name)

    # auto batch
    if encoded_data.shape[0] > batch_size:
        num_batches = int(np.ceil(encoded_data.shape[0]/batch_size))
        # print("Splitting data into batches of size 256 since more than 256 examples were provided.")
        predictions_for_data = np.zeros(encoded_data.shape[0])
        bg = batch_generator([encoded_data], batch_size, skip_last_batch=False, num_epochs=1, shuffle=False)
        for batch_num, batch_data in tqdm(enumerate(bg), total=num_batches):
            # if batch_num % 10 == 0:
            #     print("Running batch {} of {}...".format(batch_num+1, num_batches))
            batch_data = batch_data[0]
            predictions_for_batch = sess.run(predictions, feed_dict={input_seqs_ph: batch_data})
            start_index = batch_num * batch_size
            end_index = start_index + batch_size
            predictions_for_data[start_index:end_index] = np.squeeze(predictions_for_batch)
    else:
        predictions_for_data = np.squeeze(sess.run(predictions, feed_dict={input_seqs_ph: encoded_data}))

    if created_sess:
        sess.close()

    return predictions_for_data

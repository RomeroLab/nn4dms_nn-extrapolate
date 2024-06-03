""" Prep a saved TF1 checkpoint/graph for inference by freezing the graph and
    optimizing for inference using tensorflow's built in utility

    pieced together from:
    https://github.com/tensorflow/tensorflow/issues/19838

    and the source code for the optimization utils
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/freeze_graph.py
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/optimize_for_inference_lib.py
    """

from os.path import basename, join

import tensorflow as tf
from tensorflow.python.tools import optimize_for_inference_lib
from tensorflow.python.tools import freeze_graph


def prep_for_inference(ckpt_prefix_fn):

    # create a fresh graph for use with this session
    graph = tf.Graph()
    with graph.as_default():
        # reload graph from saved meta file
        saver = tf.compat.v1.train.import_meta_graph(ckpt_prefix_fn + ".meta", clear_devices=True)

    # set up the session and restore the parameters
    sess = tf.compat.v1.Session(graph=graph)
    saver.restore(sess, ckpt_prefix_fn)

    # it gets a little weird because we are calling some tensorflow utils that are designed to be run as
    # standalone programs... so we need to save some intermediate files to disk

    # save graph to disk -- this can also be saved as a pb, but save as txt to differentiate from frozen pb
    graph_def_fn = ckpt_prefix_fn + ".pbtxt"
    tf.compat.v1.train.write_graph(sess.graph.as_graph_def(), '.', graph_def_fn, as_text=True)

    # freeze graph converts all variables in a graph and checkpoint into constants
    frozen_graph_fn = ckpt_prefix_fn + ".pb"
    freeze_graph.freeze_graph(graph_def_fn, "", False, ckpt_prefix_fn, "output/BiasAdd",
                              "save/restore_all", "save/Const:0", frozen_graph_fn, True, "")

    # reload the frozen graph from disk
    frozen_graph = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(frozen_graph_fn, "rb") as f:
        frozen_graph.ParseFromString(f.read())

    # optimizing for inference removes training nodes, fuses operations, and performs other optimizations for inference
    input_names = ['raw_seqs_placeholder', 'training_ph']
    output_names = ['output/BiasAdd']
    new_graph = optimize_for_inference_lib.optimize_for_inference(
        frozen_graph, input_names, output_names, [tf.float32.as_datatype_enum, tf.bool.as_datatype_enum])

    # Write the optimized graph to disk
    optimized_fn = ckpt_prefix_fn + "_optimized.pb"
    with tf.io.gfile.GFile(optimized_fn, "wb") as f:
        f.write(new_graph.SerializeToString())


def main():
    ckpt_prefix_fn = "saved_models/gb1_cnn_ckpt_27"
    prep_for_inference(ckpt_prefix_fn)


if __name__ == "__main__":
    main()
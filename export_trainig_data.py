# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Runs a trained audio graph against a WAVE file and reports the results.

The model, labels and .wav file specified in the arguments will be loaded, and
then the predictions from running the model against the audio data will be
printed to the console. This is a useful script for sanity checking trained
models, and as an example of how to use an audio model from Python.

Here's an example of running it:

python tensorflow/examples/speech_commands/label_wav.py \
--graph=/tmp/my_frozen_graph.pb \
--labels=/tmp/speech_commands_train/conv_labels.txt \
--wav=/tmp/speech_dataset/left/a5d485dc_nohash_0.wav

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import os
import csv

import tensorflow as tf
# pylint: disable=unused-import
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
# pylint: enable=unused-import

FLAGS = None


def load_graph(filename):
  """Unpersists graph from file as default graph."""
  with tf.gfile.FastGFile(filename, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')


def load_labels(filename):
  """Read in labels, one label per line."""
  return [line.rstrip() for line in tf.gfile.GFile(filename)]




def label_wav(dir, labels, graph, output_file):
  """Check input directory"""
  if not dir or not tf.gfile.IsDirectory(dir):
    tf.logging.fatal('Diretory does not exist %s', dir)
  if not labels or not tf.gfile.Exists(labels):
    tf.logging.fatal('Labels file does not exist %s', labels)

  if not graph or not tf.gfile.Exists(graph):
    tf.logging.fatal('Graph file does not exist %s', graph)

  labels_list = load_labels(labels)

  fieldnames = labels_list.copy()
  fieldnames.insert(0,'filename')
  fieldnames.append('y')
  csvfile = open(output_file, 'w');
  writer = csv.DictWriter(csvfile, dialect='excel', fieldnames=fieldnames)
  writer.writeheader()
  
  # load graph, which is stored in the default session
  load_graph(graph)
  with tf.Session() as sess:
      softmax_tensor = sess.graph.get_tensor_by_name('labels_softmax:0')
      for label_dir in os.listdir(dir):
        if tf.gfile.IsDirectory(label_dir):
          continue 
        full_path = dir + '/' + label_dir + '/'
        for wav in os.listdir(full_path):
          if not wav.endswith(".wav"):
            continue  
          wav_path= full_path + wav
          """Loads the model and labels, and runs the inference to print predictions."""
          if not wav_path or not tf.gfile.Exists(wav_path):
            tf.logging.fatal('Audio file does not exist %s', wav_path)
        
          with open(wav_path, 'rb') as wav_file:
            wav_data = wav_file.read()
            predictions, = sess.run(softmax_tensor, {'wav_data:0': wav_data})
            dict_predctions = dict( zip( labels_list, predictions))
            dict_predctions['filename'] = wav
            dict_predctions['y'] = label_dir
            writer.writerow(dict_predctions)



def main(_):
  """Entry point for script, converts flags to arguments."""
  label_wav(FLAGS.dir, FLAGS.labels, FLAGS.graph, FLAGS.output_file)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--dir', type=str, default='', help='Input directory.')
  parser.add_argument(
      '--graph', type=str, default='', help='Model to use for identification.')
  parser.add_argument(
      '--labels', type=str, default='', help='Path to file containing labels.')
  parser.add_argument(
      '--output_file',
      type=str,
      default="results.csv",
      help='Prediction output file')
 
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)


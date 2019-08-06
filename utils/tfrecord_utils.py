"""
TF records help methods.

Author: Daniel Koguciuk
"""

from __future__ import print_function

import tensorflow as tf


def wrap_int64(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def wrap_string(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def wrap_bytes(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))


def wrap_float(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def wrap_floats(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def point_cloud_to_tfexample(pc_data, pc_objects):

    data = {'lidar/ravel': wrap_floats(pc_data.ravel()),
            'lidar/shape0': wrap_int64(pc_data.shape[0]), 'lidar/shape1': wrap_int64(pc_data.shape[1]),
            'bboxes/type': wrap_string(tf.compat.as_bytes('-'.join([obj['type'] for obj in pc_objects]))),
            'bboxes/x': wrap_floats([obj['x'] for obj in pc_objects]),
            'bboxes/y': wrap_floats([obj['y'] for obj in pc_objects]),
            'bboxes/z': wrap_floats([obj['z'] for obj in pc_objects]),
            'bboxes/w1': wrap_floats([obj['w1'] for obj in pc_objects]),
            'bboxes/w2': wrap_floats([obj['w2'] for obj in pc_objects]),
            'bboxes/l1': wrap_floats([obj['l1'] for obj in pc_objects]),
            'bboxes/l2': wrap_floats([obj['l2'] for obj in pc_objects]),
            'bboxes/h1': wrap_floats([obj['h1'] for obj in pc_objects]),
            'bboxes/h2': wrap_floats([obj['h2'] for obj in pc_objects]),
            'bboxes/ry': wrap_floats([obj['ry'] for obj in pc_objects]),
            }
    return tf.train.Example(features=tf.train.Features(feature=data))


def tfexample_to_point_cloud(example_proto):
    # Parse single example
    features = {'lidar/ravel': tf.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
                'lidar/shape0': tf.FixedLenFeature((), tf.int64), 'lidar/shape1': tf.FixedLenFeature((), tf.int64),
                'bboxes/type': tf.FixedLenFeature((), tf.string),
                'bboxes/x': tf.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
                'bboxes/y': tf.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
                'bboxes/z': tf.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
                'bboxes/w1': tf.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
                'bboxes/w2': tf.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
                'bboxes/l1': tf.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
                'bboxes/l2': tf.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
                'bboxes/h1': tf.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
                'bboxes/h2': tf.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
                'bboxes/ry': tf.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
                }
    parsed_features = tf.parse_single_example(example_proto, features)
    # Reshape data
    parsed_features['bboxes/type'] = tf.strings.split([parsed_features['bboxes/type']], sep='-').values
    x = tf.reshape(parsed_features['lidar/ravel'], [parsed_features['lidar/shape0'], parsed_features['lidar/shape1']])

    y = {'3dbbox': tf.stack([parsed_features['bboxes/x'], parsed_features['bboxes/y'], parsed_features['bboxes/z'],
                             parsed_features['bboxes/w1'], parsed_features['bboxes/w2'], parsed_features['bboxes/l1'],
                             parsed_features['bboxes/l2'], parsed_features['bboxes/h1'], parsed_features['bboxes/h2'],
                             parsed_features['bboxes/ry']], axis=-1),
         'types': parsed_features['bboxes/type']}

    # Return
    return x, y

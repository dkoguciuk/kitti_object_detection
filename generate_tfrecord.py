"""
TF records help methods.

Author: Charles R. Qi, Kui Xu
Modified by: Daniel Koguciuk
"""

from __future__ import print_function

import os
import argparse
import numpy as np
from tqdm import tqdm
import tensorflow as tf

import utils.viz_utils as viz_utils
import utils.kitti_utils as kitti_utils
import utils.tfrecord_utils as tfrecord_utils


class KittiObject(object):
    '''Load and parse object data into a usable format.'''

    def __init__(self, root_dir, split='training', args=None):
        '''root_dir contains training and testing folders'''
        self.root_dir = root_dir
        self.split = split
        print(root_dir, split)
        self.split_dir = os.path.join(root_dir, split)

        if split == 'training':
            self.num_samples = 7481
        elif split == 'testing':
            self.num_samples = 7518
        else:
            print('Unknown split: %s' % (split))
            exit(-1)

        lidar_dir = 'velodyne'
        depth_dir = 'depth'
        pred_dir  = 'pred'
        if args is not None:
            lidar_dir = args.lidar
            depth_dir = args.depthdir
            pred_dir  = args.preddir

        self.image_dir = os.path.join(self.split_dir, 'image_2')
        self.label_dir = os.path.join(self.split_dir, 'label_2')
        self.calib_dir = os.path.join(self.split_dir, 'calib')

        self.depthpc_dir = os.path.join(self.split_dir, 'depth_pc')
        self.lidar_dir = os.path.join(self.split_dir, lidar_dir)
        self.depth_dir = os.path.join(self.split_dir, depth_dir)
        self.pred_dir  = os.path.join(self.split_dir, pred_dir)


    def __len__(self):
        return self.num_samples

    def get_image(self, idx):
        assert(idx<self.num_samples)
        img_filename = os.path.join(self.image_dir, '%06d.png'%(idx))
        return kitti_utils.load_image(img_filename)

    def get_lidar(self, idx, dtype=np.float64, n_vec=4):
        assert(idx<self.num_samples)
        lidar_filename = os.path.join(self.lidar_dir, '%06d.bin'%(idx))
        return kitti_utils.load_velo_scan(lidar_filename, dtype, n_vec)

    def get_calibration(self, idx):
        assert(idx<self.num_samples)
        calib_filename = os.path.join(self.calib_dir, '%06d.txt'%(idx))
        return kitti_utils.Calibration(calib_filename)

    def get_label_objects(self, idx):
        assert(idx<self.num_samples) # and self.split=='training'
        label_filename = os.path.join(self.label_dir, '%06d.txt'%(idx))
        return kitti_utils.read_label(label_filename)

    def get_pred_objects(self, idx):
        assert(idx<self.num_samples) #and self.split=='training'
        pred_filename = os.path.join(self.pred_dir, '%06d.txt'%(idx))
        is_exist = os.path.exists(pred_filename)
        if is_exist:
            return kitti_utils.read_label(pred_filename)
        else:
            return None

    def get_depth(self, idx):
        assert(idx<self.num_samples)
        img_filename = os.path.join(self.depth_dir, '%06d.png'%(idx))
        return kitti_utils.load_depth(img_filename)

    def get_depth_image(self, idx):
        assert(idx<self.num_samples)
        img_filename = os.path.join(self.depth_dir, '%06d.png'%(idx))
        return kitti_utils.load_depth(img_filename)

    def get_depth_pc(self, idx):
        assert(idx<self.num_samples)
        lidar_filename = os.path.join(self.depthpc_dir, '%06d.bin'%(idx))
        is_exist = os.path.exists(lidar_filename)
        if is_exist:
            return kitti_utils.load_velo_scan(lidar_filename), is_exist
        else:
            return None, is_exist
        #print(lidar_filename, is_exist)
        #return utils.load_velo_scan(lidar_filename), is_exist

    def get_top_down(self, idx):
        pass

    def isexist_pred_objects(self, idx):
        assert(idx<self.num_samples and self.split=='training')
        pred_filename = os.path.join(self.pred_dir, '%06d.txt'%(idx))
        return os.path.exists(pred_filename)
    def isexist_depth(self, idx):
        assert(idx<self.num_samples and self.split=='training')
        depth_filename = os.path.join(self.depth_dir, '%06d.txt'%(idx))
        return os.path.exists(depth_filename)


def find_free_anchor_on_rect(pc_rect, anchor_center_rect, w, l, h, alpha, verbose=False):

    # Move coordinate frame to the anchor point
    pc_anchor_rect = pc_rect - anchor_center_rect

    # rotate pc_anchor with alpha
    pc_anchor_r_x = pc_anchor_rect[:, 0] * np.cos(alpha) - pc_anchor_rect[:, 2] * np.sin(alpha)
    pc_anchor_r_y = pc_anchor_rect[:, 1]
    pc_anchor_r_z = pc_anchor_rect[:, 0] * np.sin(alpha) + pc_anchor_rect[:, 2] * np.cos(alpha)
    pc_anchor_r = np.transpose(np.vstack([pc_anchor_r_x, pc_anchor_r_y, pc_anchor_r_z]))

    # Find points inside 3DBbox
    mask_l = (np.abs(pc_anchor_r[:, 0]) <= l / 2)
    mask_h = (np.abs(pc_anchor_r[:, 1]) <= h / 2)
    mask_w = (np.abs(pc_anchor_r[:, 2]) <= w / 2)
    points_2_draw = pc_rect[mask_l & mask_h & mask_w]

    # Find closest
    anchor_free_rect = None
    if len(points_2_draw):
        d_xyz = np.linalg.norm(points_2_draw, axis=-1)
        closest_idx = np.argmin(d_xyz)
        anchor_free_rect = points_2_draw[closest_idx]

    # Verbose mode
    if verbose and anchor_free_rect is not None:
        # Draw points inside bbox
        scale = np.ones(len(points_2_draw))
        mlab.points3d(points_2_draw[:, 0], points_2_draw[:, 1], points_2_draw[:, 2], scale, scale_factor=0.1)
        # Draw free anchor
        mlab.points3d([anchor_free_rect[0]], [anchor_free_rect[1]], [anchor_free_rect[2]], 1.0, scale_factor=0.1,
                      color=(1, 0, 0))

    # Return
    return anchor_free_rect


def compute_free_bbox_on_rect(center_anchor_on_rect, free_anchor_on_rect, w, l, h, alpha, verbose=False):
    """

    Calculate new bbox format for free anchor point inside the bbox.

    Args:
        center_anchor_on_rect (np.array): Center point of a bbox in rect coordinate frame.
        free_anchor_on_rect (np.array): Free point inside the bbox in rect coordinate frame -
            this will be the new anchor point.
        w (float): Width of a bbox in old format.
        l (flaot): Lenght of a bbox in old format.
        h (float): Height of a bbox in old format.
        alpha (float): Angle of a bbox in old format.
        verbose (bool): Should I be verbose?

    Returns:
        float: First component of width in new format.
        float: Second component of width in new format.
        float: First component of length in new format.
        float: Second component of length in new format.
        float: First component of height in new format.
        float: Second component of height in new format.

    """

    cx = free_anchor_on_rect[0] - center_anchor_on_rect[0]
    cz = free_anchor_on_rect[2] - center_anchor_on_rect[2]
    cx_prime = cx * np.cos(alpha) - cz * np.sin(alpha)
    cz_prime = cx * np.sin(alpha) + cz * np.cos(alpha)
    w1 = w / 2 - cz_prime
    w2 = w / 2 + cz_prime
    l1 = l / 2 - cx_prime
    l2 = l / 2 + cx_prime
    h1 = h / 2 + (free_anchor_on_rect[1] - center_anchor_on_rect[1])
    h2 = h / 2 - (free_anchor_on_rect[1] - center_anchor_on_rect[1])
    if verbose:
        print('cx: {}, cz: {}'.format(cx, cz))
        print('cx_prime: {}, cy_prime: {}'.format(cx_prime, cz_prime))
        print('l: {:{width}.{prec}f}, l1: {:{width}.{prec}f}, l2: {:{width}.{prec}f}, l1+l2: {:{width}.{prec}f}'.format(
            l, l1, l2, l1+l2, width=6, prec=3))
        print('w: {:{width}.{prec}f}, w1: {:{width}.{prec}f}, w2: {:{width}.{prec}f}, w1+w2: {:{width}.{prec}f}'.format(
            w, w1, w2, w1+w2, width=6, prec=3))
        print('h: {:{width}.{prec}f}, h1: {:{width}.{prec}f}, h2: {:{width}.{prec}f}, h1+h2: {:{width}.{prec}f}'.format(
            h, h1, h2, h1+h2, width=6, prec=3))
    return w1, w2, l1, l2, h1, h2


def generate_lidar_with_free_anchor(pc_velo, objects, calib, verbose=False):
    """

    Args:
        pc_velo:    LiDAR pointcloud in velo coordinate system
        objects:    List of bbjects.
        calib:      Calibration object.
        verbose:    Show mlab figure?

    Returns:

        (pc_rect, list of bboxes)

    """

    # Transform to rect coordinate system
    pc_rect = calib.project_velo_to_rect(pc_velo[:, :3])

    # For every object
    labels = []
    for obj in objects:

        # Leave don't care's
        if obj.type == 'DontCare':
            continue

        # Get anchor of the object (on the ground plane)
        anchor_base_on_rect = np.array(obj.t)

        # Move to the center of the object
        anchor_center_on_rect = anchor_base_on_rect + [0, -obj.h/2, 0]

        # Find free anchor
        anchor_free_on_rect = find_free_anchor_on_rect(pc_rect, anchor_center_on_rect, obj.w, obj.l, obj.h, obj.ry,
                                                       verbose=False)

        if anchor_free_on_rect is not None:

            # Calculate new bbox params
            w1, w2, l1, l2, h1, h2 = compute_free_bbox_on_rect(anchor_center_on_rect, anchor_free_on_rect, obj.w, obj.l,
                                                               obj.h, obj.ry, verbose=False)

            # Remember
            label = {'x': anchor_free_on_rect[0], 'y': anchor_free_on_rect[1], 'z': anchor_free_on_rect[2], 'w1': w1,
                     'w2': w2, 'l1': l1, 'l2': l2, 'h1': h1, 'h2': h2, 'ry': obj.ry, 'type': obj.type}
            labels.append(label)

    # Display bbox
    if verbose:
        viz_utils.draw_point_cloud_with_bboxes(pc_rect, labels)

    return pc_rect, labels


def dataset_generate_tfrecords(root_dir, arguments):

    # Create kitti object
    dataset = KittiObject(root_dir, split=args.split, args=arguments)

    # tfrecords count
    num_per_shard = 100
    num_shards = int(np.ceil(len(dataset)/num_per_shard))

    # Start TF session
    with tf.Graph().as_default():
        with tf.Session('') as _:
            for shard_id in range(num_shards):
                shard_filename = '%s_%s_%05d-of-%05d.tfrecord' % (args.split, 'kitti', shard_id, num_shards)
                shard_filepath = os.path.join(args.tfrecords_path, shard_filename)
                with tf.python_io.TFRecordWriter(shard_filepath) as tfrecord_writer:
                    start_idx = shard_id * num_per_shard
                    end_idx = min((shard_id + 1) * num_per_shard, len(dataset))
                    for data_idx in tqdm(range(start_idx, end_idx)):

                        # Load pc_velo
                        n_vec = 4
                        dtype = np.float32
                        pc_velo = dataset.get_lidar(data_idx, dtype, n_vec)[:, 0:n_vec]

                        # Load calibration
                        calib = dataset.get_calibration(data_idx)

                        # Load labels from dataset
                        objects = []
                        if args.split == 'training':
                            objects = dataset.get_label_objects(data_idx)

                        # Generate data
                        verbose = arguments.vis
                        x, y = generate_lidar_with_free_anchor(pc_velo, objects, calib, verbose=verbose)

                        # Convert to tf record
                        tf_example = tfrecord_utils.point_cloud_to_tfexample(x, y)
                        tfrecord_writer.write(tf_example.SerializeToString())


if __name__ == '__main__':
    import mayavi.mlab as mlab

    parser = argparse.ArgumentParser(description='PyTorch Training RPN')
    parser.add_argument('-d', '--dir', type=str, default="data/obj", metavar='N', help='input  (default: data/object)')
    parser.add_argument('-i', '--ind', type=int, default=0, metavar='N', help='input  (default: data/object)')
    parser.add_argument('-p', '--pred', action='store_true', help='show predict results')
    parser.add_argument('-s', '--stat', action='store_true', help=' stat the w/h/l of point cloud in gt bbox')
    parser.add_argument('-l', '--lidar', type=str, default="velodyne", metavar='N',
                        help='velodyne dir  (default: velodyne)')
    parser.add_argument('-e', '--depthdir', type=str, default="depth", metavar='N', help='depth dir  (default: depth)')
    parser.add_argument('-r', '--preddir', type=str, default="pred", metavar='N',
                        help='predicted boxes  (default: pred)')
    parser.add_argument('--vis',  action='store_true', help='show images')
    parser.add_argument('--depth',  action='store_true', help='load depth')

    parser.add_argument('--split',  type=str, default="training", help='split: training/testing')
    parser.add_argument('--tfrecords_path', type=str, help='output path where to put generated tfrecords')
    args = parser.parse_args()

    dataset_generate_tfrecords(args.dir, args)

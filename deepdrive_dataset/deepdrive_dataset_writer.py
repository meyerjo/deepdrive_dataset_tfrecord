import os
import re
import json
from os.path import expanduser

import zipfile
import datetime
import tensorflow as tf
import numpy as np

from utils import mkdir_p
from deepdrive_dataset_download import DeepdriveDatasetDownload
from deepdrive_versions import DEEPDRIVE_LABELS
from tf_features import *
from PIL import Image


class DeepdriveDatasetWriter(object):
    feature_dict = {
        'image/height': None,
        'image/width': None,
        'image/object/bbox/id': None,
        'image/object/bbox/xmin': None,
        'image/object/bbox/xmax': None,
        'image/object/bbox/ymin': None,
        'image/object/bbox/ymax': None,
        'image/object/bbox/truncated': None,
        'image/object/bbox/occluded': None,
        'image/object/class/label/name': None,
        'image/object/class/label/id': None,
        'image/object/class/label': None,
        'image/encoded': None,
        'image/format': None,
        'image/id': None,
        'image/source_id': None,
        'image/filename': None,
    }


    @staticmethod
    def feature_dict_description(type='feature_dict'):
        """
        Get the feature dict. In the default case it is filled with all the keys and the items set to None. If the
        type=reading_shape the shape description required for reading elements from a tfrecord is returned)
        :param type: (anything = returns the feature_dict with empty elements, reading_shape = element description for
        reading the tfrecord files is returned)
        :return:
        """
        obj = DeepdriveDatasetWriter.feature_dict
        if type == 'reading_shape':
            obj['image/height'] = tf.FixedLenFeature((), tf.int64, 1)
            obj['image/width'] = tf.FixedLenFeature((), tf.int64, 1)
            obj['image/object/bbox/id'] = tf.VarLenFeature(tf.int64)
            obj['image/object/bbox/xmin'] = tf.VarLenFeature(tf.float32)
            obj['image/object/bbox/xmax'] = tf.VarLenFeature(tf.float32)
            obj['image/object/bbox/ymin'] = tf.VarLenFeature(tf.float32)
            obj['image/object/bbox/ymax'] = tf.VarLenFeature(tf.float32)
            obj['image/object/bbox/truncated'] = tf.VarLenFeature(tf.string)
            obj['image/object/bbox/occluded'] = tf.VarLenFeature(tf.string)
            obj['image/encoded'] = tf.FixedLenFeature((), tf.string, default_value='')
            obj['image/format'] = tf.FixedLenFeature((), tf.string, default_value='')
            obj['image/filename'] = tf.FixedLenFeature((), tf.string, default_value='')
            obj['image/id'] = tf.FixedLenFeature((), tf.string, default_value='')
            obj['image/source_id'] = tf.FixedLenFeature((), tf.string, default_value='')
            obj['image/object/class/label/id'] = tf.VarLenFeature(tf.int64)
            obj['image/object/class/label'] = tf.VarLenFeature(tf.int64)
            obj['image/object/class/label/name'] = tf.VarLenFeature(tf.string)
        return obj

    def __init__(self):
        self.input_path = os.path.join(expanduser('~'), '.deepdrive')

    def unzip_file_to_folder(self, filename, folder, remove_file_after_creating=True):
        assert(os.path.exists(filename) and os.path.isfile(filename))
        assert(os.path.exists(folder) and os.path.isdir(folder))
        with zipfile.ZipFile(filename, 'r') as zf:
            zf.extractall(folder)
        if remove_file_after_creating:
            print('\nRemoving file: {0}'.format(filename))
            os.remove(folder)

    def get_image_label_folder(self, fold_type=None, version=None):
        """
        Returns the folder containing all images and the folder containing all label information
        :param fold_type:
        :param version:
        :return: Raises BaseExceptions if expectations are not fulfilled
        """
        assert(fold_type in ['train', 'test', 'val'])
        version = '100k' if version is None else version
        assert(version in ['100k', '10k'])

        download_folder = os.path.join(self.input_path, 'download')
        expansion_images_folder = os.path.join(self.input_path, 'images')
        expansion_labels_folder = os.path.join(self.input_path, 'labels')
        #
        if not os.path.exists(expansion_images_folder):
            mkdir_p(expansion_images_folder)
        if not os.path.exists(expansion_labels_folder):
            mkdir_p(expansion_labels_folder)

        full_labels_path = os.path.join(expansion_labels_folder, 'bdd100k', 'labels', '100k')
        full_images_path = os.path.join(expansion_images_folder, 'bdd100k', 'images')
        if version in [None, '100k']:
            full_images_path = os.path.join(full_images_path, '100k', fold_type)
        else:
            full_images_path = os.path.join(full_images_path, '10k', fold_type)

        extract_files = True
        if len(DeepdriveDatasetDownload.filter_folders(full_labels_path)) == 2 and \
                len(DeepdriveDatasetDownload.filter_files(full_images_path)) > 0:
            print('Do not check the download folder. Pictures seem to exist.')
            if fold_type != 'test':
                full_labels_path = os.path.join(full_labels_path, fold_type)
            extract_files = False
        elif os.path.exists(download_folder):
            files_in_directory = DeepdriveDatasetDownload.filter_files(download_folder, False, re.compile('\.zip$'))
            if len(files_in_directory) < 2:
                raise BaseException('Not enough files found in {0}. All files present: {1}'.format(
                    download_folder, files_in_directory
                ))
        else:
            mkdir_p(download_folder)
            raise BaseException('Download folder: {0} did not exist. It had been created. '
                                'Please put images, labels there.'.format(download_folder))

        # unzip the elements
        if extract_files:
            print('Starting to unzip the files')
            self.unzip_file_to_folder(os.path.join(download_folder, 'bdd100k_labels.zip'), expansion_labels_folder,
                                      False)
            self.unzip_file_to_folder(os.path.join(download_folder, 'bdd100k_images.zip'), expansion_images_folder,
                                      False)

        if fold_type == 'test':
            return full_images_path, None
        return full_images_path, full_labels_path

    def filter_boxes_from_annotation(self, annotations):
        """

        :param annotations:
        :return: boxes, attributes
        """
        box = []
        if annotations is None:
            return box
        attributes = annotations['attributes']
        for frame in annotations['frames']:
            for obj in frame['objects']:
                if 'box2d' in obj:
                    box.append(obj)
        return dict(boxes=box, attributes=attributes)

    def _get_boundingboxes(self, annotations_for_picture_id):
        boxid, xmin, xmax, ymin, ymax, label_id, label, truncated, occluded =\
            [], [], [], [], [], [], [], [], []
        if annotations_for_picture_id is None:
            return boxid, xmin, xmax, ymin, ymax, label_id, label, truncated, occluded
        assert(len(annotations_for_picture_id['frames']) == 1)
        for frame in annotations_for_picture_id['frames']:
            for obj in frame['objects']:
                if 'box2d' not in obj:
                    continue
                boxid.append(obj['id'])
                xmin.append(obj['box2d']['x1'])
                xmax.append(obj['box2d']['x2'])
                ymin.append(obj['box2d']['y1'])
                ymax.append(obj['box2d']['y2'])
                label.append(obj['category'])
                label_id.append(DEEPDRIVE_LABELS.index(obj['category']) + 1)

                attributes = obj['attributes']
                truncated.append(attributes.get('truncated', False))
                occluded.append(attributes.get('occluded', False))
        return boxid, xmin, xmax, ymin, ymax, label_id, label, truncated, occluded


    def _get_tf_feature_dict(self, image_id, image_path, image_format, annotations):
        boxid, xmin, xmax, ymin, ymax, label_id, label, truncated, occluded = \
            self._get_boundingboxes(annotations)
        truncated = np.asarray(truncated)
        occluded = np.asarray(occluded)

        # convert things to bytes
        label_bytes = [tf.compat.as_bytes(l) for l in label]
        im = Image.open(image_path)
        image_width, image_height = im.size
        image_filename = os.path.basename(image_path)
        image_fileid = re.search('^(.*)(\.jpg)$', image_filename).group(1)

        tmp_feat_dict = DeepdriveDatasetWriter.feature_dict
        tmp_feat_dict['image/id'] = bytes_feature(image_fileid)
        tmp_feat_dict['image/source_id'] = bytes_feature(image_fileid)
        tmp_feat_dict['image/height'] = int64_feature(image_height)
        tmp_feat_dict['image/width'] = int64_feature(image_width)
        with open(image_path, 'rb') as f:
            tmp_feat_dict['image/encoded'] = bytes_feature(f.read())
        tmp_feat_dict['image/format'] = bytes_feature(image_format)
        tmp_feat_dict['image/filename'] = bytes_feature(image_filename)
        tmp_feat_dict['image/object/bbox/id'] = int64_feature(boxid)
        tmp_feat_dict['image/object/bbox/xmin'] = float_feature(xmin)
        tmp_feat_dict['image/object/bbox/xmax'] = float_feature(xmax)
        tmp_feat_dict['image/object/bbox/ymin'] = float_feature(ymin)
        tmp_feat_dict['image/object/bbox/ymax'] = float_feature(ymax)
        tmp_feat_dict['image/object/bbox/truncated'] = bytes_feature(truncated.tobytes())
        tmp_feat_dict['image/object/bbox/occluded'] = bytes_feature(occluded.tobytes())
        tmp_feat_dict['image/object/class/label/id'] = int64_feature(label_id)
        tmp_feat_dict['image/object/class/label'] = int64_feature(label_id)
        tmp_feat_dict['image/object/class/label/name'] = bytes_feature(label_bytes)

        return tmp_feat_dict


    def _get_tf_feature(self, image_id, image_path, image_format, annotations):
        feature_dict = self._get_tf_feature_dict(
            image_id, image_path, image_format, annotations)
        return tf.train.Features(feature=feature_dict)

    def write_tfrecord(self, fold_type=None, version=None, max_elements_per_file=1000, write_masks=False):
        output_path = os.path.join(self.input_path, 'tfrecord', version if version is not None else '100k', fold_type)
        if not os.path.exists(output_path):
            mkdir_p(output_path)

        full_images_path, full_labels_path = self.get_image_label_folder(fold_type, version)

        # get the files
        image_files = DeepdriveDatasetDownload.filter_files(full_images_path, True)

        def get_annotation(picture_id):
            if full_labels_path is None:
                return None
            with open(os.path.join(full_labels_path, picture_id + '.json'), 'r') as f:
                return json.loads(f.read())

        image_filename_regex = re.compile('^(.*)\.(jpg)$')
        tfrecord_file_id, writer = 0, None
        tfrecord_filename_template = os.path.join(output_path, 'output_{version}_{{iteration:06d}}.tfrecord'.format(
            version=fold_type + ('100k' if version is None else version)
        ))
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for file_counter, f in enumerate(image_files):
                if file_counter % max_elements_per_file == 0:
                    if writer is not None:
                        writer.close()
                        tfrecord_file_id += 1
                    tmp_filename_tfrecord = tfrecord_filename_template.format(iteration=tfrecord_file_id)
                    print('{0}: Create TFRecord filename: {1} after processing {2}/{3} files'.format(
                        str(datetime.datetime.now()), tmp_filename_tfrecord, file_counter, len(image_files)
                    ))
                    writer = tf.python_io.TFRecordWriter(tmp_filename_tfrecord)
                elif file_counter % 250 == 0:
                    print('\t{0}: Processed file: {1}/{2}'.format(
                        str(datetime.datetime.now()), file_counter, len(image_files)))
                # match the filename with the regex
                m = image_filename_regex.search(f)
                if m is None:
                    print('Filename did not match regex: {0}'.format(f))
                    continue

                picture_id = m.group(1)
                picture_id_annotations = get_annotation(picture_id)
                picture_id_boxes = self.filter_boxes_from_annotation(picture_id_annotations)

                feature = self._get_tf_feature(
                    picture_id, os.path.join(full_images_path, f),
                    m.group(2), picture_id_annotations)
                example = tf.train.Example(features=feature)
                writer.write(example.SerializeToString())

            # Close the last files
            if writer is not None:
                writer.close()

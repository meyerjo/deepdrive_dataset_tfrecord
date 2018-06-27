import tensorflow as tf
import numpy as np
from PIL import Image
import argparse
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from deepdrive_dataset.deepdrive_dataset_reader import DeepdriveDatasetReader
from deepdrive_dataset.deepdrive_versions import DEEPDRIVE_FOLDS, DEEPDRIVE_VERSIONS

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, help='Batch Size to use', default=4)
    parser.add_argument('--fold_type', type=str, choices=DEEPDRIVE_FOLDS, default='train')
    parser.add_argument('--version', type=str, default='100k', choices=DEEPDRIVE_VERSIONS)
    FLAGS = parser.parse_args()

    reader = DeepdriveDatasetReader(batch_size=FLAGS.batch_size)
    if FLAGS.fold_type == 'train':
        iterator = reader.load_train_data_bbox(FLAGS.version, False)
    elif FLAGS.fold_type == 'val':
        iterator = reader.load_val_data_bbox(FLAGS.version, False)
    elif FLAGS.fold_type == 'test':
        iterator = reader.load_test_data_bbox(FLAGS.version, False)
    else:
        raise BaseException('Unknown fold type: {0}'.format(FLAGS.fold_type))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        out = sess.run(iterator)
        out_labels = DeepdriveDatasetReader.parsing_boundingboxes(None, 'labels')
        out_dict = dict(zip(out_labels, out))

        # Visualize the output
        images = out_dict['image']
        f, axarr = plt.subplots(images.shape[0], 2, dpi=600)
        for im_id in range(0, images.shape[0]):
            plt.sca(axarr[im_id, 0])
            plt.imshow(images[im_id, :])
            # Boundingbox order is ymin, xmin, ymax, xmax
            for box in out_dict['bboxes'][im_id, :]:
                rect = Rectangle(
                    xy=(box[1], box[0]), width=box[3] - box[1], height=box[2] - box[0], fill=False)
                axarr[im_id, 0].add_patch(rect)

            #
            plt.sca(axarr[im_id, 1])
            shape = out_dict['image_shape'][im_id]
            plt.imshow(images[im_id, 0:shape[1], 0:shape[0], :])
            for box in out_dict['bboxes'][im_id, :]:
                rect = Rectangle(
                    xy=(box[1], box[0]), width=box[3] - box[1], height=box[2] - box[0], fill=False)
                axarr[im_id, 1].add_patch(rect)

            if im_id == 0:
                axarr[im_id, 0].set_title('Padded')
                axarr[im_id, 1].set_title('Unpadded')
            axarr[im_id, 0].set_xticks([])
            axarr[im_id, 0].set_yticks([])
            axarr[im_id, 1].set_xticks([])
            axarr[im_id, 1].set_yticks([])
            axarr[im_id, 0].set_ylabel('Batch-Id: {0}'.format(im_id))
            axarr[im_id, 1].set_xlabel(str(out_dict['image_ids'][im_id]))
        plt.show()

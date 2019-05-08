import argparse
import logging

from deepdrive_dataset.deepdrive_dataset_writer import DeepdriveDatasetWriter
from deepdrive_dataset.deepdrive_versions import DEEPDRIVE_FOLDS, DEEPDRIVE_VERSIONS

if __name__ == '__main__':
    logging.getLogger(__name__).setLevel(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold_type', type=str,
                        default='train', choices=DEEPDRIVE_FOLDS)
    parser.add_argument('--version', type=str,
                        default='100k', choices=DEEPDRIVE_VERSIONS)
    parser.add_argument(
        '--elements_per_tfrecord', type=int, default=1000,
        help='Number of Pictures per tfrecord file. '
             'Multiple files help in the shuffling process.')
    parser.add_argument(
        '--number_images_to_write', type=int, default=None,
        help='Restricts the number of files to be written. '
             '[E.g. to create smaller files to test overfitting]')
    parser.add_argument(
        '--weather', type=str, default=None,
        help='Only write files with this specific weather type'
    )
    parser.add_argument(
        '--scene_type', type=str, default=None,
        help='Only write files with this specific scene type'
    )
    parser.add_argument(
        '--daytime', type=str, default=None,
        help='Only write files with this specific daytime'
    )

    parser.add_argument(
        '--classes', type=str, default=None,
        help='Only write boundingboxes of this specific classes '
             '(Comma separated). Classlabels: ['
             '"`bus`, `traffic light`, `traffic sign`, `person`, `bike`, '
             '`truck`, `motor`, `car`, `train`, `rider`"]'
    )

    FLAGS = parser.parse_args()

    dd = DeepdriveDatasetWriter()
    dd.write_tfrecord(
        FLAGS.fold_type, version=FLAGS.version,
        max_elements_per_file=FLAGS.elements_per_tfrecord,
        small_size=FLAGS.number_images_to_write,
        weather_type=FLAGS.weather, scene_type=FLAGS.scene_type,
        daytime_type=FLAGS.daytime, classes=FLAGS.classes
    )

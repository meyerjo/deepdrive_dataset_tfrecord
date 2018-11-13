import argparse

from deepdrive_dataset.deepdrive_dataset_writer import DeepdriveDatasetWriter
from deepdrive_dataset.deepdrive_versions import DEEPDRIVE_FOLDS, DEEPDRIVE_VERSIONS

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold_type', type=str, default='train', choices=DEEPDRIVE_FOLDS)
    parser.add_argument('--version', type=str, default='100k', choices=DEEPDRIVE_VERSIONS)
    parser.add_argument('--elements_per_tfrecord', type=int, default=1000)
    FLAGS = parser.parse_args()

    dd = DeepdriveDatasetWriter()
    dd.write_tfrecord(
        FLAGS.fold_type, version=FLAGS.version,
        max_elements_per_file=FLAGS.elements_per_tfrecord)

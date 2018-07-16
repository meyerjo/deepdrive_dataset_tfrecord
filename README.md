# Deepdrive Dataset to TFRecord
Convert the Berkeley Deepdrive dataset to a TFRecord file. (Specifically for the Object Detection Task)

This repository shall help to create a tfrecord file for the berkeley deep drive dataset. I have no affiliation with Berkeley and/or the deep drive team.

## Download dataset

1. Register at http://bdd-data.berkeley.edu/login.html . NOTE: The server does not serve an SSL ceritificate.
2. Go to the Download page http://bdd-data.berkeley.edu/portal.html#download
3. Accept the License and Download: "Images", "Labels"
4. Create a folder ~/.deepdrive/download
5. Place both zip files in that folder

## Create dataset

You can use the script create_tfrecord.py in order to create the TFRecord file you need.

--fold_type = \['train', 'val', 'test'\] : Select for which fold you want to create the tfrecord (default=train)


--version = \['100k', '10k'\] : The Berkeley Deepdrive Dataset comes in two sizes. (default=100k)

--elements_per_tfrecord = integer : You can specify, how many images are put into one tfrecord file. Multiple TFRecord files are generated.

The resulting TFRecord files can be found in :
~/.deepdrive/tfrecord/\[version\]/\[fold_type\]/

## Read dataset

Using read_data.py you can check your TFRecord file.

--batch_size = int: Specify the batch-size

--fold_type = see above

--version = see above

It will plot all images, and all boundingboxes.

import os
import warnings

import cv2
from six import raise_from
import xml.etree.ElementTree as ET
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tqdm
import pandas as pd

from generators.common import Generator
from generators.pascal import _findNode

CLASS_NAME = {
    "u": 0,
    "o": 0,
    "r": 0,
    "g": 0,
    "cell": 0,
    'cancer': 0
}


class TumorCellGenerator(Generator):
    """
            Initialize a Tumor cells data generator.

            Args:
                xml_dirs: the path of directory which contains bounding box xml files
                image_dirs: the path of directory which contains BrightField images
                classes: class names tos id mapping
                image_extension: image filename ext
                **kwargs:
            """

    def __init__(self,
                 xml_dirs,
                 image_dirs,
                 classes=CLASS_NAME,
                 multiple=1,
                 **kwargs):
        self.classes = classes
        self.image_dirs = image_dirs
        self.xml_dirs = xml_dirs
        self.labels = {}
        self.multiple = multiple
        for key, value in self.classes.items():
            self.labels[value] = key
        super(TumorCellGenerator, self).__init__(**kwargs)

    def size(self):
        """
        Size of the dataset.
        """
        return len(self.image_dirs)

    def num_classes(self):
        """
        Number of classes in the dataset.
        """
        return 1

    def has_label(self, label):
        """
        Return True if label is a known label.
        """
        return label in self.labels

    def has_name(self, name):
        """
        Returns True if name is a known class.
        """
        return name in self.classes

    def name_to_label(self, name):
        """
        Map name to label.
        """
        return self.classes[name]

    def label_to_name(self, label):
        """
        Map label to name.
        """
        return self.labels[label]

    def image_aspect_ratio(self, image_index):
        """
        Compute the aspect ratio for an image with image_index.
        """
        path = self.image_dirs[image_index]
        image = cv2.imread(path)
        h, w = image.shape[:2]
        return float(w) / float(h)

    def load_image(self, image_index):
        """
        Load an image at the image_index.
        """
        path = self.image_dirs[image_index]
        image = cv2.imread(path)
        return image

    def preprocess_image(self, image):
        # image, RGB
        image_height, image_width = image.shape[:2]
        if image_height > image_width:
            scale = self.image_size / image_height
            resized_height = self.image_size
            resized_width = int(image_width * scale)
        else:
            scale = self.image_size / image_width
            resized_height = int(image_height * scale)
            resized_width = self.image_size

        image = cv2.resize(image, (resized_width, resized_height))
        image = image.astype(np.uint8)
        # image /= 255.
        # mean = [0.485, 0.456, 0.406]
        # std = [0.229, 0.224, 0.225]
        # image -= mean
        # image /= std
        pad_h = self.image_size - resized_height
        pad_w = self.image_size - resized_width
        image = np.pad(image, [(0, pad_h), (0, pad_w), (0, 0)], mode='constant')
        return image, scale

    def __parse_annotation(self, element):
        """
        Parse an annotation given an XML element.
        """
        truncated = _findNode(element, 'truncated', parse=int)
        difficult = _findNode(element, 'difficult', parse=int)

        class_name = _findNode(element, 'name').text.lower()
        if class_name not in self.classes:
            raise ValueError('class name \'{}\' not found in classes: {}'.format(class_name, list(self.classes.keys())))

        box = np.zeros((4,))
        label = self.name_to_label(class_name)

        bndbox = _findNode(element, 'bndbox')
        box[0] = _findNode(bndbox, 'xmin', 'bndbox.xmin', parse=float) - 1
        box[1] = _findNode(bndbox, 'ymin', 'bndbox.ymin', parse=float) - 1
        box[2] = _findNode(bndbox, 'xmax', 'bndbox.xmax', parse=float) - 1
        box[3] = _findNode(bndbox, 'ymax', 'bndbox.ymax', parse=float) - 1

        return truncated, difficult, box, label

    def __parse_annotations(self, xml_root):
        """
        Parse all annotations under the xml_root.
        """
        annotations = {'labels': np.empty((0,), dtype=np.int32),
                       'bboxes': np.empty((0, 4))}
        for i, element in enumerate(xml_root.iter('object')):
            try:
                truncated, difficult, box, label = self.__parse_annotation(element)
                annotations['bboxes'] = np.concatenate([annotations['bboxes'], [box * self.multiple]])
                annotations['labels'] = np.concatenate([annotations['labels'], [label]])
            except ValueError as e:
                raise_from(ValueError('could not parse object #{}: {}'.format(i, e)), None)

        return annotations

    def load_annotations(self, image_index):
        """
        Load annotations for an image_index.
        """
        filename = self.xml_dirs[image_index]
        try:
            tree = ET.parse(filename)
            return self.__parse_annotations(tree.getroot())
        except ET.ParseError as e:
            raise_from(ValueError('invalid annotations file: {}: {}'.format(filename, e)), None)
        except ValueError as e:
            raise_from(ValueError('invalid annotations file: {}: {}'.format(filename, e)), None)

    def filter_annotations(self, image_group, annotations_group, group):
        """
        Filter annotations by removing those that are outside of the image bounds or whose width/height < 0.
        """
        # test all annotations
        image_shape = image_group[0].shape[0]
        for index in range(len(image_group)):
            annotations_group[index]['bboxes'] = np.clip(annotations_group[index]['bboxes'], 0, image_shape)

        for index, (image, annotations) in enumerate(zip(image_group, annotations_group)):
            # test x2 < x1 | y2 < y1 | x1 < 0 | y1 < 0 | x2 <= 0 | y2 <= 0 | x2 >= image.shape[1] | y2 >= image.shape[0]

            invalid_indices = np.where(
                (annotations['bboxes'][:, 2] <= annotations['bboxes'][:, 0]) |
                (annotations['bboxes'][:, 3] <= annotations['bboxes'][:, 1])
            )[0]

            # delete invalid indices
            if len(invalid_indices):
                warnings.warn('Image with id {} (shape {}) contains the following invalid boxes: {}.'.format(
                    group[index],
                    image.shape,
                    annotations['bboxes'][invalid_indices, :]
                ))
                for k in annotations_group[index].keys():
                    annotations_group[index][k] = np.delete(annotations[k], invalid_indices, axis=0)

        return image_group, annotations_group

    def compute_inputs(self, image_group, annotations_group):
        """
        Compute inputs for the network using an image_group.
        """
        images = np.array(image_group).astype(np.uint8)
        encoded = []
        for i in range(images.shape[0]):
            encoded.append(tf.io.encode_png(images[i], compression=-1))
        return encoded

    def compress_label(self, targets):
        shape_target0 = targets[0].shape[1:]
        shape_target1 = targets[1].shape[1:]
        return targets[0].reshape((targets[0].shape[0], -1)), shape_target0, \
               targets[1].reshape((targets[1].shape[0], -1)), shape_target1

    def __getitem__(self, index):
        """
        Keras sequence method for generating batches.
        """
        group = self.groups[index]
        inputs, targets = self.compute_inputs_targets(group)
        indices = [self.image_dirs[idx].split("/")[-1].split(".")[0] for idx in group]
        comp_target1, target1_shape, comp_target2, target2_shape = self.compress_label(targets)
        return inputs, comp_target1, target1_shape, comp_target2, target2_shape, indices


# Three types of data can be stored in TFRecords: bytestrings, integers and floats
# They are always stored as lists, a single data element will be a list of size 1

def _bytestring_feature(list_of_bytestrings):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=list_of_bytestrings))


def _int_feature(list_of_ints):  # int64
    return tf.train.Feature(int64_list=tf.train.Int64List(value=list_of_ints))


def _float_feature(list_of_floats):  # float32
    return tf.train.Feature(float_list=tf.train.FloatList(value=list_of_floats))


def to_tfrecord(img, classification_target, classification_target_shape, regression_target, regression_target_shape,
                index):
    feature = {
        "image": _bytestring_feature([img]),
        "classification_target": _float_feature(classification_target.tolist()),
        "classification_shape": _int_feature(list(classification_target_shape)),
        "regression_target": _float_feature(regression_target.tolist()),
        "regression_shape": _int_feature(list(regression_target_shape)),
        "index": _bytestring_feature([bytes(index, 'utf-8')]),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def read_tfrecord(example):
    features = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "classification_target": tf.io.VarLenFeature(tf.float32),
        "classification_shape": tf.io.VarLenFeature(tf.int64),
        "regression_target": tf.io.VarLenFeature(tf.float32),
        "regression_shape": tf.io.VarLenFeature(tf.int64),
        "index": tf.io.FixedLenFeature([], tf.string),
    }

    # decode the TFRecord
    example = tf.io.parse_single_example(example, features)

    # FixedLenFeature fields are now ready to use: exmple['size']
    # VarLenFeature fields require additional sparse_to_dense decoding

    image = tf.io.decode_png(example['image'])
    classification = tf.sparse.to_dense(example['classification_target'])
    classification_shape = tf.sparse.to_dense(example['classification_shape'])
    classification = tf.reshape(classification, classification_shape)
    regression = tf.sparse.to_dense(example['regression_target'])
    regression_shape = tf.sparse.to_dense(example['regression_shape'])
    regression = tf.reshape(regression, regression_shape)
    image_index = example['index']
    return image, classification, regression, image_index


if __name__ == '__main__':

    group = "test"
    seed = "10"
    dirs_dataframe = pd.read_csv(
        "/Users/kittipodpungcharoenkul/Downloads/CTC_data/bounding_box/colon_data_split_seed_{}.csv".format(seed))

    dirs_dataframe_selected = dirs_dataframe[dirs_dataframe['group'] == group]

    xml_base_dir = "/Users/kittipodpungcharoenkul/Downloads/CTC_data/bounding_box/colon_annotation/"
    xml_dirs = [os.path.join(xml_base_dir, filename + ".xml") for filename in dirs_dataframe_selected['filename']]

    image_base_dir = "/Users/kittipodpungcharoenkul/Downloads/CTC_data/"
    image_dirs = [os.path.join(image_base_dir, directory) for directory in dirs_dataframe_selected['img_dirs']]

    train_generator = TumorCellGenerator(
        xml_dirs,
        image_dirs,
        phi=0,
        batch_size=len(dirs_dataframe_selected),
        misc_effect=None,
        visual_effect=None,
        multiple=1
    )

    for batch_inputs, comp_target1, target1_shape, comp_target2, target2_shape, indices in tqdm.tqdm(train_generator):
        filename = "/Users/kittipodpungcharoenkul/Downloads/" \
                   "CTC_data/bounding_box/bbox_tfrecord/colon-output-seed{}-{}-{}.tfrec".format(
            seed, group, len(batch_inputs))
        with tf.io.TFRecordWriter(filename) as out_file:
            for i in tqdm.tqdm(range(len(batch_inputs))):
                example = to_tfrecord(batch_inputs[i].numpy(),
                                      comp_target1[i],
                                      target1_shape,
                                      comp_target2[i],
                                      target2_shape,
                                      indices[i])
                out_file.write(example.SerializeToString())

    option_no_order = tf.data.Options()
    option_no_order.experimental_deterministic = False

    filenames = tf.io.gfile.glob(
        '/Users/kittipodpungcharoenkul/Downloads/CTC_data/bounding_box/bbox_tfrecord/colon-output*seed21*.tfrec')
    train_dsr = tf.data.TFRecordDataset(filenames, num_parallel_reads=tf.data.experimental.AUTOTUNE)
    train_dsr = train_dsr.with_options(option_no_order)
    train_dsr = train_dsr.map(read_tfrecord, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    test_data = list(train_generator)
    #
    anchors = train_generator.anchors
    #
    # for image, classification, regression, image_index in train_dsr.take(10):
    #     classification = classification.numpy()
    #     regression = regression.numpy()
    #     image = image.numpy()
    #     # plt.imshow(image)
    #     # plt.show()
    #     valid_ids = np.where(regression[:, -1] == 1)[0]
    #     if valid_ids.shape[0] == 0:
    #         continue
    #     # print(valid_ids)
    #     boxes = anchors[valid_ids]
    #     deltas = regression[valid_ids]
    #     class_ids = np.argmax(classification[valid_ids], axis=-1)
    #     mean_ = [0, 0, 0, 0]
    #     std_ = [0.2, 0.2, 0.2, 0.2]
    #
    #     width = boxes[:, 2] - boxes[:, 0]
    #     height = boxes[:, 3] - boxes[:, 1]
    #
    #     x1 = boxes[:, 0] + (deltas[:, 0] * std_[0] + mean_[0]) * width
    #     y1 = boxes[:, 1] + (deltas[:, 1] * std_[1] + mean_[1]) * height
    #     x2 = boxes[:, 2] + (deltas[:, 2] * std_[2] + mean_[2]) * width
    #     y2 = boxes[:, 3] + (deltas[:, 3] * std_[3] + mean_[3]) * height
    #     for x1_, y1_, x2_, y2_, class_id in zip(x1, y1, x2, y2, class_ids):
    #         x1_, y1_, x2_, y2_ = int(x1_), int(y1_), int(x2_), int(y2_)
    #         start_point = (x1_, y1_)
    #         end_point = (x2_, y2_)
    #         cv2.rectangle(image, start_point, end_point, (0, 255, 0), 2)
    #         class_name = train_generator.labels[class_id]
    #         label = class_name
    #         ret, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)
    #         cv2.rectangle(image, (x1_, y2_ - ret[1] - baseline), (x1_ + ret[0], y2_), (255, 255, 255), -1)
    #         cv2.putText(image, label, (x1_, y2_ - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    #     plt.imshow(image)
    #     plt.show()

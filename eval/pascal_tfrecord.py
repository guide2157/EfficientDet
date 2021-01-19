# import keras
import tensorflow as tf
import numpy as np

from map_metric.MetricBuilder import MeanAveragePrecision2d

tfk = tf.keras


def _get_detections(generator, model):
    """
    Get the detections from the model using the generator.

    Args:
        generator: The generator used to run images through the model.
        model: The model to run on the images.

    Returns:
        List of predictions and their corresponding ground truth label.

    """
    detections = []
    truths = []
    for image, bounding_boxes, labels in generator:
        boxes, scores, pred_labels = model.predict(image)
        detections.append(np.hstack((boxes, pred_labels, scores)))
        truths.append(np.hstack((bounding_boxes, labels)))
    return truths, detections


def evaluate(
        generator,
        model,
        iou_threshold=0.5,
        max_detections=100
):
    """
    Evaluate a given dataset using a given model.

    Args:
        generator: The generator that represents the dataset to evaluate.
        model: The model to evaluate.
        iou_threshold: The threshold used to consider when a detection is positive or negative.
        max_detections: The maximum number of detections to use per image.

    Returns:
        A dict mapping class names to mAP scores.

    """
    # gather all detections and annotations
    truths, detections = \
        _get_detections(generator, model, iou_threshold=iou_threshold, max_detections=max_detections)
    num_classes = np.shape(np.unique(detections[0][:, -2]))[0]
    map_metric = MeanAveragePrecision2d(num_classes)
    for i in range(len(truths)):
        map_metric.add(detections[i], truths[i])
    return map_metric.value(iou_thresholds=[0.5])['mAP']
    # return average_precisions


class Evaluate(tfk.callbacks.Callback):
    """
    Evaluation callback for arbitrary datasets.
    """

    def __init__(
            self,
            tfrecord_generator,
            model,
            iou_threshold=0.5,
            max_detections=100,
            save_path=None,
            tensorboard=None,
            weighted_average=False,
            verbose=1
    ):
        """
        Evaluate a given dataset using a given model at the end of every epoch during training.

        Args:
            tfrecord_generator: The generator that represents the dataset to evaluate.
            iou_threshold: The threshold used to consider when a detection is positive or negative.
            max_detections: The maximum number of detections to use per image.
            save_path: The path to save images with visualized detections to.
            tensorboard: Instance of keras.callbacks.TensorBoard used to log the mAP value.
            weighted_average: Compute the mAP using the weighted average of precisions among classes.
            verbose: Set the verbosity level, by default this is set to 1.
        """
        self.tfrecord_generator = tfrecord_generator
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections
        self.save_path = save_path
        self.tensorboard = tensorboard
        self.weighted_average = weighted_average
        self.verbose = verbose
        self.active_model = model

        super(Evaluate, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        # run evaluation
        self.mean_ap = evaluate(
            self.tfrecord_generator,
            self.active_model,
            iou_threshold=self.iou_threshold,
            max_detections=self.max_detections
        )

        if self.tensorboard is not None:
            if tf.version.VERSION < '2.0.0' and self.tensorboard.writer is not None:
                summary = tf.Summary()
                summary_value = summary.value.add()
                summary_value.simple_value = self.mean_ap
                summary_value.tag = "mAP"
                self.tensorboard.writer.add_summary(summary, epoch)
            else:
                tf.summary.scalar('mAP', self.mean_ap, epoch)

        logs['mAP'] = self.mean_ap

        if self.verbose == 1:
            print('mAP: {:.4f}'.format(self.mean_ap))

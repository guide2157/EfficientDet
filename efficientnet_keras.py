from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os


def find_feature_layers(backbone, block_nums=None):
    model_layers = backbone.layers
    if block_nums is None:
        block_nums = ["1", "2", "3", "5", "7"]
    feature_layers = []
    previous_num = 0
    for idx, layer in enumerate(model_layers):
        layer_name = layer.name
        current_num = layer_name[5]
        if not current_num.isnumeric():
            if previous_num in block_nums:
                feature_layers.append(model_layers[idx - 1].name)
        else:
            if previous_num != current_num and previous_num in block_nums:
                feature_layers.append(model_layers[idx - 1].name)
            previous_num = current_num
        if len(feature_layers) == len(block_nums):
            break
    feature_layers = [backbone.get_layer(layer_name).output for layer_name in feature_layers]
    return feature_layers


def initialize_model(backbone, input_tensor, weights=None):
    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')
    if weights == 'imagenet':
        backbone_model = backbone(weights=weights, include_top=False, input_tensor=input_tensor)
    else:
        backbone_model = backbone(weights=None, include_top=False, input_tensor=input_tensor)
        if weights is not None:
            backbone_model.load_weights(weights)
    return backbone_model

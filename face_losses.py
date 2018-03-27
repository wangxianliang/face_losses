# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Define custom losses used in face recognition"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import math
import tensorflow as tf

slim = tf.contrib.slim


def triplet_loss(anchor, positive, negative, alpha,
                 weights=1.0, scope='triplet_loss'):
    """Calculate the triplet loss according to the FaceNet paper.

    Args:
        anchor: the embeddings for the anchor images.
                2D tensor of size [batch_size, embedding_size].
        positive: the embeddings for the positive images.
                  2D tensor of size [batch_size, embedding_size].
        negative: the embeddings for the negative images.
                  2D tensor of size [batch_size, embedding_size].
        weights: loss weight.

    Return:
      the triplet loss according to the FaceNet paper as a float tensor.
    """
    with tf.variable_scope(scope):
        pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=1)
        neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=1)

        basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
        loss = tf.maximum(basic_loss, 0.0)
        loss = tf.multiply(weights, tf.reduce_mean(loss, axis=0), 'triplet_loss')
        tf.losses.add_loss(loss) # Letting TF-Slim know about the triplet loss.        
        return loss


def center_loss(features, labels, alpha, num_classes,
                weights=1.0,
                scope='center_loss',
                reuse=None):
    """Center loss based on the paper 
        "A Discriminative Feature Learning Approach for Deep Face Recognition"
       (http://ydwen.github.io/papers/WenECCV16.pdf)
    """
    with tf.variable_scope(scope, reuse=reuse):
        feature_dim = features.get_shape()[1]
        # Each output layer use one independed center: scope/centers
        centers = tf.get_variable('centers',
                                  [num_classes, feature_dim],
                                  dtype=tf.float32,
                                  initializer=tf.constant_initializer(0),
                                  trainable=False)
        # added to model variable collections
        tf.contrib.framework.add_model_variable(centers)
        labels = tf.reshape(labels, [-1])
        centers_batch = tf.gather(centers, labels)
        diff = (1 - alpha) * (centers_batch - features)
        centers = tf.scatter_sub(centers, labels, diff)
        loss = tf.multiply(weights,
                           tf.reduce_mean(tf.square(features - centers_batch)),
                           'center_loss')
        tf.losses.add_loss(loss) # Letting TF-Slim know about center loss.
        return loss, centers


def siamense_loss(anchor, positive, negative, dist_thresh,
                  margin, weights=1.0, scope='siamense_loss'):
    """Calculate the siamense loss.
       loss = max(0, |a - p| - dist_thresh + margin) + 
              max(0, dist_thresh + margin - |a - n|)

    Args:
        anchor: the embeddings for the anchor images.
                2D tensor of size [batch_size, embedding_size].
        positive: the embeddings for the positive images.
                  2D tensor of size [batch_size, embedding_size].
        negative: the embeddings for the negative images.
                  2D tensor of size [batch_size, embedding_size].
        weights: loss weight.

    Return:
      the siamense loss.
    """
    with tf.variable_scope(scope):
        pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=1)
        neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=1)
        pos_loss = tf.maximum(pos_dist - dist_thresh + margin, 0.0)
        neg_loss = tf.maximum(dist_thresh + margin - neg_dist, 0.0)
        loss = weights * tf.reduce_mean(pos_loss + neg_loss, axis=0)
        tf.losses.add_loss(loss) # Letting TF-Slim know about the siamense loss.
        return loss


def large_margin_cosine_loss(inputs, labels, num_outputs,
                             scale=60.0, margin=0.35,
                             weights_initializer=slim.xavier_initializer(),
                             weights_regularizer=None,
                             reuse=None,
                             variables_collections=None,
                             outputs_collections=None,
                             trainable=True,
                             weights=1.0,
                             scope='large_margin_cosine_loss'):
    """
    Implement large margin cosine loss based on the paper
    "CosFace: Large Margin Cosine Loss for Deep Face Recognition"
    (https://arxiv.org/abs/1801.09414) &
    "Additive Margin Softmax for Face Verification"
    (https://arxiv.org/abs/1801.05599)

    Args:
        inputs: input embeddings.
        labels: Tensor of shape [d_0, d_1, ..., d_{r-1}] (where r is rank of labels 
        and result) and dtype int32 or int64. Each entry in labels must be an 
        index in [0, num_classes). Other values will raise an exception when this
        op is run on CPU, and return NaN for corresponding loss and 
        gradient rows on GPU.
        num_outputs: number of classes.
    """
    assert isinstance(scale, float) and scale > 0, (
        'scale must be larger than 0.')
    assert isinstance(margin, float) and 0 <= margin <= 0.8, (
        'margin must be in [0, 0.8]')
    with tf.variable_scope(scope, [inputs, labels],
                            reuse=reuse) as sc:
        inputs_dim = inputs.get_shape()[1]
        if inputs_dim is None:
            inputs_dim = tf.shape(inputs)[1]

        # kernel has shape (inputs_dim, num_outputs)
        kernel = tf.get_variable('kernel',
                                 [inputs_dim, num_outputs],
                                 dtype=inputs.dtype,
                                 initializer=weights_initializer,
                                 regularizer=weights_regularizer,
                                 trainable=trainable,
                                 collections=variables_collections)
        # added to model variable collections
        tf.contrib.framework.add_model_variable(kernel)
        # compute normalized kernel, (inputs_dim, num_outputs)
        kernel = tf.nn.l2_normalize(x=kernel, axis=0, epsilon=1e-10)
        # compute normalized input, (batch_size, inputs_dim)
        inputs = tf.nn.l2_normalize(x=inputs, axis=1, epsilon=1e-10)
        # compute cos(theta) = XW/(||X||*||W||), (batch_size, num_outputs)
        cos_theta = tf.matmul(inputs, kernel)
        #cos_theta = tf.clip_by_value(cos_theta, -1.0, 1.0)
        # compute s*cos(theta)
        scale_cos_theta = scale * cos_theta

        # one hot labels
        one_hot_labels = tf.one_hot(labels, num_outputs,
                                    on_value=scale * margin, off_value=0.0)
        logits = scale_cos_theta - one_hot_labels
        loss = tf.losses.sparse_softmax_cross_entropy(
            labels, logits, weights, scope)
        return loss


def additive_angular_margin_loss(inputs, labels, num_outputs,
                                 scale=64.0, margin=0.5,
                                 easy_margin=False,
                                 weights_initializer=slim.xavier_initializer(),
                                 weights_regularizer=None,
                                 reuse=None,
                                 variables_collections=None,
                                 outputs_collections=None,
                                 trainable=True,
                                 weights=1.0,
                                 scope='additive_angular_margin_loss'):
    """
    Implement additive angular margin loss based on the paper
    "ArcFace : Additive Angular Margin Loss for Deep Face Recognition"
    (https://arxiv.org/abs/1801.07698)

    Args:
        inputs: input embeddings.
        labels: Tensor of shape [d_0, d_1, ..., d_{r-1}] (where r is rank of labels 
        and result) and dtype int32 or int64. Each entry in labels must be an 
        index in [0, num_classes). Other values will raise an exception when this
        op is run on CPU, and return NaN for corresponding loss and 
        gradient rows on GPU.
        num_outputs: number of classes.
    """
    assert isinstance(scale, float) and scale > 0, (
        'scale must be larger than 0.')
    assert isinstance(margin, float) and 0 <= margin <= 0.8, (
        'margin must be in [0, 0.8]')
    cos_m = math.cos(margin)
    sin_m = math.sin(margin)
    # compute scale * margin * sin(pi - margin)
    scale_margin_sin_margin = scale * margin * math.sin(math.pi - margin)
    with tf.variable_scope(scope, [inputs, labels],
                           reuse=reuse) as sc:
        # inputs = tf.stop_gradient(inputs)
        inputs_dim = inputs.get_shape()[1]
        if inputs_dim is None:
            inputs_dim = tf.shape(inputs)[1]

        # kernel has shape (inputs_dim, num_outputs)
        kernel = tf.get_variable('kernel',
                                 [inputs_dim, num_outputs],
                                 dtype=inputs.dtype,
                                 initializer=weights_initializer,
                                 regularizer=weights_regularizer,
                                 trainable=trainable,
                                 collections=variables_collections)
        # added to model variable collections
        tf.contrib.framework.add_model_variable(kernel)
        # compute normalized kernel, (inputs_dim, num_outputs)
        kernel = tf.nn.l2_normalize(x=kernel, axis=0, epsilon=1e-10)
        # compute normalized input, (batch_size, inputs_dim)
        inputs = tf.nn.l2_normalize(x=inputs, axis=1, epsilon=1e-10)
        # compute cos(theta) = XW/(||X||*||W||), (batch_size, num_outputs)
        cos_theta = tf.matmul(inputs, kernel)
        # cos_theta = tf.clip_by_value(cos_theta, -1.0, 1.0)
        # compute s*cos(theta)
        scale_cos_theta = scale * cos_theta
    
        # compute sin(theta) = sqrt(1 - cos(theta) * cos(theta))
        sin_theta = tf.sqrt(1.0 - cos_theta * cos_theta)
        # compute scale*cos(theta+margin)=s*[cos(theta)*cos(m)-sin(theta)*sin(m)]
        cos_theta_margin = cos_theta * cos_m - sin_theta * sin_m
        scale_cos_theta_margin = scale * cos_theta_margin
    
        # the condition ensures that theta + margin lies in [0, PI],
        #  -margin <= theta <= PI - margin
        if easy_margin:   # easy margin
            cond = tf.cast(tf.nn.relu(cos_theta), dtype=tf.bool)
            keep_val = scale_cos_theta
        else:
            threshold = math.cos(math.pi - margin)
            cond = tf.cast(tf.nn.relu(cos_theta - threshold), dtype=tf.bool)
            keep_val = scale_cos_theta - scale_margin_sin_margin

        cond = tf.stop_gradient(cond)
        scale_cos_theta_margin = tf.where(cond, scale_cos_theta_margin, keep_val)
        mask = tf.one_hot(labels, num_outputs)
        logits = scale_cos_theta * (1.0 - mask) + scale_cos_theta_margin * mask
        loss = tf.losses.sparse_softmax_cross_entropy(
            labels, logits, weights, scope)
        return loss

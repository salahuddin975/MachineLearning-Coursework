from __future__ import absolute_import, division, print_function, unicode_literals

import os
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import math
from tensorflow.keras.models import load_model


def get_dataset(url, batch_size, column_names, label_names):
    train_dataset_fp = tf.keras.utils.get_file(fname=os.path.basename(url), origin=url)
    print("Local copy of the dataset file: {}".format(train_dataset_fp))

    train_dataset = tf.data.experimental.make_csv_dataset(
        train_dataset_fp,
        batch_size,
        column_names=column_names,
        label_name=label_names,
        num_epochs=1)

    return train_dataset         #returns (features, label) pairs; where features is a dictionary: {'feature_name': value}


def pack_features_vector(features, labels):
    features = tf.stack(list(features.values()), axis=1)
    return features, labels


def sigmoid(wx):
    return (1/(1+ math.exp(wx * -1)))


# Propagating Forward: Hidden Layer. σ ( WX )
def hidden_layer_node_update(feature, first_hidden_layer, all_activation_value):
    for i in range(first_hidden_layer_size):
        wx = hidden_layer_weights[i] * feature
        sigma_wx = sigmoid(sum(wx))
        first_hidden_layer.append(sigma_wx)
        all_activation_value.append(sigma_wx)

    return first_hidden_layer


# Propagating Forward: Output Layer σ ( WX )
def output_layer_node_update(first_hidden_layer, output_layer):
    for i in range(output_layer_size):
        wx = output_layer_weights[i] * first_hidden_layer
        sigma_wx = sigmoid(sum(wx))
        output_layer.append(sigma_wx)
#        all_activation_value.append(sigma_wx)

    return output_layer


# Backpropagation: Output layer delta δk = Ok (1-Ok)(tk – Ok)
def compute_output_layer_delta(output_layer, true_label, output_delta):
    output_bool = [0] * output_layer_size
    output_bool[true_label.numpy()] = 1

    for i in range(output_layer_size):
        delta = output_layer[i] * (1 - output_layer[i]) * (output_bool[i] - output_layer[i])
        output_delta.append(delta)


# Backpropagation: Weight of hidden layer to output layer; Wji  =  Wji  +  ΔWji
# ΔWji = η δj Xji
def update_output_layer_weight(first_hidden_layer, output_delta):
    for i in range(first_hidden_layer_size):
        for j in range(output_layer_size):
            delta_w = learn_rate * first_hidden_layer[i] * output_delta[j]
            output_layer_weights[j][i] = output_layer_weights[j][i] + delta_w


# Backpropagation: Hidden layer delta, δh = Oh(1-Oh) Σk Wkh δk
def compute_hidden_layer_delta(first_hidden_layer, output_delta, hidden_delta):
    for i in range(first_hidden_layer_size):
        sum_delta = 0
        for j in range(output_layer_size):
            sum_delta = sum_delta + output_layer_weights[j][i] * output_delta[j]
        delta = first_hidden_layer[i] * (1 - first_hidden_layer[i]) * sum_delta
        hidden_delta.append(delta)


# Backpropagation: Weight of input layer to hidden layer; Wji  =  Wji  +  ΔWji
# ΔWji = η δj Xji
def update_hidden_layer_weight(input_layer, hidden_delta):
    for i in range(input_layer_size):
        for j in range(first_hidden_layer_size):
            delta_w = learn_rate * input_layer[i] * hidden_delta[j]
            hidden_layer_weights[j][i] = hidden_layer_weights[j][i] + delta_w


def gradient_function(predicted_output, true_output):
    with tf.GradientTape() as tape:
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        loss_value = loss_object(y_true=true_output, y_pred=predicted_output)

    gradients = tape.gradient(loss_value, model.trainable_variables)
    return loss_value, gradients


def calculate_error(predicted_value, true_value):
    return    ((predicted_value[true_value.numpy()] - 1) * (predicted_value[true_value.numpy()] - 1))



def train_the_model(train_dataset):
    for features, labels in train_dataset:
        all_activation_value = []
        for epoch in range(num_epochs):
            error = 0
            train_error = []
            for i in range(train_batch_size):
                input_layer = features[i]
                label = labels[i]

                hidden_layer = []
                output_layer = []
                hidden_layer_node_update(input_layer, hidden_layer, all_activation_value)
                output_layer_node_update(hidden_layer, output_layer)
    #            print(hidden_layer)
    #            print (output_layer)

                error = error + pow((output_layer[label.numpy()] -1), 2)
#                print (error)
                output_delta = []
                compute_output_layer_delta(output_layer, label, output_delta)
                update_output_layer_weight(hidden_layer, output_delta)

                hidden_delta = []
                compute_hidden_layer_delta(hidden_layer, output_delta, hidden_delta)
                update_hidden_layer_weight(features[i], hidden_delta)

            train_error.append(error/2)
            print ("epoch: ", epoch, "; error: ", error/2)
    #        print (first_hidden_layer_weights)
    #            print (output_layer_weights)
        histogram_with_activation_value(all_activation_value)
        break


def histogram_with_activation_value(all_activation_value):
    print (len(all_activation_value))
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    plt.hist(all_activation_value, bins, histtype='bar', linewidth=0)
    plt.xticks(np.arange(0, 1.01, step=0.1))
    plt.show()


def build_model():
    train_dataset = get_dataset(train_dataset_url, train_batch_size, column_names, label_name)
#    plot_dataset(train_dataset)
    train_dataset = train_dataset.map(pack_features_vector)
    train_the_model(train_dataset)


def predict():
    test_dataset = get_dataset(test_dataset_url, test_batch_size, column_names, label_name)
    test_dataset = test_dataset.map(pack_features_vector)

    j=0
    for (features, label) in test_dataset:
        for i in range(test_batch_size):
            first_hidden_layer = []
            output_layer = []
            hidden_layer_node_update(features[i], first_hidden_layer)
            output_layer_node_update(first_hidden_layer, output_layer)

            max_value = max(output_layer)
            max_index = output_layer.index(max_value)

            print (label[i])
            print(output_layer)

            if(label[i].numpy() == max_index):
                j = j+ 1

        break
    print ("Currect prediction: ", j)


if __name__ == '__main__':
    train_dataset_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv"
    test_dataset_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv"

    activation_unit = tf.nn.relu
    learn_rate = 0.01
    num_epochs = 300
    train_batch_size = 110
    test_batch_size = 30
    model_name = 'hw2_trained_model.h5'

    class_names = ['Iris setosa', 'Iris versicolor', 'Iris virginica']
    column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
    feature_names = column_names[:-1]
    label_name = column_names[-1]

    input_layer_size = len(feature_names)
    first_hidden_layer_size = 10
    output_layer_size = len(class_names)

    hidden_layer_weights = np.random.random((first_hidden_layer_size, input_layer_size))
    output_layer_weights = np.random.random((output_layer_size, first_hidden_layer_size))

    build_model()

#    predict()
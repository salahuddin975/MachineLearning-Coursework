from __future__ import absolute_import, division, print_function, unicode_literals

import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model


def pack_features_vector(features, labels):
    features = tf.stack(list(features.values()), axis=1)
    return features, labels


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


def create_the_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(first_hidden_layer_size, activation=activation_unit, input_shape=(input_layer_size,)),
#        tf.keras.layers.Dense(second_hidden_layer_size, activation=activation_unit),
        tf.keras.layers.Dense(output_layer_size)
    ])

    return model

def loss(model, loss_object, features, true_output):
    predicted_output = model(features)
    return loss_object(y_true=true_output, y_pred=predicted_output)


def gradient_function(model, loss_object, input_features, true_output):
    with tf.GradientTape() as tape:
        loss_value = loss(model, loss_object,input_features, true_output)

    gradients = tape.gradient(loss_value, model.trainable_variables)
    return loss_value, gradients


def train_the_model(model, train_dataset, test_dataset):
    train_loss_results = []
    test_loss_results = []
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    for epoch in range(num_epochs):
        train_loss_avg = tf.keras.metrics.Mean()
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

        for features, label in train_dataset:
            loss_value, gradients = gradient_function(model, loss_object, features, label)       # Calculate single optimization step
            optimizer = tf.keras.optimizers.Adam(learning_rate=learn_rate)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            # Track progress
            train_loss_avg(loss_value)                               # Add current batch loss
            train_accuracy(label, model(features))                   # Compare predicted label to actual label

        train_loss_results.append(train_loss_avg.result())

        test_loss_avg = get_test_loss_avg(model, loss_object, test_dataset)
        test_loss_results.append(test_loss_avg.result())

        if epoch % 25 == 0:
            print("Epoch {:03d}: Train Loss: {:.3f}, Train accuracy: {:.3%}, Test Loss: {:.3f}".format(epoch,
                train_loss_avg.result(), train_accuracy.result(), test_loss_avg.result()))

    return model, train_loss_results, test_loss_results


#========================================= Plots ================================

def plot_dataset(dataset):
    features, labels = next(iter(dataset))
    print(features)

    plt.scatter(features['petal_length'],
                features['sepal_length'],
                c=labels,
                cmap='viridis')

    plt.xlabel("Petal length")
    plt.ylabel("Sepal length")
    plt.show()


def visualize_loss_function_over_time(train_loss_results, train_accuracy_results):
    fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
    fig.suptitle('Training Metrics')

    axes[0].set_ylabel("Loss", fontsize=14)
    axes[0].plot(train_loss_results)

    axes[1].set_ylabel("Accuracy", fontsize=14)
    axes[1].set_xlabel("Epoch", fontsize=14)
    axes[1].plot(train_accuracy_results)
    plt.show()


def error_chart(train_lost_result, test_loss_result):
    plt.plot(train_lost_result, label="Train error")
    plt.plot(test_loss_result, label="Test error")

    plt.xlabel("Iteration")
    plt.ylabel("Error")
    plt.legend()
    plt.show()


#============================= Testing ========================

def get_test_loss_avg(model, loss_object, test_dataset):
    test_loss_avg = tf.keras.metrics.Mean()

    for (features, label) in test_dataset:
        loss_value = loss(model, loss_object, features, label)
        test_loss_avg(loss_value)                               # Add current batch loss

    return test_loss_avg


def make_prediction(model, class_names):
    predict_dataset = tf.convert_to_tensor([
        [5.1, 3.3, 1.7, 0.5,],
        [5.9, 3.0, 4.2, 1.5,],
        [6.9, 3.1, 5.4, 2.1]
    ])

    predictions = model(predict_dataset)

    for i, logits in enumerate(predictions):
        class_idx = tf.argmax(logits).numpy()
        p = tf.nn.softmax(logits)[class_idx]
        name = class_names[class_idx]
        print("Example {} prediction: {} ({:4.1f}%)".format(i, name, 100*p))


def build_model():
    train_dataset = get_dataset(train_dataset_url, train_batch_size, column_names, label_name)
#    plot_dataset(train_dataset)

#========= for training and test error chart ==============
    test_dataset = get_dataset(test_dataset_url, test_batch_size, column_names, label_name)
    #    plot_dataset(test_dataset)
    test_dataset = test_dataset.map(pack_features_vector)
#==============

    train_dataset = train_dataset.map(pack_features_vector)
    model = create_the_model()
    model, train_loss_results, test_loss_results = train_the_model(model, train_dataset, test_dataset)

#    visualize_loss_function_over_time(train_loss_results, train_accuracy_results)
    error_chart(train_loss_results, test_loss_results)

    model.save(model_name)
    return model


def test_the_model(model_name):
    model = load_model(model_name, compile=False)

    test_dataset = get_dataset(test_dataset_url, test_batch_size, column_names, label_name)
#    plot_dataset(test_dataset)
    test_dataset = test_dataset.map(pack_features_vector)
    evaluate_test_data(model, test_dataset)

#    make_prediction(model, class_names)


if __name__ == '__main__':
    train_dataset_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv"
    test_dataset_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv"

    activation_unit = tf.nn.relu
    learn_rate = 0.001
    num_epochs = 2000
    train_batch_size = 110
    test_batch_size = 30
    model_name = 'hw2_trained_model.h5'

    class_names = ['Iris setosa', 'Iris versicolor', 'Iris virginica']
    column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
    feature_names = column_names[:-1]
    label_name = column_names[-1]

    input_layer_size = len(feature_names)
    first_hidden_layer_size = 10
    second_hidden_layer_size = 10
    output_layer_size = len(class_names)

    model = build_model()
#    test_the_model(model_name)





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

    train_dataset = tf.data.experimental.make_csv_dataset(           #The make_csv_dataset function returns a tf.data.Dataset of (features, label) pairs, where features is a dictionary: {'feature_name': value}
        train_dataset_fp,
        batch_size,
        column_names=column_names,
        label_name=label_names,
        num_epochs=1)

    return train_dataset


def create_the_model():
    model = tf.keras.Sequential([
      tf.keras.layers.Dense(10, activation=activation_unit, input_shape=(4,)),  # input shape required
      tf.keras.layers.Dense(10, activation=activation_unit),
      tf.keras.layers.Dense(3)
    ])

    return model


def loss(model, x, y):
  y_ = model(x)
  loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

  return loss_object(y_true=y, y_pred=y_)


def gradient_function(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)

    return loss_value, tape.gradient(loss_value, model.trainable_variables)


def train_the_model(model, train_dataset):
    train_loss_results = []
    train_accuracy_results = []

    for epoch in range(num_epochs):
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

        for x, y in train_dataset:
            # Optimize the model
            optimizer = tf.keras.optimizers.Adam(learning_rate=learn_rate)
            loss_value, grads = gradient_function(model, x, y)       # Calculate single optimization step
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # Track progress
            epoch_loss_avg(loss_value)  # Add current batch loss
            # Compare predicted label to actual label
            epoch_accuracy(y, model(x))
        # End epoch

        train_loss_results.append(epoch_loss_avg.result())
        train_accuracy_results.append(epoch_accuracy.result())

        if epoch % 50 == 0:
            print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch, epoch_loss_avg.result(), epoch_accuracy.result()))

    return model, train_loss_results, train_accuracy_results

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

#============================= Testing ========================

def evaluate_test_data(model, test_dataset):
    test_accuracy = tf.keras.metrics.Accuracy()

    for (x, y) in test_dataset:
        logits = model(x)
        prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
        test_accuracy(prediction, y)

    print("Test set accuracy: {:.3%}".format(test_accuracy.result()))


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

    model = create_the_model()

    train_dataset = train_dataset.map(pack_features_vector)
    model, train_loss_results, train_accuracy_results = train_the_model(model, train_dataset)
#    visualize_loss_function_over_time(train_loss_results, train_accuracy_results)

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

    num_epochs = 201
    learn_rate = 0.01
    activation_unit = tf.nn.relu

    train_batch_size = 110
    test_batch_size = 30
    model_name = 'hw2_trained_model.h5'

    class_names = ['Iris setosa', 'Iris versicolor', 'Iris virginica']
    column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
    feature_names = column_names[:-1]
    label_name = column_names[-1]

    model = build_model()
    test_the_model(model_name)





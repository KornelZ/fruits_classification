import numpy as np
import tensorflow as tf
import sklearn.datasets
import sklearn.metrics
import keras_preprocessing.image as img
import matplotlib.pyplot as plt
import datetime
import itertools

TRAIN_PATH = "fruits-360/Training"
TEST_PATH = "fruits-360/Test" #"fruits-360/RawImages"
USE_SMALL_DATASET = False
SMALL_DATASET_CLASS_NUM = 10

TRAIN_PERCENT = 0.7
PIXEL_DEPTH = 255
CLASSIFIER = "CNN"  # or DNN
IMG_SIZE = 100

LEARNING_RATE = 0.001

#CNN parameters

BREAK_EARLY = False
CNN_LAYERS = [
    [3, 5, 16],
    [16, 5, 32],
    [32, 5, 64]
]

MAX_POOLING = True
# MLP parameters
SIZE_1 = 200
SIZE_2 = 100

BATCH_SIZE = 32
EPOCHS = 20


def get_small_dataset(x, y):
    """Zwraca maly dataset o liczbie klas SMALL_DATASET_CLASS_NUM"""
    if not USE_SMALL_DATASET:
        return x, y
    x, y = zip(*((data, label) for data, label in zip(x, y) if label < SMALL_DATASET_CLASS_NUM))
    return np.array(x), np.array(y)


def get_figure_file_name(fig_name):
    """Generuje nazwe pliku rok-miesiac-dzien-godz-min-sek"""
    cur_day = datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
    return fig_name + "_" + cur_day + ".png"


def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
        """Rysuje confusion matrix dla malego datasetu"""
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        # digit format
        fmt = 'd'
        # threshold for coloring the figure
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(get_figure_file_name("cm"))
        print('Confusion matrix saved.')
        plt.close()


def plot_history(epochs, y, line, ylabel):
    """Rysuje wykres wartosci parametru w zaleznosci od iteracji"""
    ep = np.arange(0, epochs)
    for i in range(len(y)):
        plt.plot(ep, y[i], line[i])

    plt.xlabel("EPOCHS")
    plt.ylabel(ylabel)
    plt.savefig(get_figure_file_name(ylabel))
    plt.show()


def load(path):
    """Laduje dataset z danej sciezki"""
    dataset = sklearn.datasets.load_files(path)
    return np.array(dataset["filenames"]),\
            np.array(dataset["target"])


def one_hot(labels, num_classes):
    """Przeprowadza one-hot encoding [0, ..., 0, 1..., 0] na etykietach"""
    return np.eye(num_classes, dtype=np.float32)[labels]


def load_images(paths):
    """Laduje obrazy ze wskazanych sciezek jako NumPy array"""
    return np.array([img.img_to_array(img.load_img(p)) for p in paths],
                    dtype=np.float32)


def norm(data):
    """Normalizuje obraz do skali [0-1]"""
    return data / PIXEL_DEPTH


def batch(data, labels, randomize=True):
    """Generuje batch danych, losowy dla treningu"""
    if randomize:
        perm = np.random.permutation(len(labels))
        data, labels = data[perm], labels[perm]

    for i in range(0, len(labels), BATCH_SIZE):
        if i + BATCH_SIZE >= len(labels):
            yield data[i:], labels[i:]
        else:
            yield data[i:i+BATCH_SIZE], labels[i:i+BATCH_SIZE]


print("Loading data paths")
train_data, train_labels = get_small_dataset(*load(TRAIN_PATH))
perm = np.random.permutation(len(train_labels))
train_data, train_labels = train_data[perm], train_labels[perm]

test_data_paths, test_labels = get_small_dataset(*load(TEST_PATH))

print("One-hot encoding")
num_classes = len(np.unique(train_labels))
print("Number of classes:", num_classes)
if num_classes < len(np.unique(test_labels)):
    raise Exception("Too many classes in test labels")
train_labels = one_hot(train_labels, num_classes)
test_labels = one_hot(test_labels, num_classes)

print("Loading validation set")
train_size = int(TRAIN_PERCENT * train_labels.shape[0])
valid_data = norm(load_images(train_data[train_size:]))
valid_labels = train_labels[train_size:]
print("Loading training set")
train_data = norm(load_images(train_data[:train_size]))
train_labels = train_labels[:train_size]
print("Loading test set")
test_data = norm(load_images(test_data_paths))

print("Train size {}, Valid size {}, Test size {}"
      .format(train_data.shape, valid_data.shape, test_data.shape))


def cnn(num_classes):
    """Tworzy siec konwolucyjną dla danej liczby klas"""
    print("Building CNN")
    graph = tf.Graph()
    with graph.as_default():
        x = tf.placeholder(tf.float32, shape=(None, IMG_SIZE, IMG_SIZE, 3))
        y = tf.placeholder(tf.float32, shape=(None, num_classes))

        def conv(prev, prev_channels, filter_size, num_filters):
            w = tf.Variable(
                tf.truncated_normal(shape=[filter_size,
                                           filter_size,
                                           prev_channels,
                                           num_filters]))
            b = tf.Variable(tf.constant(0.0, shape=[num_filters]))

            conv = tf.nn.conv2d(input=prev,
                                filter=w,
                                strides=[1, 1, 1, 1],
                                padding="SAME")
            conv = tf.nn.relu(conv + b)

            def pool(input_layer, ksize, strides, padding):
                if MAX_POOLING:
                    return tf.nn.max_pool(value=input_layer,
                                          ksize=ksize,
                                          strides=strides,
                                          padding=padding)
                else:
                    return tf.nn.avg_pool(value=input_layer,
                                          ksize=ksize,
                                          strides=strides,
                                          padding=padding)
            return pool(input_layer=conv,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding="SAME")

        def dense(prev, prev_size, out_size):
            w = tf.Variable(
                tf.truncated_normal(shape=[prev_size, out_size])
            )
            b = tf.Variable(
                tf.constant(0.0, shape=[out_size])
            )
            return tf.matmul(prev, w) + b
        net = x
        for layer in CNN_LAYERS:
            net = conv(net, prev_channels=layer[0], filter_size=layer[1], num_filters=layer[2])

        dense_size = int(net.shape[1] * net.shape[2] * net.shape[3])
        net = tf.reshape(net, [-1, dense_size])
        net = dense(net, prev_size=dense_size, out_size=num_classes)
        pred = tf.argmax(net, axis=1)
        print("Last layer {} : in size {}".format(dense_size, net.shape))
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(logits=net, labels=y)
        )
        accuracy = tf.reduce_mean(tf.cast(tf.equal(
            tf.argmax(net, axis=1),
            tf.argmax(y, axis=1)
        ), dtype=np.float32))
        global_step = tf.Variable(0)
        optimizer = tf.train.AdamOptimizer(LEARNING_RATE)\
            .minimize(loss, global_step=global_step)

        def batch_run(data, labels, get_pred=False):
            valid_loss = 0
            valid_acc = 0
            i = 0
            predictions = []
            for batch_data, batch_labels in batch(data, labels, not get_pred):
                feed_dict = {x: batch_data, y: batch_labels}
                length = len(batch_labels)
                v_pred, v_loss, v_acc = session.run([pred, loss, accuracy], feed_dict=feed_dict)
                valid_loss += v_loss * length
                valid_acc += v_acc * length
                i += length
                predictions.append(v_pred)
            valid_acc /= i
            valid_loss /= i
            if get_pred:
                return valid_loss, valid_acc, np.concatenate(predictions)
            return valid_loss, valid_acc

        session = tf.Session()
        with session.as_default():
            session.run(tf.global_variables_initializer())
            train_loss_history = []
            train_acc_history = []
            valid_loss_history = []
            valid_acc_history = []
            total = int(np.ceil(len(train_labels) / BATCH_SIZE))
            last_valid_acc = -1
            eps = 0
            for ep in range(EPOCHS):
                eps += 1
                print("EPOCH: ", ep + 1)
                i = 0
                train_loss = 0
                train_acc = 0
                total_length = 0
                for batch_data, batch_labels in batch(train_data, train_labels):
                    feed_dict = {x: batch_data, y: batch_labels}
                    length = len(batch_labels)
                    _, t_loss, t_acc = session.run([optimizer, loss, accuracy], feed_dict=feed_dict)
                    train_loss += t_loss * length
                    train_acc += t_acc * length
                    total_length += length
                    i += 1
                    if i % 50 == 0 or i == total:
                        print("Batch {}/{} loss {}, batch acc {}".format(i, total,
                                                                         t_loss, t_acc))
                train_acc /= total_length
                train_loss /= total_length
                train_loss_history.append(train_loss)
                train_acc_history.append(train_acc)
                print("EPOCH: ", ep + 1)
                print("Train avg loss {}, train avg acc {}".format(train_loss, train_acc))
                #VALIDATION
                valid_loss, valid_acc = batch_run(valid_data, valid_labels)
                valid_loss_history.append(valid_loss)
                valid_acc_history.append(valid_acc)
                if valid_acc < last_valid_acc and BREAK_EARLY:
                    print("Early break, valid accuracy dropped")
                    break
                last_valid_acc = valid_acc
                print("Valid loss {}, valid acc {}".format(valid_loss, valid_acc))
            test_loss, test_acc, test_pred = batch_run(test_data, test_labels, True)
            print("Test loss {}, test acc {}".format(test_loss, test_acc))
            plot_history(eps, [train_loss_history, valid_loss_history], ["r--", "b--"], "LOSS")
            plot_history(eps, [train_acc_history, valid_acc_history], ["r:", "b:"], "ACCURACY")
            if USE_SMALL_DATASET:
                cm = sklearn.metrics.confusion_matrix(np.argmax(test_labels, axis=1), test_pred)
                plot_confusion_matrix(cm, np.unique(train_labels))


def mlp(num_classes):
    """Tworzy wielowarstwowy perceptron dla danej liczby klas"""
    print("Building MLP")
    graph = tf.Graph()
    with graph.as_default():
        x = tf.placeholder(tf.float32, shape=(None, IMG_SIZE, IMG_SIZE, 3))
        y = tf.placeholder(tf.float32, shape=(None, num_classes))

        def dense(prev, prev_size, out_size):
            w = tf.Variable(
                tf.truncated_normal(shape=[prev_size, out_size])
            )
            b = tf.Variable(
                tf.constant(0.0, shape=[out_size])
            )
            return tf.matmul(prev, w) + b

        net = tf.reshape(x, [-1, IMG_SIZE*IMG_SIZE*3])
        net = (dense(net, prev_size=IMG_SIZE*IMG_SIZE*3, out_size=SIZE_1))
        net = (dense(net, prev_size=SIZE_1, out_size=SIZE_1))
        net = (dense(net, prev_size=SIZE_1, out_size=SIZE_2))
        net = dense(net, prev_size=SIZE_2, out_size=num_classes)
        pred = tf.argmax(net, axis=1)
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(logits=net, labels=y)
        )
        accuracy = tf.reduce_mean(tf.cast(tf.equal(
            tf.argmax(tf.nn.softmax(net), axis=1),
            tf.argmax(y, axis=1)
        ), dtype=np.float32))
        global_step = tf.Variable(0)
        optimizer = tf.train.AdamOptimizer(LEARNING_RATE)\
            .minimize(loss, global_step=global_step)

        def batch_run(data, labels, get_pred=False):
            valid_loss = 0
            valid_acc = 0
            i = 0
            predictions = []
            for batch_data, batch_labels in batch(data, labels, not get_pred):
                feed_dict = {x: batch_data, y: batch_labels}
                length = len(batch_labels)
                v_pred, v_loss, v_acc = session.run([pred, loss, accuracy], feed_dict=feed_dict)
                valid_loss += v_loss * length
                valid_acc += v_acc * length
                i += length
                predictions.append(v_pred)
            valid_acc /= i
            valid_loss /= i
            if get_pred:
                return valid_loss, valid_acc, np.concatenate(predictions)
            return valid_loss, valid_acc


        session = tf.Session()
        with session.as_default():
            session.run(tf.global_variables_initializer())
            train_loss_history = []
            train_acc_history = []
            valid_loss_history = []
            valid_acc_history = []
            eps = 0
            last_valid_acc = -1
            total = int(np.ceil(len(train_labels) / BATCH_SIZE))
            for ep in range(EPOCHS):
                eps += 1
                print("EPOCH: ", ep + 1)
                i = 0
                train_loss = 0
                train_acc = 0

                for batch_data, batch_labels in batch(train_data, train_labels):
                    feed_dict = {x: batch_data, y: batch_labels}

                    _, t_loss, t_acc = session.run([optimizer, loss, accuracy], feed_dict=feed_dict)
                    train_loss += t_loss
                    train_acc += t_acc
                    if (i + 1) % 10 == 0 or i + 1 == total:
                        print("Batch {}/{} loss {}, batch acc {}".format(i + 1, total, t_loss, t_acc))
                    i += 1
                train_acc /= i
                train_loss /= i
                train_loss_history.append(train_loss)
                train_acc_history.append(train_acc)
                print("EPOCH: ", ep + 1)
                print("Train loss {}, train acc {}".format(train_loss, train_acc))
                # VALIDATION
                valid_loss, valid_acc = batch_run(valid_data, valid_labels)
                valid_loss_history.append(valid_loss)
                valid_acc_history.append(valid_acc)
                if valid_acc < last_valid_acc and BREAK_EARLY:
                    print("Early break, valid accuracy dropped")
                    break
                last_valid_acc = valid_acc
                print("Valid loss {}, valid acc {}".format(valid_loss, valid_acc))
            test_loss, test_acc, test_pred = batch_run(test_data, test_labels, True)
            print("Test loss {}, test acc {}".format(test_loss, test_acc))
            plot_history(eps, [train_loss_history, valid_loss_history], ["r--", "b--"], "LOSS")
            plot_history(eps, [train_acc_history, valid_acc_history], ["r:", "b:"], "ACCURACY")
            if USE_SMALL_DATASET:
                cm = sklearn.metrics.confusion_matrix(np.argmax(test_labels, axis=1), test_pred)
                plot_confusion_matrix(cm, np.unique(train_labels))


if CLASSIFIER == "CNN":
    #grid_search()
    cnn(num_classes)
else:
    mlp(num_classes)

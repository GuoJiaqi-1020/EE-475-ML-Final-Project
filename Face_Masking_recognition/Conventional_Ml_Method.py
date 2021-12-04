import sys
from lib.math_optimization_library import static_plotter
import autograd.numpy as np
from autograd.misc.flatten import flatten_func
from autograd import grad as gradient
from lib import edge_extract
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

sys.path.append('..')

plotter = static_plotter.Visualizer()


def linear_model(x, w):
    a = w[0] + np.dot(x.T, w[1:])
    return a.T


def multiclass_softmax(w, x, y, iter):
    x_p = x[:, iter]
    y_p = y[:, iter]
    all_evals = linear_model(x_p, w)
    a = np.log(np.sum(np.exp(all_evals), axis=0))
    b = all_evals[y_p.astype(int).flatten(), np.arange(np.size(y_p))]
    cost = np.sum(a - b)
    return cost / float(np.size(y_p))


class MNIST_Classification(object):
    def __init__(self, data_name, img_size):
        self.gray = []
        self.red = []
        self.blue = []
        self.green = []
        self.mismatching_his = []
        file_path = "../Data/Pixel" + str(img_size[0]) + "/"
        x, y = self.fetchData(file_path,data_name, img_size)
        self.feature_extraction(x, img_size)
        self.y = y
        self.x = np.array(self.gray).T
        self.shuffle_data(n_sample=4500, x=self.x, y=self.y)
        self.standard_normalizer(self.x_rand.T)
        self.x_rand = self.normalizer(self.x_rand.T).T
        self.x_edge = self.hog_extractor(self.x_rand)

        self.cost_function = multiclass_softmax

    @staticmethod
    def fetchData(file_path, data_name, img_size):
        y = []
        x = np.empty(shape=(0, img_size[0], img_size[1], 3))
        tag = 0
        number = 0
        for name in data_name:
            data_subset = np.load(file_path+name)
            x = np.append(x, data_subset, axis=0)
            y.extend(np.shape(data_subset)[0] * [tag])
            number += np.shape(data_subset)[0]
            tag += 1
        return x, np.reshape(np.array(y), (1, number))

    def feature_extraction(self, data, img_size):
        for i in range(np.shape(data)[0]):
            self.gray_image(data[i, :], img_size)

    def gray_image(self, img, img_size):
        blue = []
        green = []
        red = []
        for i in range(img_size[0]):
            for j in range(img_size[1]):
                red.append(img[i][j][0])
                green.append(img[i][j][1])
                blue.append(img[i][j][2])
        gray = list(0.07 * np.array(blue) + 0.72 * np.array(green) + 0.21 * np.array(red))
        self.gray.append(gray)
        self.red.append(red)
        self.green.append(green)
        self.blue.append(blue)

    def shuffle_data(self, n_sample, x, y):
        inds = np.random.permutation(y.shape[1])[:n_sample]
        self.x_rand = np.array(x)[:, inds]
        self.y_rand = y[:, inds]

    def standard_normalizer(self, x):
        x_ave = np.nanmean(x, axis=1)[:, np.newaxis]
        x_std = np.nanstd(x, axis=1)[:, np.newaxis]
        self.normalizer = lambda data: (data - x_ave) / x_std

    @staticmethod
    def hog_extractor(x):
        kernels = np.array([
            [[-1, -1, -1],
             [0, 0, 0],
             [1, 1, 1]],
            [[-1, -1, 0],
             [-1, 0, 1],
             [0, 1, 1]],
            [[-1, 0, 1],
             [-1, 0, 1],
             [-1, 0, 1]],
            [[0, 1, 1],
             [-1, 0, 1],
             [-1, -1, 0]],
            [[1, 0, -1],
             [1, 0, -1],
             [1, 0, -1]],
            [[0, -1, -1],
             [1, 0, -1],
             [1, 1, 0]],
            [[1, 1, 1],
             [0, 0, 0],
             [-1, -1, -1]],
            [[1, 1, 0],
             [1, 0, -1],
             [0, -1, -1]]])
        extractor = edge_extract.tensor_conv_layer()
        x_transformed = extractor.conv_layer(x.T, kernels).T
        return x_transformed

    def gradient_descent(self, loss_fun, w, x_train, y_train, alpha, max_its, batch_size):
        g_flat, unflatten, w = flatten_func(loss_fun, w)
        grad = gradient(g_flat)
        num_train = y_train.size
        w_hist = [unflatten(w)]
        train_hist = [g_flat(w, x_train, y_train, np.arange(num_train))]
        num_batches = int(np.ceil(np.divide(num_train, batch_size)))
        for k in range(max_its):
            for b in range(num_batches):
                batch_inds = np.arange(b * batch_size, min((b + 1) * batch_size, num_train))
                grad_eval = grad(w, x_train, y_train, batch_inds)
                grad_eval.shape = np.shape(w)
                w = w - (alpha / (k + 1)) * grad_eval
            train_cost = g_flat(w, x_train, y_train, np.arange(num_train))
            w_hist.append(unflatten(w))
            train_hist.append(train_cost)
        return w_hist, train_hist

    def misclass_counting(self, x, y, weight_his):
        mis_his = []
        for w in weight_his:
            all_evals = linear_model(x, w)
            y_predict = (np.argmax(all_evals, axis=0))[np.newaxis, :]
            count = np.shape(np.argwhere(y != y_predict))[0]
            mis_his.append(count)
        return mis_his

    # Plotting
    def confusion_matrix(self, mis_history, x, y, weight_his, labels, normalize=False, title='Confusion Matrix',
                         precision="%0.f"):
        ind = np.argmin(mis_history)
        w_p = weight_his[ind]
        tick_marks = np.array(range(len(labels))) + 0.5
        all_evals = linear_model(x, w_p)
        y_predict = np.argmax(all_evals, axis=0)
        count = np.shape(np.argwhere(y != y_predict))[0]
        acc = 1 - (count / np.shape(all_evals)[1])
        print("the prediction accuracy is:" + str(acc))
        cm = confusion_matrix(y[0], y_predict)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            title = "Normalized " + title
            precision = "%0.2f"
        plt.figure(figsize=(12, 8), dpi=120)
        ind_array = np.arange(len(labels))
        x, y = np.meshgrid(ind_array, ind_array)
        for x_val, y_val in zip(x.flatten(), y.flatten()):
            c = cm[y_val][x_val]
            if c > 0.0:
                plt.text(x_val, y_val, precision % (c,), color='k', fontsize=17, va='center', ha='center')
        plt.gca().set_xticks(tick_marks, minor=True)
        plt.gca().set_yticks(tick_marks, minor=True)
        plt.gca().xaxis.set_ticks_position('none')
        plt.gca().yaxis.set_ticks_position('none')
        plt.grid(True, which='minor', linestyle='-')
        plt.gcf().subplots_adjust(bottom=0.15)
        plt.imshow(cm, interpolation='nearest', cmap='Blues')
        font = {'size': 13}
        plt.title(title, font)
        plt.colorbar()
        xlocations = np.array(range(len(labels)))
        plt.xticks(xlocations, labels, rotation=0)
        plt.yticks(xlocations, labels)
        plt.ylabel('True label', font)
        plt.xlabel('Predicted label', font)
        plt.show()

    @staticmethod
    def weight_normalizer(w):
        w_norm = sum([v ** 2 for v in w[1:]]) ** 0.5
        return [v / w_norm for v in w]


if __name__ == "__main__":
    data_name = ['Correct.npy', 'Incorrect.npy', 'NoMask.npy', ]
    mnist = MNIST_Classification(data_name, img_size=[100, 100])
    N = mnist.x_rand.shape[0]
    C = len(np.unique(mnist.y_rand))
    w = 0.1 * np.random.randn(N + 1, C)
    weight_his, cost_his = mnist.gradient_descent(mnist.cost_function, w, mnist.x_rand, mnist.y_rand, alpha=0.02,
                                                  max_its=100, batch_size=300)
    N = mnist.x_edge.shape[0]
    w = 0.1 * np.random.randn(N + 1, C)
    weight_edge_his, cost_edge_his = mnist.gradient_descent(mnist.cost_function, w, mnist.x_edge, mnist.y_rand,
                                                            alpha=0.02,
                                                            max_its=100, batch_size=300)
    mis1 = mnist.misclass_counting(mnist.x_rand, mnist.y_rand, weight_his)
    mnist.confusion_matrix(mis1, mnist.x_rand, mnist.y_rand, weight_his,
                           labels=["Correct Masking", "Incorrect Masking", "No Masking"],
                           normalize=True,
                           title="Confusion matrix: Raw Dataset")

    mis2 = mnist.misclass_counting(mnist.x_edge, mnist.y_rand, weight_edge_his)
    mnist.confusion_matrix(mis2, mnist.x_edge, mnist.y_rand, weight_edge_his,
                           labels=["Correct Masking", "Incorrect Masking", "No Masking"],
                           normalize=True,
                           title="Confusion matrix: HoG Dataset")

    plotter.plot_mismatching_histories(histories=[mis1, mis2], start=1,
                                       labels=['Raw', 'Hog'],
                                       title="Training Mis-classification History")
    plotter.plot_cost_histories(histories=[cost_his, cost_edge_his], start=0,
                                labels=['Raw', 'Hog'],
                                title="Training Cost History")

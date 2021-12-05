import autograd.numpy as np
import matplotlib.pyplot as plt


def confusion_matrix(cm_collection, labels, precision="%0.f", normalize=True):
    tick_marks = np.array(range(len(labels))) + 0.5
    title = ["Image Size = 20", "Image Size = 50", "Image Size = 100"]
    plt.figure(figsize=(16, 8), dpi=120)
    for i in range(len(cm_collection)):
        cm = cm_collection[i]
        plt.subplot(1, len(cm_collection), i + 1)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            title[i] = "Normalized " + title[i]
            precision = "%0.2f"
        ind_array = np.arange(len(labels))
        x, y = np.meshgrid(ind_array, ind_array)
        for x_val, y_val in zip(x.flatten(), y.flatten()):
            c = cm[y_val][x_val]
            if c > 0.0:
                plt.text(x_val, y_val, precision % (c,), color='k', fontsize=11, va='center', ha='center')
        plt.gca().set_xticks(tick_marks, minor=True)
        plt.gca().set_yticks(tick_marks, minor=True)
        plt.gca().xaxis.set_ticks_position('none')
        plt.gca().yaxis.set_ticks_position('none')
        plt.grid(True, which='minor', linestyle='-')
        plt.gcf().subplots_adjust(bottom=0.15)
        plt.imshow(cm, interpolation='nearest', cmap='Blues')
        font = {'size': 9}
        plt.title(title[i], font)
        xlocations = np.array(range(len(labels)))
        plt.xticks(xlocations, labels, rotation=45)
        if i == 0:
            plt.yticks(xlocations, labels)
        else:
            plt.yticks([])

    plt.show()


if __name__ == "__main__":
    cm1 = np.array([[1240, 130, 163], [252, 1035, 196], [274, 212, 998]])
    cm_collection = [cm1, cm1, cm1]
    confusion_matrix(cm_collection, labels=["Correct Masking", "Incorrect Masking", "No Masking"],
                     precision="%0.f", normalize=True)

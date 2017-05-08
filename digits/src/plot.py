import matplotlib.pyplot as plt


def plot_loss(history):
    loss = history["loss"]
    val_loss = history["val_loss"]
    epochs = range(1, len(loss) + 1)

    try:
        fig = plt.figure()
        plt.plot(epochs, loss, 'r+')
        plt.plot(epochs, val_loss, 'b+')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.show()
        plt.savefig("../plots/loss.png", dpi=fig.dpi)
    except Exception as e:
        print(e)


def plot_acc(history):
    loss = history["acc"]
    val_loss = history["val_acc"]
    epochs = range(1, len(loss) + 1)

    try:
        fig = plt.figure()
        plt.plot(epochs, loss, 'r+')
        plt.plot(epochs, val_loss, 'b+')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.show()
        plt.savefig("../plots/loss.png", dpi=fig.dpi)
    except Exception as e:
        print(e)


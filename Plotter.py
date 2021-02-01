import matplotlib.pyplot as plt


class Plotter:
    @staticmethod
    def plot_data(image_datasets, x_train, y_train, kind='train'):
        # plot train data with labels
        R, C = 1, x_train.size(0)
        fig, ax = plt.subplots(R, C)
        fig.suptitle('Training Data with corresponding labels')
        for i, plot_cell in enumerate(ax):
            plot_cell.grid(False)
            plot_cell.axis('off')
            plot_cell.set_title(image_datasets[kind].classes[y_train[i].item()])
            plot_cell.imshow(x_train[i][0], cmap='gray')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_losses(loss_history):
        plt.plot(loss_history)
        plt.title('Loss variation over increasing epochs')
        plt.xlabel('epochs')
        plt.ylabel('loss value')
        plt.show()

    @staticmethod
    def plot_stats(fh, ffa):
        plt.plot(ffa, fh, 'o')
        plt.title('Fh vs Ffa')
        plt.xlabel('Ffa')
        plt.ylabel('Fh')
        plt.show()

    @staticmethod
    def plot_sample(x, y):
        # # plot predicted data with
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle('Predicted Data')
        ax1.grid(False)
        ax1.axis('off')
        ax1.set_title('Actual')
        ax1.imshow(x, cmap='gray')

        ax2.grid(False)
        ax2.axis('off')
        ax2.set_title('Predicted')
        ax2.imshow(y.detach().numpy(), cmap='gray')

        plt.tight_layout()
        plt.show()

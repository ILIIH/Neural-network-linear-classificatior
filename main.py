from matplotlib import pyplot as plt
import numpy as np
from dataset import load_svhn

def main():
    train_X, train_y, test_X, test_y = load_svhn("data", max_train=1000, max_test=100)
    print(train_X)
    visualise_data(train_X,train_y)

def visualise_data(train_X, train_Y):
    samples_per_class = 5  # Number of samples per class to visualize
    plot_index = 1
    for example_index in range(samples_per_class):
        for class_index in range(10):
            plt.subplot(5, 10, plot_index)
            image = train_X[train_Y == class_index][example_index]
            plt.imshow(image.astype(np.uint8))
            plt.axis('off')
            plot_index += 1

if __name__ == "__main__":
    main()

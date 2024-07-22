from dataset import load_svhn

def main():
    train_X, train_y, test_X, test_y = load_svhn("data", max_train=1000, max_test=100)
    print(train_X)

if __name__ == "__main__":
    main()

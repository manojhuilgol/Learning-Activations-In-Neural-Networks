

class LoadDataset():
    def __init__(self, type="MNIST"):
        self.type = type
        self.x = None
        self.y = None

    def _load_train_and_test(self):
        if self.type == "MNIST":
            self.x, self.y = datasets.fetch_openml('mnist_784', version=1, return_X_y=True)
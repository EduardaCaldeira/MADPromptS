from .loaders import FaceDataset, TestFaceDataset

def get_dataset(rank, **kwargs):
    # create a list of test datasets
    testset = [TestFaceDataset(path, is_train=False) for path in kwargs["test_dataset_path"]]
    return testset
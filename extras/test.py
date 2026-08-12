import numpy as np

y_train = np.load(
    "/home/qortjsdn/projects/ecg_project/dataset/y_train.npy"
)

y_test = np.load(
    "/home/qortjsdn/projects/ecg_project/dataset/y_test.npy"
)

print(np.unique(y_train, return_counts=True))
print(np.unique(y_test, return_counts=True))
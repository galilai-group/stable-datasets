import scipy.io as sio


data_path = "/Users/zhanghaodong/.stable_datasets/downloads/test_32x32.mat"
mat = sio.loadmat(str(data_path), squeeze_me=True)

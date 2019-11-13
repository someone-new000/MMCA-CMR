from torch.utils.data.dataset import Dataset
from scipy.io import loadmat
from torch.utils.data import DataLoader
import numpy as np

class CustomDataSet(Dataset):
    def __init__(
            self,
            images,
            texts,
            labels):
        self.images = images
        self.texts = texts
        self.labels = labels

    def __getitem__(self, index):
        img = self.images[index]
        text = self.texts[index]
        label = self.labels[index]
        return img, text, label

    def __len__(self):
        count = len(self.images)
        assert len(
            self.images) == len(self.labels)
        return count

def scale2vec(ind):
    ind = np.asarray(ind)
    N = ind.shape[0]
    Ma = ind.max()
    Mi = ind.min()
    if Mi == 0:
        M = Ma + 1
    else:
        M = Ma
    label = np.zeros((N,M))
    for i in range(N):
        num = ind[i,0]
        label[i, num-1] = 1
    return label

def get_loader(path, batch_size):
    img_train = loadmat(path+"icptv4_log_norzm.mat")['img_tr']
    img_test = loadmat(path + "icptv4_log_norzm.mat")['img_te']
    text_train = loadmat(path+"data_norzm.mat")['txt_tr']
    text_test = loadmat(path + "data_norzm.mat")['txt_te']
    label_train = loadmat(path+"data_norzm.mat")['label_tr']
    label_test = loadmat(path + "data_norzm.mat")['label_te']

    text_train = text_train.astype(np.single)
    text_test = text_test.astype(np.single)

    label_train = scale2vec(label_train).astype(np.int)
    label_test = scale2vec(label_test).astype(np.int)

    imgs = {'train': img_train, 'test': img_test}
    texts = {'train': text_train, 'test': text_test}
    labels = {'train': label_train, 'test': label_test}
    dataset = {x: CustomDataSet(images=imgs[x], texts=texts[x], labels=labels[x])
               for x in ['train', 'test']}

    shuffle = {'train': False, 'test': False}

    dataloader = {x: DataLoader(dataset[x], batch_size=batch_size,
                                shuffle=shuffle[x], num_workers=0) for x in ['train', 'test']}

    img_dim = img_train.shape[1]
    text_dim = text_train.shape[1]
    num_class = label_train.shape[1]

    input_data_par = {}
    input_data_par['img_test'] = img_test
    input_data_par['text_test'] = text_test
    input_data_par['label_test'] = label_test
    input_data_par['img_train'] = img_train
    input_data_par['text_train'] = text_train
    input_data_par['label_train'] = label_train
    input_data_par['img_dim'] = img_dim
    input_data_par['text_dim'] = text_dim
    input_data_par['num_class'] = num_class
    return dataloader, input_data_par
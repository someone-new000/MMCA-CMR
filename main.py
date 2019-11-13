import torch
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
import time
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from datetime import datetime
from model import IDCM_NN
from train_model import train_model
from load_data import get_loader
#from load_wiki import get_loader
#from load_mir import get_loader
#from load_nus import get_loader
from evaluate import fx_calc_map_label
from util.myPCA import myPCA
from lsdr_mcplst import lsdr_mcplst
#from cplst import lsdr_cplst
from SAE import SAE
from util.test_s_map import test_s_map

######################################################################
# Start running

if __name__ == '__main__':

    # environmental setting: setting the following parameters based on your experimental environment.
    dataset = 'wiki'   #  using dataset: pascal sentence datasets  ;;; pascal ;; wiki ;; mirflickr ;; nuswide
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # data parameters
    DATA_DIR = 'data/' + dataset + '/'
    alpha = 1e-3
    beta = 1e-1
    MAX_EPOCH = 500   #500
    batch_size = 100
    # batch_size = 512
    lr = 1e-4
    betas = (0.5, 0.999)
    weight_decay = 0

    #load data
    print('...Data loading is beginning...')

    data_loader, input_data_par = get_loader(DATA_DIR, batch_size)   #  load the datasets .mat files pretrained by the CNN networks which contain the image, text, label information

    print('...Data loading is completed...')

    # set train model
    model_ft = IDCM_NN(img_input_dim=input_data_par['img_dim'], text_input_dim=input_data_par['text_dim'], output_dim=input_data_par['num_class']).to(device)
    params_to_update = list(model_ft.parameters())

    # Observe that all parameters are being optimized
    optimizer = optim.Adam(params_to_update, lr=lr, betas=betas)

    print('...Training is beginning...')
    # Train and evaluate
    model_ft, img_acc_hist, txt_acc_hist, loss_hist = train_model(model_ft, data_loader, optimizer, alpha, beta, MAX_EPOCH)
    print('...Training is completed...')

    img_feature_tr, txt_feature_tr, img_predict_tr, txt_predict_tr = model_ft(torch.tensor(input_data_par['img_train']).to(device), torch.tensor(input_data_par['text_train']).to(device))
    img_feature_te, txt_feature_te, img_predict_te, txt_predict_te = model_ft(torch.tensor(input_data_par['img_test']).to(device), torch.tensor(input_data_par['text_test']).to(device))

    img_feature_tr = img_feature_tr.data.cpu().numpy()
    img_feature_te = img_feature_te.data.cpu().numpy()
#    img_predict_tr = img_predict_tr.data.cpu().numpy()
#    img_predict_te = img_predict_te.data.cpu().numpy()

    txt_feature_tr = txt_feature_tr.data.cpu().numpy()
    txt_feature_te = txt_feature_te.data.cpu().numpy()
#    txt_predict_tr = txt_predict_tr.data.cpu().numpy()
#    txt_predict_te = txt_predict_te.data.cpu().numpy()

    img_original_tr = input_data_par['img_train']
    img_original_te = input_data_par['img_test']
    txt_original_tr = input_data_par['text_train']
    txt_original_te = input_data_par['text_test']


    # perform PCA
    pca = 1
    if pca == 1:
        options= {}
        options['PCARatio'] = 0.99
        [eigvector,eigvalue] = myPCA(img_feature_tr, options)
        img_tr = np.dot(img_feature_tr, eigvector)
        img_te = np.dot(img_feature_te, eigvector)
        [eigvector,eigvalue] = myPCA(txt_feature_tr, options)
        txt_tr = np.dot(txt_feature_tr, eigvector)
        txt_te = np.dot(txt_feature_te,eigvector)


    # set parameters
    h = input_data_par['num_class'] # hidden dimension
    max_iter = 20 # max number of iterations  5
    alph = 0.01
    beta1 = 1

    label_tr = input_data_par['label_train']
    label_te = input_data_par['label_test']

    # training
    # in here, we use data matrix where each colume is a sample.
    V = img_tr.T
    T = txt_tr.T
    L = label_tr.T

    # stage 1
    C = lsdr_mcplst(V.T, T.T, L.T, h, max_iter, alph, beta1)


    # stage 2
    S = C.T
    U = S

    for i in range(max_iter):
        Pi = SAE(V, U, alph)
        Pt = SAE(T, U, alph)


        U = np.linalg.solve((alph * np.dot(Pi, Pi.T) + alph * np.dot(Pt, Pt.T) + (2 + beta1) * np.identity(h)), ((1 + alph) * np.dot(Pi, V) + (1 + alph) * np.dot(Pt, T) + beta1 * S))
        print(i + 1)

    # testing
    img_te_proj = np.dot(img_te, Pi.T)
    txt_te_proj = np.dot(txt_te, Pt.T)

    start = time.time()
    fout = open('record_OURS_pascal.txt', 'a')
    fout.write('Start Time: {}\n'.format(start))
    fout.write('-' * 20)
    fout.write('\n')

    # test img2txt
    print('img search txt:\n')
    fout.write('img search txt:\n')
    smatrix = np.dot(img_te_proj, txt_te_proj.T)
    test_s_map(smatrix, label_te, label_te, fout)
    # test txt2img
    print('txt search img:\n')
    fout.write('txt search img:\n')
    test_s_map(smatrix.T, label_te, label_te, fout)

    end = time.time()
    time_spend = end - start
    fout.write('End Time: {}\n'.format(end))
    fout.write('Testing complete in {:.0f}m {:.0f}s\n'.format(time_spend // 60, time_spend % 60))
    fout.close()

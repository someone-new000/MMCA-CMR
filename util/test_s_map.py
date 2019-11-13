# test retrieval using score matrix
# smatrix: score matrix, N_que x N_doc
# calculate agree matrix first

import numpy as np

def test_s_map(smatrix, label_que, label_doc, fd):
    N_que, N_doc = np.shape(smatrix)
    c = label_que.shape[1]

    top_n = [50-1, N_doc-1]   #  50, N_doc

    dist = np.sort(smatrix)
    idx = []
    for i in range(N_que):
        idex = np.lexsort([-1*smatrix[i,:]])
        if i == 0:
            idx = idex
        else:
            idx = np.row_stack((idx, idex))

    if c == 1:
        agree = ((label_que == label_doc[idx[:,:]]).all())
    else:
        agree = np.zeros((N_que, N_doc))
        for j in range(N_que):
            q_label = label_que[j, :]
            dist_label = label_doc[idx[j, :], :]
            A = dist_label[:,q_label > 0]
            B = np.zeros((A.shape[0],1))
            for num in range(A.shape[0]):
                B[num,0] = np.sum(A[num,:]) / A.shape[1]
            agree[j, :] = (B > 0).T

    #map
    rele = np.cumsum(agree, axis = 1)

#   rele=
    col, row = np.shape(rele)

    prec = np.empty((col,row))
    for a in range(row):
        prec[:,a] = rele[:,a] / (a+1)

#   map =
    One = np.cumsum(prec*agree, axis=1)
    Ones = np.ones((rele.shape[0], rele.shape[1]))
    Two = np.maximum(rele, Ones)

    map = np.empty((One.shape[0], One.shape[1]))
    for e in range(One.shape[0]):
        for f in range(One.shape[1]):
            map[e,f] = One[e,f] / Two[e,f]
#   mapk =
    mapk = np.mean(map[:, top_n], axis = 0)

#    max = 0
    for m in range(len(top_n)):
        fd.write('{}\t'.format(mapk[m]))

    print(mapk)
    #fd.write('{}\t'.format(mapk))
    fd.write('\n')
    return mapk



# map
#rele = cumsum(agree, 2);
#prec = bsxfun(@ldivide, (1:N_doc), rele);
#map = bsxfun(@rdivide, cumsum(prec .* agree, 2), max(rele, 1));
#mapk = mean(map(:, top_n), 1);

#for i = 1 : size(top_n, 2)
#    fprintf('%f\t', mapk(i));
#end
#fprintf('\n');

#if nargin > 3 % write to file
#    for i = 1 : size(top_n, 2)
#        fprintf(fd, '%f\t', mapk(i));
#    end
#    fprintf(fd, '\n');
#end
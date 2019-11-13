import numpy as np
import scipy.sparse as sp

if __name__ == '__main__':
#    list1 = np.array([[ 2, 7, 4, 2],
#                      [35, 9, 1, 5],
#                      [22, 12, 3, 2]])

#    print(list1)

#    list2 = np.sort(list1)

#    print(list2)

#    list3 = sorted(list1, key = lambda x: x[1], reverse = True)
#    list3 = np.array(list3)

#    print(list3)

#    idex = np.lexsort([-1 * list1[0,:]])

#    print(idex)

#    list4 = list1[:, idex]

#    print(list4)

#    idex2 = np.lexsort([-1 * list4[1, :]])

#    print(idex2)

#    list6 = list4[:, idex2]

#    print(list6)

#    idexa = np.row_stack((idex, idex2))

#    print(idexa)

#    line = list1.shape[0]
#    print(line)
#    print(list1)

#    idx = []
#    for i in range(line):
#        print(i)
#        idex = np.lexsort([-1 * list1[i, :]])
#        if i == 0:
#            idx = idex
#        else:
#            idx = np.row_stack((idx, idex))
#    print(idx)

#    A = np.matrix([[1,	1,	0,	3,	4,	5,	6,	6,	7],
#                   [1,	2,	3,	4,	5,	6,	7,	8,	9],
#                   [1,	2,	3,	4,	5,	6,	7,	8,	9],
#                   [0,	0,	0,	0,	1,	1,	1,	1,	1],
#                   [0,	0,	0,	0,	0,	0,	0,	1,	1],
#                   [1,	2,	3,	4,	5,	6,	7,	8,	9],
#                   [0,	0,	0,	0,	0,	0,	0,	0,	0],
#                   [0,	0,	0,	0,	0,	0,	0,	0,	0],
#                   [1,	2,	3,	4,	5,	6,	7,	8,	9]])

#    print(A)

#    col, row = np.shape(A)

#    B = np.array(A)
#    line = 0
#    for i in range(col):
#        for j in range(row):
#            print(A[i,j])
#            if A[i,j] == 0:
#                B = np.delete(B, line, axis=0)
#                line = line - 1
#                break

#        line = line + 1
#        print(line)
#        print(B)

#    print(B)

    A = np.matrix([[1],
                   [2],
                   [3],
                   [4],
                   [5],
                   [6]])

    length = A.shape[0]
    print(length)

    B = np.zeros((length,length))
    for i in range(length):
        B[i,i] = A[i,0]

    print(A)
    print(B)






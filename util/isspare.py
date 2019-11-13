#coding=utf-8

# 声明稀疏元素和稀疏系数。在本程序中，稀疏元素是0，稀疏系数是0.5（也就是说，当稀疏元素占总元素的比重小于稀疏系数时，代码判定该矩阵不为 # 稀疏矩阵）
SPARE_ELEMENT = 0
SPARE_RATE = 0.5

# 判断输入的矩阵是否为稀疏矩阵
def isSparse(matrix):
    """
    Judge spare matrix.
    :param matrix: matrix
    :return: boolean
    """
    sum = len(matrix) * len(matrix[0])
    spare = 0

    for row in range(len(matrix)):
        for column in range(len(matrix[row])):
            if matrix[row][column] == SPARE_ELEMENT:
                spare += 1

    if spare / sum >= SPARE_RATE:
        return True
    else:
        return False
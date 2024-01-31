import numpy as np
import matplotlib.pyplot as plt

# ´´½¨Ò»¸ö¾ØÕó
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# ÏÔÊ¾¾ØÕóÍ¼Ïñ
plt.imshow(matrix, cmap='viridis', interpolation='nearest')

# ÔÚÃ¿¸öÎ»ÖÃÏÔÊ¾¾ØÕóµÄ¶ÔÓ¦ÊýÖµ
for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        plt.text(j, i, matrix[i, j], ha='center', va='center', color='w')

# Òþ²Ø×ø±êÖá
plt.xticks([])
plt.yticks([])

# ÏÔÊ¾Í¼Ïñ
plt.show()
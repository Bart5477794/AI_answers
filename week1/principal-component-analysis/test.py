import numpy as np

m = np.array([[2.2, 1.4], [4.3, 7.3]])
p = np.array([[2.2, 1.4], [4.3, 7.3], [2.3, 3.4]])
n = 3 * m
a = np.array([1, 2, 4])
b = np.array([4, 7, 8])
q = np.array([])

mat = np.outer(a, b)

mat2 = np.append(q, a, 0)
mat3 = np.append([mat2], [b], 0)
mat4 = np.append(mat3, [a], 0)

# for i in range(len(m)):
#     m[i] = m[i] / np.dot(m[i], m[i])**0.5

# mean = np.mean(m, 0)

# print(mean)
# print(mat2)
# print(mat3)
print(mat4)
print(np.sum(a[:2]))
# print(np.matmul(m, p.T))

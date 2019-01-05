import numpy as np
import matplotlib.pyplot as plt
# import seaborn; seaborn.set()


x = np.random.randint(0,10,(10,2))

plt.figure(1)
plt.scatter(x[:,0], x[:,1])
# plt.show()
print(x)

diff = x[:, np.newaxis, :] - x[np.newaxis,:,:]
diff = diff ** 2
dist_sq = diff.sum(2)

# print(dist_sq)
# print(dist_sq.shape)

nearest = np.argsort(dist_sq, 1)
print(nearest)
index = np.arange(10)
index = np.tile(index,(10,1))
index = index.transpose()
# print(dist_sq[index,nearest])

K = 2
nearest_partition = np.argpartition(dist_sq, K+1, axis=1)
print(nearest_partition)

plt.figure(2)
plt.scatter(x[:, 0], x[:,1])
for i in range(x.shape[0]):
    for j in nearest_partition[i, :K+1]:
        plt.plot(*zip(x[j], x[i]), color='black')
plt.show()




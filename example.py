from cu_gaussian_process import GaussianProcess
import numpy as np
import matplotlib.pyplot as plt

sample_X = np.random.uniform(-10, 10, (1000, 1))
sample_y = np.sin(sample_X[:, 0])

gaussian_process = GaussianProcess(1., 1e-6)
gaussian_process.fit(sample_X, sample_y)
estimate_y, m_covariance = gaussian_process.predict(sample_X)

plt.scatter(sample_X, sample_y, c='b')
plt.scatter(sample_X, estimate_y, c='r')
plt.show()


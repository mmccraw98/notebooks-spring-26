import numpy as np
import matplotlib.pyplot as plt

num_segments = 100
num_answer_choices = 5
tokens = 200
slope = 0.357
print(slope * num_segments * num_answer_choices * tokens / 1e3)

cost = 105
size = 12670 * 63
training_slope = cost / size
print(training_slope * 100)
print(training_slope * (1000 * 100))
print(training_slope)

print(training_slope * 100_000 * 100)

print(45 / 1000 * 5 * 100)


# data = np.load('data.npz')
# print(np.sum(data['mass'][..., None] * data['vel'] ** 2, axis=(-1, -2)))
# plt.plot(data['vel'][:, 0])
# plt.savefig('test.png')
from collections import namedtuple
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import numpy as np

bias = -618133.6567896605

a = np.loadtxt("ab_res.txt")
b = np.loadtxt("opt_res.txt")

print(np.abs(np.mean(a)-bias))
print(np.abs(np.mean(b)-bias))


c = []
d = []
for i in range(10):
    c.append(np.std(a[1000*i:1000*(i+1)]))
    d.append(np.std(b[1000*i:1000*(i+1)]))
print(np.mean(c))
print(np.mean(d))
print(np.std(c))
print(np.std(d))


xticks = ["Bernoulli randomization", "optimal"]
#means = [np.abs(np.mean(a)-bias), np.abs(np.mean(b)-bias)]
#stds = [np.std(a)/10000, np.std(b)/10000]
colors = ['#3498db', '#e74c3c']

means = [np.mean(c), np.mean(d)]
stds = [np.std(c), np.std(d)]

fig, ax = plt.subplots()

error_config = {'ecolor': '0.3', 'capsize': 30}

rects1 = ax.bar(xticks, means,
                color=colors, width=0.5,
                yerr=stds, error_kw=error_config)


ax.set_xlabel('Experiment')
ax.set_ylabel('Variance')
ax.set_title('Variance of Bernoulli randomization and optimal experiment')
ax.legend()

fig.tight_layout()
plt.show()

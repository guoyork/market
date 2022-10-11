import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set()


exp1 = np.zeros((10, 5))
var1 = np.zeros((10, 5))
for i in range(10):
    for j in range(5):
        temp1 = np.loadtxt("correlation/ground_truth_"+str(i)+str(j)+".txt")
        temp2 = np.loadtxt("correlation/opt_"+str(i)+str(j)+".txt")
        exp1[i][j] = np.average(np.abs(temp1-temp2[:, 0]))
        var1[i][j] = np.average(temp2[:, 1])


n = len(var1)
x = np.asarray(range(5))/5
y = np.asarray(range(10))/10+1

ax = sns.heatmap(exp1, cmap='OrRd', annot=True, annot_kws={'color': 'black'}, vmin=0, vmax=3)
ax.set_xticklabels(np.asarray(range(5))/5, fontsize=9)
ax.set_yticklabels(np.asarray(range(10))/10+1, fontsize=9)
ax.set_title('bias under different parameters')
plt.xlabel('consistency rate')
plt.ylabel('budget cost rate')

plt.show()

'''
n = len(var1)
x = np.asarray(range(n))+1
plt.xlabel('supply demand rate')
plt.ylabel('variance')


plt.plot(x, var1, marker='.',  color='#3498db', label='abtest')

plt.legend(loc='lower right')

plt.grid()
plt.show()
'''

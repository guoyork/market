import matplotlib.pyplot as plt
import numpy as np

exp1 = []
exp2 = []
error3 = []
error4 = []
var1 = []
var2 = []
error1 = []
error2 = []
for i in range(30):
    temp1 = np.loadtxt("online/ground_truth_"+str(i)+".txt")
    temp2 = np.loadtxt("online/opt_"+str(i)+".txt")
    temp3 = np.loadtxt("online/theory_opt_"+str(i)+".txt")
    exp1.append(np.average(np.abs(temp1-temp2[:, 0])/(i+1)))
    exp2.append(np.average(np.abs(temp1-temp3[:, 0])/(i+1)))
    error3.append(np.std(np.abs(temp1-temp2[:, 0])/(i+1)))
    error4.append(np.std(np.abs(temp1-temp3[:, 0])/(i+1)))
    var1.append(np.average(temp2[:, 1]))
    var2.append(np.average(temp3[:, 1]))
    error1.append(np.std(temp2[:, 1]))
    error2.append(np.std(temp3[:, 1]))

n = len(var1)
x = np.asarray(range(n))+1
plt.xlabel('supply demand rate')
plt.ylabel('average bias')


plt.errorbar(x, exp1, error3, marker='.', capsize=3, color='#3498db', label='offline')
plt.errorbar(x, exp2, error4, marker='.', capsize=3, color='#e74c3c', label='online')

plt.legend(loc='upper right')

plt.grid()
plt.show()

'''
n = len(var1)
x = np.asarray(range(n))+1
plt.xlabel('supply demand rate')
plt.ylabel('variance')


plt.errorbar(x, var1, error1, marker='.', capsize=3, color='#3498db', label='offline')
plt.errorbar(x, var2, error2, marker='.', capsize=3, color='#e74c3c', label='online')

plt.legend(loc='lower right')

plt.grid()
plt.show()
'''

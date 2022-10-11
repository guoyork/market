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
for i in range(10):
    temp1 = np.loadtxt("budget/ground_truth_"+str(i)+".txt")
    temp2 = np.loadtxt("budget/opt_"+str(i)+".txt")
    temp3 = np.loadtxt("budget/theory_opt_"+str(i)+".txt")
    exp1.append(np.average(np.abs(temp1-temp2[:, 0])))
    exp2.append(np.average(np.abs(temp1-temp3[:, 0])))
    var1.append(np.average(temp2[:, 1]))
    var2.append(np.average(temp3[:, 1]))
'''
n = len(var1)
x = np.asarray(range(n))/10+1
plt.xlabel('budget cost rate')
plt.ylabel('bias')


plt.plot(x, exp1, marker='.',  color='#3498db', label='solution with budget')
plt.plot(x, exp2, marker='.',  color='#e74c3c', label='solution without budget')

plt.legend(loc='upper right')

plt.grid()
plt.show()

'''
n = len(var1)
x = np.asarray(range(n))/10+1
plt.xlabel('budget cost rate')
plt.ylabel('variance')


plt.plot(x, var1, marker='.',  color='#3498db', label='solution with budget')
plt.plot(x, var2, marker='.',  color='#e74c3c', label='solution without budget')

plt.legend(loc='lower right')

plt.grid()
plt.show()

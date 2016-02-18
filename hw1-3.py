import numpy as np
import time
import matplotlib.pyplot as plt


def matrix_prod(a,b):
    
    ans = np.zeros((int(a.shape[0]),int(b.shape[1])))

    if not(isinstance(a,np.ndarray) & isinstance(b,np.ndarray)):
        raise TypeError('arguments are not of numpy.ndarray type!')
    if a.shape[1] != b.shape[0]:
        raise ValueError('dimensions of the two matrices do not match!')

    for i,x in enumerate(a):
        for j,y in enumerate(b.T):
            ans[i,j] = np.dot(x,y)
    
    return ans

def mprod(a,b):
    result = np.zeros((a.shape[0],b.shape[1]))
    for i in range(a.shape[0]):
        for j in range(b.shape[1]):
            result[i,j] = np.dot(a[i,:],b[:,j])
    return result

a = np.array([[1.5,2,3],[2.2,3,4]])
b = np.array([[3,2.9],[-1,2],[-3.2,4]])

t0 = time.clock()
ans1 = matrix_prod(a,b)
t1 = time.clock()
ans2 = a.dot(b)
t2 = time.clock()

print 'Answer 1\n', ans1
print '\nAnswer 2\n', ans2
print('\nNew implementation took %f.' % float(t1-t0))
print('Built-in implementation took %f.' % float(t2-t1))

print('Are the answers the same? %s.' % np.allclose(ans1, ans2))

t = np.zeros(100)
for i in range(1,100,1):
    a = np.random.rand(i,i)
    b = np.random.rand(i,i)
    t0 = time.clock()
    ans1 = matrix_prod(a,b)
    t1 = time.clock()
    ans2 = a.dot(b)
    t2 = time.clock()
    
    t[i] = (t1-t0)-(t2-t1) # New minus built-in

plt.figure(figsize=(7,5))
plt.plot(range(100),t)
plt.xlabel('Matrix dimension',fontsize=14)
plt.ylabel('Difference in computational time (sec)',fontsize=14)
plt.show()    
    
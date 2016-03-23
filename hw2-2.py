import numpy as np
import matplotlib.pyplot as plt
import time

def reduce_matrix(A):
    a11 = A[0,0]
    A21 = np.mat(A[1:,0])
    A22 = np.mat(A[1:,1:])
    return A22 - A21*A21.T/a11

def my_cholesky(A):
    
    np.linalg.eigvals(A) # error would be raised if not spd
        
    n = A.shape[0]
    if n == 1:
        return np.sqrt(A)
    l11 = np.sqrt(A[0,0])
    A21 = np.mat(A[1:,0])
    return np.bmat([[np.mat(l11), np.zeros((1,n-1))],
                     [A21/l11, my_cholesky(reduce_matrix(A))]])

# test
A = np.mat(np.random.randn(3,3))
A = A*A.T

# this will not work
# A = np.mat([[3,1,2],[1,3,2],[2,2,3]])

Ans1 = my_cholesky(A)
Ans2 = np.linalg.cholesky(A)

print("Answers agree? %s" % np.allclose(Ans1,Ans2))

t = np.zeros(50)
for i in range(2,52,1):
    A = np.mat(np.random.randn(i,i))
    A = A*A.T
    t0 = time.clock()
    ans1 = my_cholesky(A)
    t1 = time.clock()
    ans2 = np.linalg.cholesky(A)
    t2 = time.clock()
    
    t[i-2] = (t1-t0)-(t2-t1) # New minus built-in

plt.figure(figsize=(7,5))
plt.plot(range(50),t)
plt.xlabel('Matrix dimension',fontsize=14)
plt.ylabel('Difference in computational time (sec)',fontsize=14)
plt.show()
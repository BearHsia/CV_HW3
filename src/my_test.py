import numpy as np
"""
N = 2

u = np.array([[11,12],[14,15]])
#print(u)
u = np.pad(u,((0,0),(0,1)),constant_values=1)
#print(u)
A = np.zeros((2*N,9))

A[0,3:6] = u[0,:]*-1
A[0,6:9] = u[0,:]*10
print(A)
vhl = np.array([1,2,3,4,5,6,7,8,9])
print(vhl.shape)
vhl = np.reshape(vhl, (3,3))
print(vhl)
"""
"""
N = 3
M = 2
na = [10,11,12]
ma = range(M)
xv, yv = np.meshgrid(na, ma)
zv = np.ones((M,N))
#print(zv.shape)
#print(xv)
#print(yv)
c = np.stack((xv,yv,zv))
print(c.shape)
print(c)
d = np.reshape(c,(3,6))
print(d.shape)
print(d)
"""
"""
a = np.array([[1,2,3],[4,5,6]])
b = np.array([[11,12,13],[14,15,16]])
c = np.stack((a,b))
print(c.shape)
print(c)
d = np.reshape(c,(2,6))
print(d.shape)
print(d)

e = d[0,:]>3
print(e)
f = d[1,:]<16
print(f)
g = e & f
print(g)
"""
"""
raw = np.array([[11,12,13,14,15,16],[21,22,23,24,25,26],[1,1,1,1,1,1]])
mas = np.array([True,True,False,True,False,True])
sel = raw[:,mas]
print(sel)
"""
"""
dst = range(2*3*3)
dst = np.reshape(dst,(2,3,3))
src = dst*100
print(dst)
row_idx = [1,0]
col_idx = [2,0]
#print(dst[row_idx,col_idx,:])
dst[row_idx,col_idx,:] = src[row_idx,col_idx,:]
print(dst)
"""
"""
aa = range(0,10)
bb = range(10,20)
print(list(zip(aa,bb)))
"""
"""
a = [[1,2],[3,4],[5,6],[7,8]]
a = np.array(a)
print(a)
b= np.reshape(a,(2,4))
print(b)
c = np.transpose(a)
print(c)
"""
"""
raw = np.array([[11,12,13,14,15,16],[21,22,23,24,25,26],[1,1,1,1,1,1]])
dist = np.linalg.norm(raw[:2,:],axis=0,ord=1)
print(dist)
"""
"""
raw = np.array(range(5))
print(np.sum(raw>2))
"""
"""
NN = np.zeros((3,2,2))
for i in range(3):
    N_tmp = np.ones((2,2))*i
    NN[i,...] = N_tmp
print(NN)
"""
aaa = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(aaa)
print(aaa[[0,2],[0,2]])
aaa[[0,2],[0,2]] = 1

print(aaa)
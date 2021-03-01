title: 非线性维度下降之局部线性插值
date: 2015-08-14 09:18:41
tags: 机器学习
---
非线性维度下降的目的是发现隐藏在高维数据中的低维结构，例如手写字体的方向，弯曲程度，写字风格（比如2右下角带不带环）。维度下降的基本假设是数据点位于高维空间中的一个很薄的流体上或者其附近，这个流体的维度就是数据的内在维度，远低于空间的维度，维度下降算法就是用来重构这个内在维度。<!-- more -->线性维度下降，例如PCA, ICA, 是假设这个流体是一个低维平面，即一组正交的基向量生成的一个空间，数据点可以由这些基向量的不同线性组合来表示，非线性维度下降则是应对非这种平面的流体的一种降维方法，一般称为流体学习（mandifold learning）。本文介绍一种局部线性插值（locally linear embedding）的流体学习方法及它的一个改进。这个方法有很好的几何直观性，数据点在高维空间如果能由它的邻近数据点的加权得到，那么在低维空间同样可以用相同的权重由它的邻近数据点加权得到。算法的推导和实现如下：  
1. 计算k-近邻  
 找到每个数据点K-个最近的邻近数据点，K的选择影响算法最后得到的低维结构。python code:  
 ```Python
 import numpy as np
 from numpy.matlib import repmat
 #X = data as D x N matrix (D = dimensionality, N = #points)
 D,N=X.shape
 X2=np.sum(X**2,0).reshape(1,N)
 distance=repmat(X2,N,1)+repmat(X2.T,1,N)-2*np.dot(X.T,X)
 index=np.argsort(distance,0)
 neighborhood=index[1:1+K,:]
 ```
2. 计算权重矩阵  
 对每个数据点$\mathbf{x}$，K个最近邻用$\mathbf{x}\_\boldsymbol{j}$表示，最小化它的重构误差，  
 $$\varepsilon(\mathbf{w})=||\mathbf{x}-\sum\_j w \_j \mathbf{x}\_\boldsymbol{j} ||^2     \ \ \ \      s.t.   \ \   \sum \_j w\_j = 1$$  
 $$\varepsilon(\mathbf{w})=||\sum \_j w \_j (\mathbf{x}-\mathbf{x}\_\boldsymbol{j})||^2 = \sum \_ {jk} w \_j w \_k (\mathbf{x}-\mathbf{x}\_\boldsymbol{j})^T (\mathbf{x}-\mathbf{x}\_\boldsymbol{k})$$  
 令$\mathbf{G}=[\centerdot\centerdot\centerdot, (\mathbf{x}-\mathbf{x}\_\boldsymbol{j}),\centerdot\centerdot\centerdot]$，可定义局部协方差矩阵$\mathbf{C}$（local covariance matrix）为  
 $$\mathbf{C} = \mathbf{G}^T \mathbf{G} $$
 此时有  
 $$\varepsilon(\mathbf{w})=\mathbf{w}^\boldsymbol{T} \mathbf{C} \mathbf{w} \ \ \  s.t.\ \  ||\mathbf{w}||=1$$  
 可以利用拉格朗日乘数法求最优解，构造函数  
 $$f(\mathbf{w},\lambda)=\mathbf{w}^\boldsymbol{T} \mathbf{C} \mathbf{w} - 2*\lambda (\boldsymbol{1}^T \mathbf{w}-1)$$  
 求偏导并令之等于零可得，  
 $$\mathbf{C} \mathbf{w}=\lambda\boldsymbol{1}$$  
 根据$ \mathbf{w}$的模为1可解出$\lambda=\frac{1}{\boldsymbol{1}^T \mathbf{C}^{-1} \boldsymbol{1}}$，即矩阵$\mathbf{C}$的逆所有元素相加的和。进而可以求出$\mathbf{w}$  
 $$\mathbf{w}=\frac{\mathbf{C}^{-1}\boldsymbol{1}}{\boldsymbol{1}^T \mathbf{C}^{-1} \boldsymbol{1}}$$  
 可以看出分母是分子的和，所以等同于求$\mathbf{C}\mathbf{w}=\boldsymbol{1}$的解，然后归一化使$\mathbf{w}$的模为1。当K大于数据点的维度时，每个数据点可以由不同的邻近数据点的不同组和线性表示出来，即权重矩阵不唯一，矩阵$\mathbf{C}$为奇异矩阵，无法求逆，此时应加入正则项。python code:  
 ```Python
 if K>D:
     tol=1e-3
 else:
     tol=0
 W=np.zeros((K,N))
 for ii in range(N):
     z=X[:,neighborhood[:,ii]]-repmat(X[:,ii].reshape(D,1),1,K)
     C=np.dot(z.T,z)
     C=C+np.eye(K)*tol*np.trace(C)
     W[:,ii]=np.dot(np.linalg.inv(C),np.ones((K,1))).reshape((12,))
     W[:,ii]=W[:,ii]/sum(W[:,ii])
```
3. 求低维空间的坐标
 最小化插值误差函数  
 \\[\begin{split}\Phi(\mathbf{Y})&=\sum \_i ||\mathbf{y} \_i - \sum \_j w \_j \mathbf{y} \_j||^2=tr((\mathbf{Y}-\mathbf{WY})^T(\mathbf{Y}-\mathbf{WY})) \\\\ &=tr(\mathbf{Y}^T(\mathbf{I}-\mathbf{W})^T(\mathbf{I}-\mathbf{W})\mathbf{Y}) =\sum \_i ^d \mathbf{Y}\_i ^T \mathbf{M}\mathbf{Y} _i \end{split}\\]  
 令$\mathbf{M}=(\mathbf{I}-\mathbf{W})^T(\mathbf{I}-\mathbf{W})$，并且约束$\mathbf{Y}$有零均值，单位协方差矩阵，那么最小化此二次型当且仅当$\mathbf{Y}$为$\mathbf{M}$的最小d+1个特征值对应的特征向量（Rayleigh-Ritz theorem），舍去第一个零特征值对应的特征向量。python code:  
 ```Python
 import scipy.sparse as sp
 from scipy.sparse.linalg.eigen.arpack import eigsh
 M=sp.csr_matrix(np.eye(N))
 for ii in range(N):
     w=W[:,ii]
     jj=neighborhood[:,ii]
     M[ii,jj]=M[ii,jj]-w
     M[jj,ii]=M[jj,ii]-w.reshape(K,1)
     M[np.ix_(jj,jj)]=M[np.ix_(jj,jj)]+np.dot(w.reshape(K,1),w.reshape(1,K))
 print 'begin solving eigenvectors'
 eigenvals, Y=eigsh(M,d+1,sigma=0.0,tol=1e-6)
 Y_r=Y[:,1:].T*(np.sqrt(N))
```
以上各部分可整合成一个函数lle(X,K,d)，X是输入数据，K是近邻数，d是流体的维度。  
下面对瑞士卷(swissroll)和S-曲线(s-curve)使用lle算法。
```Python
N=2000
K=12
d=2

tt0 = (3*np.pi/2)*(1+2*np.arange(0,1.01,0.02))
hh = np.arange(0,1.01,0.125)*30
xx = np.dot((tt0*np.cos(tt0)).reshape(len(tt0),1),np.ones((1,len(hh))))
yy = np.dot(np.ones((len(tt0),1)),hh.reshape(1,len(hh)))
zz = np.dot((tt0*np.sin(tt0)).reshape(len(tt0),1),np.ones((1,len(hh))))
cc = np.dot(tt0.reshape(len(tt0),1),np.ones((1,len(hh))))

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

color_map = cm.jet
scalarMap = cm.ScalarMappable(norm=Normalize(vmin=cc.min(), vmax=cc.max()), cmap=color_map)
C_colored = scalarMap.to_rgba(cc)

fig = plt.figure()
ax = fig.add_subplot(2,3,1,projection='3d')
ax._axis3don = False
surf = ax.plot_surface(xx, yy, zz, rstride=1, cstride=1,facecolors=C_colored,zorder=1)
surf.set_edgecolor('k')
ax.view_init(20,-72)
ax.set_xlim(-15,20)
ax.set_ylim(0,32)
ax.set_zlim(-15,15)
ax.plot([-15,-15],[0,32],[-15,-15],'k-',lw=2,clip_on=True,clip_box=surf,zorder=-1)
ax.plot([-15,20],[0,0],[-15,-15],'k-',linewidth=2)
ax.plot([-15,-15],[0,0],[-15,15],'k-',linewidth=2)

tt = (3*np.pi/2)*(1+2*np.random.rand(1,N))
height = 21*np.random.rand(1,N)
X = np.row_stack((tt*np.cos(tt),height,tt*np.sin(tt)))

ax=fig.add_subplot(2,3,2,projection='3d')
sca=ax.scatter(X[0,:],X[1,:],X[2,:],s=12,c=tt,marker='+',cmap=cm.jet,
           norm=Normalize(vmin=tt.min(), vmax=tt.max()))
ax.view_init(20,-72)
ax._axis3don=False
ax.set_xlim(-15,20)
ax.set_ylim(0,32)
ax.set_zlim(-15,15)
ax.plot([-15,-15],[0,32],[-15,-15],'k-',lw=2,zorder=-1)
ax.plot([-15,20],[0,0],[-15,-15],'k-',linewidth=2)
ax.plot([-15,-15],[0,0],[-15,15],'k-',linewidth=2)

Y=lle(X,K,d)
ax=fig.add_subplot(2,3,3)
ax.scatter(Y[0,:],Y[1,:],s=12,c=tt,marker='+',cmap=cm.jet,
           norm=Normalize(vmin=tt.min(), vmax=tt.max()))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.xticks([])
plt.yticks([])

#S-curve
tt=np.arange(-1,0.51,0.1)*np.pi
uu=np.arange(0.5,-1.1,-0.1)*np.pi
hh=np.arange(0,1.01,0.1)*5
xx=np.dot(np.row_stack((np.cos(tt), -1*np.cos(uu))).reshape(32,1),np.ones((1,11)))
yy=np.dot(np.ones((32,1)),hh.reshape(1,11))
zz=np.dot(np.row_stack((np.sin(tt), 2-np.sin(uu))).reshape(32,1),np.ones((1,11)))
cc=np.dot(np.row_stack((tt,uu)).reshape(32,1),np.ones((1,11)))

color_map = cm.jet
scalarMap = cm.ScalarMappable(norm=Normalize(vmin=cc.min(), vmax=cc.max()), cmap=color_map)
C_colored = scalarMap.to_rgba(cc)

ax = fig.add_subplot(2,3,4,projection='3d')
ax._axis3don = False
surf = ax.plot_surface(xx, yy, zz, rstride=1, cstride=1,
                       facecolors=C_colored,zorder=1)
surf.set_edgecolor('k')
ax.view_init(10,-70)
ax.set_xlim(-1,1)
ax.set_ylim(0,5)
ax.set_zlim(-1,3)

angle=np.pi*(1.5*np.random.rand(1,N/2)-1)
height=5*np.random.rand(1,N)
X=np.row_stack((np.row_stack((np.cos(angle),-1*np.cos(angle))).reshape(1,N),
                height,
                np.row_stack((np.sin(angle),2-np.sin(angle))).reshape(1,N)))

ax=fig.add_subplot(2,3,5,projection='3d')
ax.scatter(X[0,:],X[1,:],X[2,:],s=12,c=np.row_stack((angle,angle)).reshape(1,N),
               marker='+',cmap=cm.jet,
               norm=Normalize(vmin=angle.min(), vmax=angle.max()))
ax.view_init(10,-70)
ax._axis3don=False
ax.set_xlim(-1,1)
ax.set_ylim(0,5)
ax.set_zlim(-1,3)

Y=lle(X,K,d)

ax=fig.add_subplot(2,3,6)
ax.scatter(Y[1,:],Y[0,:],s=12,c=np.row_stack((angle,angle)).reshape(1,N),marker='+',cmap=cm.jet,
           norm=Normalize(vmin=angle.min(), vmax=angle.max()))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.xticks([])
plt.yticks([])
plt.show()
```
结果如下：  
![lle](/img/lle.png)  
lle算法得到的低维结构有些变形，如上图所示，瑞士卷不是长方形的而是从宽到窄，相对而言，S型曲线的结果要好一些。下面介绍modified locally linear embedding算法来解决这个问题。  
lle算法计算权重矩阵时用到的重构误差函数为$||\mathbf{Gw}||$，理论上$\boldsymbol{1}$在矩阵$\mathbf{G}$的零空间上的正交投影都可以归一化为符合要求的权重向量，那么权重最优解的近似值的个数为矩阵$\mathbf{G}$零空间的维度，可以利用这样的多个权重矩阵描述每个数据点的局部结构，然后来求数据的低维表示。程序参照了sklearn包里面的locally_linear_embedding函数，如下，  
```Python
from scipy.linalg import eigh, svd, qr, solve

def mlle(X,K,d,modified_tol=1E-12):
    D,N=X.shape
    #STEP1: COMPUTE PAIRWISE DISTANCES & FIND NEIGHBORS
    X2=np.sum(X**2,0).reshape(1,N)
    distance=repmat(X2,N,1)+repmat(X2.T,1,N)-2*np.dot(X.T,X)
    index=np.argsort(distance,0)
    neighbors=index[1:1+K,:].T
    
    
    V = np.zeros((N, K, K))
    nev = min(D, K)
    evals = np.zeros([N, nev])
    
    use_svd = (K > D)
    X=X.T
    if use_svd:
        for i in range(N):
            X_nbrs = X[neighbors[i]] - X[i]
            V[i], evals[i], _ = svd(X_nbrs,
                                    full_matrices=True)
        evals **= 2
    else:
        for i in range(N):
            X_nbrs = X[neighbors[i]] - X[i]
            C_nbrs = np.dot(X_nbrs, X_nbrs.T)
            evi, vi = eigh(C_nbrs)
            evals[i] = evi[::-1]
            V[i] = vi[:, ::-1]
    
    reg = 1E-3 * evals.sum(1)

    tmp = np.dot(V.transpose(0, 2, 1), np.ones(K))
    tmp[:, :nev] /= evals + reg[:, None]
    tmp[:, nev:] /= reg[:, None]

    w_reg = np.zeros((N, K))
    for i in range(N):
        w_reg[i] = np.dot(V[i], tmp[i])
    w_reg /= w_reg.sum(1)[:, None]
    
    rho = evals[:, d:].sum(1) / evals[:, :d].sum(1)
    eta = np.median(rho)
    
    s_range = np.zeros(N, dtype=int)
    evals_cumsum = np.cumsum(evals, 1)
    eta_range = evals_cumsum[:, -1:] / evals_cumsum[:, :-1] - 1
    for i in range(N):
        s_range[i] = np.searchsorted(eta_range[i, ::-1], eta)
    s_range += K - nev
    
    M = np.zeros((N, N), dtype=np.float)
    for i in range(N):
        s_i = s_range[i]

        #select bottom s_i eigenvectors and calculate alpha
        Vi = V[i, :, K - s_i:]
        alpha_i = np.linalg.norm(Vi.sum(0)) / np.sqrt(s_i)

            #compute Householder matrix which satisfies
            #  Hi*Vi.T*ones(n_neighbors) = alpha_i*ones(s)
            # using prescription from paper
        h = alpha_i * np.ones(s_i) - np.dot(Vi.T, np.ones(K))

        norm_h = np.linalg.norm(h)
        if norm_h < modified_tol:
            h *= 0
        else:
            h /= norm_h

            #Householder matrix is
            #  >> Hi = np.identity(s_i) - 2*np.outer(h,h)
            #Then the weight matrix is
            #  >> Wi = np.dot(Vi,Hi) + (1-alpha_i) * w_reg[i,:,None]
            #We do this much more efficiently:
            Wi = (Vi - 2 * np.outer(np.dot(Vi, h), h)
                  + (1 - alpha_i) * w_reg[i, :, None])

            #Update M as follows:
            # >> W_hat = np.zeros( (N,s_i) )
            # >> W_hat[neighbors[i],:] = Wi
            # >> W_hat[i] -= 1
            # >> M += np.dot(W_hat,W_hat.T)
            #We can do this much more efficiently:
            nbrs_x, nbrs_y = np.meshgrid(neighbors[i], neighbors[i])
            M[nbrs_x, nbrs_y] += np.dot(Wi, Wi.T)
            Wi_sum1 = Wi.sum(1)
            M[i, neighbors[i]] -= Wi_sum1
            M[neighbors[i], i] -= Wi_sum1
            M[i, i] += s_i
    M=sp.csr_matrix(M)
    print 'begin solving eigenvectors'
    eigenvals, Y=eigsh(M,d+1,sigma=0.0,tol=1e-12)
    Y_r=Y[:,1:].T*(np.sqrt(N))
    return Y_r
    ```
用mlle算法重新跑瑞士卷和S-曲线的数据，结果如下，  
![mlle](/img/mlle.png)  
得到的2维结构正好是长方形。  
**参考文献**：  
1. S. Roweis and L Saul. Nonlinear dimensionality reduction by locally linear embedding. Science, 290: 2323–2326, 2000.
2. Zhang, Z. & Wang, J. MLLE: Modified Locally Linear Embedding Using Multiple Weights. [http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.70.382](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.70.382)  
3. [http://www.cs.nyu.edu/~roweis/lle/](http://www.cs.nyu.edu/~roweis/lle/)
4. [http://scikit-learn.org/stable/modules/generated/sklearn.manifold.locally_linear_embedding.html](http://scikit-learn.org/stable/modules/generated/sklearn.manifold.locally_linear_embedding.html)
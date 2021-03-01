title: t-SNE:高维数据可视化
date: 2015-08-17 22:26:30
tags: data visualization
banner: /img/digits_tsne-generated.png
---
t-SNE，即t-distributed stochastic neighbor embedding，也是一种流体学习方法（manifold learning），通过保持数据点的相邻关系把数据从高维空间中降低到2维平面上，对高维数据可视化的效果非常好。本文使用python的机器学习包sklearn来对这个算法进行简单的介绍。<!-- more -->导入所需要的包，
```Python
# That's an impressive list of imports.
import numpy as np
from numpy import linalg
from numpy.linalg import norm
from scipy.spatial.distance import squareform, pdist

# We import sklearn.
import sklearn
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale

# We'll hack a bit with the t-SNE code in sklearn 0.15.2.
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.manifold.t_sne import (_joint_probabilities,
                                    _kl_divergence)
from sklearn.utils.extmath import _ravel
# Random state.
RS = 20150101

# We'll use matplotlib for graphics.
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects

# We import seaborn to make nice plots.
import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})

# We'll generate an animation with matplotlib and moviepy.
from moviepy.video.io.bindings import mplfig_to_npimage
import moviepy.editor as mpy
```
首先介绍一个可视化手写字体的例子。数据使用sklearn包里面带的一个数据集，共有1797张手写字体的图片，每张图片的像素为8*8=64。
```Python
digits = load_digits()
print(digits.shape)   #1797, 64
print(digits['DESCR'])
nrows, ncols = 2, 5
plt.figure(figsize=(6,3))
plt.gray()
for i in range(ncols * nrows):
    ax = plt.subplot(nrows, ncols, i + 1)
    ax.matshow(digits.images[i,...])
    plt.xticks([]); plt.yticks([])
    plt.title(digits.target[i])
plt.savefig('digits-generated.png', dpi=150)
```
![digits](/img/digits-generated.png)
接着运行t-SNE算法
```Python
# We first reorder the data points according to the handwritten numbers.
X = np.vstack([digits.data[digits.target==i]
               for i in range(10)])
y = np.hstack([digits.target[digits.target==i]
               for i in range(10)])
digits_proj = TSNE(random_state=RS).fit_transform(X)
```
写绘图函数，用转换后的数据点绘图。
```Python
def scatter(x, colors):
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", 10))

    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40,
                    c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # We add the labels for each digit.
    txts = []
    for i in range(10):
        # Position of each label.
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts
```
展示结果，可以看到不同数字被分成了不同的簇。
```Python
scatter(digits_proj, y)
plt.savefig('images/digits_tsne-generated.png', dpi=120)
```
![digits-tSNE](/img/digits_tsne-generated.png)
#算法原理  
对每个数据点i，和每个潜在的邻居j，首先计算一个条件概率$p\_{j|i}$，表示i选择j为邻居的概率：
$$p\_{j|i}=\frac{exp(-||x\_i-x\_j||^2/2\sigma\_i ^2)}{\sum\_{k\neq i}exp(-||x\_i-x\_k||^2/2\sigma\_i ^2)}$$
$\sigma\_i$是以点$x\_i$为中心的高斯分布的方差，是通过二分法找到的能使分布的熵等于$log(Perp)$，$Perp$是perplexity，用来衡量点$x\_i$有效邻居的个数，值可以设置在5-50之间，是$\sigma\_i$的单调函数。
$$Perp(P\_i)=2^{H(P\_i)} \ \ \ \ H(P\_i)=-\sum\_j p\_{j|i}log\_2 p\_{j|i}$$
接着我们定义数据点间的相似性为对称的条件概率，
$$p\_{ij}=\frac{p\_{j|i}+p\_{i|j}}{2N}$$
#相似性矩阵（similarity matrix）  
下面我们比较一下数据点的距离矩阵，$\sigma$为常量的相似性矩阵和$\sigma$为变量时的相似性矩阵。
```Python
def _joint_probabilities_constant_sigma(D, sigma):
    P = np.exp(-D**2/2 * sigma**2)
    P /= np.sum(P, axis=1)
    return P
# Pairwise distances between all data points.
D = pairwise_distances(X, squared=True)
# Similarity with constant sigma.
P_constant = _joint_probabilities_constant_sigma(D, .002)
# Similarity with variable sigma.
P_binary = _joint_probabilities(D, 30., False)
# The output of this function needs to be reshaped to a square matrix.
P_binary_s = squareform(P_binary)

plt.figure(figsize=(12, 4))
pal = sns.light_palette("blue", as_cmap=True)

plt.subplot(131)
plt.imshow(D[::10, ::10], interpolation='none', cmap=pal)
plt.axis('off')
plt.title("Distance matrix", fontdict={'fontsize': 16})

plt.subplot(132)
plt.imshow(P_constant[::10, ::10], interpolation='none', cmap=pal)
plt.axis('off')
plt.title("$p_{j|i}$ (constant $\sigma$)", fontdict={'fontsize': 16})

plt.subplot(133)
plt.imshow(P_binary_s[::10, ::10], interpolation='none', cmap=pal)
plt.axis('off')
plt.title("$p_{j|i}$ (variable $\sigma$)", fontdict={'fontsize': 16})
plt.savefig('similarity-generated.png', dpi=120)
```
![similarity](/img/similarity-generated.png)
可以看到数据已经呈现出10组，分别对应10个数字。  
现在使用1自由度的t-分布来定义低维空间数据点的相似性矩阵，
$$q\_{ij}=\frac{(1+||y\_i -y\_j||^2)^{-1}}{\sum\_{k\neq l}(1+||y\_k-y\_l||^2)^{-1}}$$
我们的目的是找到低维空间的一组坐标使得这两个分布尽可能相似，一个自然的度量就是这两个分布的Kullback-Leiber divergence:
$$C=KL(P||Q)=\sum \_{i,j}p\_{ij} \frac{p\_{ij}}{q\_{ij}}$$
我们使用梯度下降最小化这个得分，原始论文中有详细的求导过程，这里我们只写出结果：
$$\frac{\delta C}{\delta y\_i}=4\sum \_j (p\_{ij}-q\_{ij})(y\_i -y\_j)(1+||y\_i -y\_j||^2)^{-1}$$
这个公式有很好物理意义。$p\_{ij}-q\_{ij}$为正，说明$x\_i$在高维空间和$x\_j$相似度高，而低维空间相似度低，$x\_i$将会靠近$x\_j$，反之则会远离$x\_j$。下面用一个动画来阐明这个过程。
首先monkey-patch sklearn包的t-SNE里面的_gradient_descent()函数，把每次迭代的结果保存下来。
```Python
# This list will contain the positions of the map points at every iteration.
positions = []
def _gradient_descent(objective, p0, it, n_iter, n_iter_without_progress=30,
                      momentum=0.5, learning_rate=1000.0, min_gain=0.01,
                      min_grad_norm=1e-7, min_error_diff=1e-7, verbose=0,
                      args=[]):
    # The documentation of this function can be found in scikit-learn's code.
    p = p0.copy().ravel()
    update = np.zeros_like(p)
    gains = np.ones_like(p)
    error = np.finfo(np.float).max
    best_error = np.finfo(np.float).max
    best_iter = 0

    for i in range(it, n_iter):
        # We save the current position.
        positions.append(p.copy())

        new_error, grad = objective(p, *args)
        error_diff = np.abs(new_error - error)
        error = new_error
        grad_norm = linalg.norm(grad)

        if error < best_error:
            best_error = error
            best_iter = i
        elif i - best_iter > n_iter_without_progress:
            break
        if min_grad_norm >= grad_norm:
            break
        if min_error_diff >= error_diff:
            break

        inc = update * grad >= 0.0
        dec = np.invert(inc)
        gains[inc] += 0.05
        gains[dec] *= 0.95
        np.clip(gains, min_gain, np.inf)
        grad *= gains
        update = momentum * update - learning_rate * grad
        p += update

    return p, error, i
sklearn.manifold.t_sne._gradient_descent = _gradient_descent

X_proj = TSNE(random_state=RS).fit_transform(X)

X_iter = np.dstack(position.reshape(-1, 2)
                   for position in positions)
```
使用moviepy包生成gif动画
```Python
f, ax, sc, txts = scatter(X_iter[..., -1], y)

def make_frame_mpl(t):
    i = int(t*40)
    x = X_iter[..., i]
    sc.set_offsets(x)
    for j, txt in zip(range(10), txts):
        xtext, ytext = np.median(x[y == j, :], axis=0)
        txt.set_x(xtext)
        txt.set_y(ytext)
    return mplfig_to_npimage(f)

animation = mpy.VideoClip(make_frame_mpl,
                          duration=X_iter.shape[2]/40.)
animation.write_gif("d:/dataVisualization/animation.gif", fps=20)
```
![animation](/img/animation.gif)
接着创建低维空间数据点的相似性矩阵的动画
```Python
n = 1. / (pdist(X_iter[..., -1], "sqeuclidean") + 1)
Q = n / (2.0 * np.sum(n))
Q = squareform(Q)

f = plt.figure(figsize=(6, 6))
ax = plt.subplot(aspect='equal')
im = ax.imshow(Q, interpolation='none', cmap=pal)
plt.axis('tight')
plt.axis('off')

def make_frame_mpl(t):
    i = int(t*40)
    n = 1. / (pdist(X_iter[..., i], "sqeuclidean") + 1)
    Q = n / (2.0 * np.sum(n))
    Q = squareform(Q)
    im.set_data(Q)
    return mplfig_to_npimage(f)

animation = mpy.VideoClip(make_frame_mpl,
                          duration=X_iter.shape[2]/40.)
animation.write_gif("d:/dataVisualization/animation_matrix.gif", fps=20)
```
![animation](/img/animation_matrix.gif)
可以看到相似性矩阵越来越接近原始数据的相似性矩阵。  
#t-分布
下面解释一下选择1自由度t-分布的原因。一个半径为r的N维球体的体积与$r^N$成正比，当N非常大的时候，如果我们在这个球体中以均匀分布选择一些点，那么大多数点都会非常靠近球的表面，非常少的点会在中心附近。可以用模拟的方法阐释这个现象：
```Python
npoints = 1000
plt.figure(figsize=(15, 4))
for i, D in enumerate((2, 5, 10)):
    # Normally distributed points.
    u = np.random.randn(npoints, D)
    # Now on the sphere.
    u /= norm(u, axis=1)[:, None]
    # Uniform radius.
    r = np.random.rand(npoints, 1)
    # Uniformly within the ball.
    points = u * r**(1./D)
    # Plot.
    ax = plt.subplot(1, 3, i+1)
    ax.set_xlabel('Ball radius')
    if i == 0:
        ax.set_ylabel('Distance from origin')
    ax.hist(norm(points, axis=1),
            bins=np.linspace(0., 1., 50))
    ax.set_title('D=%d' % D, loc='left')
plt.savefig('spheres-generated.png', dpi=100, bbox_inches='tight')
```
![spheres](/img/spheres.png)
如果我们对高维空间中原始数据和低维空间映射的数据点使用相同的高斯分布，则会造成数据点与其邻居间距离分布的不平衡。而当算法想要在这两个空间中重复出相同的距离时，这种不平衡就会造成数据点间过度的吸引力，使数据点多靠近中心位置。而在低维空间使用1自由度的t-分布可以避免这个问题。1自由度的t-分布较高斯分布有更高尾部，补偿了因空间维度造成的距离分布的不平衡。
```Python
z = np.linspace(0., 5., 1000)
gauss = np.exp(-z**2)
cauchy = 1/(1+z**2)
plt.figure()
plt.plot(z, gauss, label='Gaussian distribution')
plt.plot(z, cauchy, label='Cauchy distribution')
plt.legend()
plt.savefig('distributions-generated.png', dpi=100)
```
![distributions](/img/distributions.png)
使用这个分布会得到更有效的数据可视化，不同的簇会有更明显的区分。  
  
#参考文献：  
1. van der Maaten, L. and Hinton, G. E. (2008). Visualizing data using
t-SNE. J. Machine Learning Res., 9.  
2. [https://beta.oreilly.com/learning/an-illustrated-introduction-to-the-t-sne-algorithm](https://beta.oreilly.com/learning/an-illustrated-introduction-to-the-t-sne-algorithm)
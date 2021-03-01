title: 分类型变量的聚类算法 - ROCK
date: 2015-07-10 23:40:22
tags: 数据挖掘
---
在数据挖掘中，聚类是发现数据分布模式的一种方式，把一组数据点按照某种合适的距离分成不同的簇，使得簇内有尽可能小的距离，而簇间有尽可能大的距离。本篇介绍一种适用于分类型变量的聚类算法 - ROCK，内容主要基于论文“ROCK: A Robust Clustering Algorithm for Categorical Attributes”。<!-- more -->  
ROCK是一种基于图的聚类算法，定义了数据点间的邻居（neighbors）和链接（links）。
**邻居（neighbors）**：两个数据点间的相似度大于一定的阈值就可以称为邻居。即$sim(p_i,p_j)\geqslant\theta$。其中$\theta$是取值在0，1之间的一个小数，用于衡量两个点相似度。$sim$是相似度函数，可以是距离函数（如$L_1$, $L_2$），也可以是一些非度量型的（如一个领域的专家提供的相似度衡量方法）。对于分类型变量，我们选择Jaccard相似性函数，即  
$$sim(T_1, T_2)=\frac{|T_1\cap T_2|}{|T_1\cup T_2|}$$  
对于分类型变量，我们把每条记录当作一个交易，如果某个属性值缺失，仅仅让这条交易不包含这个属性。当然这只是一种处理缺失值的方法，针对不同的数据会有不同的处理方式。  
Python code:  
```Python
import numpy as np    

def neighbors(filename, theta):  
    S=[]  
    P=[]  
    for line in open(filename):  
        lines=line.rstrip().split(',')  
        for i in range(1,len(lines)):  
            lines[i]+=str(i)  
        #print lines  
        if '?11' in lines:  
            lines.remove('?11')  
        S.append(set(lines[1:]))  
        P.append(lines[0])  
    A=np.zeros((len(S),len(S)),dtype='int16')  
    for i in range(len(S)-1):  
        for j in range(i+1,len(S)):  
            if float(len(S[i].intersection(S[j])))/len(S[i].union(S[j])) >= theta:  
                A[i,j]=1  
                A[j,i]=1  
    return A,P  
```
数据是UCI数据库里面用作benchmark的Mushroom数据集([https://archive.ics.uci.edu/ml/datasets/Mushroom](https://archive.ics.uci.edu/ml/datasets/Mushroom))，其中只有一个一个属性值有缺失。使用上面的函数建立neighbors矩阵B和类别向量P。  
Python code:  
```Python
B,P=neighbors('agaricus-lepiota.data.txt',0.8)  
```
**连接（links）**：两个点之间的连接数$link(p_i, p_j)$定义为这两个点共同邻居的个数，因此，这种基于连接的方法是一种考虑全局性的聚类算法。通过下面程序可以得到Mushroom数据点之间的连接矩阵A。  
Python code:  
```Python
def compute_links(A):
    link=np.zeros_like(A)
    for i in range(A.shape[0]):
        N=np.where(A[i,]>0)[0]
        for j in range(len(N)-1):
            for z in range(j+1,len(N)):
                link[N[j],N[z]]+=1
                link[N[z],N[j]]+=1
    return link
A=compute_links(B)
```
**准则函数（Criterion Function）**：寻找最好的簇等价于最大化准则函数。我们的目的是让簇内有高的连接度，同时最小化不同簇之间的连接度。因此可以采用如下函数：  
$$E\_l = \sum\_{i=1}^k n\_i \ast \sum\_{p\_q,p\_r \in C\_i} \frac{link(p\_q, p\_r)}{n\_i^{1+2 f ( \theta )}}$$

其中$C_i$代表第i个簇，左边的分母为$C_i$中期望的连接数。本文使用的$f(\theta)$为$\frac{1-\theta}{1+\theta}$。  
  
**适合度函数（Goodness Measure）**：根据准则函数，我们可以得到两个簇之间的适合度函数，即评价两个簇是否相近，以便合并在一起。如下：  
$$link[C\_i, C\_j] = \sum_{p_q \in C_i, p_r \in C_j} link(p_q, p_r)$$  
  
即两个簇之间的连接数为其中结点间连接数的总和。  

$$g(C_i,C_j)=\frac{link[C_i,C_j]}{(n_i + n_j)^{1+2 f(\theta)} - n_i^{1+2f(\theta)}-n_j^{1+2f(\theta)}}$$  
$C_i$, $C_j$两个簇合并在一起后期望的连接数为$(n_i +n_j)^{1+2f(\theta)}$，每个簇内的期望数分别为$n_i^{1+2f(\theta)}$和$n_j^{1+2f(\theta)}$，因此簇间的期望连接数为他们相减。除以期望数可以让簇以使准则函数最大的方向进行。  
Python code:  
```Python
def goodness(value, c1, c2, theta):
    return value/((c1+c2)**(1+2*(1-theta)/(1+theta))-c1**(1+2*(1-theta)/(1+theta))-c2**(1+2*(1-theta)/(1+theta)))  
def delete(Q,value):
    for i in Q:
        if i[1]==value:
            Q.remove(i)
            break

def update(Q,value,link):
    for i in range(len(Q)):
        if Q[i][1]==value:
            Q.remove(Q[i])
            Q.insert(i,(link,value))
            break

def build_heap(A,theta):
    q={}
    Q=[]
    for i in range(A.shape[0]):
        q[tuple([i])]=[]
        for j in range(A.shape[1]):
            if j==i or A[i,j]==0:
                continue
            #if tuple([i]) not in q:
             #   q[tuple([i])]=[]
            heapq.heappush(q[tuple([i])], (goodness(A[i,j],1,1,theta)*(-1),[j]))
            #print q[tuple([i])]
        if len(q[tuple([i])])>0:
            heapq.heappush(Q, (q[tuple([i])][0][0],[i]))
    return q,Q    

import heapq
import copy    
def cluster(A, k, theta):
    q={}
    Q=[]
    for i in range(A.shape[0]):
        q[tuple([i])]=[]
        for j in range(A.shape[1]):
            if j==i or A[i,j]==0:
                continue
            heapq.heappush(q[tuple([i])], (goodness(A[i,j],1,1,theta)*(-1),[j]))
        if len(q[tuple([i])])>0:
            heapq.heappush(Q, (q[tuple([i])][0][0],[i]))
    print len(Q)
    while len(Q)>k:
        print "lenQ: ",len(Q)
        print "lenq: ",len(q)
        if Q[0][0] == 0:
            return Q
        u=heapq.heappop(Q)
        print "lenQafterpop: ",len(Q)
        w=copy.deepcopy(u[1])
        print "MAXlink: ",u[0]
        u=u[1]
        if tuple(u) not in q:
            print "u: ",u
        v=q[tuple(u)][0][1]
        print "v: ",v
        delete(Q,v)
        print "lenQafterdeletev: ",len(Q)
        print "lenQafterremoveUV: ",len(Q)
        w.extend(v)
        q[tuple(w)]=[]
        U=[tuple(x[1]) for x in q[tuple(u)]]
        V=[tuple(x[1]) for x in q[tuple(v)]]
        U.extend(V)
        if len(U)==2:
            heapq.heappush(Q,(0,w))
            continue
        for x in set(U):
            if set(x).issubset(set(tuple(w))) :
                continue
            link=0
            for y in w:
                for z in x:
                    link+=A[y,z]
            if tuple(x) not in q:
                print "x: ",x
            delete(q[tuple(x)],u)
            delete(q[tuple(x)],v)
            g=goodness(link,len(x),len(w),theta)*(-1)
            q[tuple(x)].append((g,w))
            q[tuple(w)].append((g,list(x)))
            heapq.heapify(q[tuple(x)])
            heapq.heapify(q[tuple(w)])
            update(Q,list(x),q[tuple(x)][0][0])
        if len(q[tuple(w)])>0:
            Q.append((q[tuple(w)][0][0],w))
        heapq.heapify(Q)
        del q[tuple(u)]
        del q[tuple(v)]
    return Q
```
函数需要提供类的个数，当类达到要求时会结束聚类过程。当任何两个类之间没有共同连接时也会停止聚类。如下得到Mushroom数据聚类的结果。  
```Python
Q=cluster(A,20,0.8)
```
最后查看所得到类的性质。  
```Python
def printQ(Q,P):
    for i in range(len(Q)):
        stat={'p':0,'e':0}
        for j in Q[i][1]:
            stat[P[j]]+=1
        print str(i+1)+': p-'+str(stat['p'])+' e-'+str(stat['e'])
printQ(Q,P)
```
输出如下：  
1: p-256 e-0
2: p-0 e-704
3: p-0 e-1728
4: p-0 e-96
5: p-0 e-96
6: p-192 e-0
7: p-288 e-0
8: p-1728 e-0
9: p-0 e-192
10: p-0 e-768
11: p-32 e-0
12: p-36 e-0
13: p-8 e-0
14: p-0 e-192
15: p-0 e-48
16: p-0 e-288
17: p-0 e-48
18: p-72 e-32
19: p-0 e-16
20: p-8 e-0
21: p-1296 e-0  
可以看到，虽然指定了类的个数20，但算法找到了21个类，它们之间的连接数均为零，使得聚类过程无法继续进行下去。另外，可以看到，除了第18个簇，其他簇全部是有毒的(p)，或者全部是可食用的(e)，说明这种聚类算法在Mushroom数据集上得到了很好的结果。
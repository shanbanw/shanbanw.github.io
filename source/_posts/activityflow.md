---
title: 静态fMRI数据中的信息流动
date: 2016-12-27 09:36:25
tags: neuroscience
banner: /img/activityflow.png
---
大脑信号传播是通过突触完成的，具体表现是受到激发的神经元发出一个个spike，激活突触中的信号通路，这个过程一般在微秒级的时间尺度上完成的。通过测量血液中氧气浓度(BOLD信号)来间接反应大脑信号强度的fMRI技术的缺点正是时间分辨率太低，导致不能采集出神经元放电前后的变化，也就不能让我们检测到信号传导的前后顺序。即我们只能从磁共振扫描仪采集的大脑信号看出来大脑每个位置的信号随时间的波动。最近看nature neuroscience的网站，发现了一篇论文——"[Activity flow over resting-state networks shapes cognitive task activations](http://www.nature.com/neuro/journal/v19/n12/full/nn.4406.html)"，单单是名字就很吸引人了，从相对静态的fMRI数据发现信息流动，是一个很神奇的设想。<!-- more -->
论文作者把静息态网络作为大脑中相互交错的道路，大脑信号沿着这些路径在各个脑区之间流动，信号传递效率受到突触权重的影响。对于fmri，每个时间点每个脑区信号就是这种信号流动后的结果，也就是说每个脑区的信号是其他脑区信号传递到这个脑区的总和，可以用下面的模型表示，
![Activity flow mapping](/img/activityflowmapping.png)
论文首先使用了模拟的数据来检验模型，虽然用了一个非常简单的模型来模拟数据，但对于初学者而言是一个非常有趣的模型。静息态下自发的大脑活动的起源目前并不清楚，但很多研究是通过结构连接来模拟静息态的功能连接的，所以这篇论文也是先建立了脑区间的结构连接，进而调整连接权重实现可塑性的突触连接。论文模拟了300个单元，代表了300个不同的脑区，然后设置网络密度为15%建立结构连接。脑区间的连接（cross-region connections）G取值范围设为0-5，脑区自连（self-connections，也成为reccurrent connections）Indep取值范围设为0-100。接着让结点间的相互连接形成3个结构单元。对于第一个结构单元，再通过调整突触权重形成2个功能单元。这样就生成了一个大脑的突触连接网络，也就是信号传播的道路。
<pre class="brush: py;">
import numpy as np
import matplotlib.pyplot as plt 
from scipy.integrate import odeint  //用于计算积分
numRegions = 300

#Creating network model with random connectivity, 15% density with random synaptic strengths
structConnVector=np.random.uniform(0,1,(numRegions,numRegions))&gt;.85
#Add self-connections (important if adding autocorrelation later)
np.fill_diagonal(structConnVector,10)
#Create modular structural network (3 modules)
numModules=3
numRPerModule=int(round(numRegions/numModules))
lastModuleNode=-1
for moduleNum in range(0,numModules):
    for thisNodeNum in range(lastModuleNode+1,lastModuleNode+numRPerModule+1):    
        #Set this node to connect to 50 random other nodes in module
        for i in range(1,numRPerModule//2):
		    randNodeInModule=int(np.random.uniform(lastModuleNode+1,lastModuleNode+numRPerModule-1,(1,1)))
            structConnVector[thisNodeNum,randNodeInModule]=1
    lastModuleNode=lastModuleNode+numRPerModule

#Adding synaptic weights to existing structural connections (small random synapse strength variation)
synapticWeightVector=structConnVector*(1+np.random.standard_normal((numRegions,numRegions))*.001)
#Adding synaptic mini-communities (within community 1)
synapticWeightVector[0:50,50:100]=synapticWeightVector[0:50,50:100]*0.5
synapticWeightVector[50:100,0:50]=synapticWeightVector[50:100,0:50]*0.5
synapticWeightVector[0:50,0:50]=synapticWeightVector[0:50,0:50]*1.5
synapticWeightVector[50:100,50:100]=synapticWeightVector[50:100,50:100]*1.5

#Implement global coupling parameter
G=1
synapticWeightVector=G*synapticWeightVector
#Implement local independence parameter
Indep=1

plt.imshow(synapticWeightVector)
plt.colorbar()
plt.savefig('ActflowSim_StructMat.png',dpi=600,transparent=True, bbox_inches='tight')
</pre>

![ActflowSim_StructMat](/img/ActflowSim_StructMat.png)
论文接着使用经典的spiking rate model来生成自发的大脑活动和任务态的大脑活动。
$$\tau\_i\frac{dx\_i}{dt}=-x\_i+f\_i(\sum \_{j=1}^{n}w\_{ji}x\_j + bias\_i)\ \ \ i=[1..n]$$
静息态：在每个时间点，每个单元的活性都设为0，然后对所有单元同时进行刺激，让刺激引发的spike通过网络传播。自相关项设为AR(1)，系数设为0.1。
$$\varepsilon\_t=0.1*\varepsilon\_{t-1}+normal(0,1)$$
任务态：对每一个任务，选出5个连续的单元，在需要进行刺激的时间点处，把刺激强度加到静息态随机刺激强度上，然后让活动沿着网络传播开来。<font color="red">就像在一张餐巾纸上面滴一滴墨水，让其扩散开来</font>。
<pre class="brush: py;">
def sigmoid(x):
    return 1/(1+np.exp(-x))
def equation(startActivity, t, spontActivity, synapticWeightVector, bias=0):
    x=startActivity
	dxdt = -x + sigmoid(np.dot(spontActivity, synapticWeightVector)+bias)
	return dxdt
def networkModel(G=1.0, Indep = 1.0, stimTimes=[], stimRegions=None, synapticWeightVector=synapticWeightVector, numTimePoints=numTimePoints):
    bias = np.zeros(numRegions)
    simulatedTimeseries=np.zeros((numTimePoints, numRegions))
	#Each state modifies previous state (creating some autocorrelation; pink noise)
    autocorrFactor=0.10

    # Modulate synaptic weight matrix by coupling parameter, G
    GlobalCouplingMat = synapticWeightVector*G
    np.fill_diagonal(GlobalCouplingMat,0)
    # Modulate self connection 'independence' parameter, Indep
    IndepVarMat = np.identity(numRegions)*Indep
    IndepVarMat = np.multiply(IndepVarMat,synapticWeightVector)
    
    # Now reconstruct synapticWeightMatrix
    synapticWeightVector = GlobalCouplingMat + IndepVarMat
	
	# Begin computing simulation
    for thisTimePoint in range(0,numTimePoints):
	    #one time step
		t = np.linspace(0,1,2)
		#initialize network
		startActivity = np.zeros(numRegions)
        # Generate spontaneous activity for initial state
        spontActVector=np.random.normal(0,1,(numRegions,))
        stimActVector=np.zeros(numRegions)

        # Specify spontaneous input activity at this time point and task activity
        if thisTimePoint in stimTimes:
            #Include moment-to-moment variability in task stimulation
            stimAct=np.ones(len(stimRegions))*np.random.normal(1,0.5,)
            #stimAct=np.ones(len(stimRegions))*0.5   #excluding moment-to-moment variability in task stimulation
            stimActVector[stimRegions]=stimActVector[stimRegions]+stimAct
        # Add spontaneous activity vector with task stimulus
        spontActVector=(autocorrFactor*spontActVector)+np.random.normal(0,1,(numRegions,))+stimActVector
        
        if thisTimePoint==0: # set initial condition
            simulatedTimeseries[thisTimePoint,] = 0.0 # 0 for all regions
        else:
            simulatedTimeseries[i,]+=odeint(equation, startActivity, t, args=(spontActVector,synapticWeightVector,bias))[1,]

    return simulatedTimeseries

simulatedRestTimeseries = networkModel(G=G, Indep=Indep, stimTimes=[], stimRegions=None, numTimePoints=numTimePoints)
</pre>
与hrf卷积，然后下采样到2s的分辨率
<pre class="brush: py;">
def canonicalHRF(x, param={}):
    if len(param)!=5:
        param={'a1':6, 'a2':12, 'b1':0.9, 'b2':0.9, 'c':0.35}
    d1 = param['a1']*param['b1']
    d2 = param['a2']*param['b2']
    return ((x/d1)**param['a1']*np.exp(-(x-d1)/param['b1']) - param['c']*(x/d2)**param['a2']*np.exp(-(x-d2)/param['b2']))
simsample_rate=0.1
simsample_times = np.arange(0, 30, simsample_rate)
hrf = canonicalHRF(simsample_times)

simulatedTimeseries_convolved=np.ones(np.shape(simulatedRestTimeseries))
for regionNum in range(0,numRegions):
    convolved = np.convolve(simulatedRestTimeseries[:,regionNum], hrf_at_simsample)
    n_to_remove = len(hrf_at_simsample) - 1
    convolved = convolved[:-n_to_remove]
    simulatedTimeseries_convolved[:,regionNum]=convolved

#Downsample fMRI time series
TR=2
dt_rec=0.1
n_skip_BOLD = int(TR/dt_rec)
BOLD_rec = simulatedTimeseries_convolved[::n_skip_BOLD]

#Produce rest FC matrix based on produced spontaneous fMRI time series
fcMat_rest=np.corrcoef(BOLD_rec[10:,],rowvar=0)
#np.fill_diagonal(fcMat_rest,0)
plt.imshow(fcMat_rest)
plt.colorbar()
plt.savefig('ActflowSim_RestfMRI_FCMat.png',dpi=600,transparent=True, bbox_inches='tight')
</pre>

![ActflowSim_RestfMRI_FCMat](/img/ActflowSim_RestfMRI_FCMat.png)

剩下的部分非常明了，模拟任务态数据，然后运行activity flow model，接着迭代cross-region connections和self-connection的参数值，从中发现模型所需要的静息态网络的特性。
title: 非负矩阵分解（NMF）
date: 2015-06-12 23:12:26
tags: 机器学习
---
非负矩阵分解能够学习到对象的局部特征，例如把人脸分解成嘴巴，眼睛，鼻子等等不同的部分，在很多领域都有重要的应用，例如文本聚类，语音处理，协同过滤。其主要思想是把一个大的非负矩阵X分解为两个小矩阵（WH）的乘积，满足这两个小矩阵所有元素都不为负值的条件，即X=WH。然而完全相等很难实现，所以我们只要求能够充分接近就可以了。这时需要一个代价函数（cost function）来表示两者的差距。一种代价函数是基于欧式距离的，一种是分离度的，两个都有对应的迭代公式。  <!-- more -->
介绍NMF算法的文章已经有很多了，本文主要介绍一下非负矩阵分解在分析癌症突变异质性中的作用。数据来自于nature论文 ——Mutational heterogeneity in cancer and the search for new cancer-associated genes ——的附表S3，是对2892个癌症病人的96中突变类型的统计。论文分析的是3083个癌症病人的数据，而附表S3只提供了突变个数在10个以上的数据，只有2892个病人，因此不求能够得到和论文完全一致的结果。现在的数据是每个癌症样本在96种突变类型上的分布，通过非负矩阵分解，我们期望得到每个样本在不同的突变谱上的分布，所谓突变谱就是把几个突变类型综合在一起的形式，例如\*CpG岛处的突变代表了一些C参与的突变类型。下图为论文中找到的6个突变谱，包含了目前已知的突变过程。
<div class="justified-gallery">![legoplot-paper](/img/legoplot-paper.jpg)</div>
数据归一化：某个样本的某种类型的突变可能是由于这种类型的突变的测序覆盖度比较好导致的，因此需要对每种突变类型进行归一化处理。对每个样本s和突变类型c，定义\\({n\_{cs}}\\)为观察到的突变数，\\({N\_{cs}}\\)为具有足够覆盖度碱基数，则样本总的突变频率为$\mu \_s=\sum \_c{n\_{cs}} / \sum \_c{N\_{cs}}$，每种突变的相对突变率$R\_{cs}=\(n\_{cs}/N\_{cs}\)/\mu\_s$，接着可以对$R\_{cs}$矩阵进行非负矩阵分解。论文中并没有提供每个样本的每种突变类型的碱基数，只给了所有样本的平均值，所以这次计算用了这种并不合适的数据。  
R code:  
``` R
coverage <- read.table('coverage.txt',header=F)[-1]  
mutation <- read.csv('table_s3.csv',header=TRUE, row.names='name',stringsAsFactors=FALSE)  
mutation.spe <- as.matrix(mutation[,2:97])  
mutation.spectrum <- matrix(0,nrow=2892,ncol=96,dimnames=dimnames(mutation.spe))  
samlength=rep(0,2892)  
for (i in 1:nrow(mutation.spe)){  
  mius=sum(mutation.spe[i,])/sum(coverage)  
  samlength[i]=mius*1000000  
  mutation.spectrum[i,]=as.matrix((mutation.spe[i,]/coverage)/mius)  
}
```
使用R包NMF进行矩阵分解，rank=6，nrun=50
```R
library(NMF)
spectrum.fit <- nmf(mutation.spectrum, 6, nrun=50)
spectrum.coef <- coef(spectrum.fit)
#spectrum.basis <- t(spectrum.basis)

basematrix <- function(factor,base1,base2){
  base <- rep(0,16)
  index=1
  for (i in c('T','C','A','G')){
    for (j in c('T','C','A','G')){
      name=paste0(i,base1,j,'to',i,base2,j)
      base[index]=factor[name]
      index=index+1
    }
  }
  return(matrix(base,nrow=4,byrow=T))     
}

combinebase <- function(index){
  CG = basematrix(spectrum.coef[index,],'C','G')
  CA = basematrix(spectrum.coef[index,],'C','A')
  CT = basematrix(spectrum.coef[index,],'C','T')
  AT = basematrix(spectrum.coef[index,],'A','T')
  AC = basematrix(spectrum.coef[index,],'A','C')
  AG = basematrix(spectrum.coef[index,],'A','G')
  return(cbind(rbind(CG,AT),rbind(CA,AC),rbind(CT,AG)))
}


par(mfrow=c(2,3),mar=c(2,1,1,2))

#plot
for (i in 1:6){
  data = combinebase(i)
  
  data=data/max(data)
  
  # generate 'empty' persp plot
  pmat = persp(x=c(0,10), y=c(0,10), z=matrix(c(0,.001,0,.001), nrow=2), 
               xlim=c(0,10), ylim=c(0,10), zlim=c(0,1.5), 
               theta=60, phi=25, d=5, box=F,border=NA) 
  
  # define color ramp
  colorCG = matrix(rep('#FF0000',16),nrow=4)
  colorAT = matrix(rep('#7F4CB2',16),nrow=4)
  colorCA = matrix(rep('#00B2B2',16),nrow=4)
  colorAC = matrix(rep('#0033CC',16),nrow=4)
  colorCT = matrix(rep('#FFFF00',16),nrow=4)
  colorAG = matrix(rep('#19CC19',16),nrow=4)
  
  colorM = cbind(rbind(colorCG,colorAT),rbind(colorCA,colorAC),rbind(colorCT,colorAG))
  
  # draw each bar: from left to right ...
  for (i in 1:nrow(data)){
    
    # ... and back to front 
    for (j in ncol(data):1){
      
      xy = which(data == data[i,j], arr.ind=TRUE)
      
      # side facing y
      x = rep(xy[1]-0.1,4)
      y = c(xy[2]-0.9,xy[2]-0.1,xy[2]-0.1,xy[2]-0.9)
      z = c(0,0,data[i,j],data[i,j])
      polygon(trans3d(x, y, z, pmat), col=colorM[i,j], border=1)
      
      #  side facing x
      x = c(xy[1]-0.9,xy[1]-0.1,xy[1]-0.1,xy[1]-0.9)
      y = rep(xy[2]-0.9,4)
      z = c(0,0,data[i,j],data[i,j])
      polygon(trans3d(x, y, z, pmat), col=colorM[i,j], border=1)
      
      # top side
      x = c(xy[1]-0.9,xy[1]-0.1,xy[1]-0.1,xy[1]-0.9)
      y = c(xy[2]-0.9,xy[2]-0.9,xy[2]-0.1,xy[2]-0.1)
      z = rep(data[i,j],4)
      polygon(trans3d(x, y, z, pmat), col=colorM[i,j], border=1)
      
    }
  }
}
```
<div class="justified-gallery">![legoplot](/img/legoplot.png)</div>

从上图可以看出算法找到的突变谱分别为Tp\*C->mut, Tp\*A->T, C->A, misc, misc, *CpG->T。最后查看每种类型的癌症在这六种突变谱上的分布，即确认是否有的癌症只具有一种突变谱还是多种，可以通过绘制热图来查看。这里使用论文的可视化方法，通过绘制径向图查看。R绘图包plotrix中的radial.plot函数需要提供两个参数，即径向的长度和角度（以弧度为单位）。这里用每个样本的总突变率为长度。对于每个样本s，使$i\_{sr}$表示在六个突变谱中第r-th个最大的权重，角度可以通过这个公式计算$\alpha\_s=2\pi\sum \_{r=0}^K{i\_r\(1/K\)^r}$。第一个最大的权重决定样本在整个圆周的哪个扇区，第二大权重决定在上个扇区中的小扇区，以此推导。  
R code:  
```R
spectrum.sample <- basis(spectrum.fit)
samlength=rep(0,2892)
radial=rep(0,2892)
for (i in 1:2892){
  ir=order(spectrum.sample[i,1:6],decreasing = T)
  radial[i]=2*pi*sum(ir*(1/6)**(1:6))
}
colorlist <- list('AML'='#12A6DA','Bladder'='#E5E515','Breast'='#F37F81','CLL'='#A94399','Colorectal'='#0F9B5A',
'Carcinoid'='#6F2011','Cervical'='#F58020','DLBCL'='#2C2C81',
'Oesophageal adenocarcinoma'='#5DBB46','Ewing sarcoma'='#664182',
'Glioblastoma multiforme'='#8B715D','Head and neck'='#364EA1',
'Kidney clear cell'='#543C1C','Kidney papillary cell'='#302636','Low-grade glioma'='#8D5967',
'Lung adenocarcinoma'='#B12124','Lung squamous cell carcinoma'='#ED1F24',
'Multiple myeloma'='#808233','Medulloblastoma'='#16304A','Melanoma'='#231F20',
'Neuroblastoma'='#39276B','Ovarian'='#CC80B4','Pancreas'='#543C1C','Prostate'='#B3B48C',
'Rhabdoid tumor'='#9999A6','Stomach'='#69BD45','Thyroid'='#A1A6D2')
tumor_type=table(mutation[,1])
collevel=rep('',27)
for (i in 1:length(colorlist)){
  collevel[i]=colorlist[[i]]
}
colorpoint=rep('',2892)
loc=1
for (i in 1:27){
  #tumor=names(tumor_type[i])
  colorpoint[loc:(loc+tumor_type[i]-1)]=rep(colorlist[[i]],tumor_type[i])
  loc=loc+tumor_type[i]
}
library(plotrix)
radial.plot(log(samlength,10),radial,rp.type="s",point.col=colorpoint,point.symbols=20,
            show.grid=T,grid.col="white",show.grid.labels=0,
            radial.lim=c(log10(0.04),log10(100)),
            label.pos=c(pi/2,5*pi/6,7*pi/6,3*pi/2,11*pi/6,pi/6),
            labels=c('Tp*C->mut','Tp*A->T','C->A','misc','misc','*CpG->T'))
radial.plot(rep(2,6),c(0,pi/3,pi*2/3,pi,pi*4/3,pi*5/3),rp.type='r',
            line.col = 'grey',lwd=2,add=T,radial.lim=c(log10(0.04),log10(100)))
```
<div class="justified-gallery">![radial plot](/img/radial plot.png)</div>
下面绘制图例  
```R
plot(1:30,1:30,bty='n',type='n',xaxt='n',yaxt='n',xlab='',ylab='')
legend(1,27,names(colorlist)[1:14],col=collevel[1:14],pch=19,pt.cex=1.5,bty='n')
legend(18,27,names(colorlist)[15:27],col=collevel[15:27],pch=19,pt.cex=1.5,bty='n')
```
<div class="justified-gallery">![legend](/img/legend.png)</div>
从图中可以看到一些有意义的结果，例如AML主要分布在Tp*A->T的突变，而肺癌主要是C->A的突变。然后由于数据不完整，归一化做的比较，最终得到的结果也不是太好，并没有很好地解释癌症突变谱异质性。
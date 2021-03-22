---
title: Spatial Partition
date: 2021-03-10 15:30:49
tags: Optimization Pattern
---
根据空间位置把对象放进特定的空间数据结构来高效定位对象。

* 分割模式
    * 平坦分割，均匀分成cells
    * 分层分割，递归地分成regions，使得每个region包含的对象个数不超过一个阈值。

<!-- More -->

* 边界选择
    * 边界固定，选中间位置 - Grid, Quadtree, Octree
    * 根据对象个数和位置选择分割平面 - BSP, k-d tree, bounding volume hierarchy
        * 确保平衡分割，获得稳定的帧率
        * 做场景剔除算法，高效渲染 - Unity, UE, Quake3
    * 边界固定，分层依赖于对象 - Quadtree, Octree

* Essentials
    * Grid - Persisitent <font color="blue">bucket sort</font>
    * BSP, k-d tree, bounding volume hierarchy - <font color="blue">Binary search tree</font>
    * Quadtree, Octree - <font color=blue>Tries</font>



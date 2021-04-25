---
title: Physically Based Shading
date: 2021-04-23 16:41:39
tags: Shader
---
# 渲染方程
$$L_{o}(p, \omega_{o}) = \int_{\Omega}f(p, \omega_{o}, \omega_{i})L_{i}(p, \omega_{i})|\cos\theta_{i}|\mathrm{d}\omega_{i}$$

$L_{o}(p, \omega_{o})$, $L_{i}(p, \omega_{i})$ is radiance.
$f(p, \omega_{o}, \omega_{i})$ is BRDF.
$\cos\theta_{i}$ is the dot between normal and light direction.
<!-- more -->
基于物理的着色算法就是计算视线方向的Radiance.
Rayleigh scattering - 粒子直径远小于光的波长
Mie scattering - 粒子直径与波长相近
Geometry scattering - 粒子直径远大于波长
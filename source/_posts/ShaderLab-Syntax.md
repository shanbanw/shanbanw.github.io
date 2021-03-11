---
title: ShaderLab Syntax
date: 2021-03-11 14:09:20
tags: Unity SRP
---

# Syntax
```HLSL
Shader "name" { [Properties] Subshaders [Fallback] [CustomEditor] }
```
[]代表可选

<!-- more -->

# Properties
```HLSL
Properties { Property [Property] ... }
```
#### Numbers and Sliders
```HLSL
name ("display name", Range (min, max)) = number
name ("display name", Float) = number
name ("display name", Int) = number
```
#### Colors and Vectors
```HLSL
name ("display name", Color) = (number, number, number, number)
name ("display name", Vector) = (number, number, number, number)
```
#### Textures
```HLSL
name ("display name", 2D) = "defaulttexture" {}
name ("display name", Cube) = "defaulttexture" {}
name ("display name", 3D) = "defaulttexture" {}
```

#### 说明
* unity中shader property name一般以_开头。
* 2D Textures - 默认值可以是空字符串，或是built-in 默认贴图: "white", "black", "gray", "bump" (RGBA: 0.5, 0.5, 1, 0.5) and "red" (RGBA: 1, 0, 0, 0).
* Non-2D Textures (Cube, 3D, 2DArray) - 默认值是空字符串。

Properties里面的参数是和材质数据一起序列化的，外面的参数不会被保存，用于runtime脚本控制(such as Material.SetFloat)

#### Property attributes and drawers
* [HideInInspector] - 不出现在material inspector
* [NoScaleOffset] - 不显示Texture tiling/offset
* [Normal] - normal map
* [HDR] - high-dynamic range (HDR) texture
* [Gamma] 
* [PerRendererData] - come from pre-renderer data, read-only
* [MainTexture] - unity默认使用_MainTex作为main texture,如果有多个，unity使用第一个。
* [MainColor] - unity默认使用_Color作为main color,如果有多个，unity使用第一个。



# SubShaders & Fallback
每个shader可以有多个subshaders. 当Unity加载shader时，会从subshaders中找到第一个机器支持的subshader，如果没有，就会使用fall back shader.
```HLSL
Subshader { [Tags] [CommonState] Passdef [Passdef] ...}
```

#### Any statements that are allowed in a Pass definition can also appear in Subshader block. This will make all passes use this “shared” state.


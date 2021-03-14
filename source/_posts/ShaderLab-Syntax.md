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

#### SubShader Tags - to tell how and when they expect to be rendered to the rendering engine.
```HLSL
Tags { "TagName1" = "Value1" "TagName2" = "Value2" }
```
以下tags必须放在subshader中， 不能Pass中
* #### Queue - Rendering Order
    * Background - 1000
    * Geometry - 2000
    * AlphaTest - 2450
    * Transparent - 3000
    * Overlay - 4000

```HLSL
Tags { "Queue" = "Geometry+1" }
```

* #### DisableBatching - draw call batching transforms all geometry into world space
    * True - always disables batching for this shader
    * False - does not disable batching; default
    * LODFading - disable batching when LOD fading is active; mostly used on trees

* #### ForceNoShadowCasting
    * True - an object that is rendered using this subshader will never cast shadows.

* #### IgnoreProjector
    * True - will not be effected by Projectors.

* #### CanUseSpriteAltas
    * False - meant for sprites, not work when they are packed into atlases

* #### PreviewType - how the material inspector preview should display the material.
    * Sphere - default
    * Plane - 2D
    * Skybox - skybox

* #### RenderType - Categorize shaders into several perdefined groups, used by shader replacement.
    * Opaque
    * Transparent
    * TransparentCutout - masked transparency shaders
    * Background - skybox
    * Overlay - Halo, flare shaders
    * TreeOpaque - terrain engine tree bark
    * TreeTransparentCutout - terrain engine billboarded trees
    * Grass - terrain engine grass
    * GrassBillboard - terrain engine billboarded grass

#### Any statements that are allowed in a Pass definition can also appear in Subshader block. This will make all passes use this “shared” state.

* Three Type of Pass
    * Regular Pass
    * Use Pass
    * Grab Pass

#### Regular Pass
每个pass渲染一次geometry
```HLSL
Pass { [Name and Tags] [RenderSetup] }
```
#### Name
```HLSL
Name "PASSNAME"
```
定义一个名字，一般用大写，用于UsePass
#### Tags
```HLSL
Tags { "TagName1" = "Value1" "TagName2" = "Value2" }
```
用于确定渲染方式和渲染时间
以下Tags必须放在Pass中，不能放在SubShader中
* #### LightMode
    * Always - Always rendered; no lighting is applied.
    * ForwardBase - Used in Forward rendering, ambient, main directional light, vertex/SH lights and lightmaps are applied.
    * ForwardAdd - Used in Forward rendering; additive per-pixel lights are applied, one pass per light.
    * Deferred - Used in Deferred Shading; renders g-buffer.
    * ShadowCaster - Renders object depth into shadowmap or a depth texture.
    * MotionVectors - Used to calculate per-object motion vectors.

* #### PassFlags - 指定rendering pipeline如何给pass传递数据
    * OnlyDirectional - only the main directional light and ambient/lightprobe data is passed into shader.

* #### RequireOptions - 指定满足某些条件时才渲染
    * SoftVegetation - Render this pass only if Soft Vegetation is on in the quality window.

#### Render State Set-up
```HLSL
// Set polygon culling mode
Cull Back | Front | Off
// Set depth buffer testing mode, default (LEqual)
ZTest (Less | Greater | LEqual | GEqual | Equal | NotEqual | Always)
// Set depth buffer writing mode, default (On)
ZWrite On | Off
// Set Z buffer depth offset, Factor scales the maximum Z slope, with respect to X or Y of the polygon; units scale the minimum resolvable depth buffer value.
Offset Factor, Units
// Sets alpha blending, alpha operation, and alpha-to-coverage modes
Blend sourceBlendMode destBlendMode
Blend sourceBlendMode destBlendMode, alphaSourceBlendMode alphaDestBlendMode
BlendOp colorOp
BlendOp colorOp, alphaOp
AlphaToMask On | Off
// Set conservative rasterization on and off
Conservative True | False
// Set color channel writing mask. Writing ColorMask 0 turns off rendering to all color channels. Default mode is writing to all channels (RGBA), but for some special effects you might want to leave certain channels unmodified, or disable color writes completely.

When using multiple render target (MRT) rendering, it is possible to set up different color masks for each render target, by adding index (0–7) at the end. For example, ColorMask RGB 3 would make render target #3 write only to RGB channels.

ColorMask RGB | A | 0 | any combination of R, G, B, A
```
#### Debugging Normals
first we render the object with normal vertex lighting, then we render the backfaces in bright pink. This has the effects of highlighting anywhere your normals need to be flipped.
```HLSL
Shader "Reveal Backfaces" {
    Properties {
        _MainTex ("Base (RGB)", 2D) = "white" { }
    }
    SubShader {
        // Render the front-facing parts of the object.
        // We use a simple white material, and apply the main texture.
        Pass {
            Material {
                Diffuse (1,1,1,1)
            }
            Lighting On
            SetTexture [_MainTex] {
                Combine Primary * Texture
            }
        }

        // Now we render the back-facing triangles in the most
        // irritating color in the world: BRIGHT PINK!
        Pass {
            Color (1,0,1,1)
            Cull Front
        }
    }
}
```

#### 
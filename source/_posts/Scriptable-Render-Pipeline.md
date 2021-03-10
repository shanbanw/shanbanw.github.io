---
title: Scriptable Render Pipeline
date: 2021-03-10 16:57:02
tags: Unity SRP
---
渲染管线是把场景物体显示在屏幕上的一系列技术的总称。简单来说，包含以下三大项

   *  ## Culling
   +  ## Rendering Objects
   -  ## Post processing

## SRP - schedule and configure rendering commands using C# scripts.
<!-- more -->

* 每个Custom SRP有两个关键的文件
    * Render Pipeline Asset - 存储configuration data
    * Render Pipeline Instance - 继承RenderPipeline的类，其Render函数为SRP入口

渲染从调用Render函数开始
```CSharp
public class BasicPipeInstance : RenderPipeline
{
   public override void Render(ScriptableRenderContext context, Camera[] cameras)
   {

   }
}
```
**ScriptableRenderContext**用来存放和执行设置的渲染命令队列。
```CSharp
// Create a new command buffer that can be used
// to issue commands to the render context
var cmd = new CommandBuffer();
 
// issue a clear render target command
cmd.ClearRenderTarget(true, false, Color.green);
 
// queue the command buffer
context.ExecuteCommandBuffer(cmd);
```

* ## Culling
    * Frustum culling
    * Occlusion culling

渲染开始，以相机为单位进行剔除造作，返回能被渲染的物体和灯光。
```CSharp
// Create an structure to hold the culling paramaters
ScriptableCullingParameters cullingParams;
 
//Populate the culling paramaters from the camera
if (!CullResults.GetCullingParameters(camera, stereoEnabled, out cullingParams))
    continue;
 
// if you like you can modify the culling paramaters here
cullingParams.isOrthographic = true;
 
// Create a structure to hold the cull results
CullResults cullResults = new CullResults();
 
// Perform the culling operation
CullResults.Cull(ref cullingParams, context, ref cullResults);
```

* ## Drawing
    * HDR vs LDR
    * Linear vs Gamma
    * MSAA vs Post Process AA
    * PBR Materials vs Simple Materials
    * Lighting vs No Lighting
    * Lighting Technique
    * Shadowing Technique

## Render Queue
用来确定渲染顺序，unity定义的有5个
* Background - 最早渲染的, index: 1000
* Geometry - 不透明物体, index: 2000
* AlphaTest - alpha tested geometry 比不透明的晚, index: 2450
* Transparent - 透明物体, index: 3000
* Overlay - 最后渲染的overlay effects, index: 4000

## Filtering: Render Buckets and Layers
筛选渲染哪些队列和layers
```CSharp
// Get the opaque rendering filter settings
var opaqueRange = new FilterRenderersSettings();
 
//Set the range to be the opaque queues
opaqueRange.renderQueueRange = new RenderQueueRange()
{
    min = 0,
    max = (int)UnityEngine.Rendering.RenderQueue.GeometryLast,
};
 
//Include all layers
opaqueRange.layerMask = ~0;
```
## Draw Setting: How things should be drawn
控制渲染方式
* Sorting - 以什么顺序渲染, back to front, or, front to back
* Per Render flags - pass from unity to shader
* Rendering flags - 选择batching算法，instancing or non-instanceing
* Shader Pass - 指定当前draw call用哪个shader pass

```CSharp
// Create the draw render settings
// note that it takes a shader pass name
var drs = new DrawRendererSettings(Camera.current, new ShaderPassName("Opaque"));

// enable instancing for the draw call
drs.flags = DrawRendererFlags.EnableInstancing;

// pass light probe and lightmap data to each renderer
drs.rendererConfiguration = RendererConfiguration.PerObjectLightProbe | RendererConfiguration.PerObjectLightmaps;

// sort the objects like normal opaque objects
drs.sorting.flags = SortFlags.CommonOpaque;
```

* # Draw Call
    * Cull results
    * Filtering rules
    * Drawing rules

draw call 需要放进context
```CSharp
// draw all of the renderers
context.DrawRenderers(cullResults.visibleRenderers, ref drs, opaqueRange);

// submit the context, this will execute all of the queued up commands.
context.Submit();
```

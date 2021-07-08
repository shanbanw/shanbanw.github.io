---
title: Asset
date: 2021-03-14 18:26:03
tags: Unity
---
## Asset Processing
Unity reads and processes any files that you add to the Assets folder, converting the contents of the file to internal game-ready data. The asset files themselves remain unchanged, and the internal data is stored in the project’s Library folder.
Unity stores the internal representation of your assets in the Library folder which is like a cache folder.

<!-- More -->

## AssetDatabase
开发阶段使用
AssetDatabase.LoadAssetAtPath()

## Resource Folders
Inclued in the built Unity Player.
所有的Assets和dependencies都会存进resources.assets里面。
加载使用 Resources.Load()
卸载使用 Resources.UnloadUnusedAssets()

## StreamingAssets
Hiden in Editor.
获取文件夹位置 - Application.streamingAssetsPath
可以用UnityWebRequest来读取

## AssetBundles
External collection of assets
减小安装包的大小
build - BuildPipeline.BuildAssetBundles()
UnityWebRequestAssetBundle使用disk cache
AssetBundle.LoadFromFile, AssetBundle.LoadFromFileAsync使用memory cache
load - AssetBundle.LoadAsset()
unload - AssetBundle.Unload() AssetBundle.UnloadAsync(bool)

## Addressable Asset System
[Unity Blog: Addressable打包策略](https://blogs.unity3d.com/2021/03/31/tales-from-the-optimization-trenches-saving-memory-with-addressables/)

Content catalog是一个序列化的ResourceLocationMap(IResourceLocator), 保存address与存放位置(ResourceLocationBase, IResourceLocation)的对象关系, 使用LoadContentCatalogAsync加载;
每个IResourceLocation里面保存了ResourceProvider的，用ResourceManager.GetResourceProvider获取到对应的provider;


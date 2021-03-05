---
title: Prototype Pattern
date: 2021-03-03 20:03:33
tags: Creational Design Pattern
---
从一个对象复制出一个拷贝
``` CSharp
using UnityEngine;

public class Spawner
{
    public static T Spawn<T>(T prototype) where T : Object
    {
        return Object.Instantiate(prototype);
    }
}
```

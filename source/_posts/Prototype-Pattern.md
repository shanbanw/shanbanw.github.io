---
title: Prototype Pattern
date: 2021-03-03 20:03:33
tags: Creational Design Pattern
---
从一个对象复制出一个拷贝
``` CSharp
using EngineEngine;

public class Spawner
{
    public T Spawn<T>(T prototype)
    {
        return Instantiate(prototype);
    }
}
```

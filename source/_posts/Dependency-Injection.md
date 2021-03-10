---
title: Dependency Injection
date: 2021-03-10 12:03:29
tags: Decoupling Pattern
---
通常一个类需要其他类的实例来完成特定的功能，此时不再自己实例化一个对象，而是通过函数参数接收一个实例达到解耦的目的。
当有很多依赖的实例的时候，可以用IoC(Inversion of Control) container来管理。
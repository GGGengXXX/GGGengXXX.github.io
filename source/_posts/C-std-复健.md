---
title: C++ std 复健
date: 2026-02-20 01:02:51
tags: 算法题
---

每经过一段时间会忘记 `C++` 语法，刷题的时候 `C++` 语法也不能使用 ai，所以打算做点笔记，总结一下常用的 `C++` 语法，主要是 `std`

## vector

- 如何初始化一个数组赋初值

```c++
vector<int> v = {1,2,3};
vector<int>v{1,2,3};
```

- 如何初始化一个数组，指定数组的大小

  - 一维数组

  - ```c++
    vector<int> a(5);
    ```

  - 二维数组

  - ```c++
    vector<vector<int>>matrix(rows, vector<int>(cols));
    ```


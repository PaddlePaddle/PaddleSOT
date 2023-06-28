# SideEffect

## 什么是 SideEffect

> 在计算机科学中，函数副作用（side effect）指当调用函数时，除了返回可能的函数值之外，还对主调用函数产生附加的影响。例如修改全局变量（函数外的变量），修改参数，向主调方的终端、管道输出字符或改变外部存储信息等。
>
> <p align="right">—— <a href="https://zh.wikipedia.org/wiki/%E5%89%AF%E4%BD%9C%E7%94%A8_(%E8%AE%A1%E7%AE%97%E6%9C%BA%E7%A7%91%E5%AD%A6)">WikiPedia 副作用_(计算机科学)</a></p>

如上是 WikiPedia 中 SideEffect 的定义，简单来说就是除去输出，还对外部环境产生了其他影响，比如

```python
def foo():
    print("Hello World")
```

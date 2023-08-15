# CALL 相关字节码适配

## CALL 相关字节码

函数调用主要涉及 LOAD 和 CALL 两类字节码，在 Python 3.10 及之前（以下简称 Python 3.10）和 Python 3.11 Python 生成的字节码发生了变化

在 Python 3.10，对于 function call 和 method call 会生成两种不同的 LOAD + CALL 字节码，而在 Python 3.11 将 CALL 进行了统一，具体如下

| code | `b(1)` | `a.b(1)` |
|-|-|-|
| 3.10 | `LOAD_GLOBAL` <br/> - <br/>`CALL_FUNCTION` | `LOAD_METHOD` <br/> - <br/>`CALL_METHOD` |
| 3.11 | `PUSH_NULL` <br/> `LOAD_GLOBAL` <br/> - <br/>`PRECALL` <br/> `CALL` | `LOAD_METHOD` <br/> - <br/>`PRECALL` <br/> `CALL` |

> **Note**
>
> - function call 指 `b(1)` 这种形式，method call 指 `a.b(1)` 这种形式，注意后者虽然是 method call，但 `a.b` 不一定是 method，也可能只是普通的 function，比如 `paddle.abs`，在编译时时我们无法知道它具体的类型，只是从语法结构上我们会认为其是 method call，要注意语法形式上的 function call 和 method call 以及运行时 function 和 method 的区别，后者区别见 [函数和方法](../../notes/function-and-method.md)
> - 实际使用 dis 在 Python 3.11 下 `b(1)` 的字节码会发现字节码是 `LOAD_GLOBAL  1 (NULL + b)`，而其实际上只是 `PUSH_NULL` + `LOAD_GLOBAL` 字节码序列经过一个 pass 优化后的结果（见 [cpython 3.11 compile.c - optimize_basic_block](https://github.com/python/cpython/blob/3.11/Python/compile.c#L9034-L9040)），实际上等价于 `PUSH_NULL` + `LOAD_GLOBAL`
> - 注意 `LOAD_GLOBAL` 只是其中一种 LOAD 指令而已，实际上该处可能是 `LOAD_FAST` 等指令

## Python 3.10 相关字节码的行为

在 Python 3.10，CALL 有两种，一种是 `CALL_FUNCTION`，简单来说就是把栈上的函数取出来直接 CALL，其往往会搭配 `LOAD_GLOBAL` 等 LOAD 指令

另一种是 `CALL_METHOD`，其往往会搭配 `LOAD_METHOD`，因为在运行时才能知道它具体是 function 还是 method，因此在 `LOAD_METHOD` 时候会根据情况来判断具体向栈上放什么元素，相关源码见 [cpython 3.10 ceval.c - LOAD_METHOD](https://github.com/python/cpython/blob/3.10/Python/ceval.c#L4122-L4157)，具体如下：

- 如果是 method，那么向栈上放

    ```
    meth | self | arg1 | ... | argN
    ```

- 如果是 function，那么向栈上放

    ```
    NULL | meth | arg1 | ... | argN
    ```

两者的栈布局是完全不同的

`CALL_METHOD` 时，则会根据栈的布局来判断这是一个 function 还是 method，相关源码见 [cpython 3.10 ceval.c - CALL_METHOD](https://github.com/python/cpython/blob/3.10/Python/ceval.c#L4159-L4207)

其实就是看 `-oparg-2` 位置是不是 `NULL` 而已，如果是就认为其是一个 function，否则认为其是一个 method

## Python 3.11 相关字节码的行为

Python 3.10 为两种语法形式生成了不同的 CALL 字节码，Python 3.11 则是将两者进行了统一，统一生成字节码 `PRECALL` + `CALL`，其实就是将 `CALL_METHOD` 拆成两部分，`PRECALL` 用于根据栈的布局来判断是 function 还是 method，如果是 function 布局，但其实际上是一个 method，就将其调整成 method 布局，之后 `CALL` 会进行函数调用，具体代码见 [cpython 3.11 ceval.c - PRECALL](https://github.com/python/cpython/blob/3.11/Python/ceval.c#L4657-L4701)

那么「如果是 function 布局，但其实际上是一个 method，就将其调整成 method 布局」是指什么呢？

对于 `LOAD_GLOBAL` + `PRECALL` + `CALL` 的 function call 布局，`LOAD_GLOBAL` 可能 `LOAD` 任何对象，当然可能其本身就已经是一个 method 了，比如函数作为一个参数传入

```python
def foo(method, x):
    method(x)
```

这里 method 是通过 `LOAD_FAST`（和 `LOAD_GLOBAL` 是同一类）LOAD 到栈上的，在 CALL 的时候其栈布局必然是 function call 的布局，但其实际上是一个 method，在这种情况下 `PRECALL` 便会调整其布局，将其变为 method call 布局

在 `PRECALL` 之后，通过栈的布局是否是 method call 布局就可以完全确定调用对象是否是 method 了，`CALL` 时对 method 对象的处理是统一的，即 `A.b(a, *args)`，在 Python 3.10 之前，通过 method call 形式 LOAD 到栈上的 method 同样是 `A.b(a, *args)` 调用的，而通过 function call 形式 LOAD 到栈上的 method（`a.b`）则是直接 `a.b(*args)` 调用的

不过 `LOAD_GLOBAL` 和 `LOAD_METHOD` 在处理 function 时是有一点差别的，就是 `LOAD_METHOD` 在处理 function 时会先 push 一个 `NULL` 到栈上，为了能够完全统一两者，在遇到 function call 形式时，编译时在生成 `LOAD_GLOBAL` 之前会先插入一条 `PUSH_NULL`，这样两者就一致了～

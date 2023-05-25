# Closure Implementation

## Closure Example in Python

以下是对一个闭包函数的处理的demo，以及它对应的字节码 :

```python
import dis

def func():
    free_x = 1
    free_y = 2

    def local(y):
        return y + free_x + free_y
    return local(1)

dis.dis(func)
```

```text
  4           0 LOAD_CONST               1 (1)
              2 STORE_DEREF              0 (free_x)

  5           4 LOAD_CONST               2 (2)
              6 STORE_DEREF              1 (free_y)

  7           8 LOAD_CLOSURE             0 (free_x)
             10 LOAD_CLOSURE             1 (free_y)
             12 BUILD_TUPLE              2
             14 LOAD_CONST               3 (<code object local at 0x1055f9240, file "tmp.py", line 7>)
             16 LOAD_CONST               4 ('func.<locals>.local')
             18 MAKE_FUNCTION            8 (closure)
             20 STORE_FAST               0 (local)

  9          22 LOAD_FAST                0 (local)
             24 LOAD_CONST               1 (1)
             26 CALL_FUNCTION            1
             28 RETURN_VALUE

Disassembly of <code object local at 0x1055f9240, file "tmp.py", line 7>:
  8           0 LOAD_FAST                0 (y)
              2 LOAD_DEREF               0 (free_x)
              4 BINARY_ADD
              6 LOAD_DEREF               1 (free_y)
              8 BINARY_ADD
             10 RETURN_VALUE

```

上述字节码可以先进行猜测：一切闭包都是通过额外的字节码进行构建的。

- STORE_DEREF : 将TOS存储到cell中

- LOAD_CLOSURE：将Cell构建为闭包

- LOAD_DEREF：将CELL中的值读取出来

## Closure Implementation Bytecode overview in Python

在Python3.8中，闭包是通过字节码实现的，与Closure相关的字节码有如下几个：

|     ByteCode    |   功能  |
|  -------------  | ------- |
|   LOAD_CLOSURE  |    N    |
|    LOAD_DEREF   |    N    |
| LOAD_CLASSDEREF |    N    |
|   STORE_DEREF   |    N    |
|   DELETE_DEREF  |    N    |


## Closure bytecode implementation in detail


```python
        case TARGET(LOAD_CLOSURE): {
            PyObject *cell = freevars[oparg];
            Py_INCREF(cell);
            PUSH(cell);
            DISPATCH();
        }
```


```python
        case TARGET(LOAD_DEREF): {
            PyObject *cell = freevars[oparg];
            PyObject *value = PyCell_GET(cell);
            if (value == NULL) {
                format_exc_unbound(tstate, co, oparg);
                goto error;
            }
            Py_INCREF(value);
            PUSH(value);
            DISPATCH();
        }
```

```python
        case TARGET(STORE_DEREF): {
            PyObject *v = POP();
            PyObject *cell = freevars[oparg];
            PyObject *oldobj = PyCell_GET(cell);
            PyCell_SET(cell, v);
            Py_XDECREF(oldobj);
            DISPATCH();
        }
```

## Conclusion：the implementation of python in detail。

对闭包进行总结，弄明白他的整体的工作方式。

首先是外层函数给inner函数准备闭包环境：CELLs，外层函数遇到 STORE_DEREF 指令就会将栈顶元素压入到freevars准备好的CELLS中。然后每个CELL中可以存储一个python obj，存储完毕之后会对old存储的对象进行减引用。

> Notes: freevars 的定义如下，指代 `freevars = f->f_localsplus + co->co_nlocals;`

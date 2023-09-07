# Closure 适配


## Python 中的闭包示例

以下是在新版本中闭包函数处理的demo，以及它对应的字节码 :

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

```bash
              0 MAKE_CELL                1 (free_x)
              2 MAKE_CELL                2 (free_y)

  9           4 RESUME                   0

 10           6 LOAD_CONST               1 (1)
              8 STORE_DEREF              1 (free_x)

 11          10 LOAD_CONST               2 (2)
             12 STORE_DEREF              2 (free_y)

 13          14 LOAD_CLOSURE             1 (free_x)
             16 LOAD_CLOSURE             2 (free_y)
             18 BUILD_TUPLE              2
             20 LOAD_CONST               3 (<code object local at 0x1022e0100, file "/Users/gouzi/Documents/git/paddle-symbolic-trace/tests/demo2.py", line 13>)
             22 MAKE_FUNCTION            8 (closure)
             24 STORE_FAST               0 (local)

 15          26 PUSH_NULL
             28 LOAD_FAST                0 (local)
             30 LOAD_CONST               1 (1)
             32 PRECALL                  1
             36 CALL                     1
             46 RETURN_VALUE

Disassembly of <code object local at 0x1022e0100, file "/Users/gouzi/Documents/git/paddle-symbolic-trace/tests/demo2.py", line 13>:
              0 COPY_FREE_VARS           2

 13           2 RESUME                   0

 14           4 LOAD_FAST                0 (y)
              6 LOAD_DEREF               1 (free_x)
              8 BINARY_OP                0 (+)
             12 LOAD_DEREF               2 (free_y)
             14 BINARY_OP                0 (+)
             18 RETURN_VALUE
```

## 新版本中对字节码的改动:

### 首先是语义上的改动

LOAD_CLOSURE: 新版本不再是`co_cellvars + co_freevars`长度偏移量, 而是`LOAD_FAST`的一个别名

LOAD_DEREF: 加载包含在 locals 中的元素

STORE_DEREF: 存储 TOS 到 locals 中

### 新增字节码

MAKE_CELL: 如果元素不存在于 locals 则从 co_freevars 和 co_cellvars 中加载

COPY_FREE_VARS: 复制 co_freevars 和 co_cellvars 中的元素到 locals

## 分析

从字节码上的改动来看，在 python3.11 中, 闭包将数据存储在 locals 中，而不是 cell 中，这样做的好处是可以减少一次间接寻址，提高性能。

## 实现

LOAD_CLOSURE: 作为`LOAD_FAST`的别名，所以直接调用

LOAD_DEREF: 改为从 `self._locals` 中加载元素到 TOS 中

STORE_DEREF: 改为存储 TOS 到 `self._locals` 中

MAKE_CELL: 从 `self._cells` 中加载元素到 `self._locals`

COPY_FREE_VARS: 从 `self._code.co_freevars` 拿到 key 在 `self._cells` 中找到元素存储到 `self._locals`

从实现上来看`MAKE_CELL` 和 `COPY_FREE_VARS` 的定位有点奇怪，因为它们都是从 `self._cells` 中加载元素, 并存储到 `self._locals` 中

## TODO: Fallback


## 单测

没有改动

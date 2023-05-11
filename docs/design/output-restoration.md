# 输出恢复机制

## 为什么需要恢复输出

比如对于一个需要转换的函数 `foo`，我们转换后的函数主要包含了以下几部分：

1. LOAD 编译后的 SIR 函数
2. LOAD SIR 函数的输入
3. CALL_FUNCTION，此时 SIR 函数的输出放在栈上
4. UNPACK_SEQUENCE 将 SIR 函数各个输出元素 unpack 到栈上
5. 循环遍历所有输出 Tensor，STORE 各个 Tensor 到 `f_locals`
6. LOAD `foo` 函数的输出
7. Side Effect 的处理（TODO）
8. RETURN_VALUE

在拥有输出恢复机制以前，我们是没有 4、5、6 步的，这就要求用户的 `foo` 函数只能返回 Tensor，这样 CALL_FUNCTION 的输出就是在栈上的，所以直接 RETURN_VALUE 就可以了。

但实际上用户函数是多种多样的，我们不能假设用户的输出一定是一个 Tensor，这就需要我们在 CALL_FUNCTION 之后，通过输出恢复机制来将输出恢复到栈上，之后再 RETURN_VALUE。

## 恢复输出的实现方式

### 输出恢复的出发点

输出恢复机制与输入恢复机制很相似，都是从一些源头开始逐渐构建得到的，有一点稍微不同的是，输入恢复是从 frame 的初始状态出发的，而输出恢复则相对于输入恢复多了一个从 SIR 输出结果出发的可能。

比如对于如下代码：

```python
def foo(x: paddle.Tensor):
    return x
```

这里的 `x` 是没有参加组网的，完全可以直接通过其 tracker 索引到并从 frame 中直接恢复。

而对于下面的代码：

```python
def foo(x: paddle.Tensor):
    return x + 1
```

这里的 `x` 参加了组网，所以这个 `x` 应该从 SIR 的输出中恢复。

这样对于输出恢复机制，相对于输入恢复机制，我们只需要额外实现一个从 SIR 输出结果恢复的机制即可。

### 从 SIR 输出结果出发的恢复方式

为了能够恢复从 SIR 输出结果中恢复，我们只需要在 CALL_FUNCTION 结束以后先将各个输出结果 Tensor STORE 到 `f_locals` 中，比如 `f_locals["___SIR_out_var_0"]`、`f_locals["___SIR_out_var_1"]`，这样之后需要恢复 SIR 中的输出的时候，只需要从 `f_locals` 里 LOAD 回来即可

### 容器类型的恢复

```python
def foo(x: paddle.Tensor, y: paddle.Tensor, z: int):
    a = x + 1
    b = z + 1
    l = [1, a, b, y]
    return l
```

最终输出是一个 `ListVariable`，对于这种情况，我们可以递归地处理：

- 循环 LOAD 各个元素到栈上
- BUILD_LIST

这样就可以得到最终的输出。

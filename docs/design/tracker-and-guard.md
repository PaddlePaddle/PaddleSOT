# Guard 收集机制

## 为什么需要 Guard？

在整个实现过程中，我们需要对用户函数的原始字节码转换成转换后的字节码，但如果每次运行都完整地转写全部字节码只会导致性能的浪费，也无法实现 JIT 的效果，因此我们需要一个缓存机制，来复用已经转换后的字节码。

但并不是说任何字节码成功转换一次后都是可以复用的，因为我们的字节码变换操作是通过模拟执行得到的，而模拟执行的起点是 Eval Frame 的初始状态，主要就是函数的输入，对于不同的输入，我们得到的字节码转换结果是可能不同的，因此我们需要有一个机制来判断转换后的字节码是否有效。

由于转换的过程与输入是强相关的，在函数 Eval Frame 初始阶段，我们可以从 `frame` 中拿到函数的输入，之后我们只需要通过 `guard` 来判断一个已经缓存的字节码是否有效即可，即 `guard(frame)`，如果结果是 `True`，则认为缓存命中。

guard 签名如下：

```python
Guard = Callable[[types.FrameType], bool]
```

## Guard 的收集机制

在模拟执行过程中，我们会根据字节码执行不同的操作，每一个字节码都会对应一个操作，如果我们将整个操作的链条构建起来，形成一个 DAG，就可以在任何时刻追踪到我们需要的 Guard。

我们使用 Tracker 来承载追踪操作的功能，Tracker 的 `inputs` 会持有其相关输入的 Variable，该 Tracker 将会由输出的 Variable 持有，相关数据结构如下：

```python
class VariableBase:
    tracker: Tracker

class Tracker:
    inputs: list[VariableBase]
```

比如对于如下的代码

```python
def foo(a: list[Tensor], b: int, c: int):
    d = a[b]
    e = d + c
    return e
```

最终构建的 Python 端 DAG 如下：

<p align="center">
    <img alt="Tracker" src="https://user-images.githubusercontent.com/38436475/237019099-a8e40aa6-5d0a-42d4-8330-ccee247835cb.png" width="500px"/>
</p>

有了 DAG 之后，我们只需要从需要的结点出发，找到全部需要的结点，并按照拓扑序收集一下即可～

## DummyTracker

上图中可以看到有 DummyTracker，而 DummyTracker 相关的路径也标成了虚线，那么什么情况需要 DummyTracker 呢？

对于 LocalTracker、GetItemTracker 来说，除去 Guard 的收集，有很重要的一点是，我们可以通过这些 Tracker 还原从 frame 初始状态出发，获取这些值的方法，这包括了如下两点：

- 在生成函数的字节码前，需要将输入 LOAD 到栈上，我们需要根据 Tracker 来生成 LOAD 这些输入的字节码
- 在调用 Guard 时，需要根据 Tracker 来索引到新的 Frame 里的相同变量的值，这样才能进行 Guard 的判断（`new_value == old_value`）

我们可以将这种索引机制称为 Source，而大多数中间结点是经过计算得到的，我们并不需要去还原这些中间结点，比如 `c = a + b`，`c` 是由 `BINARY_ADD` 构建得到的，我们的 Source 只需要分别索引 `a` 和 `b` 的来源，而我们的 Guard 也只需要分别 Guard 住 `a` 和 `b` 即可。

因此对于这种中间结点，我们只需要知道它是由什么构建得到即可，即只需要知道 inputs 是什么，对于这些结点，我们使用 DummyTracker 来作为连接结点，DummyTracker 不会承担 Source 的索引功能，只会承担 DAG 的连接功能，以便 Guard 的收集。

## Guard 收集的短路机制

对于如下的 case

```python
def foo(x):
    if x < 4:
        ...
    else:
        ...

foo(9)
foo(10)
```

如果我们的 Guard 收集机制是遍历全部结点的话，会强制 Guard 住 `x == 9`，所以第二次调用 `foo(10)` 时会 cache miss。

为了减少 cache miss 的概率，我们增加了一个短路机制，当一个 Tracker 的所有输入都不是 DummyTracker 时，可以认为从该 Tracker 上所获得的 Guard 会从其 inputs 所获得的更加精准，就不需要再从其 inputs 收集 Guard 了，可以大大降低重新编译的概率。

# 字符串化 Guard

## Guard 原设计及其遇到的问题

我们的原来的 Guard 设计是，每个子 Guard 都是一个 lambda 函数，签名如下：

```python
Guard = Callable[[types.FrameType], bool]
```

在 Guard 收集时，我们可以通过 `compose_guards` 来整合成一个总 Guard，这个总 Guard 也是一个 lambda 函数，它的实现非常简单，就是多个 Guard 的 `and` 串联：

```python
def compose_guards(guards: list[Guard]) -> Guard:
    def composed_guard_fn(frame: types.FrameType) -> bool:
        ret = True
        for guard in guards:
            ret = ret and guard(frame)
        return ret

    return composed_guard_fn
```

这个设计在正确性上没有太大问题，但是经测试发现，该设计会造成非常大的性能开销。我们分别测试了将总 Guard 直接设置为 `lambda _: True` 和每个 子 Guard 设置为 `lambda _: True`，后者比前者仅仅多了函数调用逻辑，不过我们发现后者仍然会比前者多出不少的性能开销，这很好理解，因为每个子 Guard 都需要一次函数调用，函数调用的开销是非常大的。

## 字符串化 Guard 的设计

为了避免 Guard 中的函数调用性能开销，我们将每个子 Guard 表示为字符串，在最后汇总的时候使用 `eval` 来生成一个 lambda 函数，这样最后的总 Guard 只是一个函数调用，没有了多个函数调用的开销。

为了能够让每个 Guard 都字符串化，Guard 中所使用的 `trace_value_from_frame` 也需要字符串化。

另外，对于 lambda 函数来说，可利用闭包来捕获自由变量，但字符串是没有这一能力的，为了能够让字符串化的 Guard 也能够捕获自由变量，我们需要在字符串化的同时，将自由变量一并保存下来，因此子 Guard 和 `trace_value_from_frame` 不仅会返回字符串，还会返回自由变量 dict。为了方便管理，将该数据结构命名为 `StringifyExpression`：

```python
@dataclass
class StringifyExpression:
    expr: str
    free_vars: dict[str, Any]
```

比如编写 BuiltinTracker 时，对比如下：

```diff
  class BuiltinTracker:
      def trace_value_from_frame(self):
-         return lambda frame: builtins.__dict__[self.name]
+         return StringifyExpression(
+             f"builtins.__dict__[{self.name}]", {"builtins": builtins}
+         )
```

这里 `builtins` 是一个本应通过闭包捕获的自由变量，在字符串化后，通过 `free_vars` 字段来保存。

Variable 和 Tracker 相关函数变化成如下：

```python
class VariableBase:
    def make_stringify_guard(self) -> StringifyExpression:
        ...

class Tracker:
    def trace_value_from_frame(self) -> StringifyExpression:
        ...
```

最终合并后的总 Guard 签名不变，仍然是 `Guard`：

```python
Guard = Callable[[types.FrameType], bool]

def make_guard(stringify_guards: list[StringifyExpression]) -> Guard:
    free_vars = union_free_vars(
        *[guard.free_vars for guard in stringify_guards]
    )
    num_guards = len(stringify_guards)
    if not num_guards:
        return lambda frame: True
    guard_string = f"lambda frame: {' and '.join([guard.expr for guard in stringify_guards])}"
    guard = eval(
        guard_string,
        free_vars,
    )
    log(3, f"[Guard]: {guard_string}\n")
    assert callable(guard), "guard must be callable."

    return guard
```

实现也很简单，就是字符串上的 `and` 拼接，之后 `eval` 并传入自由变量即可。

## 字符串化 Guard 书写的注意点

1. 应注意捕获自由变量，字符串无法自动捕获自由变量
2. 应注意字符串化前后的比较逻辑可能有所不同
3. 应注意尽可能将计算在「编译时」就计算好，编码在 Guard 字符串中，而不是传到运行时再进行计算，尽可能降低运行时开销

# Builtin 函数派发机制

## 什么是 BuiltinVariable？

什么是 BuiltinVariable 呢？最开始我们以为 BuiltinVariable 应该就是 builtin 这个 namespace 里的各种变量，比如 `int`、`abs` 等等这些不需要 import 就可以直接使用的变量，但是实际上这一角色已经由 BuiltinTracker 承担了，BuiltinVariable 就和 BuiltinTracker 定位重复了。

对于现有的其他 CallableVariable，每一个 Variable 的定位都很清晰，实现方式也很清晰，比如以 UserDefinedFunctionVariable 为代表的 inline call 方式、以 PaddleApiVariable 为代表的组网方式、以及部分需要子图打断的 API。但是 BuiltinVariable 不是这样的，对于大多数 builtin 函数，在 Python 执行时会去调用对应的 magic method，比如 `len` 会调用 `__len__`，此时执行的效果是与变量类型强相关的，比如用户重载了一个自定义对象的 `__len__`，此时应该去尝试 inline call，因为用户的代码中包含各种各样的情况，而部分对象的 `__len__` 不是写在 Python 端的，此时我们应该去模拟其执行效果。

另外值得注意的是，Python 会认为不在 Python 端定义的、没有字节码的函数的类型为 `types.BuiltinFunctionType`，这样 BuiltinVariable 的定位已经很清晰了，即**没有字节码，无法通过字节码来模拟执行的函数**。

## 为什么需要派发机制

如果无法直接模拟字节码的话，我们在模拟执行时要怎么做呢？起初我们直接利用 Python 的派发机制，在 Variable 上重载 magic method，实现自动派发：

```python
class BuiltinVariable:
    def call_function(self, *args: VariableBase, **kwargs: VariableBase):
        self.value(*args, **kwargs)
```

但这样问题很明显，比如其实 Python 对部分 magic method 输出是有强制类型检查的，比如 `__bool__` 强制是 bool 类型、`__len__` 强制是 int 类型。而按照上述的实现，我们应该返回的是 VariableBase，这就导致部分 magic method 是无法复用 Python 内建的派发机制的。另一方面，也因为很多 Variable 还没有实现相应的 magic method 而报错。

为了避免这些问题，我们添加了一个类似 magic method 的派发机制，来将 BuiltinVariable 的调用转发某一个具体的函数上，这个派发机制会去尝试匹配参数的类型，如果找不到匹配的参数类型，就会尝试从类上获取相应的 magic method 来 inline call，如果仍然找不到，则会产生一个 BreakGraphError 来打断子图。

## 派发机制的实现和使用方式

派发机制的本质是对参数类型进行匹配，对于一个函数，我们会有多种允许的类型签名（Pattern）以及对应的 handler，而在调用这个函数的时候，我们会根据参数类型来派发到相应的 handler 上，主体代码如下：

```python
class Pattern:
    type_strings: Args[str]
    kwtype_strings: Kwargs[str]

    def __init__(
        self,
        *types: str,
        **kwtypes: str,
    ):
        self.type_strings = types
        self.kwtype_strings = kwtypes

class Dispatcher:
    handlers: dict[
        Callable[..., Any], list[tuple[Pattern, Callable[..., Any]]]
    ] = {}

    @classmethod
    def register(
        cls,
        fn: Callable[..., Any],
        types: tuple[str, ...],
        kwtypes: dict[str, str],
        handler: Callable[..., Any],
    ):
        if fn not in cls.handlers:
            cls.handlers[fn] = []
        cls.handlers[fn].append((Pattern(*types, **kwtypes), handler))

    @classmethod
    def dispatch(
        cls, fn: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Callable[..., Any] | None:
        if fn not in cls.handlers:
            return None
        for pattern, handler in cls.handlers[fn]:
            if pattern.match_inputs(*args, **kwargs):
                return handler
        return None
```

这样，我们只需要调用 Dispatcher 将函数签名和 handler 注册到 Dispatcher 上即可：

```python
Dispatcher.register(
    len,
    ("ContainerVariable",),
    {},
    lambda var: var.len(),
)
```

为了方便使用，我们还提供了一个装饰器的模式：


```python
if TYPE_CHECKING:
    from .variables import ContainerVariable

@Dispatcher.register_decorator(len)
def dispatch_len(var: ContainerVariable):
    return var.len()
```

对于一些复杂的函数，是比较推荐装饰器模式的。

## 利用派发机制简化现有代码

在实现派发机制之前，我们在很多地方会有重复的代码，比如 Python 的 `a.b` 和 `getattr(a, "b")` 是等价的，但是字节码层面是完全不同的，前者是 `LOAD_ATTR`，后者则是 `CALL_FUNCTION`，此前我们也是各自实现的。

在实现了派发机制之后，我们完全可以利用派发机制来实现 `LOAD_ATTR`：

```python
    @call_break_graph_decorator(push_n=1)
    def LOAD_ATTR(self, instr):
        attr_name = instr.argval
        obj = self.pop()
        self.push(
            BuiltinVariable(
                getattr, graph=self._graph, tracker=DanglingTracker()
            )(obj, attr_name)
        )
```

这样可以极大的简化代码，降低维护成本。

> **Note**
>
> 我们现在代码中仍然有很多地方使用了旧的派发机制（利用 Python magic method 直接派发），这些将在之后逐步替换

# SideEffect

## 什么是 SideEffect

> 在计算机科学中，函数副作用（side effect）指当调用函数时，除了返回可能的函数值之外，还对主调用函数产生附加的影响。例如修改全局变量（函数外的变量），修改参数，向主调方的终端、管道输出字符或改变外部存储信息等。
>
> <p align="right">—— <a href="https://zh.wikipedia.org/wiki/%E5%89%AF%E4%BD%9C%E7%94%A8_(%E8%AE%A1%E7%AE%97%E6%9C%BA%E7%A7%91%E5%AD%A6)">WikiPedia 副作用_(计算机科学)</a></p>

如上是 WikiPedia 中 SideEffect 的定义，简单来说就是除去输出，还对外部环境产生了其他影响，比如

修改全局变量：

```python
global_a = 0

def update_global_a():
    global global_a
    global_a += 1

print(global_a)
update_global_a()
print(global_a)
```

修改输入的可变数据：

```python
def update_mutatable_data(x):
    x.append(0)

x = [0]
print(x)
update_mutatable_data(x)
print(x)
```

输出到标准输出：

```python
def write_stdout():
    print("side effect")

write_stdout()
write_stdout()
```

这些函数都不是纯函数（即没有副作用的函数），对于这些函数，我们会采取两种策略：

- 打断子图 / Fallback，让该段代码在动态图环境下执行，如 `print`
- 记录副作用，并在生成的代码里进行恢复，如上面的全局变量、可变数据的修改

## SideEffect 的恢复所需要的信息

### list、dict 的恢复

对于 list、dict 而言，我们分别可以通过如下方式来进行一次性地副作用恢复：

```python
old_list[:] = new_list

old_dict.clear()
old_dict.update(new_dict)
```

这分别需要拿到这个 list/dict 的原始引用，即 traceable 的 Variable 从 frame 里引用到的变量名，以及 list/dict 更新后的值。我们可以分别通过 `reconstruct` 和 `_reconstruct` 来完成这两个操作。

### 自定义对象的恢复

对于对象而言，我们当然没有一种方法类似 list、dict 那样直接一次性恢复，我们需要记录每一次属性变化，并按照变化的顺序进行恢复。

## 含 SideEffect 的 Variable 模拟

在模拟执行过程中，我们不应该对用户传入的对象进行修改，如果用户的代码中包含 SideEffect，我们应该使用记录操作替代直接操作。为此，我们设计了一个数据中间层（或者说代理）MutableData，该中间层持有原始数据，在对数据进行修改时，会记录下修改，而在对数据进行读取时，会返回修改后的数据。

```python
class MutableData:
    """
    An intermediate data structure between data and variable, it records all the mutations.
    """

    class Empty:
        def __repr__(self):
            return "Empty()"

    def __init__(self, data: Any, getter: DataGetter):
        self.original_data = data
        self.getter = getter
        self.records = []

    def is_empty(self, value):
        return isinstance(value, MutableData.Empty)

    @property
    def version(self):
        return len(self.records)

    @property
    def has_changed(self):
        return self.version != 0

    def rollback(self, version: int):
        assert version <= self.version
        self.records[:] = self.records[:version]

    def get(self, key):
        raise NotImplementedError()

    def set(self, key, value):
        raise NotImplementedError()
```

MutableData 具有版本（version）的概念，每当发生一次修改操作时，版本号会 +1，这样我们就可以在任何时刻通过版本号来 rollback 到想要的状态（比如 inline call 失败时所需要的回滚）。

由于 list 的模拟强依赖于顺序关系，因此 MutableData 主要分为 MutableListLikeData 和 MutableDictLikeData 两种模式。

## SideEffect 的恢复方式

以 dict 为例，按照下面代码的设计：

```python
old_dict.clear()
old_dict.update(new_dict)
```

我们可以写出如下的实现方式：

```python
def restore_side_effects(self, variables: list[VariableBase]):
    for var in variables:
        # skip inner variables
        if not var.tracker.is_traceable():
            continue
        if isinstance(var, DictVariable):
            # old_dict.clear()
            var.reconstruct(self.pycode_gen)
            self.pycode_gen.gen_load_method("clear")
            self.pycode_gen.gen_call_method(0)
            self.pycode_gen.gen_pop_top()

            # old_dict.update(new_dict)
            var.reconstruct(self.pycode_gen)
            self.pycode_gen.gen_load_method("update")
            var._reconstruct(self.pycode_gen)
            self.pycode_gen.gen_call_method(1)
            self.pycode_gen.gen_pop_top()
```

看起来很合理，但是实际执行时会发现是有问题的，比如

```python
def foo(x):
    x[0] = 42

foo({0: 1, 2: 3})
```

这里 `1` 和 `3` 分别通过 `x[0]` 和 `x[2]` 得到，我们根据 Tracker 生成的代码中类似：

```python
x.clear()
x.update({0: 42, 2: x[2]})
```

这里很明显，在 `update` 的时候 `x` 已经是 `{}` 了，因此 `x[2]` 会报 `KeyError`。

这仅仅是单个 Variable 自身因为变化而导致之后不再能引用到原来的值，假如不同发生 SideEffect 的 Variable 之间（即 for 循环中不同的 `var` 之间）存在依赖关系，那么这种恢复方式就会出现更多问题。

为了确保在 LOAD 需要的变量的时候能够得到需要的值，在此时我们不能进行任何 SideEffect 的应用，我们需要确保所有需要的变量都 LOAD 到栈上之后，再应用 SideEffect，对应代码如下：

```python
def restore_side_effects(self, variables: list[VariableBase]):
    if not variables:
        return

    var = variables[0]
    # skip inner variables
    if not var.tracker.is_traceable():
        self.restore_side_effects(variables[1:])
        return
    if isinstance(var, DictVariable):
        # old_dict.clear()
        # old_dict.update(new_dict)

        # Reference to the original dict.
        # load old_dict.update and new_dict to stack.
        var.reconstruct(self.pycode_gen)
        self.pycode_gen.gen_load_method("update")
        # Generate dict by each key-value pair.
        var._reconstruct(self.pycode_gen)
        # load old_dict.clear to stack.
        var.reconstruct(self.pycode_gen)
        self.pycode_gen.gen_load_method("clear")

        # Generate side effects of other variables.
        self.restore_side_effects(variables[1:])

        # Call methods to apply side effects.
        self.pycode_gen.gen_call_method(0)  # call clear
        self.pycode_gen.gen_pop_top()
        self.pycode_gen.gen_call_method(1)  # call update
        self.pycode_gen.gen_pop_top()
```

这里我们将生成的代码分为两部分，一部分是 LOAD 需要的变量、方法，一部分是应用 SideEffect（调用方法），根据栈的特性，先调用的方法应该在栈顶，即后入栈的数据因此生成代码的布局为：

```
LOAD 方法二
LOAD 方法一

递归处理下一个 Variable

CALL 方法一
CALL 方法二
```

这样就能确保在生成的代码中，前面全是 LOAD，后面全是 CALL，LOAD 的时候可以确保全都是原始值，不会出现找不到或者找到错误的结果的问题。

# 函数和方法

## method 与 bind

什么是 function 就不必说了，这里说明一下 method 和 function 的区别

简单来说，method 就是 function 的第一个位置 bind 一个 object（即 self），其行为上有点类似用 partial 绑定了第一个参数的 function

```python
class A:
    def b(self, x):
        return x

a = A()
a.b(1)

# equivalent to
A.b(a, 1)

# it is like
partial_b = partial(A.b, a) # bind a to the first position
partial_b(1)
```

注意 `a.b(a)` 和 `A.b(a, 1)` 是完全等价的，这里 `A.b` 是一个 function，也称 unbound method，而 `a.b` 则是一个 bound method，根据名字也能看出来，就是 bind 了 self 的 method

## method 与 descriptor

method 的实际是利用了 descriptor，比如

```python
A.b.__get__(a, A)

# equivalent to
a.b
```

由于 `A.b` 是一个 descriptor，因此在其实例获取属性 `b` 时，自然会调用 `A.b.__get__`，在此时便会将原来的 function 和 object 绑定在一起，获得一个 bound method

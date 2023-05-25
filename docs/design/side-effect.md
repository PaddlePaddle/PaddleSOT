# SideEffect

## Cases 1

```python
a = [[1,2], [3,4]]
def func(a)
    a[0][0] = 4
```

嵌套的 SideEffect 需要能够影响到这个Object的来源

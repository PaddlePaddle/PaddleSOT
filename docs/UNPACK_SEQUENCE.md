# UNPACK_SEQUENCE

## 1. python3.8 的 UNPACK_SEQUENCE

1.   只处理三种情况：list，tuple，iterator
2.   从右向左入栈，即最终 list[0] 在栈顶 （iter是第一次 next 的返回值在栈顶）

```py
        case TARGET(UNPACK_SEQUENCE): {
            PREDICTED(UNPACK_SEQUENCE);
            PyObject *seq = POP(), *item, **items;
            if (PyTuple_CheckExact(seq) &&
                PyTuple_GET_SIZE(seq) == oparg) {
                items = ((PyTupleObject *)seq)->ob_item;
                while (oparg--) {
                    item = items[oparg];
                    Py_INCREF(item);
                    PUSH(item);
                }
            } else if (PyList_CheckExact(seq) &&
                       PyList_GET_SIZE(seq) == oparg) {
                items = ((PyListObject *)seq)->ob_item;
                while (oparg--) {
                    item = items[oparg];
                    Py_INCREF(item);
                    PUSH(item);
                }
            } else if (unpack_iterable(tstate, seq, oparg, -1,
                                       stack_pointer + oparg)) {
                STACK_GROW(oparg);
            } else {
                /* unpack_iterable() raised an exception */
                Py_DECREF(seq);
                goto error;
            }
            Py_DECREF(seq);
            DISPATCH();
        }
```



## 2. 遇到的问题

从 iterator 中 unpack 出来的元素，其 source 是什么？



## 3. torch 的做法

在 torch 中 unpack 的逻辑是和 Varaible 绑定的（作为成员方法）

只支持：

1.   const
2.   dict
3.   list （BaseListVariable，RangeVariable，SizeVariable，ListIteratorVariable）
     -   tuple 也在这个文件，但是它的 iterator 也不能 unpack （应该说根本没有 TupleIteratorVariable）
     -   SizeVairable 是 TupleVariable 的子类
4.   nn_module
5.   tensor

对于迭代器类型，只支持 ListIterator，所以并没有实现 unpack iterator

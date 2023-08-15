# `co_linetable` 字段适配

继 Python 3.10 将 `co_lnotab` 修改为 `co_linetable` 并修改其格式之后，Python 3.11 再次对 `co_linetable` 进行了较大的变动。相比于之前版本，Python 3.11 不仅将行号信息编码到 `co_linetable`，还将列号信息编码到了 `co_linetable` 中。具体编码格式见 [cpython 3.11 - locations.md](https://github.com/python/cpython/blob/3.11/Objects/locations.md)，这篇文档介绍已经比较详细了，但是示例太少不是太容易理解

## `co_linetable` 包含的信息

`co_linetable` 包含了所有字节码的行号和列号信息，我们可以通过 `co_positions` 来查看解码后的每条字节码对应的位置信息：

```python
import dis


def foo(x):
    x = x + 1
    y = x + 2
    return y


dis.dis(foo)
for bytecode_pos in foo.__code__.co_positions():
    print(bytecode_pos)
print(foo.__code__.co_linetable)
```

注意空行需要完全对应，不然输出可能会有些不一样，输出如下：

```text
  4           0 RESUME                   0

  5           2 LOAD_FAST                0 (x)
              4 LOAD_CONST               1 (1)
              6 BINARY_OP                0 (+)
             10 STORE_FAST               0 (x)

  6          12 LOAD_FAST                0 (x)
             14 LOAD_CONST               2 (2)
             16 BINARY_OP                0 (+)
             20 STORE_FAST               1 (y)

  7          22 LOAD_FAST                1 (y)
             24 RETURN_VALUE
(4, 4, 0, 0)
(5, 5, 8, 9)
(5, 5, 12, 13)
(5, 5, 8, 13)
(5, 5, 8, 13)
(5, 5, 4, 5)
(6, 6, 8, 9)
(6, 6, 12, 13)
(6, 6, 8, 13)
(6, 6, 8, 13)
(6, 6, 4, 5)
(7, 7, 11, 12)
(7, 7, 4, 12)
b'\x80\x00\xd8\x08\t\x88A\x89\x05\x80A\xd8\x08\t\x88A\x89\x05\x80A\xd8\x0b\x0c\x80H'
```

这段代码共 13 条字节码，这里每个四元组对应了一条字节码的「开始行号」、「结束行号」、「开始列号」、「结束列号」

当然仅仅知道解码结果是不够的，因为我们是需要编码成 `co_linetable` 字节流的，我们接下来会逐渐解析字节码的编码方式

## `co_linetable` 的编码

### entry

`co_linetable` 字节序列包含了多个 entry，每个 entry 由不定长的多个字节组成

对于每个 entry 来说，第一个字节的最高位为 1，之后所有字节的最高位都为 0，通过这种方式我们可以发现上述 `co_linetable` 可以划分为如下几个 entry：

```python
def get_entries(linetable_bytes: bytes):
    buffer = []
    for byte in linetable_bytes:
        if not 0x80 & byte:
            buffer.append(byte)
            continue
        if buffer:
            yield tuple(buffer)
            buffer.clear()
        buffer.append(byte)
    yield tuple(buffer)
    buffer.clear()

linetable_bytes = b'\x80\x00\xd8\x08\t\x88A\x89\x05\x80A\xd8\x08\t\x88A\x89\x05\x80A\xd8\x0b\x0c\x80H'

for entry_bytes in get_entries(linetable_bytes):
    for b in entry_bytes:
        print(f"{int(b):08b}", end=" ")
    print()
```

输出如下：

```text
10000000 00000000
11011000 00001000 00001001
10001000 01000001
10001001 00000101
10000000 01000001
11011000 00001000 00001001
10001000 01000001
10001001 00000101
10000000 01000001
11011000 00001011 00001100
10000000 01001000
```

这样我们就解码出来了 11 个 entry，咦？为啥是 11 个？我们不是有 13 条字节码么？这是因为每个 entry 会用于表示一到多条字节码的位置信息，当多个连续字节码的位置信息是一样时，会编码在同一个 entry 里

### entry head 信息

对于每个 entry，第一个字节除去最高位固定是 `1`，剩余 7 位存储了编码方式以及 entry 所表示的字节码数量信息，具体如下：

```text
1 | 1011 | 000
│     │     │
│     │     └──────────────────── 0-2 位，用于表示相关字节码数量（因为数量 >= 1，因此这里数值为字节码数量 - 1，这里 `000` 表示 包含 1 个字节码）
│     └────────────────────────── 3-6 位，用于表示编码类型（这里表示编码类型 11）
└────────────────────────────────   7 位，entry head 固定为 1
```

根据这些信息我们可以将所有 entry 信息都解码出来

```python
for entry_bytes in get_entries(linetable_bytes):
    head = entry_bytes[0]
    code = (head & 0x78) >> 3
    num_bytecode = (head & 0x07) + 1
    print(f"code is: {code:2}, num_bytecode is: {num_bytecode:2}, ", end="")
    for b in entry_bytes:
        print(f"{int(b):08b}", end=" ")
    print()
```

输出如下：

```text
code is:  0, num_bytecode is:  1, 10000000 00000000
code is: 11, num_bytecode is:  1, 11011000 00001000 00001001
code is:  1, num_bytecode is:  1, 10001000 01000001
code is:  1, num_bytecode is:  2, 10001001 00000101
code is:  0, num_bytecode is:  1, 10000000 01000001
code is: 11, num_bytecode is:  1, 11011000 00001000 00001001
code is:  1, num_bytecode is:  1, 10001000 01000001
code is:  1, num_bytecode is:  2, 10001001 00000101
code is:  0, num_bytecode is:  1, 10000000 01000001
code is: 11, num_bytecode is:  1, 11011000 00001011 00001100
code is:  0, num_bytecode is:  1, 10000000 01001000
```

可以看到有两个 entry 包含两条字节码，其余都是只包含一条字节码，加在一起刚刚好是 13 条字节码的信息

### 数据的编码方式

除去 head 外，之后的不定长字节序列用来表示相关字节码的位置信息，这部分主要使用了两种编码方式，对于无符号整型，可以编码成 `varint`，对于有符号整型，可以编码成 `svarint`

#### varint

由于后续数据字节最高位一定是 1，因此有效的数据编码位只有 7 位，对于 varint 来说，其编码是除了最后一个字节外，其余字节都是次高位（6 位）为 1

也就是说，是 `01xxxxxx 01xxxxxx 01xxxxxx ... 01xxxxxx 00xxxxxx` 的序列

对于每个字节来说，实际用来表示数据的只剩 6 位了，这个编码实现也很简单

```python
def encode_varint(num: int):
    continue_flag = 0b01 << 6
    stop_flag = 0b00 << 6
    while num >= 0x40:
        yield (num & 0x3F) | continue_flag
        num >>= 6
    yield num | stop_flag


def display_integers(encodes: list[int]):
    for encode in encodes:
        print(f"0x{encode:02x}", end=" ")
    print()


display_integers(list(encode_varint(63)))
display_integers(list(encode_varint(200)))
```

输出如下

```text
0x3f
0x48 0x03
```

#### svarint

svarint 用于表示有符号整型，其编码方式是首先将有符号整型编码成无符号整型（即将符号位编码在数据里），之后再用 varint 编码，其实现也很简单：

```python
def encode_svarint(num: int):
    unsigned_value = (((-num) << 1) | 1) if num < 0 else (num << 1)
    yield from encode_varint(unsigned_value)

display_integers(list(encode_svarint(63)))
display_integers(list(encode_svarint(200)))
display_integers(list(encode_svarint(-20)))
display_integers(list(encode_svarint(-200)))
```

输出如下：

```text
0x7e 0x01
0x50 0x06
0x29
0x51 0x06
```

数据可能会选择使用 varint 或者 svarint 进行编码，这个需要根据编码方式来判断

### 编码方式表

这里直接贴文档中的表格，主要有如下五种编码方式

| Code | Meaning | Start line | End line | Start column | End column |
| - | - | - | - | - | - |
| 0-9 | Short form | Δ 0 | Δ 0 | See below | See below |
| 10-12 | One line form | Δ (code - 10) | Δ 0 | unsigned byte | unsigned byte |
| 13 | No column info | Δ svarint | Δ 0 | None | None |
| 14 | Long form | Δ svarint | Δ varint | varint | varint |
| 15 | No location |  None | None | None | None |

这里的 Code 就是前面所说的编码方式，由于其共占据 4 位，因此共有 16 种 Code 可选值

部分编码方式有多种可选 Code，是因为会在 Code 里编码一些位置信息，以节省编码长度。比如上面的 `10-12`，其中 `code - 10` 编码了开始行的变化，比如编码 12 表示开始行号相对于上一个字节码增加了 2，而该编码方式 One line form 用于表示在同一行，因此没有结束行号信息，因此行号信息使用 Code 已经完全表示了，剩余的列号信息将会使用其他字节来表示。

其他的不一一列举，因为大多我们用不到。

## `co_linetable` 适配

Python 3.11 增加了这么多信息和这么复杂的编码方式，难道我们还需要考虑使用哪种 Code 嘛？不不不，其实我们可以和 3.10 之前一样只关注行号，我们可以发现编码 `13` No column info	刚好用于表示无列号信息，因此我们将所有字节码 entry code 都设置为 `13` 即可，这样 `co_linetable` 的编码就和 3.8-3.10 非常相似了。

不过值得注意的一点是，Python 3.11 是对每条字节码都会进行编码的，而不是换新行才会生成新的编码，因此遍历字节码时，每条字节码都需要传入 `calc_linetable_py311` 中进行编码，即便 `starts_line is None`

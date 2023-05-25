# Eval Frame 设计和实现

## 疑难问题定位和分析

### Generator Early Return Causes `System Error`

待完善，可以见如流知识库

### Resnet 支持后出现 segement fault

- [x] 果然是 stacksize 的问题

初步定位，发现第一个Step的翻译过程已经走完，出现问题的环节应该是 `eval frame` 中的第二步。

#### 如何查找线索？

面临的第一个问题是，如何寻找到定位问题的线索。eval frame 中没有对应的 log 可以查看，所以我们很难进行定位那里出现了问题，但是因为 Python 的独立编译过，所以可以很方便的在Pytho中打印出我们需要的变量。

所以我们有如下两个探索方式：

1. 在 Eval frame 中进行日志探索。

2. 在 Python 源码中插入信息搜集的点。

理论上，有了上述两个方法，我们可以找到一切问题。包含 segment 错误。

#### 定位 segment fault 位置

segment fault 问题的首要任务就是找到错误的地点。才可以逐渐分析出错误原因。

**没有跑到动转静组网**: 首先需要明确，segment fault 位置在 `CALL_FUNCTION` 字节码之前。

```python
import os
os.main
```

int main


#### 问题猜测

- [x] 果然是这个stacksize的问题！！

这个问题是否与 eval frame 中的 stack size 有关系，因为stack size不够大，导致栈溢出了。这样的解释也是比较合理的。

# PaddleSOT

**Paddle** **S**ymbolic **O**pcode **T**ranslator.

PaddleSOT 是一个 Opcode-Based 的动转静孵化项目，借助 Symbolic Opcode Translator（简称：SOT）在运行时将 PaddlePaddle 动态图组网代码转换为静态图组网代码，具体设计参见：[PaddleSOT 项目介绍](https://github.com/PaddlePaddle/community/tree/master/pfcc/paddle-code-reading/symbolic_opcode_translator)

## Install

```bash
git clone https://github.com/PaddlePaddle/PaddleSOT.git
cd PaddleSOT/
pip install -e .
```

此外由于我们有部分特性依赖于最新的 PaddlePaddle，因此你需要安装 Nightly build 版本的 PaddlePaddle，你可以在[官网安装页面](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/develop/install/pip/linux-pip.html)根据自己的平台找到相应的安装方式

## Usage

你可以通过运行 `examples/` 下的示例来了解 PaddleSOT 的使用方法。

```bash
python examples/trace_basic.py
```

## Contributing

请参考 [PaddleSOT 贡献指南](./CONTRIBUTING.md)

# PaddleSOT 贡献指南

很高兴你对参与 PaddleSOT 的贡献感兴趣，在提交你的贡献之前，请花一点点时间阅读本指南

## 本地调试

### Fork Repo 到自己 GitHub 账户

为了方便提交 PR，建议你在 clone 之前先在自己的 GitHub 创建一个 fork，你可以前往 [paddle-symbolic-trace/fork](https://github.com/2742195759/paddle-symbolic-trace/fork) 来创建一个 Fork。

> **Note**
>
> 由于历史原因，我们的 PaddleSOT 项目 repo 早期命名为 paddle-symbolic-trace，这将会在未来迁移到 Paddle 后统一修改。

### Clone Repo 到本地

```bash
git clone git@github.com:<YOUR_USER_NAME>/paddle-symbolic-trace.git            # 将你的 repo clone 到本地
cd paddle-symbolic-trace/                                                      # cd 到该目录
git remote add upstream git@github.com:2742195759/paddle-symbolic-trace.git    # 将原分支绑定在 upstream
```

### 环境配置

PaddleSOT 目前支持 Python 3.8、3.9、3.10，因此你需要先创建一个 3.8-3.10 的环境（推荐 Python 3.8），这可以通过使用 virtualenv、conda 等工具来快速完成：

```bash
# 对于 conda
# 需事先自行安装 Anaconda 或者等效替代品 miniconda、miniforge、mambaforge 等工具
conda create -n py38 python=3.8                 # 创建环境
conda activate py38                             # 激活环境
conda deactivate                                # 退出环境

# 对于 virtualenv
pip install virtualenv                          # 安装 virtualenv
virtualenv .venv --python=python3.8             # 创建环境（需要保证 python3.8 可以访问）
source .venv/bin/activate                       # 激活环境
deactivate                                      # 退出环境
```

你可以在激活环境后运行 `python --version` 确保 Python 版本正确。

### 安装依赖

目前 Paddle SOT 主体部分是独立于 Paddle 开发的，因此开发过程中不需要和 Paddle 一起编译，你只需要安装 Paddle 的 wheel 包即可。

但由于我们有部分特性（Eval Frame 相关部分）是放在 Paddle C++ 端编译的，所以需要依赖于最新的 Paddle wheel 包（即 nightly build），你可以在[官网安装页面](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/develop/install/pip/linux-pip.html)根据自己的平台找到相应的安装方式

### 运行单测

为了确保你的环境配置正确，你可以尝试先运行全量单测：

```bash
cd tests/
bash run_all.sh
```

如果环境配置没有问题，这里所有单测都应该通过。

### 代码风格检查工具配置

Paddle SOT 使用 pre-commit 来自动运行 Linter 和 Formatter，我们基本保持了与 Paddle 一样的配置，使用 black 作为主 Formatter，isort 作为 import 区域 Formatter，ruff 作为主 Linter。你可以在 [.pre-commit-config.yaml](./.pre-commit-config.yaml) 中查看详细的版本配置。

在提交 PR 之前，你需要确保自己的代码风格检查可以通过

```bash
pip install pre-commit
pre-commit run --all-files
```

你也可以直接运行 `pre-commit install` 来保证在每次 commit 之前自动运行 `pre-commit`。

> **Note**
>
> 以上全流程在 M1 macOS 和 Linux 上均已测试过，Windows 上略有出入，需要稍微自行调整下

## 示例与教程

### 基础示例

你可以通过如下方式来运行示例代码：

```bash
LOG_LEVEL=3 PYTHONPATH=. python examples/trace_basic.py
```

运行如上示例，你可以看到如下的 log：

```text
# Eval Frame Callback 在 foo 函数执行前被调用
[eval_frame_callback] start to translate: foo
# 查找 foo 的 CodeObject 对应的 Cache，Cache 没命中，开始转换
[Cache]: Cache miss
# 函数原始的字节码如下（通过 `dis.dis(foo)` 即可查看）
OriginCode:
  8           0 LOAD_FAST                0 (x)
              2 LOAD_FAST                1 (y)
              4 BINARY_ADD
              6 STORE_FAST               2 (z)

  9           8 LOAD_FAST                2 (z)
             10 LOAD_CONST               1 (1)
             12 BINARY_ADD
             14 RETURN_VALUE
# 开始转换 foo 函数字节码（模拟执行）
start execute opcode: <code object foo at 0x104659c90, file "/Users/xxx/Projects/paddle-symbolic-trace/examples/trace_basic.py", line 7>
# 依次执行 foo 函数的字节码
# 模拟执行过程中，log 中同时展示了模拟栈的状态
# 在此过程中，我们同时收集了 SIR（Paddle 组网相关信息）和 Variable 关系信息（含 Guard、Tracker 等）
[TraceExecution]: LOAD_FAST, stack is []
[TraceExecution]: LOAD_FAST, stack is [TensorVariable(shape: [2, 3], dtype: paddle.float32, stop_gradient: True) var_0]
[TraceExecution]: BINARY_ADD, stack is [TensorVariable(shape: [2, 3], dtype: paddle.float32, stop_gradient: True) var_0, TensorVariable(shape: [2, 3], dtype: paddle.float32, stop_gradient: True) var_1]
[TraceExecution]: STORE_FAST, stack is [TensorVariable(shape: [2, 3], dtype: paddle.float32, stop_gradient: True) var_2]
[TraceExecution]: LOAD_FAST, stack is []
[TraceExecution]: LOAD_CONST, stack is [TensorVariable(shape: [2, 3], dtype: paddle.float32, stop_gradient: True) var_2]
[TraceExecution]: BINARY_ADD, stack is [TensorVariable(shape: [2, 3], dtype: paddle.float32, stop_gradient: True) var_2, ConstantVariable(1)]
[TraceExecution]: RETURN_VALUE, stack is [TensorVariable(shape: [2, 3], dtype: paddle.float32, stop_gradient: True) var_3]
# 遇到 RETURN_VALUE，打断子图，触发子图编译（本示例只有一个完整子图），如下是收集到的 SIR（Paddle 组网信息），利用 to_static 编译并挂载到 f_globals 中
start subgraph compile and execution.
StatmentIR: SIR_0
  inputs: ['var_0', 'var_1']
  outputs: ['var_3']
  statements:
    method     || var_2 = __add__ ((var_0, var_1), {})
    method     || var_3 = __add__ ((var_2, 1), {})
# Guard 如下，用来保证当前轮次编译的字节码的有效性
[Guard]: lambda frame: str(MetaInfo.from_tensor(frame.f_locals['y'])) == '(shape: [2, 3], dtype: paddle.float32, stop_gradient: True)' and str(MetaInfo.from_tensor(frame.f_locals['x'])) == '(shape: [2, 3], dtype: paddle.float32, stop_gradient: True)' and 1 == 1
# 编译好的字节码如下
  7           0 LOAD_GLOBAL              0 (SIR_0)                  # LOAD 编译好的组网代码
              2 LOAD_FAST                0 (x)                      # LOAD 相关输入
              4 LOAD_FAST                1 (y)
              6 BUILD_TUPLE              2                          # 打包成 tuple，准备传入 SIR_0
              8 CALL_FUNCTION            1                          # 调用 SIR_0，传入栈上的 x、y
             10 UNPACK_SEQUENCE          1                          # 解包返回值
             12 STORE_FAST               3 (___SIR_out_var_3)       # 将参数依次存入 f_locals
             14 LOAD_FAST                3 (___SIR_out_var_3)       # 依次 LOAD 输出
             16 RETURN_VALUE                                        # RETURN 栈上变量
# Eval Frame callback 运行结束，回到 Eval Frame，编译好的 CodeObject 构造一个新的 shadow frame，Python 默认 Eval Frame 执行 shadow frame
# 执行动转静函数 `SIR_0`，调用新执行器执行静态图
I0601 15:07:44.390908 4192656896 interpretercore.cc:237] New Executor is Running.
```

### 子图打断

当我们遇到无法或没有必要继续模拟执行的情况时，会将当前的子图打断，立即触发子图编译，并将之后的代码抽离到一个新的函数中，这主要发生在如下几种情形中：

- **D**ata-**D**ependence **C**ontrol **F**low（简称：DDCF），即对于控制流 If、For 依赖 Tensor 的场景，需要打断构图并触发子图打断
- **Uns**u**p**port **S**imulation（简称：UNSPS），即对于我们无法模拟的一些函数调用，需要触发子图打断
- **C**ustomize **D**efined **B**lack**L**ist 机制强制触发（简称：CDBL），即用户可以手动指定某些API，使得在遇到这些 API 时强制触发子图打断

这里主要展示 DDCF 的情况：

```bash
LOG_LEVEL=3 PYTHONPATH=. python examples/graph_break.py
```

`foo` 函数的字节码如下：

```text
  8           0 LOAD_FAST                1 (x)
              2 LOAD_CONST               1 (1)
              4 INPLACE_ADD
              6 STORE_FAST               1 (x)

  9           8 LOAD_FAST                0 (cond)
             10 POP_JUMP_IF_FALSE       22

 10          12 LOAD_FAST                1 (x)
             14 LOAD_CONST               1 (1)
             16 INPLACE_ADD
             18 STORE_FAST               1 (x)
             20 JUMP_FORWARD             8 (to 30)

 12     >>   22 LOAD_FAST                1 (x)
             24 LOAD_CONST               1 (1)
             26 INPLACE_SUBTRACT
             28 STORE_FAST               1 (x)

 13     >>   30 LOAD_FAST                1 (x)
             32 RETURN_VALUE
```

这里的 `cond` 是 Tensor，我们发现 `POP_JUMP_IF_FALSE` 所依赖的栈顶元素是一个 Tensor，因此会在此处打断子图，并将之后的代码抽离到新的函数中，编译后的代码如下：

```text
  7           0 LOAD_GLOBAL              0 (SIR_0)
              2 LOAD_FAST                1 (x)
              4 BUILD_TUPLE              1
              6 CALL_FUNCTION            1
              8 UNPACK_SEQUENCE          1
             10 STORE_FAST               2 (___SIR_out_var_2)
             12 LOAD_FAST                0 (cond)
             14 LOAD_FAST                2 (___SIR_out_var_2)
             16 POP_TOP
             18 POP_JUMP_IF_FALSE       28
             20 LOAD_GLOBAL              1 (__resume_fn_0)
             22 LOAD_FAST                2 (___SIR_out_var_2)
             24 CALL_FUNCTION            1
             26 RETURN_VALUE
        >>   28 LOAD_GLOBAL              2 (__resume_fn_1)
             30 LOAD_FAST                2 (___SIR_out_var_2)
             32 CALL_FUNCTION            1
             34 RETURN_VALUE
```

其中 `__resume_fn_0` 和 `__resume_fn_1` 分别对应 JUMP / NOT JUMP 两种情况的代码。字节码分别如下：

```text
  7           0 JUMP_ABSOLUTE           14

  8           2 LOAD_FAST                0 (x)
              4 LOAD_CONST               1 (1)
              6 INPLACE_ADD
              8 STORE_FAST               0 (x)

  9          10 LOAD_FAST                1 (cond)
             12 POP_JUMP_IF_FALSE       24

 10     >>   14 LOAD_FAST                0 (x)
             16 LOAD_CONST               1 (1)
             18 INPLACE_ADD
             20 STORE_FAST               0 (x)
             22 JUMP_FORWARD             8 (to 32)

 12     >>   24 LOAD_FAST                0 (x)
             26 LOAD_CONST               1 (1)
             28 INPLACE_SUBTRACT
             30 STORE_FAST               0 (x)

 13     >>   32 LOAD_FAST                0 (x)
             34 RETURN_VALUE
```

```text
  7           0 JUMP_ABSOLUTE           24

  8           2 LOAD_FAST                0 (x)
              4 LOAD_CONST               1 (1)
              6 INPLACE_ADD
              8 STORE_FAST               0 (x)

  9          10 LOAD_FAST                1 (cond)
             12 POP_JUMP_IF_FALSE       24

 10          14 LOAD_FAST                0 (x)
             16 LOAD_CONST               1 (1)
             18 INPLACE_ADD
             20 STORE_FAST               0 (x)
             22 JUMP_FORWARD             8 (to 32)

 12     >>   24 LOAD_FAST                0 (x)
             26 LOAD_CONST               1 (1)
             28 INPLACE_SUBTRACT
             30 STORE_FAST               0 (x)

 13     >>   32 LOAD_FAST                0 (x)
             34 RETURN_VALUE
```

各段字节码整体关系如下：

<p align="center">
    <img alt="graph break" src="https://github.com/2742195759/paddle-symbolic-trace/assets/38436475/bd7c0730-fa4d-402a-b789-1c7e25135f79" width="500px"/>
</p>

这里代码抽离逻辑很简单，只是在原有代码的基础上添加了 `JUMP_ABSOLUTE`，跳转到不同分支的目标即可。

我们在 log 中可以发现 `foo` 函数、`__resume_fn_0`（图中 `resume_to_block_1`）函数都执行了 eval frame callback，各自也都有一个子图，这里的执行流程如下：

- 执行 `foo`
  - 进入 Eval Frame Callback，开始模拟执行 foo 的字节码
  - 遇到 DDCF 的情况，打断子图，并利用已经模拟执行收集的信息（SIR + Tracker 等）编译生成 `compiled BLOCK 0` 部分
  - 生成两个分支的字节码，分别对应 JUMP / NOT JUMP 两种情况（注意不应该仅仅认为是 if/else 情况，while、and 等情况的字节码也是同样的，他们在字节码层面的处理逻辑是一致的）
  - 退出 Eval Frame Callback，Python 执行编译好的字节码（即 `compiled foo`）
- 执行 `resume_to_block_1`（`compiled foo` 执行过程中调用的）
  - 进入 Eval Frame Callback，开始模拟执行 `resume_to_block_1` 的字节码
  - 遇到 `RETURN_VALUE`，打断子图（该函数中只包含一个完整的子图），编译生成对应的字节码
  - 退出 Eval Frame Callback，Python 执行编译好的字节码

对于该示例是只有一个 JUMP 的情况，但这种处理已经涵盖了嵌套 JUMP 的问题，假如 `resume_to_block_1` 中也有 JUMP，那么只需要递归地处理即可。

### Guard 及缓存机制

```bash
# 通过环境变量 SHOW_TRACKERS 你可以看到 Python 端所有 Variable 依赖关系
# 不过在此之前你需要先按照 https://github.com/2742195759/paddle-symbolic-trace/pull/82 说明安装好 graphviz 库和 dot 可执行文件
SHOW_TRACKERS=out LOG_LEVEL=3 PYTHONPATH=. python examples/guard.py
```

在 log 中，我们可以看到第二次和第三次执行分别发生了 `cache hit` 和 `cache miss`，这是因为我们在第一次执行生成字节码的同时，还生成了相应的 Guard，我们可以在 log 中找到 Guard 代码：

```python
lambda frame: str(MetaInfo.from_tensor(frame.f_locals['x'])) == '(shape: [1], dtype: paddle.float32, stop_gradient: True)' and str(MetaInfo.from_tensor(frame.f_locals['y'])) == '(shape: [2, 3], dtype: paddle.float32, stop_gradient: True)'
```

可以发现它保证了 `x` 和 `y` 的 meta 信息（shape 等）是不变的，当我们第二次执行时，`x` 和 `y` 的 meta 信息都没有发生变化，因此可以命中 Guard，而第三次执行 `x` 的 meta 信息发生了变化，因此 Guard 失效，会触发重新模拟执行，编译生成新的字节码（当然也会有相应的 Guard）。

值得注意的是，`z` 并不在 Guard 中，这是因为 Guard 的收集是从组网代码输入开始的，而 `z` 并没有参与组网，因此不会被收集到 Guard 中。你可以在生成的 `out.png` 中看到所有 Variable 之间的关系：

<p align="center">
    <img alt="trackers" src="https://github.com/2742195759/paddle-symbolic-trace/assets/38436475/bae5b86d-14c2-4c1e-857b-3241fb71b3ad" width="800px"/>
</p>

> **Note**
>
> 绿框内表示参与 SIR 组网部分，但目前的组网判定逻辑有些问题，`z` 并未参与组网，但仍然被划分到了绿框内

关于 Guard 设计的一些细节，你可以阅读 [Guard 收集机制](./docs/design/tracker-and-guard.md) 和[字符串化 Guard](./docs/design/stringify-guard.md) 来了解。

### 更多示例（建设中）

TODO...

### 技术细节

你可以通过阅读 [PaddleSOT 孵化项目说明](https://github.com/PaddlePaddle/community/tree/master/pfcc/paddle-code-reading/symbolic_opcode_translator)来了解我们项目的背景以及整体架构，此外还可以通过阅读 [docs](./docs/) 来了解我们在设计过程中的一些技术细节，这可以帮助你更好地理解我们的设计思路。

## 项目结构

现在，你可以尝试阅读代码来更进一步地了解 PaddleSOT 了，如下是我们当前的项目结构

```text
.
├── README.md
├── docs                                              # 文档，目前主要存放一些技术细节
│   ├── design
│   └── instructions
├── pyproject.toml
├── requirements.txt
├── symbolic_trace
│   ├── __init__.py
│   ├── infer_meta.py                                 # Infer Meta 模块，利用静态图/动转静进行 Tensor 的 Meta 信息推导
│   ├── opcode_translator                             # 「编译期」代码转换模块
│   │   ├── __init__.py
│   │   ├── executor                                  # 模拟执行模块
│   │   │   ├── __init__.py
│   │   │   ├── function_graph.py                     # 组网功能模块
│   │   │   ├── guard.py                              # Guard 相关数据结构
│   │   │   ├── instr_flag.py                         # 指令相关 Flags，与 CPython 保持一致
│   │   │   ├── opcode_executor.py                    # 模拟执行器，包含了主要的字节码模拟执行逻辑
│   │   │   ├── opcode_inline_executor.py             # Inline 模拟执行器，主要用来处理嵌套的函数调用
│   │   │   ├── pycode_generator.py                   # 字节码 CodeGen 模块
│   │   │   ├── tracker.py                            # Tracker 相关数据结构
│   │   │   ├── tracker_viewer.py                     # Tracker 可视化模块（based on Graphviz）
│   │   │   ├── variable_monkey_patch.py              # Variable magic method monkey patch 模块
│   │   │   └── variables.py                          # 模拟执行过程中包装的各种 Variable 数据结构
│   │   ├── instruction_utils                         # 字节码相关功能模块
│   │   │   ├── __init__.py
│   │   │   ├── instruction_utils.py
│   │   │   ├── opcode_analysis.py
│   │   │   └── opcode_info.py
│   │   ├── skip_files.py
│   │   └── transform.py                              # Eval Frame Callback 入口
│   ├── symbolic                                      # symbolic 模块，包含 SIR 数据结构及解释执行操作
│   │   ├── compile_cache.py
│   │   ├── interpreter.py
│   │   ├── statement_ir.py
│   │   └── symbolic_context.py
│   ├── trace.py                                      # 功能入口
│   └── utils
│       ├── __init__.py
│       ├── exceptions.py
│       ├── monkey_patch.py
│       ├── paddle_api_config.py
│       └── utils.py
└── tests                                             # 单测目录
    ├── run_all.sh                                    # 单测运行脚本
    ├── test_*.py                                     # 单测文件
    ├── error_*.py                                    # `error_` 前缀表示待解决的 case
    └── tests_legacy                                  # 旧方案的一些单测，可忽略
```

## 提交 PR

经过上述步骤，相信你已经对我们的项目有了一个整体的了解，可以尝试根据自己的兴趣提一个 PR 了～

```bash
git checkout -b <NEW_BRANCH>                                # 新建一个分支，名称随意，最好含有你本次改动的语义
git push origin <NEW_BRANCH>                                # 将该分支推送到 origin （也就是你 fork 后的 repo）
# 对源码进行修改、并通过测试
# 此时可以在 GitHub 发起 PR
```

之后只需要静待 CI 通过和我们的 review 即可～

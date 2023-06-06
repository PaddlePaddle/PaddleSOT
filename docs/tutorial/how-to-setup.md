# 背景

当前的 Paddle Symbolic Opcode Translator 已经可以处理常见的场景。其中包含Paddle的API调用、Paddle Method调用、Layer 调用甚至Resnet也可以以单子图的形式运行。因此Paddle Symbolic Opcode Translator进入了正确率和成功率的保证阶段：对Paddle原有动转静的所有单测进行兼容测试，保证 <font color='red'>100%</font> 的兼容。

# 单测调研现状

在Paddle侧提了`AST to static` 和 `PSOT to static` 的统一PR之后，我们进行了一次正确率测试，目前的成功率是：<font color='red'>47.5%</font>（上述是开了非严格模式下的成功率）。一共 120 个单测只完整通过了 57个。


目前还存在问题的单测如下： [单测完整情况](https://ku.baidu-int.com/knowledge/HFVrC7hq1Q/yKeL8Lljko/5K6Iojo8fU/8WiP4r3K6L6Nvr)


为了加速单测的测试，并且让更多的开发者参与进来，我们将这些单测的修复以任务的形式发布，大家可以自主认领，并参与开发。这里将介绍如何联合 Paddle 进行 PSOT 仓库的单测复现和单测修复。


# 环境搭建部分

这个章节讲述如何进行单测修复环境的搭建。

## Step1: 下载并编译Paddle

因为单测是依赖Paddle的，因此我们需要首先编译Paddle，并且获取最新的 PSOT pr。

1. 使用下面命令clone paddle仓库。
```bash
git clone https://github.com/PaddlePaddle/Paddle
git remote add upstream https://github.com/PaddlePaddle/Paddle
```

2. 拉取最新的 Paddle 补丁，[PSOT 对 Paddle 的 PR](https://github.com/PaddlePaddle/Paddle/pull/54202)。这个PR让`paddle.jit.to_static` 接口默认调用 PSOT 进行动转静，而不是原来的AST to static。
```bash
git fetch upstream pull/54202/head:pr_54202
git checkout pr_54202
```

3. 编译 paddle，具体的编译问题可以查看[paddle的官网](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/install/compile/linux-compile.html#anchor-0)。下面给出推荐的编译脚本文件(rebuild.sh)。CPU和GPU都可以，注意我们默认的Python版本是3.8，大家记得安装对应的环境，推荐使用 Conda。
```bash
set -e
cmake .. -DPY_VERSION=3.8 -DWITH_GPU=OFF -DWITH_TESTING=ON -DCMAKE_BUILD_TYPE=Release
make install -j 16 #> /home/data/error2 2>&1
```

上述命令都走完了，我们可以使用如下的命令来测试是否成功安装了Paddle:

```bash
cd ~ # 切换到别的目录，防止复用了原来的Paddle目录。
PYTHONPAYH=$PADDLE_ROOT/build/python python
```

如果上述代码import paddle成功，那么说明paddle安装成功了。注意将你的 PADDLE_ROOT 替换为你git clone 的目录


## Step2: 联合PSOT仓库

编译完毕Paddle之后，我们需要配置 PSOT仓库，让Paddle使用最新的PSOT 仓库代码。

1. 拉取最新的 PSOT 仓库
```bash
git clone https://github.com/2742195759/paddle-symbolic-trace
git checkout origin/develop
```

2. 找到 tests/run_all.sh 文件，修改第2行的`export PYTHONPATH=$PYTHONPATH:../` 为 `export PYTHONPATH=$PYTHONPATH:../:$PADDLE_ROOT/build/python` 然后运行，查看是否可以成功运行所有的单测，如果提示Paddle找不到，请认真查看PYTHONPATH的路径是否设置正确。
```bash
bash run_all.sh
```

3. 软链到最新的PSOT仓库：
```bash
cd $PADDLE_ROOT/build/python/paddle/jit/
rm -rf symbolic_trace
ln -sf $PSOT_ROOT/symbolic_trace
```

4. 在`$PSOT_ROOT/`目录下创建脚本: start_test.sh，输入如下内容
```bash
PYTHONPATH=$PADDLE_ROOT/build/python/ python $PADDLE_ROOT/test/dygraph_to_static/$1 $2
```

5. 运行单测，看是否成功：
```bash
cd $PSOT_ROOT/
./start_test.sh test_tensor_hook.py
```

如果在第5步成功运行了单测，说明成功了，可以进行单测的修复了。

# 单测修复指导

单测的修复需要对PSOT仓库有一定的了解，但是大部分单测我们都给出的可能的问题和解决方法，所以可以在修复单测的过程中注重看对应的部分，然后进行修复，使用 pdb 来进行流程的跟踪还是很推荐的。

## AST Static Only 单测

由于Paddle的存量单测都是 AST 下的单测，所有并不是所有的单测我们都需要保证正确，有的单测就是专门在AST下运行的，如果在运行对应的单测出现了：
```log
ERROR: test_switch_eval_and_train (__main__.TestWithTrainAndEval)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/home/ssd2/xiongkun/Paddle/test/dygraph_to_static/test_partial_program.py", line 135, in test_switch_eval_and_train
    _, train_partial_layer = linear_net.forward.program_cache.last()[-1]
  File "/home/ssd2/xiongkun/Paddle/build/python/paddle/jit/dy2static/program_translator.py", line 717, in program_cache
    raise_error_template("program_cache")()
  File "/home/ssd2/xiongkun/Paddle/build/python/paddle/jit/dy2static/program_translator.py", line 664, in _raise_error
    raise RuntimeError(error_template.format(func=func_str))
RuntimeError: Can't call program_cache when enable_fallback=True.Use paddle.jit.to_static(enable_fallback=False) instead.

----------------------------------------------------------------------
```
从报错中我们可以看到，出错的原因是enable fallback=True，这类问题可以直接对单测进行装饰来修复：

```python
@AST_ONLY #<-- 添加这个装饰器就可以了，这个单测将只跑 AST to static
class TestPruneUnusedParamInProgram(unittest.TestCase):
    def test_prune(self):
        input_ids = np.array([[15, 11, 6, 3, 18, 13]]).astype("float32")

        place = fluid.CPUPlace()
        with fluid.dygraph.guard(place):
            model = GPT2LMHeadModel()
            model.eval()
            input_ids = paddle.to_tensor(input_ids)
            out = model(input_ids)
            np.testing.assert_array_equal(out.numpy(), [[15, 11]])
```


## API 列表问题

有的问题是因为

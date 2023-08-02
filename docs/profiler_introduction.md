## 链接：
    https://github.com/feifei-111/json2flame


## Event：

   Event 应该在 SOT 代码中进行标注，当运行指定代码时触发。

### 注册方式：
1. 使用 EventGuard
2. 使用 event_register 装饰器
3. 使用 event_start 和 event_end 函数


### Event 的参数：
1. event_name 用于标识 Event 的类型
2. event_level 类似于 log_level 的机制, 默认为 0

### 示例
```py
# 使用 Guard，执行 Guard 内的代码视为一个 Event
with EventGuard("add_1"):
    x += 1

# 使用装饰器, 调用该函数视为一个 Event
@event_register("add_2")
def add_2(x):
    x += 2
add_2(x)

# 使用函数，在 event_start 和 event_end 之间的代码视为一个 Event
new_event = event_start("add_3")
x += 3
event_end(new_event)
```


## Profiler：

Profiler 是一个事件观测者，与事件是否发生无关。
如果事件发生了，但是没有观测者，那么事件也不会被记录。

### 创建 Profiler 的方法：
1. 构造 SotProfiler 类型实例
2. 使用 ProfilerGuard

构造Profiler时可以传入 outpath 指定输出路径

### 使用方法：
需要通过 enable 和 disable 接口来开关 profiler
enable 能够接受一个 tag 参数 （一个 string，如果开关多次，可以在输出信息中进行区分， 默认为 "Main"）

```py
profiler = SotProfiler()
profiler.enable()

net(data)

# 可以调用 profiler.disable() 关闭，也可以等它析构时自动关闭

############################
# 也可以用 guard 形式进行监控
with ProfilerGuard():
    net(data)
```

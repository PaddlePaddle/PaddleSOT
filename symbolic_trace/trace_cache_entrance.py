import paddle
from paddle.jit.dy2static.convert_operators import convert_load

from .symbolic.symbolic_context import SymbolicTraceContext
from .symbolic.statement_ir import Symbol, SIRRuntimeCache
from .utils import no_eval_frame, log, map_if
from .infer_meta import MetaInfo
from .proxy_tensor import ProxyTensor, ProxyTensorContext, convert_arguments


def trace_cache(func):
    @no_eval_frame
    def call_with_cache(*args, **kwargs):
        args, kwargs = convert_arguments(args), convert_arguments(kwargs)
        args, kwargs, inputs, outter_names = construct_inner_proxy_tensor(func.__name__, *args, **kwargs)

        if frame_enter(func.__name__, inputs):
            return cache_and_return(func.__name__, outter_names)
        ret = func(*args, **kwargs)
        frame_leave(func.__name__, outter_names, ret)
        return ret
    return call_with_cache


def construct_inner_proxy_tensor(func_name, *args, **kwargs):
    flat_args = paddle.utils.flatten(args)
    flat_kwargs = paddle.utils.flatten(kwargs)
    outter_names = []
    inner_proxy_tensor = []
    name_i = 0
    for i, v in enumerate(flat_args):
        if isinstance(v, ProxyTensor):
            name = '{}_input_{}'.format(func_name, name_i)
            outter_names.append(v.name)
            flat_args[i] = ProxyTensor(name, v.meta)
            inner_proxy_tensor.append(flat_args[i])
            name_i = name_i + 1
    for i, v in enumerate(flat_kwargs):
        if isinstance(v, ProxyTensor):
            name = '{}_input_{}'.format(func_name, name_i)
            outter_names.append(v.name)
            flat_kwargs[i] = ProxyTensor(name, v.meta)
            inner_proxy_tensor.append(flat_kwargs[i])
            name_i = name_i + 1

    args = paddle.utils.pack_sequence_as(args, flat_args)
    kwargs = paddle.utils.pack_sequence_as(kwargs, flat_kwargs)

    return args, kwargs, inner_proxy_tensor, outter_names

@no_eval_frame
# should generate a unique name for every function
def frame_enter(name, inputs):
    key_name = gen_inputs_key(name, inputs)
    SymbolicTraceContext().sir_key_stack.append(key_name)

    # if hit cache
    if sir_hit_cache(key_name):
        return True

    # create sir with specific name, 
    new_sir = SymbolicTraceContext().statement_factory.create(key_name)

    # add new_sir to stack
    SymbolicTraceContext().sir_stack.append(new_sir)

    # gen symbol inputs for new_sir
    flat_inputs = paddle.utils.flatten(inputs)
    new_sir.inputs = [Symbol(x.name) for x in flat_inputs if isinstance(x, ProxyTensor)]

    # gen and save origin_inputs (keep origin structure)
    origin_inputs = map_if(
        inputs,
        pred=lambda x: isinstance(x, ProxyTensor),
        true_fn=lambda x: x.meta,
        false_fn=lambda x: x,
    )
    SIRRuntimeCache().set_origin_inputs(key_name, origin_inputs)

    return False


@no_eval_frame
def frame_leave(name, outter_names, outputs):
    key_name = SymbolicTraceContext().sir_key_stack[-1]
    SymbolicTraceContext().sir_key_stack.pop()

    # fetch cur_sir at top of stack
    cur_sir = SymbolicTraceContext().sir_stack[-1]
    assert key_name == cur_sir.name

    # pop sir from stack
    SymbolicTraceContext().sir_stack.pop()

    # gen symbol outputs for cur_sir
    flat_outputs = paddle.utils.flatten(outputs)
    cur_sir.outputs = [Symbol(x.name) for x in flat_outputs if isinstance(x, ProxyTensor)]

    # gen and save origin_outputs for cur_sir
    origin_outputs = map_if(
        outputs, 
        pred=lambda x: isinstance(x, ProxyTensor), 
        true_fn=lambda x: x.meta, 
        false_fn=lambda x: x,
    )
    SIRRuntimeCache().set_origin_outputs(key_name, origin_outputs)

    # analyse free_vars
    all_inputs = cur_sir.analyse_inputs()
    free_vars = [x for x in all_inputs if x not in set(cur_sir.inputs)]
    if free_vars:
        if not cache_free_vars(cur_sir, free_vars):
            # cache sir failed.
            # 1. combine cur_sir with the sir at top of stack
            # 2. del sir in SymbolicTraceContext().statement_factory and SIRRuntimeCache
            # 3. should mark it as failed?
            return

    # at the first time, the inputs and outputs need not change
    SymbolicTraceContext().call_SIR(cur_sir.name, [Symbol(name) for name in outter_names], cur_sir.outputs)
    log(1, cur_sir, "\n")
    return


@no_eval_frame
def cache_and_return(name, outter_names):
    key_name = SymbolicTraceContext().sir_key_stack[-1]
    SymbolicTraceContext().sir_key_stack.pop()

    # find sir and it's origin_outputs
    cached_sir = SymbolicTraceContext().statement_factory[key_name]
    origin_outputs = SIRRuntimeCache().get_origin_outputs(key_name)

    # create return value
    outputs = gen_new_proxy_tensor_output(origin_outputs)

    # gen call_SIR outputs
    flat_outputs = paddle.utils.flatten(outputs)
    symbol_outputs = [Symbol(x.name) for x in flat_outputs if isinstance(x, ProxyTensor)]

    # add call_SIR
    SymbolicTraceContext().call_SIR(cached_sir.name, [Symbol(name) for name in outter_names], symbol_outputs)
    return outputs


def gen_inputs_key(name, inputs):
    key_list = []
    for inp in paddle.utils.flatten(inputs):
        if isinstance(inp, ProxyTensor):
            key_list.append(str(inp.meta))
        else:
            key_list.append(inp)

    inputs_key = hash(tuple(key_list))
    key = name + "@" + str(inputs_key)
    return key

def sir_hit_cache(key):
    if SIRRuntimeCache().has_key(key):
        origin_inputs, origin_outputs, _ = SIRRuntimeCache()[key]
        # this check is not property, maybe use class Empty instead
        if origin_inputs is not None and origin_outputs is not None:
            return True
    return False

def cache_free_vars(SIR, free_var_symbols):
    origin_inputs = SIRRuntimeCache().get_origin_inputs(SIR.name)
    flat_inputs = paddle.utils.flatten(origin_inputs)
    net = flat_inputs[0]
    if not isinstance(net, paddle.nn.Layer):
        return False

    param_name_to_var_name = {}
    runtime_map =  ProxyTensorContext().get_runtime()

    # here might slow
    for free_var_symbol in free_var_symbols:
        free_var = runtime_map[free_var_symbol.name].value_

        if not isinstance(free_var, paddle.fluid.framework.EagerParamBase):
            return False

        free_var_found = False
        for param_name, param in net.named_parameters():
            if free_var is param:
                param_name_to_var_name.update({param_name: free_var_symbol.name})
                free_var_found = True
                break

        if not free_var_found:
            return False

    def free_var_seeker():
        free_var_state = {}
        for param_name, param in net.named_parameters():
            if param_name in param_name_to_var_name.keys():
                free_var_state.update({param_name_to_var_name[param_name] : convert_load(param)})
        return free_var_state

    SIRRuntimeCache().set_free_vars(SIR.name, free_var_seeker)

    return True

def gen_new_proxy_tensor_output(origin_outputs):
    # need to find inplace operate with sir, but it is not used now
    def create_new_proxy_tensor_from_meta(meta):
        result = ProxyTensor(SymbolicTraceContext().new_varname(), meta)
        return result

    return map_if(
        origin_outputs, 
        pred=lambda x: isinstance(x, MetaInfo), 
        true_fn=create_new_proxy_tensor_from_meta,
        false_fn=lambda x: x
    )

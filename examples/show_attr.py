# TODO(zrr1999): the file should be removed before merge

import paddle
import paddle.static

tensor = paddle.to_tensor([[1, 2]])
tensor_dict = dict(tensor.__class__.__dict__)
paddle.enable_static()
var = paddle.static.data(name="x", shape=[10, 10], dtype='float32')
variable_dict = dict(var.__class__.__dict__)
paddle.disable_static()

supported_attr_list = ["T", "ndim", "shape", "size"]
gpu_attr_list = ["cuda"]
shared_keys = (
    set(tensor_dict).intersection(set(variable_dict))
    - set(supported_attr_list)
    - set(gpu_attr_list)
)

method_list = []
unbound_list = []
property_list = []
attr_list = []
magic_list = []

not_in_var_list = []

for k, v in tensor_dict.items():
    if k in shared_keys:
        if not k.startswith("_"):
            try:
                if "function" in str(v):
                    unbound_list.append((k, type(v(tensor))))
                elif "method" in str(v):
                    method_list.append((k, type(v(tensor))))
                elif "property" in str(v):
                    property_list.append(k)
                else:
                    attr_list.append(k)
            except Exception as e:
                # print("error", k, e)
                pass
        else:
            magic_list.append((k, v))
    else:
        not_in_var_list.append((k, v))
# print("method_list", method_list )
print("unbound_list", unbound_list)
# print("property_list",property_list)
# property_list ['ndim', 'size', 'T']
# print("attr_list",attr_list)
# attr_list ['name', 'stop_gradient', 'persistable', 'shape', 'place', 'dtype', 'type']

# print("magic_list",magic_list)
# print("not_in_var_list",not_in_var_list)

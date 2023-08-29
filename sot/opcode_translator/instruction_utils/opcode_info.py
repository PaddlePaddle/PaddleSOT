import opcode

REL_JUMP = {opcode.opname[x] for x in opcode.hasjrel}
ABS_JUMP = {opcode.opname[x] for x in opcode.hasjabs}
HAS_LOCAL = {opcode.opname[x] for x in opcode.haslocal}
HAS_FREE = {opcode.opname[x] for x in opcode.hasfree}
ALL_JUMP = REL_JUMP | ABS_JUMP
UNCONDITIONAL_JUMP = {"JUMP_ABSOLUTE", "JUMP_FORWARD"}


# Cache for some opcodes, it's for Python 3.11+
# https://github.com/python/cpython/blob/3.11/Include/internal/pycore_opcode.h#L41-L53
PYOPCODE_CACHE_SIZE = {
    "BINARY_SUBSCR": 4,
    "STORE_SUBSCR": 1,
    "UNPACK_SEQUENCE": 1,
    "STORE_ATTR": 4,
    "LOAD_ATTR": 4,
    "COMPARE_OP": 2,
    "LOAD_GLOBAL": 5,
    "BINARY_OP": 1,
    "LOAD_METHOD": 10,
    "PRECALL": 1,
    "CALL": 4,
}

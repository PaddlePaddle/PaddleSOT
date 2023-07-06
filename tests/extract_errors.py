import re
import sys

error_msg = sys.stdin.read()

pattern = r'File "?(.*?)"?, line (\d+),.*\n(.*?)\n(.*?)$'
for match in re.finditer(pattern, error_msg, re.MULTILINE):
    file = match.group(1)
    if not file.startswith("/"):
        file = f"tests/{file}"
    line = match.group(2)
    error_info = match.group(4)
    # error_info = match.group(3) + '\n' + match.group(4)
    output = f"::error file={file},line={line}::{error_info}"
    print(f"file={file},line={line}::{error_info}")
    print(output)

import re
import sys

error_msg = sys.stdin.read()

pattern = r'File "?(.*?)"?, line (\d+),.*\n(.*?)\n(.*?)$'
match = re.search(pattern, error_msg, re.MULTILINE)
if match:
    file = match.group(1)
    line = match.group(2)
    error_info = match.group(4)
    # error_info = match.group(3) + '\n' + match.group(4)
    output = f"::error file=tests/{file},line={line}::{error_info}"
    print(output)

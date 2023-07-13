import re
import sys

runtime_error_msg = sys.stdin.read()

pattern = r'File "?(.*?)"?, line (\d+),.*\n(.*?)\n(.*?)$'
for match in re.finditer(pattern, runtime_error_msg, re.MULTILINE):
    file = match.group(1)
    if file.startswith("./"):
        file = f"tests/{file[2:]}"
        line = match.group(2)
        error_info = match.group(4)
        # error_info = match.group(3) + '\n' + match.group(4)
        output = f"::error file={file},line={line}::Error"
        print(output)

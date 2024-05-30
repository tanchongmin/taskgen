import ast
import inspect
import re


def get_source_code_for_func(fn):
    if fn.__name__ == "<lambda>":
        source_line = inspect.getsource(fn)
        source_line = source_line.split('#')[0]
        match = re.search(r"\blambda\b[^:]+:.*", source_line).group(0)
        splits = [s for s in match.split(",") if s != ""]
        fn_code = splits[0]
        idx = 1
        while idx < len(splits):
            try:
                ast.parse(fn_code)
                break
            except SyntaxError as _:
                fn_code = fn_code + "," + splits[idx]
                idx = idx + 1
        while True:
            try:
                ast.parse(fn_code)
                break
            except SyntaxError as _:
                fn_code = fn_code[:-1]

        return fn_code
    else:
        return inspect.getsource(fn)
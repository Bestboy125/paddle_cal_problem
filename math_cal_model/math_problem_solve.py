import ast
import operator as op
import math
import re  # 添加这一行导入re模块

# 受控运算符映射表：仅允许的运算符
allowed_operators = {
    ast.Add: op.add,      # 加法
    ast.Sub: op.sub,      # 减法
    ast.Mult: op.mul,     # 乘法
    ast.Div: op.truediv,  # 真除法
    ast.Pow: op.pow       # 幂运算
}

def format_math_expression(expr):
    """
    格式化OCR识别的数学表达式，使其符合计算模块的要求。
    
    参数:
        expr (str): OCR识别的文本表达式。
    
    返回:
        str: 格式化后的表达式。
    """
    if not isinstance(expr, str):
        return ""
    
    # 转为字符串并去除首尾空格
    expr = expr.strip()
    
    # 常见符号替换
    replacements = {
        '×': '*',
        'X': '*',
        'x': '*',  # 注意小写x可能是变量，需要根据上下文处理
        '÷': '/',
        '−': '-',  # 负号
        '–': '-',  # en dash
        '—': '-',  # em dash
        'π': '3.141592653589793',
        '√': 'sqrt',
        '^': '**',
        '∧': '**',
        '≠': '!=',
        '≤': '<=',
        '≥': '>=',
        # 添加更多替换规则根据需要
    }
    
    for key, value in replacements.items():
        expr = expr.replace(key, value)
    
    # 移除所有非允许的字符（仅保留数字、运算符和括号）
    # 允许的字符：数字、点、小数点、加减乘除、括号、幂运算符
    allowed_chars_pattern = r'[^\d\.\+\-\*\/\^\(\) ]'
    expr = re.sub(allowed_chars_pattern, '', expr)
    
    # 移除多余的空格
    expr = expr.replace(' ', '')
    
    return expr

def eval_expression(expr):
    """
    安全地计算数学表达式，仅支持 allowed_operators 中定义的运算符及数字、括号。
    返回计算结果或 None (表示解析失败或不支持的操作)。
    """
    try:
        # 使用 ast.parse 将表达式解析为抽象语法树
        parsed_ast = ast.parse(expr, mode='eval')
        return _eval_ast(parsed_ast.body)
    except Exception:
        # 包括语法错误、零除错误等
        return None

def _eval_ast(node):
    """
    递归地对 AST 节点进行求值。
    仅处理我们允许的节点类型：BinOp(二元操作), UnaryOp(一元操作), Num(数字/常量), Expr(表达式)
    """
    if isinstance(node, ast.BinOp):
        left_val = _eval_ast(node.left)
        right_val = _eval_ast(node.right)
        op_type = type(node.op)

        if op_type in allowed_operators:
            # 防止除数为 0
            if op_type == ast.Div and right_val == 0:
                return None
            return allowed_operators[op_type](left_val, right_val)
        else:
            return None

    elif isinstance(node, ast.UnaryOp):
        # 一元操作（仅支持负号、正号）
        if isinstance(node.op, ast.UAdd):
            return _eval_ast(node.operand)
        elif isinstance(node.op, ast.USub):
            return -_eval_ast(node.operand)
        else:
            return None

    elif isinstance(node, ast.Num):
        # 兼容 Python < 3.8
        return node.n

    elif isinstance(node, ast.Constant):
        # Python 3.8+，数字常量统一使用 ast.Constant
        if isinstance(node.value, (int, float)):
            return node.value
        else:
            return None

    elif isinstance(node, ast.Expression):
        # 递归处理表达式节点
        return _eval_ast(node.body)

    else:
        # 其他节点不支持
        return None

def append_math_answers(associations):
    """
    在题干与 OCR 关联结果中，对 ocr_text 若为数学表达式则计算其结果，
    并在每条结果中新增 'correct_answer' 字段。
    
    参数:
        associations: [
            {
                'question_text': ...,
                'question_bbox': ...,
                'ocr_text': ...,
                'ocr_bbox': ...,
            },
            ...
        ]
    
    返回:
        与 associations 结构相同，但多了 'correct_answer' 字段:
        [
            {
                'question_text': ...,
                'question_bbox': ...,
                'ocr_text': ...,
                'ocr_bbox': ...,
                'correct_answer': 5  # 若可计算，否则 None
            },
            ...
        ]
    """
    for item in associations:
        # 取出 OCR 识别到的文本
        expr = item.get('question_text', '').strip()
        if not expr:
            # 若为空字符串，或你不认为是数学题，就直接给 None
            item['correct_answer'] = None
            continue
        
        # 格式化表达式
        formatted_expr = format_math_expression(expr)
        if not formatted_expr:
            item['correct_answer'] = None
            continue
        
        # 尝试计算表达式
        result = eval_expression(formatted_expr)
        item['correct_answer'] = result
    
    return associations

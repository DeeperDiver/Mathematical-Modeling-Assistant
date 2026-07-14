from modeling_assistant.memory.validation import validate_dynamic_ltm
from modeling_assistant.schemas.state import DynamicLTM


def test_validate_clean_ltm_passes():
    """符号定义完整、公式只引用已定义符号 → 通过。"""
    ltm = DynamicLTM(
        assumptions=["x 是自变量"],
        nomenclature={"x": "自变量", "y": "因变量"},
        equations=["y = 2 * x + 1"],
        objective="最小化 y",
        solution_outline="线性拟合",
    )
    errors = validate_dynamic_ltm(ltm)
    assert errors == []


def test_validate_undefined_symbol_in_equation():
    """公式引用了未定义符号 → 报错。"""
    ltm = DynamicLTM(
        nomenclature={"x": "自变量"},
        equations=["y = 2 * x + 1"],  # y 未在 nomenclature 中定义
        objective="最小化 y",
    )
    errors = validate_dynamic_ltm(ltm)
    assert any("y" in err for err in errors)


def test_validate_duplicate_descriptions():
    """不同符号有相同描述 → 警告歧义。"""
    ltm = DynamicLTM(
        nomenclature={
            "x1": "自变量",
            "x2": "自变量",  # 与 x1 描述相同
        },
        equations=["x1 + x2"],
        objective="最小化 x1 + x2",
    )
    errors = validate_dynamic_ltm(ltm)
    assert any("歧义" in err or "描述" in err for err in errors)


def test_validate_math_functions_are_reserved():
    """数学函数（sin、cos、log 等）不应被视为未定义符号。"""
    ltm = DynamicLTM(
        nomenclature={"x": "自变量", "y": "因变量"},  # y 也定义，避免误报
        equations=["y = sin(x) + cos(x)"],
        objective="最小化 y",
    )
    errors = validate_dynamic_ltm(ltm)
    # sin/cos 是保留字，不应出现在错误信息中
    joined = " ".join(errors)
    assert "sin" not in joined.replace("sin(x)", "")  # 排除方程文本中的 sin
    assert "cos" not in joined.replace("cos(x)", "")
    assert errors == []  # x、y 都已定义，sin/cos 是保留字 → 无错误


def test_validate_empty_ltm_passes():
    """空 LTM（无公式、无符号）应通过校验。"""
    ltm = DynamicLTM()
    errors = validate_dynamic_ltm(ltm)
    assert errors == []

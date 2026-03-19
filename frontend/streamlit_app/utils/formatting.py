from typing import Optional


def fmt_value(value, digits: int = 1, unit: str = "") -> str:
    if value is None:
        return "--"
    if isinstance(value, float):
        if value.is_integer():
            return f"{int(value)}{unit}"
        return f"{value:.{digits}f}{unit}"
    return f"{value}{unit}"


def parse_numeric_text(
    raw: str,
    label: str,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    integer: bool = False,
    default_value=None,
):
    value_text = (raw or "").strip().replace(",", "")
    if value_text == "":
        if default_value is None:
            raise ValueError(f"{label} 不能为空")
        number = int(default_value) if integer else float(default_value)
    else:
        try:
            number = int(value_text) if integer else float(value_text)
        except ValueError as exc:
            raise ValueError(f"{label} 请输入有效数字") from exc
    if min_value is not None and number < min_value:
        raise ValueError(f"{label} 不能小于 {min_value}")
    if max_value is not None and number > max_value:
        raise ValueError(f"{label} 不能大于 {max_value}")
    return number


def input_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    number = float(value)
    return str(int(number)) if number.is_integer() else f"{number:g}"

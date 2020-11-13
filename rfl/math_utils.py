from typing import Callable


def anneal_linear(src_value: float, dst_value: float, steps: int) -> Callable[[int], float]:
    return lambda k: (k / steps) * dst_value + (1 - k / steps) * src_value

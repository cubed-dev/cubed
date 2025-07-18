import math

try:
    from math import lcm  # type: ignore  # Python 3.9
except ImportError:

    def lcm(a: int, b: int) -> int:  # type: ignore
        """Implementation of `math.lcm()` for all Python versions."""
        # https://stackoverflow.com/a/51716959/809705
        return a * b // math.gcd(a, b)

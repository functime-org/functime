from __future__ import annotations

import sys
from typing import Literal

if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:  # 3.9
    from typing_extensions import TypeAlias


DetrendMethod: TypeAlias = Literal["linear", "mean"]

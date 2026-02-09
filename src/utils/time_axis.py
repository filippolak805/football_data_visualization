from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass(frozen=True)
class TimeAxis:
    origin: datetime  # global min datetime
    max_days: int

    def day_to_dt(self, day: int) -> datetime:
        return self.origin + timedelta(days=int(day))

    def dt_to_day(self, dt: datetime) -> int:
        delta = dt - self.origin
        return int(delta.total_seconds() // 86400)

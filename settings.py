"""Public application settings; secrets remain outside this model."""

import os

from pydantic import BaseModel, ConfigDict


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    search_enabled: bool = True

    @classmethod
    def from_environment(cls) -> "Settings":
        value = os.getenv("ADVISOR_SEARCH_ENABLED", "true").strip().lower()
        if value not in {"true", "false"}:
            raise ValueError("ADVISOR_SEARCH_ENABLED must be true or false.")
        return cls(search_enabled=value == "true")

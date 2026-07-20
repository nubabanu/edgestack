"""Runtime-only research-provider credentials loaded from environment or ``.env``."""

from __future__ import annotations

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class ResearchProviderCredentials(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        frozen=True,
        populate_by_name=True,
    )

    alpaca_api_key_id: SecretStr | None = Field(
        default=None, validation_alias="EDGESTACK_ALPACA_API_KEY_ID"
    )
    alpaca_api_secret_key: SecretStr | None = Field(
        default=None, validation_alias="EDGESTACK_ALPACA_API_SECRET_KEY"
    )
    fred_api_key: SecretStr | None = Field(default=None, validation_alias="EDGESTACK_FRED_API_KEY")
    sec_user_agent: str | None = Field(default=None, validation_alias="EDGESTACK_SEC_USER_AGENT")

    @staticmethod
    def reveal(value: SecretStr | None) -> str | None:
        if value is None:
            return None
        revealed = value.get_secret_value().strip()
        return revealed or None

    @property
    def alpaca_key_id(self) -> str | None:
        return self.reveal(self.alpaca_api_key_id)

    @property
    def alpaca_secret_key(self) -> str | None:
        return self.reveal(self.alpaca_api_secret_key)

    @property
    def fred_key(self) -> str | None:
        return self.reveal(self.fred_api_key)

    @property
    def sec_agent(self) -> str | None:
        value = (self.sec_user_agent or "").strip()
        return value or None

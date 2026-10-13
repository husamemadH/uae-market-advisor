"""Local session commands, independent of inference and of a particular frontend."""

from copy import deepcopy
from uuid import UUID, uuid4

from agno.db.base import SessionType
from agno.db.sqlite import SqliteDb
from agno.session.agent import AgentSession
from sqlalchemy.exc import SQLAlchemyError

from profiles import PROFILE_KEY, BusinessProfile, change_profile, profile_from_state


class SessionStorageError(RuntimeError):
    def __init__(self):
        super().__init__("Could not read or save local session storage. Please check runtime/.")


def validate_session_id(session_id: str) -> str:
    try:
        canonical = str(UUID(session_id))
    except (ValueError, AttributeError) as exc:
        raise ValueError(
            "Invalid session ID. Use the UUID printed when the conversation started."
        ) from exc
    if session_id != canonical:
        raise ValueError("Invalid session ID. Use the UUID exactly as printed by the application.")
    return canonical


class SessionStore:
    def __init__(self, db: SqliteDb):
        self.db = db

    def load(self, session_id: str) -> AgentSession:
        validate_session_id(session_id)
        try:
            session = self.db.get_session(session_id, session_type=SessionType.AGENT)
        except (OSError, SQLAlchemyError) as exc:
            raise SessionStorageError() from exc
        if session is None:
            raise ValueError(
                "Unknown session ID. Start a new conversation or use a saved session ID."
            )
        if not isinstance(session, AgentSession):
            raise ValueError("This session does not belong to the advisor.")
        # Agno may cache run objects. Callers must never mutate those shared objects.
        return deepcopy(session)

    def _save(self, session: AgentSession) -> None:
        try:
            saved = self.db.upsert_session(session)
        except (OSError, SQLAlchemyError) as exc:
            raise SessionStorageError() from exc
        if saved is None:
            raise SessionStorageError()

    def new(self) -> str:
        session_id = str(uuid4())
        self._save(
            AgentSession(
                session_id=session_id,
                session_data={
                    "session_state": {PROFILE_KEY: BusinessProfile().model_dump(mode="json")}
                },
            )
        )
        return session_id

    def profile(self, session_id: str) -> BusinessProfile:
        session = self.load(session_id)
        return profile_from_state((session.session_data or {}).get("session_state"))

    def update_profile(self, session_id: str, field: str, value: str | None) -> BusinessProfile:
        session = self.load(session_id)
        data = session.session_data if session.session_data is not None else {}
        state = data.get("session_state")
        updated = change_profile(profile_from_state(state), field, value)
        session.session_data = {
            **data,
            "session_state": {**(state or {}), PROFILE_KEY: updated.model_dump(mode="json")},
        }
        self._save(session)
        return updated

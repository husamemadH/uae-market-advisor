"""Observe the synchronous Agno boundaries without replacing its tool/model loop."""

from copy import copy
from time import perf_counter

from agno.db.sqlite import SqliteDb
from agno.models.openrouter import OpenRouter
from agno.run import RunStatus

from observability import CURRENT_TURN, RequestUsage, capture_usage, error_category
from sessions import SessionStorageError


class ObservedOpenRouter(OpenRouter):
    def invoke(self, *args, **kwargs):
        trace = CURRENT_TURN.get()
        request = RequestUsage()
        if trace is not None:
            trace.requests.append(request)
        start = perf_counter()
        try:
            result = super().invoke(*args, **kwargs)
            request.outcome = "completed"
            return result
        except (Exception, KeyboardInterrupt) as exc:
            category = error_category(exc)
            request.outcome = "cancelled" if category == "cancelled" else "error"
            if trace is not None:
                trace.error = category
            raise
        finally:
            request.duration_ms = (perf_counter() - start) * 1000

    def _parse_provider_response(self, response, response_format=None):
        trace = CURRENT_TURN.get()
        if trace is not None and trace.requests:
            capture_usage(trace.requests[-1], response)
        return super()._parse_provider_response(response, response_format)


class ObservedSqliteDb(SqliteDb):
    def _observe(self, method, *args, **kwargs):
        try:
            return method(*args, **kwargs)
        except Exception as exc:
            if trace := CURRENT_TURN.get():
                trace.storage_failed = True
            raise SessionStorageError() from exc

    def get_session(self, *args, **kwargs):
        return self._observe(super().get_session, *args, **kwargs)

    def get_run(self, *args, **kwargs):
        return self._observe(super().get_run, *args, **kwargs)

    def upsert_session(self, *args, **kwargs):
        result = self._observe(super().upsert_session, *args, **kwargs)
        if result is None:
            if trace := CURRENT_TURN.get():
                trace.storage_failed = True
            raise SessionStorageError()
        return result

    def upsert_run(self, run, *args, **kwargs):
        trace = CURRENT_TURN.get()
        if trace is not None and trace.storage_failed:
            run = copy(run)
            run.status = RunStatus.error
        return self._observe(super().upsert_run, run, *args, **kwargs)

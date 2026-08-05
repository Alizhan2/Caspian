import logging
import uuid
from contextvars import ContextVar

request_id_context: ContextVar[str] = ContextVar("request_id", default="-")


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_context.get()
        return True


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s [%(request_id)s] %(name)s: %(message)s",
    )
    request_filter = RequestIdFilter()
    root = logging.getLogger()
    root.addFilter(request_filter)
    for handler in root.handlers:
        handler.addFilter(request_filter)


def new_request_id() -> str:
    return str(uuid.uuid4())

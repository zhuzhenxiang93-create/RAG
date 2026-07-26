"""Request correlation and safe access logging."""

import logging
import re
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, Request, Response

logger = logging.getLogger("docmind.access")
REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def register_request_tracking(app: FastAPI) -> None:
    @app.middleware("http")
    async def track_request(request: Request, call_next) -> Response:
        supplied = request.headers.get("x-request-id", "")
        request_id = supplied if REQUEST_ID_PATTERN.fullmatch(supplied) else uuid4().hex
        request.state.request_id = request_id
        started = perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            duration_ms = round((perf_counter() - started) * 1000, 3)
            logger.info(
                "request_completed",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": status_code,
                    "duration_ms": duration_ms,
                },
            )
            if "response" in locals():
                response.headers["X-Request-ID"] = request_id
                response.headers["Server-Timing"] = "app;dur={}".format(duration_ms)

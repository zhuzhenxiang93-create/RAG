"""Application exceptions and normalized error responses."""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


class AppError(Exception):
    """Expected application error safe to expose to an API client."""

    def __init__(self, message: str, *, code: str = "app_error", status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code


def register_exception_handlers(app: FastAPI) -> None:
    """Register handlers for expected application failures."""

    @app.exception_handler(AppError)
    async def handle_app_error(_: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": exc.code, "message": exc.message}},
        )

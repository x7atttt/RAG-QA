import os
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.response import ResponseCode, error_response

# 静态目录（用于 404 时返回友好的 HTML 页面）
_STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "static")


class BizError(Exception):
    def __init__(self, code: int, message: str, http_status: int = 400, data: Any = None):
        self.code = code
        self.message = message
        self.http_status = http_status
        self.data = data
        super().__init__(message)


class AuthError(BizError):
    def __init__(self, code: int = ResponseCode.AUTH_FAILED, message: str = "认证失败", http_status: int = 401):
        super().__init__(code, message, http_status)


def _wants_html(request: Request) -> bool:
    """判断客户端是否期望 HTML 响应（浏览器导航场景）。"""
    accept = request.headers.get("accept", "")
    # 浏览器请求页面时 Accept 通常含 text/html；fetch/API 调用一般是 application/json 或 */*
    return "text/html" in accept and "application/json" not in accept.split(",")[0]


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(BizError)
    async def _biz_error_handler(_: Request, exc: BizError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.http_status,
            content={"code": exc.code, "message": exc.message, "data": exc.data},
        )

    @app.exception_handler(AuthError)
    async def _auth_error_handler(_: Request, exc: AuthError) -> JSONResponse:
        return JSONResponse(status_code=exc.http_status, content=error_response(exc.code, exc.message))

    @app.exception_handler(StarletteHTTPException)
    async def _http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse | FileResponse:
        # 浏览器导航到不存在的页面 → 返回友好 HTML（而非裸 JSON）
        if exc.status_code == 404 and _wants_html(request):
            not_found = os.path.join(_STATIC_DIR, "404.html")
            if os.path.exists(not_found):
                return FileResponse(not_found, status_code=404)

        message = exc.detail if isinstance(exc.detail, str) else "请求错误"
        code_map = {
            401: ResponseCode.AUTH_FAILED,
            404: ResponseCode.NOT_FOUND,
            405: ResponseCode.BAD_REQUEST,
            422: ResponseCode.VALIDATION_ERROR,
        }
        code = code_map.get(exc.status_code, ResponseCode.BAD_REQUEST)
        return JSONResponse(status_code=exc.status_code, content=error_response(code, message))

    @app.exception_handler(RequestValidationError)
    async def _validation_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=error_response(ResponseCode.VALIDATION_ERROR, "参数校验失败"),
        )

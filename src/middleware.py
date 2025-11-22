import datetime
import logging
import uuid
from datetime import timezone

import structlog
from starlette.types import ASGIApp, Receive, Scope, Send


class LoggerMiddleWare:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ):
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        logger: structlog.stdlib.BoundLogger = structlog.get_logger()
        request_start = datetime.datetime.now(timezone.utc)

        method = scope.get("method", "")
        path = scope.get("path", "")
        client = scope.get("client", ("", 0))

        request_id = None
        headers = {
            k.decode("latin1").lower(): v.decode("latin1")
            for k, v in scope.get("headers", [])
        }
        request_id = headers.get("x-request-id", str(uuid.uuid4()))
        logger = logger.bind(
            request_url=path,
            request_id=request_id,
        )

        scope["state"] = scope.get("state", {})
        scope["state"]["logger"] = logger

        response_status = [None]

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                response_status[0] = message["status"]
            await send(message)

        await self.app(scope, receive, send_wrapper)

        request_ended = datetime.datetime.now(timezone.utc)

        protocol = "https" if headers.get("x-forwarded-proto") == "https" else "http"

        logger.log(
            logging.INFO,
            "request_log",
            request_start=request_start.isoformat(),
            request_ended=request_ended.isoformat(),
            ip_address=client[0] if client else "",
            http_method=method,
            endpoint=path,
            protocol=protocol,
            status_code=response_status[0],
        )


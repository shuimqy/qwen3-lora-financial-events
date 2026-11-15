import logging

import uvicorn

from .app import create_app
from .config import Settings


def main():
    settings = Settings()
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    uvicorn.run(
        create_app(settings), host=settings.host, port=settings.port,
        log_level=settings.log_level, workers=1,
    )

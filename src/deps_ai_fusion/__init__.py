import logging

from deps_ai_fusion.settings import Settings

_LOG_LEVEL = getattr(logging, Settings().logger_level.upper(), logging.INFO)
logging.basicConfig(
    level=_LOG_LEVEL,
    format="[%(asctime)s] [%(name)s: %(levelname)s] %(message)s",  # noqa: WPS323
    datefmt="%Y-%m-%d %I:%M:%S",  # noqa: WPS323
)
logging.getLogger().setLevel(_LOG_LEVEL)

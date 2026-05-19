"""Elastic EP scaling HTTP endpoints for dp_attention deployments."""

import logging
from http import HTTPStatus

from fastapi import APIRouter, Request
from fastapi.responses import ORJSONResponse

from sglang.srt.utils.auth import AuthLevel, auth_level

logger = logging.getLogger(__name__)

router = APIRouter()


def _classify_scale_failure(message: str) -> int:
    """Map known scheduler validation failures to HTTP status codes.

    Reserve 500 for unexpected exceptions; the validations below are
    user-precondition errors (400) or transient state conflicts (409).
    """
    if "scale-down is handled separately" in message:
        return HTTPStatus.BAD_REQUEST
    if "exceeds --max-ep-size" in message:
        return HTTPStatus.BAD_REQUEST
    if "previous scale operation has not completed" in message:
        return HTTPStatus.CONFLICT
    return HTTPStatus.INTERNAL_SERVER_ERROR


@router.post("/scale_elastic_ep")
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def scale_elastic_ep(raw_request: Request):
    """Scale EP size at runtime. Returns immediately — rank joining is async."""
    try:
        body = await raw_request.json()
    except Exception as e:
        return ORJSONResponse(
            {"error": f"Invalid JSON: {e}"},
            status_code=HTTPStatus.BAD_REQUEST,
        )

    new_ep_size = body.get("new_ep_size")
    if new_ep_size is None or not isinstance(new_ep_size, int) or new_ep_size <= 0:
        return ORJSONResponse(
            {"error": "new_ep_size must be a positive integer"},
            status_code=HTTPStatus.BAD_REQUEST,
        )

    from sglang.srt.entrypoints.http_server import _global_state
    from sglang.srt.managers.io_struct import ScaleElasticEPReqInput

    req = ScaleElasticEPReqInput(new_ep_size=new_ep_size)
    result = await _global_state.tokenizer_manager.scale_elastic_ep(req)

    if not result.success:
        return ORJSONResponse(
            {"error": result.message},
            status_code=_classify_scale_failure(result.message),
        )

    return ORJSONResponse(
        {
            "message": result.message,
            "old_ep_size": result.old_ep_size,
            "new_ep_size": result.new_ep_size,
        }
    )


@router.post("/is_scaling_elastic_ep")
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def is_scaling_elastic_ep(raw_request: Request):
    """Query each DP scheduler for its scaling state and OR the results.

    The HTTP server runs in the TokenizerManager process which does NOT
    initialize ElasticEPStateManager (that lives in the scheduler / model
    worker). We therefore round-trip through the existing get_internal_state
    pipeline, which returns one dict per DP rank.
    """
    from sglang.srt.entrypoints.http_server import _global_state

    states = await _global_state.tokenizer_manager.get_internal_state()
    is_scaling = any(s.get("is_scaling_elastic_ep", False) for s in states)
    return ORJSONResponse({"is_scaling_elastic_ep": is_scaling})

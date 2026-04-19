"""Elastic EP scaling HTTP endpoints for dp_attention deployments."""

import logging
from http import HTTPStatus

from fastapi import APIRouter, Request
from fastapi.responses import ORJSONResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/scale_elastic_ep")
async def scale_elastic_ep(raw_request: Request):
    """Scale EP size at runtime. Returns immediately — rank joining is async."""
    try:
        body = await raw_request.json()
    except Exception as e:
        return ORJSONResponse(
            {"error": f"Invalid JSON: {e}"},
            status_code=HTTPStatus.BAD_REQUEST,
        )

    new_tp_size = body.get("new_tp_size")
    if new_tp_size is None or not isinstance(new_tp_size, int) or new_tp_size <= 0:
        return ORJSONResponse(
            {"error": "new_tp_size must be a positive integer"},
            status_code=HTTPStatus.BAD_REQUEST,
        )

    from sglang.srt.entrypoints.http_server import _global_state
    from sglang.srt.managers.io_struct import ScaleElasticEPReqInput

    req = ScaleElasticEPReqInput(new_tp_size=new_tp_size)
    result = await _global_state.tokenizer_manager.scale_elastic_ep(req)

    if not result.success:
        return ORJSONResponse(
            {"error": result.message},
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
        )

    return ORJSONResponse(
        {
            "message": result.message,
            "old_tp_size": result.old_tp_size,
            "new_tp_size": result.new_tp_size,
        }
    )


@router.post("/is_scaling_elastic_ep")
async def is_scaling_elastic_ep(raw_request: Request):
    from sglang.srt.elastic_ep.elastic_ep import ElasticEPStateManager

    inst = ElasticEPStateManager.instance()
    is_scaling = inst is not None and getattr(inst, "scaling_in_progress", False)
    return ORJSONResponse({"is_scaling_elastic_ep": is_scaling})

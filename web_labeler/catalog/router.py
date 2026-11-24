"""HTTP API of the Catalog. The import runs from the command line; the only writes here are admin decisions."""
from __future__ import annotations

import base64
import binascii

from typing import Any, Dict, Optional

from fastapi import APIRouter, Body, HTTPException, Query
from fastapi.responses import FileResponse, Response

from web_labeler.catalog import store
from web_labeler.catalog.examples import ExampleNotFound
from web_labeler.catalog.store import (CatalogError, CatalogSettings, CatalogUnavailable, UnknownEvidence, UnknownModel,
                                       UnknownPage)


def build_catalog_router(settings: CatalogSettings) -> APIRouter:
    router = APIRouter(prefix="/api/catalog")

    @router.get("/status")
    def catalog_status() -> Dict[str, Any]:
        return store.catalog_status(settings)

    @router.get("/models")
    def catalog_models() -> Dict[str, Any]:
        try:
            return {"models": store.list_models(settings)}
        except CatalogUnavailable as e:
            raise HTTPException(status_code=503, detail=str(e))

    @router.get("/models/{model}/classes")
    def catalog_classes(model: str) -> Dict[str, Any]:
        try:
            return store.list_model_view(settings, model)
        except UnknownModel:
            raise HTTPException(status_code=404, detail="Taj model ne postoji u Katalogu.")
        except CatalogUnavailable as e:
            raise HTTPException(status_code=503, detail=str(e))

    @router.get("/models/{model}/classes/{class_name}/variations")
    def catalog_variations(model: str, class_name: str) -> Dict[str, Any]:
        try:
            return {"model": model, "class": class_name,
                    "variations": store.list_variations(settings, model, class_name)}
        except UnknownModel:
            raise HTTPException(status_code=404, detail="Ta klasa ne postoji u Katalogu.")
        except CatalogUnavailable as e:
            raise HTTPException(status_code=503, detail=str(e))

    @router.get("/models/{model}/classes/{class_name}/details")
    def catalog_class_detail(model: str, class_name: str) -> Dict[str, Any]:
        try:
            return store.class_detail(settings, model, class_name)
        except UnknownModel:
            raise HTTPException(status_code=404, detail="Ta klasa ne postoji u Katalogu.")
        except CatalogUnavailable as e:
            raise HTTPException(status_code=503, detail=str(e))

    @router.get("/source-report")
    def catalog_source_report() -> Dict[str, Any]:
        try:
            return store.source_report(settings)
        except CatalogUnavailable as e:
            raise HTTPException(status_code=503, detail=str(e))

    @router.get("/unclassified")
    def catalog_unclassified() -> Dict[str, Any]:
        try:
            return store.list_unclassified(settings)
        except CatalogUnavailable as e:
            raise HTTPException(status_code=503, detail=str(e))

    @router.get("/models/{model}/classes/{class_name}/examples")
    def catalog_examples(model: str, class_name: str, count: int = Query(6), seed: Optional[int] = Query(None)) -> Dict[str, Any]:
        try:
            return store.class_examples(settings, model, class_name, count, seed)
        except UnknownModel:
            raise HTTPException(status_code=404, detail="Taj model ne postoji u Katalogu.")
        except CatalogUnavailable as e:
            raise HTTPException(status_code=503, detail=str(e))

    @router.get("/models/{model}/classes/{class_name}/examples/{file_name}")
    def catalog_example_image(model: str, class_name: str, file_name: str, box: int = Query(0)) -> Response:
        try:
            data = store.example_image(settings, model, class_name, file_name, box)
        except (UnknownModel, ExampleNotFound):
            raise HTTPException(status_code=404, detail="Primer nije pronađen.")
        except CatalogUnavailable as e:
            raise HTTPException(status_code=503, detail=str(e))
        return Response(content=data, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=3600"})

    @router.get("/miro-pages")
    def catalog_miro_pages() -> Dict[str, Any]:
        try:
            return store.list_miro_pages(settings)
        except CatalogUnavailable as e:
            raise HTTPException(status_code=503, detail=str(e))

    @router.get("/miro-pages/{page_key}/image")
    def catalog_miro_image(page_key: str) -> FileResponse:
        try:
            return FileResponse(store.miro_page_image(settings, page_key), media_type="image/jpeg",
                                headers={"Cache-Control": "private, max-age=86400"})
        except UnknownPage:
            raise HTTPException(status_code=404, detail="Ta strana ne postoji.")
        except CatalogUnavailable as e:
            raise HTTPException(status_code=503, detail=str(e))

    @router.post("/miro-pages/{page_key}")
    def catalog_save_miro_page(page_key: str, req: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        classes = req.get("classes")
        if req.get("reset") is not True and (not isinstance(classes, list) or not all(isinstance(c, str) for c in classes)):
            raise HTTPException(status_code=400, detail="Pošalji spisak klasa.")
        try:
            if req.get("reset") is True:
                return store.reset_miro_page(settings, page_key)
            return store.save_miro_page(settings, page_key, classes)
        except UnknownPage:
            raise HTTPException(status_code=404, detail="Ta strana ne postoji.")
        except CatalogError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except CatalogUnavailable as e:
            raise HTTPException(status_code=503, detail=str(e))

    @router.get("/without-image")
    def catalog_without_image(model: Optional[str] = Query(None)) -> Dict[str, Any]:
        try:
            return store.list_without_image(settings, model or None)
        except CatalogUnavailable as e:
            raise HTTPException(status_code=503, detail=str(e))

    @router.get("/evidence")
    def catalog_evidence(status: Optional[str] = Query("pending")) -> Dict[str, Any]:
        if status not in (None, "", "all", "pending", "confirmed", "rejected"):
            raise HTTPException(status_code=400, detail="Nepoznat status.")
        try:
            return store.list_evidence(settings, None if status in (None, "", "all") else status)
        except CatalogUnavailable as e:
            raise HTTPException(status_code=503, detail=str(e))

    @router.get("/evidence/{evidence_id}/image")
    def catalog_evidence_image(evidence_id: int) -> FileResponse:
        try:
            path, media_type = store.evidence_image(settings, evidence_id)
        except UnknownEvidence:
            raise HTTPException(status_code=404, detail="Slika nije pronađena.")
        except CatalogUnavailable as e:
            raise HTTPException(status_code=503, detail=str(e))
        return FileResponse(path, media_type=media_type, headers={"Cache-Control": "private, max-age=86400"})

    @router.post("/evidence")
    def catalog_add_evidence(req: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        identity_id, image = req.get("identity_id"), req.get("image_base64")
        if not isinstance(identity_id, int) or not isinstance(image, str):
            raise HTTPException(status_code=400, detail="Izaberi artikal i nalepi sliku.")
        if len(image) > store.EVIDENCE_MAX_BYTES * 4 // 3 + 16:
            raise HTTPException(status_code=400, detail="Slika je prevelika (najviše 10 MB). Iseci manji deo ekrana.")
        try:
            data = base64.b64decode(image.split(",", 1)[-1], validate=True)
        except (binascii.Error, ValueError):
            raise HTTPException(status_code=400, detail="Ovo nije slika koju alat može da pročita.")
        try:
            return store.add_evidence(settings, identity_id, data, req.get("added_by"), req.get("role"),
                                      req.get("model") or None, req.get("class") or None)
        except UnknownEvidence:
            raise HTTPException(status_code=404, detail="Taj artikal ne postoji u Katalogu.")
        except CatalogError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except CatalogUnavailable as e:
            raise HTTPException(status_code=503, detail=str(e))

    @router.post("/evidence/{evidence_id}/decision")
    def catalog_decide_evidence(evidence_id: int, req: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        try:
            return store.decide_evidence(settings, evidence_id, str(req.get("decision") or ""),
                                         req.get("decided_by"), str(req.get("reason") or ""))
        except UnknownEvidence:
            raise HTTPException(status_code=404, detail="Slika nije pronađena.")
        except CatalogError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except CatalogUnavailable as e:
            raise HTTPException(status_code=503, detail=str(e))

    return router

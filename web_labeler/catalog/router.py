"""HTTP API of the Catalog. The import runs from the command line; the only writes here are admin decisions."""
from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, Body, HTTPException, Query
from fastapi.responses import FileResponse, Response

from web_labeler.catalog import store
from web_labeler.catalog.examples import ExampleNotFound
from web_labeler.catalog.store import CatalogError, CatalogSettings, CatalogUnavailable, UnknownModel, UnknownPage


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

    return router

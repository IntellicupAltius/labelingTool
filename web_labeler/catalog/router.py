"""HTTP API of the Catalog. Read-only for now; the import runs from the command line."""
from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from web_labeler.catalog import store
from web_labeler.catalog.store import CatalogSettings, CatalogUnavailable, UnknownModel


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

    @router.get("/unclassified")
    def catalog_unclassified() -> Dict[str, Any]:
        try:
            return store.list_unclassified(settings)
        except CatalogUnavailable as e:
            raise HTTPException(status_code=503, detail=str(e))

    return router

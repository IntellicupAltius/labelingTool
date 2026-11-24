"""HTTP API of the Catalog. The import runs from the command line; the only writes here are admin decisions."""
from __future__ import annotations

import base64
import binascii
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, Body, HTTPException, Query
from fastapi.responses import FileResponse, Response

from web_labeler.catalog import store
from web_labeler.catalog import model_cards as _model_cards
from web_labeler.catalog.examples import ExampleNotFound
from web_labeler.catalog.store import (CatalogError, CatalogSettings, CatalogUnavailable, NotAllowed, UnknownArticle,
                                       UnknownEvidence, UnknownModel, UnknownPage)


_REPO_ROOT = Path(__file__).resolve().parents[2]
_refresh: Dict[str, Any] = {"proc": None}


def start_model_cards_refresh(settings: CatalogSettings) -> bool:
    """Rebuild the model cards in a separate process (never in the web process). False when one already runs."""
    proc = _refresh["proc"]
    if proc is not None and proc.poll() is None:
        return False
    target = store.model_cards_snapshot_path(settings)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target.with_suffix(".log"), "ab") as log:
        _refresh["proc"] = subprocess.Popen([sys.executable, "-m", "web_labeler.catalog", "model-cards"], cwd=str(_REPO_ROOT),
                                            stdout=log, stderr=log, stdin=subprocess.DEVNULL, start_new_session=True)
    return True


def build_catalog_router(settings: CatalogSettings) -> APIRouter:
    router = APIRouter(prefix="/api/catalog")

    @router.get("/model-cards")
    def catalog_model_cards() -> Dict[str, Any]:
        target = store.model_cards_snapshot_path(settings)
        if target is None:
            raise HTTPException(status_code=404, detail="Kartice modela nisu podešene na ovom serveru.")
        snapshot = _model_cards.read_snapshot(target)
        stale = snapshot is None or snapshot.get("fingerprint") != _model_cards.fingerprint(settings.model_cards)
        if stale:
            try:
                start_model_cards_refresh(settings)
            except OSError:
                pass
        return {"refreshing": stale, "snapshot": snapshot}

    def blocked(e: store.ReleaseBlocked) -> HTTPException:
        return HTTPException(status_code=409, detail={"message": str(e), "checks": e.checks})

    @router.get("/releases")
    def catalog_releases() -> Dict[str, Any]:
        try:
            return store.list_releases(settings)
        except CatalogUnavailable as e:
            raise HTTPException(status_code=503, detail=str(e))

    @router.get("/releases/next")
    def catalog_next_release() -> Dict[str, Any]:
        try:
            return store.next_release(settings)
        except CatalogUnavailable as e:
            raise HTTPException(status_code=503, detail=str(e))
        except CatalogError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @router.get("/releases/{number}/files/{path:path}")
    def catalog_release_file(number: int, path: str) -> Response:
        try:
            return Response(store.release_file(settings, number, path), media_type="text/plain; charset=utf-8")
        except store.UnknownRelease:
            raise HTTPException(status_code=404, detail="Taj fajl ne postoji u tom izdanju.")
        except CatalogUnavailable as e:
            raise HTTPException(status_code=503, detail=str(e))

    @router.post("/releases")
    def catalog_publish(req: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        try:
            return store.publish_release(settings, req.get("actor"), req.get("role"), req.get("reason"))
        except NotAllowed as e:
            raise HTTPException(status_code=403, detail=str(e))
        except store.ReleaseBlocked as e:
            raise blocked(e)
        except CatalogError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except CatalogUnavailable as e:
            raise HTTPException(status_code=503, detail=str(e))

    @router.post("/releases/{number}/rollback")
    def catalog_rollback(number: int, req: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        try:
            return store.rollback_release(settings, number, req.get("actor"), req.get("role"), req.get("reason"))
        except NotAllowed as e:
            raise HTTPException(status_code=403, detail=str(e))
        except store.UnknownRelease:
            raise HTTPException(status_code=404, detail="To izdanje ne postoji.")
        except store.ReleaseBlocked as e:
            raise blocked(e)
        except CatalogError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @router.get("/lock")
    def catalog_lock_status() -> Dict[str, Any]:
        try:
            return store.lock_status(settings)
        except CatalogUnavailable as e:
            raise HTTPException(status_code=503, detail=str(e))

    @router.post("/lock")
    def catalog_lock(req: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        action = store.unlock_catalog if req.get("action") == "unlock" else store.lock_catalog
        try:
            return action(settings, req.get("actor"), req.get("role"), req.get("reason"))
        except NotAllowed as e:
            raise HTTPException(status_code=403, detail=str(e))
        except store.ReleaseBlocked as e:
            raise blocked(e)
        except CatalogError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @router.get("/findings")
    def catalog_findings(status: Optional[str] = Query("open")) -> Dict[str, Any]:
        try:
            return store.list_findings(settings, status)
        except CatalogUnavailable as e:
            raise HTTPException(status_code=503, detail=str(e))

    @router.post("/findings/{finding_id}")
    def catalog_decide_finding(finding_id: int, req: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        try:
            return store.decide_finding(settings, finding_id, str(req.get("action") or ""), req.get("actor"), req.get("role"),
                                        req.get("reason"))
        except NotAllowed as e:
            raise HTTPException(status_code=403, detail=str(e))
        except store.UnknownFinding:
            raise HTTPException(status_code=404, detail="Taj nalaz ne postoji.")
        except CatalogError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @router.get("/day-table")
    def catalog_day_table() -> Dict[str, Any]:
        try:
            return store.day_table(settings)
        except CatalogUnavailable as e:
            raise HTTPException(status_code=503, detail=str(e))

    @router.post("/day-table/{day}")
    def catalog_mark_day(day: str, req: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        try:
            return store.mark_day(settings, day, str(req.get("action") or ""), req.get("actor"), req.get("role"), req.get("reason"))
        except CatalogError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @router.get("/pos-days")
    def catalog_pos_days() -> Dict[str, Any]:
        try:
            return store.list_pos_days(settings)
        except CatalogUnavailable as e:
            raise HTTPException(status_code=503, detail=str(e))

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

    @router.get("/class-choices")
    def catalog_class_choices() -> Dict[str, Any]:
        try:
            return {"models": store.class_choices(settings)}
        except CatalogUnavailable as e:
            raise HTTPException(status_code=503, detail=str(e))

    @router.get("/articles")
    def catalog_articles(q: Optional[str] = Query(None), state: Optional[str] = Query(None)) -> Dict[str, Any]:
        if state not in (None, "", "tracked", "ignored", "pending", "dormant", "missing"):
            raise HTTPException(status_code=400, detail="Nepoznato stanje.")
        try:
            return store.list_articles(settings, q, state or None)
        except CatalogUnavailable as e:
            raise HTTPException(status_code=503, detail=str(e))

    @router.get("/articles/{article_id}")
    def catalog_article(article_id: int, as_of: Optional[str] = Query(None)) -> Dict[str, Any]:
        try:
            return store.article_card(settings, article_id, store._valid_date(as_of) if as_of else None)
        except UnknownArticle:
            raise HTTPException(status_code=404, detail="Taj artikal ne postoji u Katalogu.")
        except CatalogError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except CatalogUnavailable as e:
            raise HTTPException(status_code=503, detail=str(e))

    @router.get("/decisions")
    def catalog_decisions(kind: Optional[str] = Query(None)) -> Dict[str, Any]:
        try:
            return store.list_decisions(settings, kind or None)
        except CatalogUnavailable as e:
            raise HTTPException(status_code=503, detail=str(e))

    @router.post("/decisions")
    def catalog_record_decision(req: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        def opt_int(key: str) -> Optional[int]:
            value = req.get(key)
            if value is None:
                return None
            if not isinstance(value, int):
                raise HTTPException(status_code=400, detail=f"{key} mora biti broj.")
            return value
        payload = req.get("payload") or {}
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="Neispravna odluka.")
        try:
            return store.record_decision(
                settings, str(req.get("kind") or ""), req.get("actor"), req.get("role"), article_id=opt_int("article_id"),
                identity_id=opt_int("identity_id"), payload=payload, reason=req.get("reason"), evidence=req.get("evidence"),
                valid_from=req.get("valid_from"), target_id=opt_int("target_id"))
        except NotAllowed as e:
            raise HTTPException(status_code=403, detail=str(e))
        except UnknownArticle:
            raise HTTPException(status_code=404, detail="Taj artikal ne postoji u Katalogu.")
        except CatalogError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except CatalogUnavailable as e:
            raise HTTPException(status_code=503, detail=str(e))

    return router

"""Assortment Catalog: a self-contained module with its own database.

It never touches the tool's loaded video or annotation state, and it never writes to RAW or to
the class list files it imports from.
"""
from web_labeler.catalog.store import (
    CatalogError,
    CatalogSettings,
    CatalogUnavailable,
    UnknownModel,
    catalog_status,
    import_class_lists,
    list_classes,
    list_models,
)

__all__ = [
    "CatalogError", "CatalogSettings", "CatalogUnavailable", "UnknownModel",
    "catalog_status", "import_class_lists", "list_classes", "list_models",
]

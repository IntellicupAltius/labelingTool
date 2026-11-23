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
    class_detail,
    import_catalog,
    import_class_lists,
    list_classes,
    list_model_view,
    list_models,
    list_unclassified,
    list_variations,
    source_report,
)

__all__ = [
    "CatalogError", "CatalogSettings", "CatalogUnavailable", "UnknownModel",
    "catalog_status", "import_catalog", "import_class_lists", "list_classes", "list_model_view",
    "list_models", "list_unclassified", "list_variations", "class_detail", "source_report",
]

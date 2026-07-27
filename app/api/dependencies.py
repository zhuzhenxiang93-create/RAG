"""Lazy shared services."""

from fastapi import Request

from app.indexing.service import IndexService
from app.plugins.intent.classifier import IntentClassifier


def get_intent_classifier(request: Request) -> IntentClassifier:
    classifier = getattr(request.app.state, "intent_classifier", None)
    if classifier is None:
        classifier = IntentClassifier(request.app.state.settings)
        request.app.state.intent_classifier = classifier
    return classifier


def get_index_service(request: Request) -> IndexService:
    service = getattr(request.app.state, "index_service", None)
    if service is None:
        settings = request.app.state.settings
        service = IndexService(
            settings.data_dir,
            intent_classifier=get_intent_classifier(request),
        )
        request.app.state.index_service = service
    return service

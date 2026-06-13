"""Shared identity-state helpers for person analytics."""

from __future__ import annotations

from core.tracker import TrackedObject

KNOWN_PERSON_KIND = "known_person"
PENDING_PERSON_KIND = "pending_person"
STRANGER_KIND = "stranger"


def is_confirmed_stranger(obj: TrackedObject) -> bool:
    """Return True only after identity resolution has confirmed an unknown person."""
    return (
        obj.class_name == "person"
        and obj.identity_kind not in {KNOWN_PERSON_KIND, PENDING_PERSON_KIND}
    )

"""
SIH26170 - Engineering Review & Audit
Phase 3 / File 8

Purpose
-------
Provides an application-layer engineering review and audit trail without
modifying the Phase 1 AI core or Phase 2 screening engines.

The module supports:
- engineer review actions
- review comments and justification
- reviewer identity
- review timestamps
- dataset/model/config metadata
- review history
- JSONL audit persistence
- review summaries
- component-level review status

Prototype governance note
-------------------------
This is an application-level prototype audit mechanism. It is not a
replacement for a qualified manufacturing quality system, controlled
database, electronic-signature system, access-control system, or validated
regulated audit trail.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


class EngineeringReview:
    """Manage engineer decisions and an append-only prototype audit trail."""

    VERSION = "3.0"
    SUPPORTED_ACTIONS = (
        "ACCEPT",
        "MONITOR",
        "INVESTIGATE",
        "REJECT",
    )

    DEFAULT_AUDIT_PATH = Path("reports") / "engineering_review_audit.jsonl"

    def __init__(
        self,
        audit_path: str | os.PathLike[str] = DEFAULT_AUDIT_PATH,
        project_root: str | os.PathLike[str] | None = None,
    ) -> None:
        self.project_root = (
            Path(project_root).resolve()
            if project_root is not None
            else Path(__file__).resolve().parents[1]
        )

        supplied_path = Path(audit_path)
        if supplied_path.is_absolute():
            self.audit_path = supplied_path
        else:
            self.audit_path = self.project_root / supplied_path

        self.audit_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Basic metadata
    # ------------------------------------------------------------------

    @property
    def version(self) -> str:
        return self.VERSION

    def get_supported_actions(self) -> List[str]:
        return list(self.SUPPORTED_ACTIONS)

    def get_review_info(self) -> Dict[str, Any]:
        return {
            "module": "EngineeringReview",
            "version": self.VERSION,
            "supported_actions": self.get_supported_actions(),
            "audit_format": "JSONL",
            "audit_path": str(self.audit_path),
            "prototype_governance": True,
        }

    # ------------------------------------------------------------------
    # Normalization helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_text(value: Any, default: str = "") -> str:
        if value is None:
            return default
        text = str(value).strip()
        if text.lower() in {"nan", "none", "null"}:
            return default
        return text

    @staticmethod
    def _json_safe(value: Any) -> Any:
        """Convert common pandas/numpy/scalar values into JSON-safe values."""
        if value is None:
            return None

        if isinstance(value, (str, int, float, bool)):
            try:
                if isinstance(value, float):
                    if value != value:
                        return None
                    if value in (float("inf"), float("-inf")):
                        return None
            except Exception:
                pass
            return value

        if isinstance(value, dict):
            return {
                str(key): EngineeringReview._json_safe(item)
                for key, item in value.items()
            }

        if isinstance(value, (list, tuple, set)):
            return [
                EngineeringReview._json_safe(item)
                for item in value
            ]

        # pandas/numpy scalar support without requiring either package.
        if hasattr(value, "item"):
            try:
                return EngineeringReview._json_safe(value.item())
            except Exception:
                pass

        if hasattr(value, "isoformat"):
            try:
                return value.isoformat()
            except Exception:
                pass

        return str(value)

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    def _normalize_action(self, action: Any) -> str:
        normalized = self._clean_text(action).upper()
        if normalized not in self.SUPPORTED_ACTIONS:
            raise ValueError(
                f"Unsupported review action '{normalized}'. "
                f"Use one of: {', '.join(self.SUPPORTED_ACTIONS)}."
            )
        return normalized

    # ------------------------------------------------------------------
    # Review record creation
    # ------------------------------------------------------------------

    def create_review_record(
        self,
        component_id: Any,
        action: str,
        reviewer: Any,
        comment: Any = "",
        justification: Any = "",
        lot_id: Any = None,
        dataset_id: Any = None,
        dataset_version: Any = None,
        model_version: Any = None,
        specification_version: Any = None,
        configuration_version: Any = None,
        decision_before_review: Any = None,
        risk_level: Any = None,
        risk_score: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Build a review record.

        This method does not persist anything. Use submit_review() to append
        the record to the audit trail.
        """
        component = self._clean_text(component_id)
        reviewer_name = self._clean_text(reviewer)

        if not component:
            raise ValueError("component_id is required.")

        if not reviewer_name:
            raise ValueError("reviewer is required.")

        normalized_action = self._normalize_action(action)

        comment_text = self._clean_text(comment)
        justification_text = self._clean_text(justification)

        if not comment_text and not justification_text:
            justification_text = (
                f"Engineer review action recorded as {normalized_action}."
            )

        record = {
            "review_id": f"REV-{uuid.uuid4().hex[:12].upper()}",
            "review_version": self.VERSION,
            "review_timestamp_utc": self._timestamp(),
            "component_id": component,
            "lot_id": self._clean_text(lot_id),
            "reviewer": reviewer_name,
            "action": normalized_action,
            "comment": comment_text,
            "justification": justification_text,
            "decision_before_review": self._clean_text(
                decision_before_review
            ),
            "risk_level": self._clean_text(risk_level),
            "risk_score": self._json_safe(risk_score),
            "dataset": {
                "dataset_id": self._clean_text(dataset_id),
                "dataset_version": self._clean_text(dataset_version),
            },
            "model": {
                "model_version": self._clean_text(model_version),
            },
            "configuration": {
                "specification_version": self._clean_text(
                    specification_version
                ),
                "configuration_version": self._clean_text(
                    configuration_version
                ),
            },
            "metadata": self._json_safe(metadata or {}),
        }

        return record

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def append_audit_record(
        self,
        record: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Append one review record as a JSONL audit event."""
        if not isinstance(record, dict):
            raise TypeError("record must be a dictionary.")

        safe_record = self._json_safe(record)

        with self.audit_path.open(
            "a",
            encoding="utf-8",
            newline="\n",
        ) as handle:
            handle.write(
                json.dumps(
                    safe_record,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )

        return safe_record

    def submit_review(
        self,
        component_id: Any,
        action: str,
        reviewer: Any,
        comment: Any = "",
        justification: Any = "",
        lot_id: Any = None,
        dataset_id: Any = None,
        dataset_version: Any = None,
        model_version: Any = None,
        specification_version: Any = None,
        configuration_version: Any = None,
        decision_before_review: Any = None,
        risk_level: Any = None,
        risk_score: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create and persist a new engineering review event."""
        record = self.create_review_record(
            component_id=component_id,
            action=action,
            reviewer=reviewer,
            comment=comment,
            justification=justification,
            lot_id=lot_id,
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            model_version=model_version,
            specification_version=specification_version,
            configuration_version=configuration_version,
            decision_before_review=decision_before_review,
            risk_level=risk_level,
            risk_score=risk_score,
            metadata=metadata,
        )

        return self.append_audit_record(record)

    # ------------------------------------------------------------------
    # Audit retrieval
    # ------------------------------------------------------------------

    def read_audit_records(
        self,
        component_id: Any = None,
        reviewer: Any = None,
        action: Any = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Read audit records with optional filters."""
        if not self.audit_path.exists():
            return []

        component_filter = (
            self._clean_text(component_id)
            if component_id is not None
            else ""
        )
        reviewer_filter = (
            self._clean_text(reviewer)
            if reviewer is not None
            else ""
        )
        action_filter = (
            self._normalize_action(action)
            if action is not None and self._clean_text(action)
            else ""
        )

        records: List[Dict[str, Any]] = []

        with self.audit_path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    # Preserve readable history even if an incomplete line
                    # exists at the end of a manually edited prototype file.
                    continue

                if (
                    component_filter
                    and str(record.get("component_id", ""))
                    != component_filter
                ):
                    continue

                if (
                    reviewer_filter
                    and str(record.get("reviewer", ""))
                    != reviewer_filter
                ):
                    continue

                if (
                    action_filter
                    and str(record.get("action", "")).upper()
                    != action_filter
                ):
                    continue

                records.append(record)

        # Most recent review first.
        records.sort(
            key=lambda item: str(
                item.get("review_timestamp_utc", "")
            ),
            reverse=True,
        )

        if limit is not None:
            if limit < 1:
                return []
            records = records[:limit]

        return records

    def get_component_history(
        self,
        component_id: Any,
    ) -> List[Dict[str, Any]]:
        """Return all review events for one component."""
        return self.read_audit_records(
            component_id=component_id,
        )

    def get_latest_review(
        self,
        component_id: Any,
    ) -> Optional[Dict[str, Any]]:
        """Return the latest review event for one component."""
        history = self.get_component_history(component_id)
        return history[0] if history else None

    # ------------------------------------------------------------------
    # Review status / summaries
    # ------------------------------------------------------------------

    def get_current_review_action(
        self,
        component_id: Any,
        default: str = "PENDING",
    ) -> str:
        latest = self.get_latest_review(component_id)

        if latest is None:
            return default

        return self._clean_text(
            latest.get("action"),
            default=default,
        ).upper()

    def summarize_reviews(
        self,
        component_ids: Optional[Iterable[Any]] = None,
    ) -> Dict[str, Any]:
        """
        Return a review summary.

        If component_ids is supplied, the summary reports the latest action
        for those components. Otherwise it summarizes all audit events.
        """
        if component_ids is None:
            records = self.read_audit_records()

            counts = {
                action: 0
                for action in self.SUPPORTED_ACTIONS
            }

            for record in records:
                action = str(
                    record.get("action", "")
                ).upper()
                if action in counts:
                    counts[action] += 1

            return {
                "review_version": self.VERSION,
                "total_review_events": len(records),
                "action_counts": counts,
                "components_with_reviews": len(
                    {
                        str(record.get("component_id", ""))
                        for record in records
                        if record.get("component_id")
                    }
                ),
            }

        latest_by_component: Dict[str, Dict[str, Any]] = {}

        for component_id in component_ids:
            component = self._clean_text(component_id)
            if not component:
                continue

            latest = self.get_latest_review(component)
            if latest is not None:
                latest_by_component[component] = latest

        counts = {
            action: 0
            for action in self.SUPPORTED_ACTIONS
        }

        pending = 0

        for latest in latest_by_component.values():
            action = str(
                latest.get("action", "")
            ).upper()

            if action in counts:
                counts[action] += 1

        for component_id in component_ids:
            if self._clean_text(component_id) not in latest_by_component:
                pending += 1

        return {
            "review_version": self.VERSION,
            "components_evaluated": len(
                list(component_ids)
            ),
            "components_with_reviews": len(
                latest_by_component
            ),
            "pending_review": pending,
            "latest_action_counts": counts,
        }

    def export_audit_records(
        self,
        output_path: str | os.PathLike[str],
        component_id: Any = None,
    ) -> Path:
        """Export filtered audit records to a JSON file."""
        records = self.read_audit_records(
            component_id=component_id,
        )

        output = Path(output_path)
        if not output.is_absolute():
            output = self.project_root / output

        output.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "audit_export_version": self.VERSION,
            "exported_at_utc": self._timestamp(),
            "record_count": len(records),
            "records": records,
        }

        output.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        return output


__all__ = ["EngineeringReview"]

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from benchmark_toolbox.boxes import BoundingBox3D, OrientedBox3D, parse_box

__all__ = [
    "BoundingBox3D",
    "OrientedBox3D",
    "SceneObject",
    "SceneOutput",
    "SceneRelation",
    "SceneSample",
    "parse_box",
]


@dataclass(frozen=True)
class SceneObject:
    object_id: str
    label: str
    bbox: "BoundingBox3D | OrientedBox3D"
    score: float = 1.0
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "label": self.label,
            "score": self.score,
            "bbox": self.bbox.to_dict(),
            "attributes": dict(self.attributes),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SceneObject:
        return cls(
            object_id=str(data["object_id"]),
            label=str(data["label"]),
            score=float(data.get("score", 1.0)),
            bbox=parse_box(data["bbox"]),
            attributes=dict(data.get("attributes", {})),
        )


@dataclass(frozen=True)
class SceneRelation:
    source_id: str
    target_id: str
    relation_type: str
    score: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation_type": self.relation_type,
            "score": self.score,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SceneRelation:
        return cls(
            source_id=str(data["source_id"]),
            target_id=str(data["target_id"]),
            relation_type=str(data["relation_type"]),
            score=float(data.get("score", 1.0)),
        )


@dataclass(frozen=True)
class SceneOutput:
    layout: "BoundingBox3D | OrientedBox3D | None"
    objects: tuple[SceneObject, ...] = ()
    relations: tuple[SceneRelation, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "layout": self.layout.to_dict() if self.layout else None,
            "objects": [scene_object.to_dict() for scene_object in self.objects],
            "relations": [relation.to_dict() for relation in self.relations],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SceneOutput:
        layout_data = data.get("layout")
        return cls(
            layout=parse_box(layout_data) if layout_data else None,
            objects=tuple(
                SceneObject.from_dict(scene_object)
                for scene_object in data.get("objects", [])
            ),
            relations=tuple(
                SceneRelation.from_dict(relation)
                for relation in data.get("relations", [])
            ),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class SceneSample:
    sample_id: str
    input_path: Path
    ground_truth: SceneOutput | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

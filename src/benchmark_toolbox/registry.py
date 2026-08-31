from __future__ import annotations

from collections.abc import Callable
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class ComponentRegistry(Generic[T]):
    def __init__(self, component_name: str) -> None:
        self.component_name = component_name
        self._factories: dict[str, Callable[[dict[str, Any]], T]] = {}

    def register(
        self, name: str
    ) -> Callable[[Callable[[dict[str, Any]], T]], Callable[[dict[str, Any]], T]]:
        normalized = name.strip().lower()

        def decorator(
            factory: Callable[[dict[str, Any]], T],
        ) -> Callable[[dict[str, Any]], T]:
            if normalized in self._factories:
                raise ValueError(
                    f"{self.component_name} '{normalized}' is already registered"
                )
            self._factories[normalized] = factory
            return factory

        return decorator

    def alias(self, name: str, *aliases: str) -> None:
        normalized = name.strip().lower()
        if normalized not in self._factories:
            raise ValueError(
                f"Cannot alias unknown {self.component_name} '{name}'; "
                f"register it first"
            )
        for candidate in aliases:
            alias_name = candidate.strip().lower()
            if alias_name in self._factories:
                raise ValueError(
                    f"{self.component_name} '{alias_name}' is already registered"
                )
            self._factories[alias_name] = self._factories[normalized]

    def create(self, name: str, parameters: dict[str, Any] | None = None) -> T:
        normalized = name.strip().lower()
        try:
            factory = self._factories[normalized]
        except KeyError as error:
            available = ", ".join(sorted(self._factories)) or "<none>"
            raise ValueError(
                f"Unknown {self.component_name} '{name}'. Available: {available}"
            ) from error
        return factory(dict(parameters or {}))

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))

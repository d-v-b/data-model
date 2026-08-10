"""GeoZarr minispec compliance validation.

Validates a Zarr store against the normative requirements of the GeoZarr
minispec (``docs/geozarr-minispec.md``) using the declarative pydantic-zarr
models from :mod:`eopf_geozarr.data_api.geozarr.store`:

- **Store root**: must declare the ``spatial`` and ``geo-proj`` conventions in
  ``zarr_conventions`` and carry ``spatial:bbox`` plus exactly one of
  ``proj:code`` / ``proj:wkt2`` / ``proj:projjson``.
- **Multiscale datasets** (groups with a ``multiscales`` attribute): must
  declare all three conventions, carry ``spatial:bbox``, ``spatial:dimensions``
  and a CRS, and every layout entry must have ``spatial:shape`` and
  ``spatial:transform`` (plus ``transform`` when ``derived_from`` is set).
  Layout assets must resolve to members of the group, and each non-``"."``
  level must itself qualify as a GeoZarr dataset.
- **Datasets** (groups or arrays using ``proj:*`` / ``spatial:*`` keys): the
  used conventions must validate and be declared in ``zarr_conventions``.

The validation snapshot is taken with :meth:`pydantic_zarr.v3.GroupSpec.from_zarr`,
so all checks run on metadata only — no chunk data is read.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

import zarr
from pydantic import ValidationError
from pydantic_zarr.v3 import ArraySpec, GroupSpec
from zarr_cm import geo_proj as geo_proj_cm
from zarr_cm import spatial as spatial_cm

from eopf_geozarr.data_api.geozarr.geoproj import Proj
from eopf_geozarr.data_api.geozarr.spatial import Spatial
from eopf_geozarr.data_api.geozarr.store import (
    GeoZarrMultiscaleGroupAttrs,
    GeoZarrStoreAttrs,
    declared_convention_uuids,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

__all__ = [
    "ValidationIssue",
    "ValidationReport",
    "validate_store",
]


@dataclasses.dataclass(frozen=True)
class ValidationIssue:
    """A single compliance violation, anchored to a Zarr node path."""

    path: str
    message: str

    def __str__(self) -> str:
        return f"{self.path}: {self.message}"


@dataclasses.dataclass
class ValidationReport:
    """Result of validating a store against the GeoZarr minispec."""

    issues: list[ValidationIssue]
    # node path -> roles detected at that node (store-root, multiscale-dataset, dataset)
    roles: dict[str, list[str]]

    @property
    def compliant(self) -> bool:
        return not self.issues

    def summary(self) -> str:
        role_counts: dict[str, int] = {}
        for node_roles in self.roles.values():
            for role in node_roles:
                role_counts[role] = role_counts.get(role, 0) + 1
        lines = [
            "GeoZarr minispec validation: " + ("COMPLIANT" if self.compliant else "NOT COMPLIANT"),
            f"  nodes checked: {len(self.roles)}",
        ]
        lines.extend(f"  {role}: {role_counts[role]}" for role in sorted(role_counts))
        if self.issues:
            lines.append(f"  issues: {len(self.issues)}")
        return "\n".join(lines)


def _format_pydantic_error(err: ValidationError) -> list[str]:
    out: list[str] = []
    for e in err.errors(include_url=False, include_input=False):
        loc = ".".join(str(part) for part in e["loc"])
        out.append(f"{loc}: {e['msg']}" if loc else e["msg"])
    return out


def _walk(
    spec: GroupSpec[Any, Any] | ArraySpec[Any],
    path: str,
    inherited: frozenset[str] = frozenset(),
) -> Iterator[tuple[str, GroupSpec[Any, Any] | ArraySpec[Any], frozenset[str]]]:
    """Yield ``(path, node, inherited_uuids)`` for every node in the hierarchy.

    ``inherited_uuids`` holds the convention UUIDs declared by the *direct
    parent* group: per the minispec, group-level convention declarations are
    inherited by direct child arrays (only), so an array using convention keys
    need not re-declare a convention its parent group already declares.
    """
    yield path, spec, inherited
    if isinstance(spec, GroupSpec):
        own = declared_convention_uuids(
            tuple(dict(spec.attributes or {}).get("zarr_conventions", ()))
        )
        for name, member in (spec.members or {}).items():
            child_path = f"{path.rstrip('/')}/{name}"
            yield from _walk(member, child_path, frozenset(own))


def _resolve_asset(
    group: GroupSpec[Any, Any], asset: str
) -> GroupSpec[Any, Any] | ArraySpec[Any] | None:
    """Resolve a multiscales layout ``asset`` path relative to *group*."""
    if asset == ".":
        return group
    node: GroupSpec[Any, Any] | ArraySpec[Any] = group
    for part in asset.split("/"):
        if part in ("", "."):
            continue
        if not isinstance(node, GroupSpec):
            return None
        members: Mapping[str, Any] = node.members or {}
        if part not in members:
            return None
        node = members[part]
    return node


def _uses_spatial(attrs: Mapping[str, Any]) -> bool:
    # Presence of the convention's own keys, not the bare `spatial:` prefix:
    # the minispec permits external keys (e.g. source metadata) in any namespace.
    return bool(spatial_cm.CONVENTION_KEYS & attrs.keys())


def _uses_proj(attrs: Mapping[str, Any]) -> bool:
    # See _uses_spatial: legacy keys such as `proj:epsg` are not convention usage.
    return bool(geo_proj_cm.CONVENTION_KEYS & attrs.keys())


def _check_used_conventions(
    attrs: Mapping[str, Any],
    path: str,
    issues: list[ValidationIssue],
    inherited: frozenset[str] = frozenset(),
) -> None:
    """Validate the ``spatial:`` / ``proj:`` conventions a node uses.

    Any node using convention keys must declare the matching convention in its
    ``zarr_conventions`` attribute (or inherit the declaration from an ancestor
    group) and the keys must validate against the convention model.
    """
    declared = declared_convention_uuids(tuple(attrs.get("zarr_conventions", ()))) | inherited
    if _uses_spatial(attrs):
        if spatial_cm.UUID not in declared:
            issues.append(
                ValidationIssue(
                    path,
                    "node uses spatial:* attributes but does not declare the "
                    "spatial convention in zarr_conventions",
                )
            )
        try:
            Spatial.model_validate(dict(attrs))
        except ValidationError as err:
            issues.extend(
                ValidationIssue(path, f"spatial convention: {msg}")
                for msg in _format_pydantic_error(err)
            )
    if _uses_proj(attrs):
        if geo_proj_cm.UUID not in declared:
            issues.append(
                ValidationIssue(
                    path,
                    "node uses proj:* attributes but does not declare the "
                    "geo-proj convention in zarr_conventions",
                )
            )
        try:
            Proj.model_validate(dict(attrs))
        except ValidationError as err:
            issues.extend(
                ValidationIssue(path, f"geo-proj convention: {msg}")
                for msg in _format_pydantic_error(err)
            )


def _check_dataset_attrs(attrs: Mapping[str, Any], path: str) -> list[ValidationIssue]:
    """Check that a node qualifies as a GeoZarr Dataset.

    Per the minispec, a Dataset must carry geospatial metadata via *both* the
    ``proj:`` and ``spatial:`` conventions, declared in ``zarr_conventions``.
    """
    issues: list[ValidationIssue] = []
    declared = declared_convention_uuids(tuple(attrs.get("zarr_conventions", ())))
    for uuid, name, model in (
        (spatial_cm.UUID, "spatial", Spatial),
        (geo_proj_cm.UUID, "geo-proj", Proj),
    ):
        if uuid not in declared:
            issues.append(
                ValidationIssue(
                    path,
                    f"multiscale level does not declare the {name} convention "
                    "in zarr_conventions (required for GeoZarr datasets)",
                )
            )
        try:
            model.model_validate(dict(attrs))
        except ValidationError as err:
            issues.extend(
                ValidationIssue(path, f"{name} convention: {msg}")
                for msg in _format_pydantic_error(err)
            )
    return issues


def _check_multiscale_group(
    spec: GroupSpec[Any, Any], path: str, issues: list[ValidationIssue]
) -> None:
    attrs = dict(spec.attributes or {})
    try:
        model = GeoZarrMultiscaleGroupAttrs.model_validate(attrs)
    except ValidationError as err:
        issues.extend(
            ValidationIssue(path, f"multiscale dataset: {msg}")
            for msg in _format_pydantic_error(err)
        )
        return

    layout = model.multiscales.layout
    for entry in layout:
        target = _resolve_asset(spec, entry.asset)
        if target is None:
            issues.append(
                ValidationIssue(
                    path,
                    f"multiscales.layout asset {entry.asset!r} does not resolve "
                    "to a member of this group",
                )
            )
            continue
        if entry.asset != ".":
            # Each level must itself qualify as a GeoZarr dataset. The "." asset
            # is the multiscale group itself, whose attributes were validated above.
            target_attrs = dict(target.attributes or {})
            target_path = f"{path.rstrip('/')}/{entry.asset}"
            issues.extend(_check_dataset_attrs(target_attrs, target_path))


def validate_store(
    store: str | zarr.Group,
    *,
    storage_options: dict[str, Any] | None = None,
) -> ValidationReport:
    """Validate a Zarr store against the GeoZarr minispec.

    Parameters
    ----------
    store : str or zarr.Group
        Path/URL of the Zarr store, or an already-open root group.
    storage_options : dict, optional
        fsspec storage options used when *store* is a path or URL.

    Returns
    -------
    ValidationReport
        Structured report; ``report.compliant`` is True when no issues were found.
    """
    issues: list[ValidationIssue] = []
    roles: dict[str, list[str]] = {}

    if isinstance(store, zarr.Group):
        root = store
    else:
        root = zarr.open_group(store, mode="r", storage_options=storage_options)

    zarr_format = root.metadata.zarr_format
    if zarr_format != 3:
        issues.append(
            ValidationIssue(
                "/",
                f"store is Zarr V{zarr_format}; the minispec targets Zarr V3 exclusively",
            )
        )
        return ValidationReport(issues=issues, roles={"/": ["store-root"]})

    spec = GroupSpec.from_zarr(root)

    for path, node, inherited in _walk(spec, "/"):
        attrs = dict(node.attributes or {})
        node_roles: list[str] = []

        if path == "/":
            node_roles.append("store-root")
            try:
                GeoZarrStoreAttrs.model_validate(attrs)
            except ValidationError as err:
                issues.extend(
                    ValidationIssue(path, f"store root: {msg}")
                    for msg in _format_pydantic_error(err)
                )

        if isinstance(node, GroupSpec) and "multiscales" in attrs:
            node_roles.append("multiscale-dataset")
            _check_multiscale_group(node, path, issues)
        elif path != "/" and (_uses_spatial(attrs) or _uses_proj(attrs)):
            node_roles.append("dataset")
            _check_used_conventions(attrs, path, issues, inherited)

        if node_roles:
            roles[path] = node_roles

    if not any("multiscale-dataset" in r or "dataset" in r for r in roles.values()):
        issues.append(
            ValidationIssue(
                "/",
                "store contains no groups or arrays using the proj:/spatial:/multiscales "
                "conventions; nothing qualifies as a GeoZarr dataset",
            )
        )

    return ValidationReport(issues=issues, roles=roles)

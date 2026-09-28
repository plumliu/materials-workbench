"""Evidence and model protocols. Only local evidence owns pixel coordinates."""

from typing import Annotated, Literal

from pydantic import ConfigDict, Field, model_validator

from .models import (
    AxisPlan,
    BBox,
    DatasetPlan,
    Model,
    Point,
    TextObservation,
    TickCandidate,
)


class Spine(Model):
    id: str
    orientation: Literal["horizontal", "vertical"]
    bbox: BBox
    coordinate: float
    width: float = Field(gt=0)
    strength: int


class Geometry(Model):
    schema_version: Literal["geometry/v1"] = "geometry/v1"
    image_size: tuple[int, int]
    preprocessing: str
    spines: list[Spine]
    ticks: list[TickCandidate]


class Evidence(Model):
    schema_version: Literal["evidence-graph/v1"] = "evidence-graph/v1"
    source_id: str
    geometry: Geometry
    texts: list[TextObservation]


AxisScale = Literal["linear", "log", "categorical"]


class AxisDirection(Model):
    """One image-truth direction; Python derives labels, units and identifiers."""

    name: str = Field(min_length=1)
    scale: AxisScale
    categories: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def valid_categories(self):
        if self.scale == "categorical" and not self.categories:
            raise ValueError("categorical directions require visible categories")
        if self.scale != "categorical" and self.categories:
            raise ValueError("numeric directions cannot carry categories")
        if len(set(self.categories)) != len(self.categories):
            raise ValueError("categories must be unique")
        return self


class AxisGroup(Model):
    """One coordinate presentation with one X mapping and one or more Y mappings."""

    x: AxisDirection
    ys: list[AxisDirection] = Field(min_length=1)


class AxisStructure(Model):
    schema_version: Literal["axis-structure/v3"] = "axis-structure/v3"
    groups: list[AxisGroup] = Field(default_factory=list)
    unresolved: list[str] = Field(default_factory=list)

    def expand(self) -> list[AxisPlan]:
        """Create deterministic internal/WPD axes without asking Qwen for IDs."""
        axes = []
        for group_index, group in enumerate(self.groups):
            root_id = f"axis_{group_index:03d}_000"
            for y_index, y in enumerate(group.ys):
                axis_id = f"axis_{group_index:03d}_{y_index:03d}"
                x_categorical = group.x.scale == "categorical"
                y_categorical = y.scale == "categorical"
                if x_categorical and y_categorical:
                    raise ValueError("an Axis cannot have two categorical directions")
                axis_type = (
                    "categorical_x_numeric_y"
                    if x_categorical
                    else "numeric_x_categorical_y"
                    if y_categorical
                    else "xy"
                )
                axes.append(
                    AxisPlan(
                        id=axis_id,
                        display_name=f"{group.x.name} - {y.name}",
                        axis_type=axis_type,
                        x_scale=group.x.scale,
                        y_scale=y.scale,
                        x_label=group.x.name,
                        y_label=y.name,
                        categories=(
                            group.x.categories if x_categorical else y.categories
                        ),
                        # Numeric X has a real calibration that sibling Y maps
                        # reuse. A categorical X is only a shared vocabulary;
                        # every result becomes an independent WPD BarAxes.
                        shared_x_axis_id=(
                            root_id
                            if y_index and group.x.scale != "categorical"
                            else None
                        ),
                    )
                )
        return axes


class DatasetItem(Model):
    """Minimal model response: ownership, final name and collection type only."""

    axis: str
    name: str
    kind: Literal[
        "scatter",
        "curve",
        "bar",
        "point_group",
        "range_boundary",
        "distribution_boundary",
    ]


class DatasetStructure(Model):
    datasets: list[DatasetItem]
    unresolved: list[str] = Field(default_factory=list)

    def expand(self) -> list[DatasetPlan]:
        """Create deterministic WPD records without asking the model for metadata."""
        return [
            DatasetPlan(
                id=f"dataset_{index:03d}",
                axis_id=item.axis,
                display_name=item.name,
                kind=item.kind,
                group_names=(
                    ["upper", "average", "lower"] if item.kind == "point_group" else []
                ),
            )
            for index, item in enumerate(self.datasets)
        ]


class Structure(Model):
    schema_version: Literal["structure/v2"] = "structure/v2"
    axes: list[AxisPlan]
    datasets: list[DatasetPlan]
    unresolved: list[str] = Field(default_factory=list)


class TickSelection(Model):
    tick_id: str
    text_id: str


class DirectionBinding(Model):
    spine_id: str
    ticks: list[TickSelection]
    calibration_tick_ids: tuple[str, str]
    grounded_tick_ids: list[str] = Field(default_factory=list)
    interval_px: tuple[float, float] | None = None


class AxisBinding(Model):
    axis_id: str
    x: DirectionBinding | None = None
    y: DirectionBinding | None = None


class Bindings(Model):
    """Local snapper output, never a model response."""

    schema_version: Literal["bindings/v2"] = "bindings/v2"
    axes: list[AxisBinding]
    unresolved: list[str] = Field(default_factory=list)


NormalizedCoordinate = Annotated[float, Field(ge=0, le=1000)]
NormalizedPoint = tuple[NormalizedCoordinate, NormalizedCoordinate]


class TickGrounding(Model):
    visible_label: str = Field(min_length=1)
    value: float
    point_2d: NormalizedPoint


class DirectionGrounding(Model):
    anchors: list[TickGrounding] = Field(min_length=2, max_length=4)


class AxisGrounding(Model):
    axis_id: str
    x: DirectionGrounding | None = None
    y: DirectionGrounding | None = None


class Grounding(Model):
    schema_version: Literal["grounding/v2"] = "grounding/v2"
    coordinate_system: Literal["normalized_0_1000"] = "normalized_0_1000"
    axes: list[AxisGrounding]
    unresolved: list[str] = Field(default_factory=list)


GroundingAnchors = Annotated[list[TickGrounding], Field(max_length=4)]


class GroundingGroup(Model):
    """Model response for one approved X with its ordered Y mappings."""

    x: GroundingAnchors = Field(default_factory=list)
    ys: list[GroundingAnchors] = Field(min_length=1)


class GroundingResponse(Model):
    """Minimal grouped model protocol; Python owns all generated Axis IDs."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, title="Grounding")
    schema_version: Literal["grounding/v3"] = "grounding/v3"
    coordinate_system: Literal["normalized_0_1000"] = "normalized_0_1000"
    groups: list[GroundingGroup]
    unresolved: list[str] = Field(default_factory=list)


class AxisVisualReview(Model):
    axis_id: str
    status: Literal["consistent", "wrong", "unclear"]
    issues: list[str]
    corrected_grounding: AxisGrounding | None

    @model_validator(mode="after")
    def check_verdict(self):
        if self.status == "consistent":
            if self.issues or self.corrected_grounding is not None:
                raise ValueError("Consistent review must have no issues or correction")
        elif not self.issues or any(not issue.strip() for issue in self.issues):
            raise ValueError("Wrong/unclear review requires a concrete issue")
        if self.status == "wrong":
            if (
                self.corrected_grounding is None
                or self.corrected_grounding.axis_id != self.axis_id
            ):
                raise ValueError("Wrong review requires correction for the same Axis")
        elif self.corrected_grounding is not None:
            raise ValueError("Only wrong verdict may supply a correction")
        return self


class VisualReview(Model):
    schema_version: Literal["visual-review/v1"] = "visual-review/v1"
    axes: list[AxisVisualReview] = Field(min_length=1)
    unresolved: list[str]


class ResolvedPoint(Model):
    tick_id: str
    text_id: str
    pixel: Point
    value: float


class Fit(Model):
    axis_id: str
    direction: Literal["x", "y"]
    scale: Literal["linear", "log"]
    spine_id: str
    slope: float
    intercept: float
    tolerance_px: float
    max_residual_px: float
    points: list[ResolvedPoint]
    endpoints: tuple[ResolvedPoint, ResolvedPoint]


class Calibration(Model):
    schema_version: Literal["calibration/v1"] = "calibration/v1"
    fits: list[Fit]

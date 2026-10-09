from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

BBox = tuple[float, float, float, float]
Point = tuple[float, float]
Identifier = Annotated[str, Field(min_length=1)]
TextSource = Literal[
    "pdf_text_layer", "rapidocr_full_page", "rapidocr_general", "rapidocr_tick_crop"
]
ReconciliationStatus = Literal[
    "agreed",
    "pdf_only",
    "ocr_only",
    "text_conflict",
    "position_conflict",
    "multiple_candidates",
]


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class ValidationIssue(Model):
    schema_version: Literal["validation-issue/v1"] = "validation-issue/v1"
    code: Identifier
    message: str
    node: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    repair: Literal["none", "semantic", "evidence"] = "none"


class TextObservation(Model):
    schema_version: Literal["text-observation/v1"] = "text-observation/v1"
    id: Identifier
    source: TextSource
    raw_text: str
    normalized_text: str
    bbox_pdf: BBox | None = None
    bbox_px: BBox | None = None
    image_size: tuple[int, int] | None = None
    method: str
    numeric_value: float | None = None
    engine_score: float | None = None
    quad_px: tuple[Point, Point, Point, Point] | None = None
    direction_pdf: Point | None = None
    direction_px: Point | None = None
    line_id: str | None = None
    pdf_span_ids: list[str] = Field(default_factory=list)
    font_names: list[str] = Field(default_factory=list)
    font_sizes_pdf: list[float] = Field(default_factory=list)
    font_flags: list[int] = Field(default_factory=list)


class TextEvidence(Model):
    schema_version: Literal["text-evidence/v1"] = "text-evidence/v1"
    source_id: str
    status: Literal["available", "unavailable", "failed"]
    reason: str | None = None
    observations: list[TextObservation] = Field(default_factory=list)


class TextMatch(Model):
    schema_version: Literal["text-match/v1"] = "text-match/v1"
    id: str
    status: ReconciliationStatus
    pdf_observation_ids: list[str] = Field(default_factory=list)
    ocr_observation_ids: list[str] = Field(default_factory=list)
    method: str


class FigureCaption(Model):
    schema_version: Literal["figure-caption/v1"] = "figure-caption/v1"
    figure_id: Annotated[str, Field(pattern=r"^\d+(?:\.[0-9A-Za-z]+)+$")]
    caption: str
    caption_source: TextSource
    caption_bbox_pdf: BBox | None = None
    caption_bbox_px: BBox | None = None
    observation_ids: list[str] = Field(default_factory=list)


class PageRecord(Model):
    schema_version: Literal["page-record/v1"] = "page-record/v1"
    source_page: int = Field(ge=1)
    single_page_pdf: str | None = None
    page_width_pdf: float = Field(gt=0)
    page_height_pdf: float = Field(gt=0)
    rotation: int = 0
    text_layer_available: bool
    text_char_count: int = Field(ge=0)
    embedded_image_count: int = Field(ge=0)
    object_summary: dict[str, int] = Field(default_factory=dict)
    figure_status: Literal["detected", "non_figure", "needs_review"] = "needs_review"
    figures: list[FigureCaption] = Field(default_factory=list)
    pdf_text_observations: str | None = None
    ocr_text_observations: str | None = None
    ocr_status: Literal["not_run", "completed", "failed"] = "not_run"
    issues: list[ValidationIssue] = Field(default_factory=list)


class IntakeInventory(Model):
    manual_id: str
    source_pdf: str
    encryption_present: bool = False
    empty_password_authenticated: bool = False
    page_count: int = Field(ge=1)
    figure_job_count: int = Field(default=0, ge=0)
    pages: list[PageRecord]

    @model_validator(mode="after")
    def physical_pages(self) -> Self:
        if [p.source_page for p in self.pages] != list(range(1, self.page_count + 1)):
            raise ValueError(
                "pages must contain each physical page exactly once, in order"
            )
        return self




class SourceAsset(Model):
    schema_version: Literal["source-asset/v1"] = "source-asset/v1"
    source_id: str
    path: str
    original_path: str | None = None
    original_size: tuple[float, float] | None = None
    rotation: int = 0
    figure_job_id: str | None = None
    kind: Literal["pdf", "image"]
    page_count: int = Field(default=1, ge=1)
    manual_id: str | None = None
    source_page: int | None = Field(default=None, ge=1)
    figure_id: str | None = None
    render_width: int | None = Field(default=None, gt=0)
    render_height: int | None = Field(default=None, gt=0)
    pdf_to_pixel_transform: tuple[float, float, float, float, float, float] | None = (
        None
    )


class TickCandidate(Model):
    schema_version: Literal["tick-candidate/v1"] = "tick-candidate/v1"
    id: Identifier
    spine_id: Identifier
    intersection_px: Point
    provenance: Literal["tick_intersection", "label_projection", "label_center"]
    text_observation_ids: list[str]
    status: ReconciliationStatus


class CalibrationBinding(Model):
    tick_id: Identifier
    value: float
    text_observation_ids: list[str] = Field(default_factory=list)


class AxisPlan(Model):
    schema_version: Literal["axis-plan/v2"] = "axis-plan/v2"
    id: Identifier
    display_name: Identifier
    axis_type: Literal[
        "xy", "categorical_x_numeric_y", "numeric_x_categorical_y", "bar"
    ]
    x_scale: Literal["linear", "log", "categorical"] = "linear"
    y_scale: Literal["linear", "log", "categorical"] = "linear"
    x_label: str = ""
    y_label: str = ""
    x_unit: str | None = None
    y_unit: str | None = None
    categories: list[str] = Field(default_factory=list)
    shared_x_axis_id: str | None = None
    calibration: dict[Literal["x", "y"], list[CalibrationBinding]] = Field(
        default_factory=dict
    )

    @model_validator(mode="after")
    def valid_scales(self) -> Self:
        for direction in ("x", "y"):
            scale = getattr(self, f"{direction}_scale")
            bindings = self.calibration.get(direction, [])
            if scale == "categorical" and bindings:
                raise ValueError("categories cannot have invented numeric calibration")
            if scale == "log" and any(b.value <= 0 for b in bindings):
                raise ValueError("log calibration values must be positive")
        if self.axis_type == "categorical_x_numeric_y" and (
            self.x_scale != "categorical" or self.y_scale == "categorical"
        ):
            raise ValueError("categorical x requires a numeric y scale")
        if self.axis_type == "numeric_x_categorical_y" and (
            self.y_scale != "categorical" or self.x_scale == "categorical"
        ):
            raise ValueError("categorical y requires a numeric x scale")
        return self


class DatasetPlan(Model):
    schema_version: Literal["dataset-plan/v1"] = "dataset-plan/v1"
    id: Identifier
    axis_id: Identifier
    display_name: Identifier
    kind: Literal[
        "scatter",
        "curve",
        "bar",
        "point_group",
        "range_boundary",
        "distribution_boundary",
    ]
    group_names: list[str] = Field(default_factory=list)
    data: list[object] = Field(default_factory=list, max_length=0)

    @model_validator(mode="after")
    def valid_groups(self) -> Self:
        if self.kind == "point_group":
            if not self.group_names or len(set(self.group_names)) != len(
                self.group_names
            ):
                raise ValueError("point groups need distinct ordered names")
        elif self.group_names:
            raise ValueError("group settings require point_group kind")
        return self


class ChartPlan(Model):
    schema_version: Literal["chart-plan/v2"] = "chart-plan/v2"
    source_asset: SourceAsset
    axes: list[AxisPlan]
    datasets: list[DatasetPlan]
    validation_issues: list[ValidationIssue] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_references(self) -> Self:
        for items in (self.axes, self.datasets):
            for field in ("id", "display_name"):
                values = [getattr(item, field) for item in items]
                if len(values) != len(set(values)):
                    raise ValueError(f"duplicate {field}")
        ids = {axis.id for axis in self.axes}
        if any(dataset.axis_id not in ids for dataset in self.datasets):
            raise ValueError("dataset references missing axis")
        if any(
            a.shared_x_axis_id is not None
            and (a.shared_x_axis_id not in ids or a.shared_x_axis_id == a.id)
            for a in self.axes
        ):
            raise ValueError("invalid shared axis reference")
        return self

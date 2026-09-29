"""WPD TAR serialization using the native version-4 format supported by WPD 5.3."""

import io
import json
import tarfile
from pathlib import Path, PurePosixPath

from chart_annotator.domain.models import ChartPlan
from chart_annotator.domain.workflow import Calibration


def project(plan: ChartPlan, calibration: Calibration) -> dict:
    if plan.validation_issues or not plan.axes:
        raise ValueError("Cannot export unresolved or empty plans")
    fits = {(f.axis_id, f.direction): f for f in calibration.fits}
    axes = []
    for axis in plan.axes:
        numeric = [
            d for d in ("x", "y") if getattr(axis, f"{d}_scale") != "categorical"
        ]
        if not numeric or any((axis.id, d) not in fits for d in numeric):
            raise ValueError("Missing numeric calibration")
        points = []
        for direction in numeric:
            fit = fits[axis.id, direction]
            if fit.max_residual_px > fit.tolerance_px or fit.scale != getattr(
                axis, f"{direction}_scale"
            ):
                raise ValueError("Invalid fit")
            if [(b.tick_id, b.value) for b in axis.calibration.get(direction, [])] != [
                (p.tick_id, p.value) for p in fit.points
            ]:
                raise ValueError("ChartPlan calibration differs from validated fit")
            for p in fit.endpoints:
                points.append(
                    {
                        "px": p.pixel[0],
                        "py": p.pixel[1],
                        "dx": str(p.value)
                        if direction == "x" and len(numeric) == 2
                        else 0,
                        "dy": str(p.value)
                        if direction == "y" or len(numeric) == 1
                        else 0,
                        "dz": None,
                    }
                )
        item = {
            "name": axis.display_name,
            "calibrationPoints": points,
            "metadata": {
                "chart_annotator": {
                    "axis_id": axis.id,
                    "categories": axis.categories,
                    "x_label": axis.x_label,
                    "y_label": axis.y_label,
                    "x_unit": axis.x_unit,
                    "y_unit": axis.y_unit,
                }
            },
        }
        if len(numeric) == 2:
            item.update(
                type="XYAxes",
                isLogX=axis.x_scale == "log",
                isLogY=axis.y_scale == "log",
                noRotation=True,
            )
        else:
            item.update(
                type="BarAxes",
                isLog=getattr(axis, f"{numeric[0]}_scale") == "log",
                isRotated=False,
            )
        axes.append(item)
    names = {a.id: a.display_name for a in plan.axes}
    datasets = []
    for dataset in plan.datasets:
        item = {
            "name": dataset.display_name,
            "axesName": names[dataset.axis_id],
            "colorRGB": [200, 0, 0, 255],
            "metadataKeys": ["label"]
            if any(
                a.id == dataset.axis_id and "categorical" in (a.x_scale, a.y_scale)
                for a in plan.axes
            )
            else [],
            "data": [],
            "autoDetectionData": None,
            "metadata": {
                "chart_annotator": {
                    "kind": dataset.kind,
                }
            },
        }
        if dataset.group_names:
            item["groupNames"] = dataset.group_names
        datasets.append(item)
    return {
        "version": [4, 2],
        "axesColl": axes,
        "datasetColl": datasets,
        "measurementColl": [],
    }


def export(path: Path, plan: ChartPlan, calibration: Calibration, image: Path) -> None:
    write_archive(path, project(plan, calibration), image)


def write_archive(path: Path, data: dict, image: Path) -> None:
    """Write a native single-source WPD project, including an unmarked PDF."""
    path.parent.mkdir(parents=True, exist_ok=True)
    info = {"version": [4, 0], "json": "wpd.json", "images": [image.name]}
    with tarfile.open(path, "w", format=tarfile.USTAR_FORMAT) as archive:
        for name, content in (
            ("info.json", json.dumps(info).encode()),
            (
                "wpd.json",
                json.dumps(data, ensure_ascii=False, allow_nan=False).encode(),
            ),
            (image.name, image.read_bytes()),
        ):
            member = tarfile.TarInfo("chart/" + name)
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))


def read(path: Path) -> tuple[dict, bytes]:
    """Read members in memory; never extract arbitrary TAR paths to disk."""
    with tarfile.open(path, "r") as archive:
        files = {}
        total_size = 0
        for member in archive:
            if member.isdir():
                continue
            parts = PurePosixPath(member.name)
            total_size += member.size
            if (
                not member.isfile()
                or parts.is_absolute()
                or ".." in parts.parts
                or "\\" in member.name
                or member.name in files
                or member.size > 100_000_000
                or total_size > 100_000_000
                or len(files) >= 50
            ):
                raise ValueError("Invalid TAR member")
            files[member.name] = archive.extractfile(member).read()
        info_paths = [name for name in files if name.endswith("/info.json")]
        if len(info_paths) != 1:
            raise ValueError("Expected one project info file")
        info_path = info_paths[0]
        info = json.loads(files[info_path])
        root = PurePosixPath(info_path).parent
        if info.get("version") != [4, 0] or len(info.get("images", [])) != 1:
            raise ValueError("Unsupported project info")
        for name in [info["json"], *info["images"]]:
            if PurePosixPath(name).name != name or "\\" in name:
                raise ValueError("Invalid project reference")
        return json.loads(files[str(root / info["json"])]), files[
            str(root / info["images"][0])
        ]


def validate(
    path: Path, plan: ChartPlan, calibration: Calibration, image: Path
) -> dict:
    loaded, pixels = read(path)
    if loaded != project(plan, calibration) or pixels != image.read_bytes():
        raise ValueError("TAR roundtrip differs from validated plan or rendered image")
    return {
        "schema_version": "wpd-roundtrip/v1",
        "status": "passed",
        "axes": len(loaded["axesColl"]),
        "datasets": len(loaded["datasetColl"]),
        "points": sum(len(d["data"]) for d in loaded["datasetColl"]),
        "image_bytes_equal": True,
    }

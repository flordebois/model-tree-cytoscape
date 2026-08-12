from pathlib import Path
from config import DIR_DATASET_UPLOAD, NEW_CSV_OPTION


def get_pmlb_options() -> list[dict]:
    return [
        {"label": "PMLB: 547_no2", "value": "547_no2.pmlb"},
        {"label": "PMLB: 294_satellite_image", "value": "294_satellite_image.pmlb"},
        {"label": "PMLB: 1199_BNG_echoMonths", "value": "1199_BNG_echoMonths.pmlb"},
        {"label": "PMLB: 537_houses", "value": "537_houses.pmlb"},
        {"label": "PMLB: 658_fri_c3_250_25", "value": "658_fri_c3_250_25.pmlb"},
        {"label": "PMLB: 505_tecator", "value": "505_tecator.pmlb"},
        {"label": "PMLB: 560_bodyfat", "value": "560_bodyfat.pmlb"},
        {"label": "PMLB: 485_analcatdata_vehicle", "value": "485_analcatdata_vehicle.pmlb"},
        {"label": "PMLB: 210_cloud", "value": "210_cloud.pmlb"},
        {"label": "PMLB: 1028_SWD", "value": "1028_SWD.pmlb"},
        {"label": "PMLB: 197_cpu_act", "value": "197_cpu_act.pmlb"},
    ]

def get_target_col(csv_path: Path) -> str:
    sidecar = csv_path.with_suffix(".target.txt")
    return sidecar.read_text().strip()


def get_uploaded_options() -> list[dict]:
    DIR_DATASET_UPLOAD.mkdir(parents=True, exist_ok=True)
    options = []
    for csv_path in sorted(DIR_DATASET_UPLOAD.glob("*.csv")):
        options.append({"label": f"CSV: {csv_path.name[:-4]}", "value": str(csv_path)})
    return options


def build_dropdown_options() -> list[dict]:
    return [
        *get_pmlb_options(),
        *get_uploaded_options(),
        {"label": "+ New dataset (CSV)", "value": NEW_CSV_OPTION},
    ]
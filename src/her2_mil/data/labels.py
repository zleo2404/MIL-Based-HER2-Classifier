"""Loading and mapping HER2 IHC labels from the clinical CSV."""
from typing import Optional

import pandas as pd

from her2_mil.config import LabelsConfig


def load_label_lookup(labels_csv: str) -> pd.DataFrame:
    df = pd.read_csv(labels_csv)
    df["cases.submitter_id"] = df["cases.submitter_id"].str.strip()
    df["her2_result"] = df["her2_result"].str.strip()
    return df


def get_slide_label(slide_id: str, label_df: pd.DataFrame, labels_cfg: LabelsConfig) -> Optional[int]:
    """Return the mapped integer label for a slide, or None if the slide
    has no matching patient row or an unrecognized her2_result value."""
    tcga_id = slide_id[0:12]  # standard TCGA case ID length
    patient_row = label_df[label_df["cases.submitter_id"].str.startswith(tcga_id)]
    if patient_row.empty:
        return None

    her2_result = patient_row["her2_result"].iloc[0]
    if her2_result not in labels_cfg.mapping:
        return None

    return labels_cfg.mapping[her2_result]

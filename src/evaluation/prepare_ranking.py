"""Check that the prepared ranking holds every submitting.parquet departure exactly once."""

import pandas as pd

ID = "MVT_ID_mvt"


def check_ids(departures: pd.DataFrame, template: pd.DataFrame) -> dict:
    ranking_ids, template_ids = set(departures[ID]), set(template[ID])
    report = {
        "ranking_departures": len(departures),
        "template_rows": len(template),
        "duplicated_ranking_ids": int(departures[ID].duplicated().sum()),
        "missing_from_ranking": len(template_ids - ranking_ids),
        "extra_in_ranking": len(ranking_ids - template_ids),
    }
    report["ok"] = (report["duplicated_ranking_ids"] == 0 and report["missing_from_ranking"] == 0
                    and report["extra_in_ranking"] == 0)
    return report

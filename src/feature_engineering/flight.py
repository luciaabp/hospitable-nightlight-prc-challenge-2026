"""Flight: airline, market segment, no-flight-plan flag and identifier format.

The identifier-format features are diagnostic only (forward ablation of 2026-09-16 was negative); they enter the model
only with ``--add-prefix-features``.
"""

import numpy as np
import pandas as pd

FEATURES = [
    "airline",
    "market_segment",
    "no_flight_plan",
    "airline_prefix_source",
    "flight_prefix_repeated",
    "flight_id_len",
    "callsign_all_alpha",
    "destination_filed",
    "flight_type",
]

# after the first three letters the second and third repeat: ITYTY680, NOSOS739, BAWW561D
REPEATED_PREFIX = r"^[A-Z]([A-Z]{2})\1"


def add_flight_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    callsign = df["CALLSIGN_flt"].astype("string")
    flight = df["FLIGHT_mvt"].astype("string")

    # the callsign starts with the 3-letter ICAO code (DLH8UL -> DLH)
    icao_from_callsign = callsign.str.extract(r"^([A-Z]{3})", expand=False)
    # fallback: flight-number prefix, which may be a 2-character IATA code (LH982 -> LH)
    prefix_from_flight = flight.str.extract(r"^([A-Z]{3}|[A-Z0-9]{2})", expand=False)
    df["airline"] = icao_from_callsign.fillna(prefix_from_flight)

    df["market_segment"] = df["MARKET_SEGMENT_flt"].astype("string")
    # flight-plan destination and flight type (scheduled, charter, military...): used by v6, not by us
    df["destination_filed"] = df["ADES_FILED_flt"].astype("string")
    df["flight_type"] = df["FLIGHT_TYPE_flt"].astype("string")
    df["no_flight_plan"] = df["FLIGHT_ID_mvt"].isna()

    # where the prefix comes from: a callsign prefix does not mean the same as a flight-number one
    df["airline_prefix_source"] = np.where(
        icao_from_callsign.notna(), "callsign", np.where(prefix_from_flight.notna(), "flight", "missing")
    )
    df["flight_prefix_repeated"] = flight.str.match(REPEATED_PREFIX).fillna(False).astype(bool)
    df["flight_id_len"] = flight.str.len().astype("Float64")
    # all-letter callsign: usually a registration or another identifier, not a commercial flight
    df["callsign_all_alpha"] = callsign.str.fullmatch(r"[A-Z]+").fillna(False).astype(bool)
    return df

import json

import pandas as pd

from chap_core.api_types import FeatureCollectionModel
from chap_core.database.dataset_tables import DataSet
from chap_core.datatypes import create_tsdataclass
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet as _DataSet


def _observation_rows_to_dicts(observations):
    rows = []
    for obs in observations:
        if hasattr(obs, "model_dump"):
            rows.append(obs.model_dump())
        elif isinstance(obs, dict):
            rows.append(obs)
        else:
            raise TypeError(f"Unsupported observation type: {type(obs)}")
    return rows


def observations_to_dataset(dataclass, observations, fill_missing=False):
    if not observations:
        raise ValueError(
            "provided_data must contain at least one observation with orgUnit, period, featureName, and value."
        )
    dataframe = pd.DataFrame(_observation_rows_to_dicts(observations))
    rename_map = {
        "org_unit": "location",
        "orgUnit": "location",
        "period": "time_period",
        "featureName": "feature_name",
    }
    dataframe = dataframe.rename(columns={k: v for k, v in rename_map.items() if k in dataframe.columns})
    required = ("location", "time_period", "feature_name", "value")
    missing = [c for c in required if c not in dataframe.columns]
    if missing:
        raise ValueError(
            f"Observations are missing required fields: {missing}. "
            f"Expected orgUnit/org_unit, period, featureName/feature_name, and value. "
            f"Columns present: {list(dataframe.columns)}"
        )
    dataframe = dataframe.set_index(["location", "time_period"])
    pivoted = dataframe.pivot(columns="feature_name", values="value").reset_index()
    new_dataset = _DataSet.from_pandas(pivoted, dataclass, fill_missing=fill_missing)
    return new_dataset


def dataset_model_to_dataset(dataset: DataSet):
    dataclass = create_tsdataclass(dataset.covariates)
    ds = observations_to_dataset(dataclass, dataset.observations)
    if dataset.geojson is not None:
        geojson = dataset.geojson
        if isinstance(geojson, str):
            geojson = json.loads(geojson)
        polygons = FeatureCollectionModel.model_validate(geojson)
        ds.set_polygons(polygons)
    return ds

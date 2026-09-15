from __future__ import annotations

import pandas as pd

from sig742_a2_project import (
    build_baseline_validation_evidence,
    build_modelling_feature_table,
    build_split_objects,
    export_submission_csv,
    make_final_forecast_submission,
    run_validation_model_suite,
    validate_forecast_actual_wide,
    wide_to_long,
)


DATA_URL = "https://raw.githubusercontent.com/tulip-lab/open-data/main/ISF-TDF2023/ISF-TDF2023.csv"
TRAIN_END = "2023M02"
VALIDATION_MONTHS = [f"2023M{month:02d}" for month in range(3, 8)]
FINAL_FORECAST_MONTHS = [f"2023M{month:02d}" for month in range(8, 13)] + [
    f"2024M{month:02d}" for month in range(1, 8)
]
DATE_COLUMN = "Date"


def main() -> None:
    raw_tourism_data = pd.read_csv(DATA_URL)
    required_destinations = [
        column for column in raw_tourism_data.columns if column != DATE_COLUMN
    ]
    final_forecast_columns = [DATE_COLUMN] + required_destinations

    (
        training_actual_wide_to_2023M02,
        validation_actual_wide_2023M03_2023M07,
        public_history_wide_to_2023M07,
    ) = build_split_objects(
        raw_tourism_data,
        train_end=TRAIN_END,
        validation_months=VALIDATION_MONTHS,
        final_cutoff="2023M07",
        date_column=DATE_COLUMN,
    )

    baseline_evidence = build_baseline_validation_evidence(
        training_actual_wide_to_2023M02=training_actual_wide_to_2023M02,
        validation_actual_wide_2023M03_2023M07=validation_actual_wide_2023M03_2023M07,
        required_destinations=required_destinations,
        validation_months=VALIDATION_MONTHS,
        date_column=DATE_COLUMN,
    )

    tourism_series_long = wide_to_long(
        public_history_wide_to_2023M07,
        date_column=DATE_COLUMN,
        value_name="demand",
    )
    modelling_data_or_feature_table = build_modelling_feature_table(
        public_history_wide_to_2023M07,
        date_column=DATE_COLUMN,
    )

    validation_forecasts, validation_metrics, model_summary = run_validation_model_suite(
        training_actual_wide_to_2023M02=training_actual_wide_to_2023M02,
        validation_actual_wide_2023M03_2023M07=validation_actual_wide_2023M03_2023M07,
        required_destinations=required_destinations,
        validation_months=VALIDATION_MONTHS,
        date_column=DATE_COLUMN,
    )

    forecast_submission_wide = make_final_forecast_submission(
        public_history_wide_to_2023M07=public_history_wide_to_2023M07,
        forecast_months=FINAL_FORECAST_MONTHS,
        required_destinations=required_destinations,
        date_column=DATE_COLUMN,
    )[final_forecast_columns]

    forecast_submission_audit = validate_forecast_actual_wide(
        forecast_submission_wide,
        actual_wide_df=None,
        required_months=FINAL_FORECAST_MONTHS,
        required_destinations=required_destinations,
        date_column=DATE_COLUMN,
    )

    export_path = export_submission_csv(
        forecast_submission_wide,
        group_id="YourGroupID",
        output_dir=".",
    )

    print("Baseline validation summary")
    print(baseline_evidence["baseline_validation_summary"].round(4).to_string(index=False))
    print("\nSelected model summary")
    print(model_summary.round(4).to_string(index=False))
    print("\nValidation metrics preview")
    print(validation_metrics.head(12).round(4).to_string(index=False))
    print("\nTourism long-format shape:", tourism_series_long.shape)
    print("Feature table preview")
    print(modelling_data_or_feature_table.tail(12).to_string(index=False))
    print("\nForecast submission audit")
    print(forecast_submission_audit)
    print("\nExported forecast CSV:", export_path)


if __name__ == "__main__":
    main()

# De-identified Data

## De-identification Procedure

All personally-identifiable information has been removed from these data files
before inclusion in this manuscript package. The following columns were replaced
with anonymous sequential identifiers:

| Original Column   | Replacement Format | Description                          |
|--------------------|--------------------|--------------------------------------|
| `participant_id`   | P001, P002, ...    | Anonymous participant label           |
| `prolific_id`      | P001, P002, ...    | Mapped to same namespace as above     |
| `session_id`        | S001, S002, ...    | Anonymous session label               |

The mapping between original identifiers and anonymous labels is **not**
included in this package and is stored separately under restricted access.

## Files

- **events_deidentified.csv** -- Trial-level event log with all behavioral
  events (clicks, rewards, timestamps, derived variables). See
  `outcome_definitions.md` in the project repository for complete column
  definitions and derivation details.

- **session_metrics.csv** -- Session-level summary metrics (one row per
  session). Includes aggregate measures such as total clicks, switch rates,
  reward rates, and phase-level statistics.

## Column Reference

For full column definitions, derivation formulae, and outcome variable
descriptions, refer to `outcome_definitions.md` in the project repository.

## Ethics Statement

[PLACEHOLDER] This study was approved by [Institution] Institutional Review
Board (Protocol #XXXX-XXXX). All participants provided informed consent prior
to participation. Data were collected via the Prolific platform. No
personally-identifiable information is included in this dataset.

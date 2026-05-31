# Manuscript Output Package

Generated: 2026-03-10T14:45:26.054540
Pipeline version: 1.0.0

- Sessions: 60
- Participants: 60
- Total events: 55454

## Contents

### figures/
- figure10_entropy.pdf
- figure10_entropy.png
- figure12_ccm.pdf
- figure12_ccm.png
- figure13_smap.pdf
- figure13_smap.png
- figure14_hmm.pdf
- figure14_hmm.png
- figure15_phenotypes.pdf
- figure15_phenotypes.png
- figure16_null_comparison.pdf
- figure16_null_comparison.png
- figure17_robustness.pdf
- figure17_robustness.png
- figure1_task_schematic.pdf
- figure1_task_schematic.png
- figure2_example_trajectories.pdf
- figure2_example_trajectories.png
- figure3_phase_transitions.pdf
- figure3_phase_transitions.png
- figure4_pulse_response.pdf
- figure4_pulse_response.png
- figure5_state_space.pdf
- figure5_state_space.png
- figure6_individual_differences.pdf
- figure6_individual_differences.png
- figure7_model_comparison.pdf
- figure7_model_comparison.png
- figure8_rqa.pdf
- figure8_rqa.png
- figure9_dfa.pdf
- figure9_dfa.png

### supplementary/
- supp_choice_acf.pdf
- supp_choice_acf.png
- supp_ici_dist.pdf
- supp_ici_dist.png
- supp_obs_vs_sim.pdf
- supp_obs_vs_sim.png
- supp_reward_trajectories.pdf
- supp_reward_trajectories.png
- supp_run_length_dist.pdf
- supp_run_length_dist.png

### tables/
- glossary_dynamical_systems.csv
- glossary_dynamical_systems.md
- table1_session_summary.csv
- table1_session_summary.md
- table2_phase_metrics.csv
- table2_phase_metrics.md
- table3_transition_metrics.csv
- table3_transition_metrics.md
- table4_pulse_response.csv
- table4_pulse_response.md
- table5_model_fits.csv
- table5_model_fits.md

### captions/
- figure1_task_schematic.md
- figure2_example_trajectories.md
- figure3_phase_transitions.md
- figure4_pulse_response.md
- figure5_state_space.md
- figure6_individual_differences.md
- figure7_model_comparison.md
- table1_session_summary.md
- table2_phase_metrics.md
- table3_transition_metrics.md
- table4_pulse_response.md
- table5_model_fits.md

### results_summary/
- manuscript_outline.md
- results_summary.md

### data/
- README.md
- events_deidentified.csv
- session_metrics.csv

## Reproducing

Run the included reproduction script:

```bash
./reproduce.sh
```

Or manually:

```bash
pip install -r requirements.txt
python run_analysis.py --config config.yaml \
    --steps validate transform metrics plots phase pulse dynamical \
           fractal models individual_differences report
```

The `analysis_config.json` file records the exact configuration used.
The `config.yaml` file is the YAML configuration consumed by the pipeline.

## De-identified Data

The `data/` directory contains de-identified versions of the behavioral
data. See `data/README.md` for details on the de-identification procedure.

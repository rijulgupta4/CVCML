# Run 24 - v0.5 Operating Policy Characterization

## Purpose

Run 24 freezes the Run 23 candidate label (`primary_or_uncertain`) and evaluates how the calibrated validation risk scores behave as operational review policies. It does not retrain the model or inspect the temporal lockbox.

## Key Validation Context

- Landmark rows evaluated: 5,694
- Positive landmark rows: 242
- Base row prevalence: 4.3%
- Positive episodes: 54

## Top 5% Row Review Policy

- Reviews: 285 landmark rows (5.0%)
- PPV: 10.2% (2.39x base prevalence)
- Row recall: 12.0%
- Episode recall: 29.6%
- False reviews per true-positive row: 8.83

## Top 10% Row Review Policy

- Reviews: 570 landmark rows (10.0%)
- PPV: 9.6% (2.27x base prevalence)
- Row recall: 22.7%
- Episode recall: 40.7%
- False reviews per true-positive row: 9.36

## Interpretation

- The model is best framed as a prioritized review-list or infection-prevention surveillance aid, not a bedside interruptive alarm.
- Top-row budgets provide the clearest operating point because they directly control review burden.
- Episode-limited policies reduce repeated reviews, but they can miss useful repeated risk signals when a patient's status evolves over time.
- Threshold policies are less portable at this stage because calibration remains development-set dependent; use review budgets until calibration is externally tested.

## Recommended Candidate Policies

- balanced_review_list: top_episode_budget / top_250_episodes | reviews=250, PPV=10.4%, episode recall=48.1%, false reviews/TP=8.62
- balanced_review_list: threshold_with_cooldown / risk_ge_0.075_cooldown_168h | reviews=266, PPV=9.0%, episode recall=44.4%, false reviews/TP=10.08
- balanced_review_list: first_alert_per_episode / first_alert_risk_ge_0.050 | reviews=382, PPV=6.8%, episode recall=48.1%, false reviews/TP=13.69
- balanced_review_list: threshold_with_cooldown / risk_ge_0.075_cooldown_72h | reviews=387, PPV=9.0%, episode recall=46.3%, false reviews/TP=10.06
- balanced_review_list: top_row_budget / top_7.5%_rows | reviews=428, PPV=10.7%, episode recall=35.2%, false reviews/TP=8.30
- episode_limited_review: top_episode_budget / top_150_episodes | reviews=150, PPV=14.0%, episode recall=38.9%, false reviews/TP=6.14
- higher_recall_surveillance: top_episode_budget / top_500_episodes | reviews=500, PPV=7.6%, episode recall=70.4%, false reviews/TP=12.16
- higher_recall_surveillance: threshold_with_cooldown / risk_ge_0.050_cooldown_168h | reviews=618, PPV=6.1%, episode recall=70.4%, false reviews/TP=15.26
- higher_recall_surveillance: top_episode_budget / top_648_episodes | reviews=648, PPV=6.6%, episode recall=79.6%, false reviews/TP=14.07
- higher_recall_surveillance: threshold_with_cooldown / risk_ge_0.040_cooldown_168h | reviews=822, PPV=5.7%, episode recall=87.0%, false reviews/TP=16.49
- low_burden_high_ppv: top_episode_budget / top_0.5%_episodes | reviews=4, PPV=25.0%, episode recall=1.9%, false reviews/TP=3.00
- low_burden_high_ppv: top_episode_budget / top_15%_episodes | reviews=98, PPV=15.3%, episode recall=27.8%, false reviews/TP=5.53

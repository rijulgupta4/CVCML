# Run 27 - ICD Agreement Label Validation

## Purpose

Run 27 evaluates whether the v0.5 microbiology/timing/source-screened CVC-BSI proxy aligns with administrative ICD-coded CLABSI or central-line infection diagnoses. This is a label-validity and framing run, not a model tuning run.

## ICD Comparator Definitions

- ICD-specific CVC bloodstream comparator: ICD-9 99931/99932 and ICD-10 T80.211* bloodstream infection due to central venous catheter.
- ICD-any CVC infection comparator: ICD-specific plus local, other, or unspecified central venous catheter infection codes such as ICD-9 99933 and ICD-10 T80.212*/T80.218*/T80.219*.
- ICD-broad comparator: ICD-any CVC infection plus broader vascular-device infection codes such as ICD-9 99662 and ICD-10 T82.7*.

## Main Agreement Results

- Episodes evaluated: 22,812.
- Primary-or-uncertain proxy positives: 291 (1.3%).
- ICD-specific positives: 677 (3.0%).
- Overlap between proxy-positive and ICD-specific positives: 60.
- Proxy-positive but ICD-specific negative: 231.
- ICD-specific positive but proxy-negative: 617.
- Positive-set Jaccard vs ICD-specific: 6.6%.
- Against the broader ICD comparator, overlap is 76 with positive-set Jaccard 6.2%.

## Interpretation

ICD-coded CLABSI is useful as an external administrative agreement check, but it should not replace the structured proxy as ground truth. Proxy-positive/ICD-negative episodes likely include clinically plausible culture/timing events that were not coded as central-line infection. ICD-positive/proxy-negative episodes may reflect coding without a qualifying reconstructed line episode, culture timing mismatch, non-bloodstream/local line infection coding, or incomplete procedureevents line documentation.

## Operational Implication

The honest language remains: strict CVC-associated BSI proxy, with ICD-coded CLABSI agreement as a secondary validation check. Strong model claims should be framed around prospective risk stratification of the proxy rather than adjudicated NHSN CLABSI.

## Lockbox Model/ICD Intersection

- In the lockbox, the top 150 episode review list selected 150 episodes.
- It captured 18 proxy-positive episodes and 6 ICD-specific positive episodes.
- Selected PPV vs proxy: 12.0%. Selected PPV vs ICD-specific: 4.0%.

## Key Output Files

- `v0_5_run27_icd_clabsi_code_candidates.csv`
- `v0_5_run27_hadm_icd_clabsi_flags.csv`
- `v0_5_run27_episode_icd_agreement.csv`
- `v0_5_run27_proxy_icd_agreement_table.csv`
- `v0_5_run27_lockbox_icd_model_intersection.csv`
- `plots/v0_5_run27_proxy_icd_overlap.png`

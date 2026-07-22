# Run 29 adjudication protocol

## Purpose
Validate the strict CVC-associated BSI proxy against structured chart evidence and, when MIMIC-IV-Note becomes available, narrative evidence. This is not official NHSN adjudication.

## Blinding
Reviewers should not see model scores while adjudicating the outcome. The queue is stratified by proxy/ICD agreement and source-screen class; it is not a prevalence sample.

## Review sequence
1. Confirm the reconstructed CVC exposure episode and at least 48 hours of observed line exposure before the candidate blood culture.
2. Confirm that the line was present on the event day or the prior calendar day where the available data permit this.
3. Confirm the blood-culture organism rule: a recognized pathogen, or repeated qualifying common commensal cultures with compatible symptoms.
4. Search the +/-3-day window for a plausible secondary source using nonblood cultures, diagnoses, procedures, antimicrobial context, and notes when available.
5. Consider MBI-LCBI plausibility using malignancy/transplant, neutropenia, and organism context; mark uncertain when the structured record is insufficient.
6. Assign one final category: likely primary CVC-associated BSI, likely secondary BSI, contaminant, or insufficient evidence.
7. Record confidence as high, medium, or low and document the decisive evidence.

## Interpretation
The balanced queue estimates failure modes and agreement within strata; it must not be used directly to estimate population PPV without applying sampling weights. Two independent reviewers and consensus resolution are preferred for a manuscript-grade validation subset.

## Required future data
MIMIC-IV-Note is not present locally. Narrative source attribution and chills documentation therefore remain pending until note access is added.

## Sources
- MIMIC-IV microbiologyevents documentation: https://mimic.mit.edu/docs/IV/modules/hosp/microbiologyevents.html
- CDC NHSN Patient Safety Component Manual, Bloodstream Infection chapter: https://www.cdc.gov/nhsn/pdfs/pscmanual/4psc_clabscurrent.pdf


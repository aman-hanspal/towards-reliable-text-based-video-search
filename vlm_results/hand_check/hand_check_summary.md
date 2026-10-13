# Hand check of the VLM test set (medium 640)

## Set A sample (blind)

- sample: 100; judged yes/no: 89; unsure: 11; blank/other: 0
- acceptable name after all (dataset 'error' is not an error): 12.4% [7.0, 20.8] (n=89)
- VLM removal rate on ALL judged sample items (as reported): 70.8% [60.6, 79.2] (n=89)
- VLM removal rate on TRUE lookalike errors only (corrected): 75.6% [65.1, 83.8] (n=78)
- VLM said yes on the acceptable-name items (agrees with you): 63.6% [35.4, 84.8] (n=11)

## Set B: real objects the VLM rejected, by reason

| reason | count | share |
|---|---:|---:|
| hand | 35 | 57% |
| surface | 9 | 15% |
| competing | 8 | 13% |
| miss | 7 | 11% |
| other | 2 | 3% |

- set B size: 600; rejected: 61 (10.2%); of which clear VLM misses: 7 (1.2% of all set B)

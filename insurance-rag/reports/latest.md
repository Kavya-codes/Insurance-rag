# Evaluation Report — insurance-rag

Generated at: `2026-11-20T14:55:05.042400+00:00`

## Summary Metrics

| Metric | Value |
| --- | --- |
| Total Questions | 40 (34 answerable, 6 out-of-scope) |
| Hit@1 | 88.2% |
| Hit@3 | 100.0% |
| Hit@5 | 100.0% |
| MRR | 0.9412 |
| Keyword Coverage | 100.0% |
| Abstention Accuracy (Out-of-Scope) | 66.7% |
| False Abstention Rate (Answerable) | 0.0% |
| LLM-as-Judge Faithfulness | Skipped (set GEMINI_API_KEY and omit --skip-judge) |

## Per-Question Breakdown

| ID | Expected Doc | Best Score | Hit@1 | Hit@3 | Hit@5 | KW Cov | Abstained | Faithfulness |
| --- | --- | ---: | :---: | :---: | :---: | ---: | :---: | ---: |
| q01 | irdai_health_insurance_master_circular_2024.md | 1.000 | Y | Y | Y | 100% | No | - |
| q02 | irdai_health_insurance_master_circular_2024.md | 0.894 | Y | Y | Y | 100% | No | - |
| q03 | irdai_health_insurance_master_circular_2024.md | 0.863 | Y | Y | Y | 100% | No | - |
| q04 | irdai_health_insurance_master_circular_2024.md | 0.920 | Y | Y | Y | 100% | No | - |
| q05 | irdai_health_insurance_master_circular_2024.md | 1.000 | Y | Y | Y | 100% | No | - |
| q06 | irdai_health_insurance_master_circular_2024.md | 0.973 | Y | Y | Y | 100% | No | - |
| q07 | irdai_health_insurance_master_circular_2024.md | 1.000 | Y | Y | Y | 100% | No | - |
| q08 | irdai_health_insurance_master_circular_2024.md | 0.885 | Y | Y | Y | 100% | No | - |
| q09 | irdai_health_insurance_master_circular_2024.md | 1.000 | Y | Y | Y | 100% | No | - |
| q10 | irdai_health_insurance_master_circular_2024.md | 0.885 | Y | Y | Y | 100% | No | - |
| q11 | irdai_health_insurance_master_circular_2024.md | 0.690 | Y | Y | Y | 100% | No | - |
| q12 | irdai_health_insurance_master_circular_2024.md | 0.690 | Y | Y | Y | 100% | No | - |
| q13 | irdai_health_insurance_master_circular_2024.md | 0.863 | Y | Y | Y | 100% | No | - |
| q14 | irdai_health_insurance_master_circular_2024.md | 1.000 | Y | Y | Y | 100% | No | - |
| q15 | irdai_health_insurance_master_circular_2024.md | 1.000 | Y | Y | Y | 100% | No | - |
| q16 | irdai_health_insurance_master_circular_2024.md | 1.000 | Y | Y | Y | 100% | No | - |
| q17 | irdai_protection_of_policyholders_interests_2024.md | 0.767 | Y | Y | Y | 100% | No | - |
| q18 | irdai_protection_of_policyholders_interests_2024.md | 0.719 | Y | Y | Y | 100% | No | - |
| q19 | irdai_protection_of_policyholders_interests_2024.md | 0.719 | Y | Y | Y | 100% | No | - |
| q20 | irdai_protection_of_policyholders_interests_2024.md | 1.000 | Y | Y | Y | 100% | No | - |
| q21 | irdai_protection_of_policyholders_interests_2024.md | 0.767 | N | Y | Y | 100% | No | - |
| q22 | irdai_protection_of_policyholders_interests_2024.md | 0.920 | Y | Y | Y | 100% | No | - |
| q23 | standard_health_policy_wordings_arogya_sanjeevani.txt | 1.000 | N | Y | Y | 100% | No | - |
| q24 | standard_health_policy_wordings_arogya_sanjeevani.txt | 0.941 | Y | Y | Y | 100% | No | - |
| q25 | standard_health_policy_wordings_arogya_sanjeevani.txt | 0.796 | Y | Y | Y | 100% | No | - |
| q26 | standard_health_policy_wordings_arogya_sanjeevani.txt | 1.000 | Y | Y | Y | 100% | No | - |
| q27 | standard_health_policy_wordings_arogya_sanjeevani.txt | 1.000 | Y | Y | Y | 100% | No | - |
| q28 | standard_health_policy_wordings_arogya_sanjeevani.txt | 0.894 | Y | Y | Y | 100% | No | - |
| q29 | standard_health_policy_wordings_arogya_sanjeevani.txt | 0.575 | N | Y | Y | 100% | No | - |
| q30 | pmjay_ab_operational_guidelines.md | 0.805 | Y | Y | Y | 100% | No | - |
| q31 | pmjay_ab_operational_guidelines.md | 0.904 | Y | Y | Y | 100% | No | - |
| q32 | pmjay_ab_operational_guidelines.md | 0.671 | Y | Y | Y | 100% | No | - |
| q33 | pmjay_ab_operational_guidelines.md | 0.690 | N | Y | Y | 100% | No | - |
| q34 | pmjay_ab_operational_guidelines.md | 0.708 | Y | Y | Y | 100% | No | - |
| q35 | OUT_OF_SCOPE | 0.230 | - | - | - | - | Yes | - |
| q36 | OUT_OF_SCOPE | 0.051 | - | - | - | - | Yes | - |
| q37 | OUT_OF_SCOPE | 0.383 | - | - | - | - | No | - |
| q38 | OUT_OF_SCOPE | 0.046 | - | - | - | - | Yes | - |
| q39 | OUT_OF_SCOPE | 0.051 | - | - | - | - | Yes | - |
| q40 | OUT_OF_SCOPE | 0.345 | - | - | - | - | No | - |

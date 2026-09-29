"""
ClinicalTrials.gov API v2 client.
Fetches NSCLC trials with biomarker eligibility criteria.
Public API, no authentication.
"""
import requests
import time
import json
from pathlib import Path

CT_API = "https://clinicaltrials.gov/api/v2/studies"


def fetch_nsclc_trials(max_trials: int = 200) -> list[dict]:
    """Fetch recruiting/active NSCLC trials with biomarker keywords."""
    biomarker_queries = [
        "EGFR lung cancer", "ALK lung cancer", "ROS1 lung cancer",
        "BRAF lung cancer", "KRAS lung cancer", "MET lung cancer",
        "RET lung cancer", "NTRK lung cancer", "HER2 lung cancer",
        "PD-L1 lung cancer osimertinib",
    ]
    all_trials = []
    seen_ids = set()

    for query in biomarker_queries:
        params = {
            "query.cond": "Non-Small Cell Lung Cancer",
            "query.term": query,
            "filter.overallStatus": "RECRUITING,ACTIVE_NOT_RECRUITING,COMPLETED",
            "fields": "NCTId,BriefTitle,OfficialTitle,OverallStatus,Phase,"
                      "EligibilityCriteria,InterventionName,PrimaryOutcomeMeasure,"
                      "BriefSummary,Keyword,Condition",
            "pageSize": 20,
            "format": "json",
        }
        try:
            r = requests.get(CT_API, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
            studies = data.get("studies", [])
            for study in studies:
                proto = study.get("protocolSection", {})
                id_module = proto.get("identificationModule", {})
                nct_id = id_module.get("nctId", "")
                if nct_id in seen_ids:
                    continue
                seen_ids.add(nct_id)

                elig = proto.get("eligibilityModule", {})
                arms = proto.get("armsInterventionsModule", {}) or {}
                interventions = arms.get("interventions", []) or []

                all_trials.append({
                    "nct_id": nct_id,
                    "title": id_module.get("briefTitle", ""),
                    "official_title": id_module.get("officialTitle", ""),
                    "status": proto.get("statusModule", {}).get("overallStatus", ""),
                    "phase": str(proto.get("designModule", {}).get("phases", [])),
                    "eligibility_criteria": elig.get("eligibilityCriteria", "")[:2000],
                    "interventions": [iv.get("name", "") for iv in interventions],
                    "conditions": proto.get("conditionsModule", {}).get("conditions", []),
                    "keywords": proto.get("conditionsModule", {}).get("keywords", []),
                })
            time.sleep(0.5)
        except Exception as e:
            print(f"  [ClinicalTrials] Error for query '{query}': {e}")

        if len(all_trials) >= max_trials:
            break

    print(f"Fetched {len(all_trials)} unique NSCLC trials from ClinicalTrials.gov")
    return all_trials


def extract_biomarker_eligibility(trials: list[dict]) -> list[dict]:
    """Extract biomarker requirements from eligibility criteria."""
    biomarkers = ["EGFR", "ALK", "ROS1", "BRAF", "MET", "RET", "NTRK", "KRAS",
                  "HER2", "ERBB2", "PD-L1", "TMB", "STK11", "KEAP1"]
    result = []
    for trial in trials:
        elig = trial.get("eligibility_criteria", "").upper()
        required = [b for b in biomarkers if b in elig]
        trial_copy = dict(trial)
        trial_copy["required_biomarkers"] = required
        trial_copy["has_biomarker_requirement"] = len(required) > 0
        result.append(trial_copy)
    return result


def fetch_and_save(output_path: str = None) -> list[dict]:
    trials = fetch_nsclc_trials()
    trials = extract_biomarker_eligibility(trials)
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(trials, f, indent=2)
        print(f"Saved {len(trials)} trials to {output_path}")
    return trials


if __name__ == "__main__":
    from pathlib import Path
    out = Path(__file__).parent.parent.parent / "data" / "knowledge_base" / "clinical_trials_nsclc.json"
    trials = fetch_and_save(output_path=str(out))
    print(f"\nSample trial:")
    if trials:
        print(json.dumps(trials[0], indent=2))

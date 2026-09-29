"""
OpenFDA API client.
Fetches FDA drug label information for NSCLC-approved targeted therapies.
Public API, no authentication.
"""
import requests
import time
import json
from pathlib import Path

OPENFDA_API = "https://api.fda.gov/drug/label.json"

NSCLC_DRUGS = {
    "osimertinib": {"biomarker": "EGFR", "alterations": ["L858R", "exon 19 deletion", "T790M"], "brand": "Tagrisso"},
    "erlotinib": {"biomarker": "EGFR", "alterations": ["L858R", "exon 19 deletion"], "brand": "Tarceva"},
    "gefitinib": {"biomarker": "EGFR", "alterations": ["L858R", "exon 19 deletion"], "brand": "Iressa"},
    "afatinib": {"biomarker": "EGFR", "alterations": ["L858R", "exon 19 deletion"], "brand": "Gilotrif"},
    "alectinib": {"biomarker": "ALK", "alterations": ["fusion", "rearrangement"], "brand": "Alecensa"},
    "crizotinib": {"biomarker": "ALK/ROS1/MET", "alterations": ["fusion", "exon 14 skipping"], "brand": "Xalkori"},
    "lorlatinib": {"biomarker": "ALK", "alterations": ["fusion", "rearrangement"], "brand": "Lorbrena"},
    "brigatinib": {"biomarker": "ALK", "alterations": ["fusion", "rearrangement"], "brand": "Alunbrig"},
    "dabrafenib": {"biomarker": "BRAF", "alterations": ["V600E"], "brand": "Tafinlar"},
    "trametinib": {"biomarker": "MEK/BRAF", "alterations": ["V600E"], "brand": "Mekinist"},
    "sotorasib": {"biomarker": "KRAS", "alterations": ["G12C"], "brand": "Lumakras"},
    "adagrasib": {"biomarker": "KRAS", "alterations": ["G12C"], "brand": "Krazati"},
    "tepotinib": {"biomarker": "MET", "alterations": ["exon 14 skipping"], "brand": "Tepmetko"},
    "capmatinib": {"biomarker": "MET", "alterations": ["exon 14 skipping"], "brand": "Tabrecta"},
    "selpercatinib": {"biomarker": "RET", "alterations": ["fusion", "rearrangement"], "brand": "Retevmo"},
    "pralsetinib": {"biomarker": "RET", "alterations": ["fusion", "rearrangement"], "brand": "Gavreto"},
    "larotrectinib": {"biomarker": "NTRK", "alterations": ["fusion"], "brand": "Vitrakvi"},
    "entrectinib": {"biomarker": "NTRK/ROS1", "alterations": ["fusion"], "brand": "Rozlytrek"},
    "pembrolizumab": {"biomarker": "PD-L1", "alterations": ["high expression", "TPS>=50%"], "brand": "Keytruda"},
    "atezolizumab": {"biomarker": "PD-L1", "alterations": ["expression"], "brand": "Tecentriq"},
}


def fetch_drug_label(drug_name: str) -> dict:
    """Fetch FDA drug label from OpenFDA."""
    drug_info = NSCLC_DRUGS.get(drug_name, {})
    params = {
        "search": f"openfda.generic_name:{drug_name}",
        "limit": 1
    }
    try:
        r = requests.get(OPENFDA_API, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        results = data.get("results", [])
        if not results:
            brand = drug_info.get("brand", "")
            if brand:
                params["search"] = f"openfda.brand_name:{brand}"
                r2 = requests.get(OPENFDA_API, params=params, timeout=30)
                r2.raise_for_status()
                data2 = r2.json()
                results = data2.get("results", [])
        if results:
            label = results[0]
            return {
                "drug_name": drug_name,
                "brand_name": drug_info.get("brand", ""),
                "biomarker": drug_info.get("biomarker", ""),
                "alterations": drug_info.get("alterations", []),
                "indications": label.get("indications_and_usage", [""])[0][:1500] if label.get("indications_and_usage") else "",
                "contraindications": label.get("contraindications", [""])[0][:500] if label.get("contraindications") else "",
                "warnings": label.get("warnings_and_cautions", [""])[0][:500] if label.get("warnings_and_cautions") else "",
                "mechanism": label.get("mechanism_of_action", [""])[0][:500] if label.get("mechanism_of_action") else "",
                "source": "FDA",
            }
    except Exception as e:
        print(f"  [FDA] Error fetching {drug_name}: {e}")

    # fallback: return known metadata even if label fetch fails
    return {
        "drug_name": drug_name,
        "brand_name": drug_info.get("brand", ""),
        "biomarker": drug_info.get("biomarker", ""),
        "alterations": drug_info.get("alterations", []),
        "indications": f"FDA-approved for NSCLC with {drug_info.get('biomarker','')} {', '.join(drug_info.get('alterations',[]))}",
        "contraindications": "",
        "warnings": "",
        "mechanism": "",
        "source": "FDA_metadata_fallback",
    }


def fetch_all_nsclc_drug_labels(output_path: str = None) -> list[dict]:
    labels = []
    for drug_name in NSCLC_DRUGS:
        print(f"  Fetching FDA label for {drug_name}...")
        label = fetch_drug_label(drug_name)
        labels.append(label)
        time.sleep(0.4)

    print(f"\nFetched {len(labels)} NSCLC drug FDA entries")
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(labels, f, indent=2)
        print(f"Saved to {output_path}")
    return labels


if __name__ == "__main__":
    from pathlib import Path
    out = Path(__file__).parent.parent.parent / "data" / "knowledge_base" / "fda_nsclc_labels.json"
    labels = fetch_all_nsclc_drug_labels(output_path=str(out))
    print(f"\nSample label:")
    if labels:
        print(json.dumps(labels[0], indent=2))

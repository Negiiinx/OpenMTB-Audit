"""
CIViC (Clinical Interpretation of Variants in Cancer) GraphQL API client.
Fetches NSCLC-relevant variant evidence records.
Public GraphQL API, no authentication required.
"""
import requests
import time
import json
from pathlib import Path

CIVIC_GRAPHQL = "https://civicdb.org/api/graphql"

NSCLC_DRUGS = [
    "Osimertinib", "Erlotinib", "Gefitinib", "Afatinib", "Dacomitinib",
    "Alectinib", "Crizotinib", "Lorlatinib", "Brigatinib", "Ceritinib",
    "Dabrafenib", "Trametinib", "Sotorasib", "Adagrasib",
    "Tepotinib", "Capmatinib", "Selpercatinib", "Pralsetinib",
    "Larotrectinib", "Entrectinib", "Pembrolizumab", "Atezolizumab",
    "Nivolumab", "Durvalumab", "Carboplatin", "Cisplatin",
    "Pemetrexed", "Docetaxel", "Paclitaxel", "Bevacizumab",
]

EVIDENCE_QUERY = """
query GetEvidenceForTherapy($therapyName: String!) {
  evidenceItems(
    first: 100
    status: ACCEPTED
    evidenceType: PREDICTIVE
    therapyName: $therapyName
  ) {
    nodes {
      id
      name
      disease { name }
      therapies { name }
      evidenceType
      evidenceLevel
      significance
      molecularProfile { name }
      source { citation }
      description
    }
  }
}
"""

GENE_EVIDENCE_QUERY = """
query GetEvidenceByGene($featureName: String!) {
  evidenceItems(
    first: 200
    status: ACCEPTED
    evidenceType: PREDICTIVE
    featureName: $featureName
  ) {
    nodes {
      id
      name
      disease { name }
      therapies { name }
      evidenceType
      evidenceLevel
      significance
      molecularProfile { name }
      source { citation }
      description
    }
  }
}
"""


def fetch_civic_evidence_for_therapy(therapy_name: str) -> list[dict]:
    try:
        r = requests.post(
            CIVIC_GRAPHQL,
            json={"query": EVIDENCE_QUERY, "variables": {"therapyName": therapy_name}},
            timeout=30
        )
        r.raise_for_status()
        data = r.json()
        nodes = data.get("data", {}).get("evidenceItems", {}).get("nodes", [])
        results = []
        for node in nodes:
            disease = node.get("disease", {}) or {}
            disease_name = disease.get("name", "")
            if not any(kw in disease_name.lower() for kw in ["lung", "nsclc", "carcinoma"]):
                continue
            results.append({
                "evidence_id": node.get("id"),
                "evidence_name": node.get("name"),
                "disease_name": disease_name,
                "therapy": therapy_name,
                "therapies": [t["name"] for t in (node.get("therapies") or [])],
                "evidence_type": node.get("evidenceType"),
                "evidence_level": node.get("evidenceLevel"),
                "significance": node.get("significance"),
                "molecular_profile": (node.get("molecularProfile") or {}).get("name", ""),
                "citation": (node.get("source") or {}).get("citation", ""),
                "description": node.get("description", "")[:500],
            })
        return results
    except Exception as e:
        print(f"  [CIViC] Error for therapy {therapy_name}: {e}")
        return []


def fetch_civic_evidence_for_gene(gene_name: str) -> list[dict]:
    try:
        r = requests.post(
            CIVIC_GRAPHQL,
            json={"query": GENE_EVIDENCE_QUERY, "variables": {"featureName": gene_name}},
            timeout=30
        )
        r.raise_for_status()
        data = r.json()
        nodes = data.get("data", {}).get("evidenceItems", {}).get("nodes", [])
        results = []
        for node in nodes:
            disease = node.get("disease", {}) or {}
            disease_name = disease.get("name", "")
            if not any(kw in disease_name.lower() for kw in ["lung", "nsclc", "carcinoma"]):
                continue
            results.append({
                "evidence_id": node.get("id"),
                "evidence_name": node.get("name"),
                "gene": gene_name,
                "disease_name": disease_name,
                "therapies": [t["name"] for t in (node.get("therapies") or [])],
                "therapy": ", ".join(t["name"] for t in (node.get("therapies") or [])),
                "evidence_type": node.get("evidenceType"),
                "evidence_level": node.get("evidenceLevel"),
                "significance": node.get("significance"),
                "molecular_profile": (node.get("molecularProfile") or {}).get("name", ""),
                "citation": (node.get("source") or {}).get("citation", ""),
                "description": node.get("description", "")[:500],
            })
        return results
    except Exception as e:
        print(f"  [CIViC] Error for gene {gene_name}: {e}")
        return []


def fetch_all_nsclc_evidence(output_path: str = None) -> list[dict]:
    """Fetch all CIViC predictive NSCLC evidence via drug and gene queries."""
    all_evidence = []
    seen_ids = set()

    # By therapy
    for drug in NSCLC_DRUGS:
        print(f"  CIViC: {drug}...")
        items = fetch_civic_evidence_for_therapy(drug)
        for item in items:
            eid = item["evidence_id"]
            if eid not in seen_ids:
                seen_ids.add(eid)
                all_evidence.append(item)
        print(f"    -> {len(items)} NSCLC items")
        time.sleep(0.3)

    # By gene (catches entries not linked to above drugs)
    for gene in ["EGFR", "ALK", "ROS1", "BRAF", "MET", "RET", "NTRK1", "KRAS", "HER2", "ERBB2"]:
        print(f"  CIViC gene: {gene}...")
        items = fetch_civic_evidence_for_gene(gene)
        added = 0
        for item in items:
            eid = item["evidence_id"]
            if eid not in seen_ids:
                seen_ids.add(eid)
                item["gene"] = gene
                all_evidence.append(item)
                added += 1
        print(f"    -> {added} new items")
        time.sleep(0.3)

    print(f"\nTotal CIViC NSCLC predictive evidence: {len(all_evidence)}")
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(all_evidence, f, indent=2)
        print(f"Saved to {output_path}")
    return all_evidence


if __name__ == "__main__":
    from pathlib import Path
    out = Path(__file__).parent.parent.parent / "data" / "knowledge_base" / "civic_nsclc_evidence.json"
    evidence = fetch_all_nsclc_evidence(output_path=str(out))
    print(f"\nSample evidence:")
    if evidence:
        print(json.dumps(evidence[0], indent=2))

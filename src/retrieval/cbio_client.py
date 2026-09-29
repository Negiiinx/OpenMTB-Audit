"""
cBioPortal API client.
Fetches public NSCLC genomic profiles (mutations, CNAs, clinical data).
ODC Open Database License — no authentication for public data.
"""
import requests
import time
import json
from pathlib import Path

CBIO_API = "https://www.cbioportal.org/api"

# Well-known public NSCLC studies in cBioPortal
NSCLC_STUDIES = [
    "luad_tcga_pan_can_atlas_2018",   # TCGA lung adenocarcinoma
    "lusc_tcga_pan_can_atlas_2018",   # TCGA lung squamous cell
    "nsclc_mskcc_2015",               # MSK cohort
    "lung_msk_2017",                  # MSK lung 2017
    "nsclc_tcga_broad_2016",          # TCGA broad 2016
]

NSCLC_DRIVER_GENES = [
    "EGFR", "KRAS", "ALK", "ROS1", "BRAF", "MET", "RET",
    "NTRK1", "NTRK2", "NTRK3", "ERBB2", "STK11",
    "KEAP1", "NF1", "TP53", "CDKN2A", "RB1", "PIK3CA",
]

# Entrez Gene IDs for NSCLC driver genes
ENTREZ_IDS = {
    "EGFR": 1956, "KRAS": 3845, "ALK": 238, "ROS1": 6098,
    "BRAF": 673, "MET": 4233, "RET": 5979, "NTRK1": 4914,
    "NTRK2": 4915, "NTRK3": 4916, "ERBB2": 2064, "STK11": 6794,
    "KEAP1": 9817, "NF1": 4763, "TP53": 7157, "CDKN2A": 1029,
    "RB1": 5925, "PIK3CA": 5290,
}


def get_available_nsclc_studies() -> list[dict]:
    """Get all available NSCLC studies in cBioPortal."""
    try:
        r = requests.get(f"{CBIO_API}/studies", params={"keyword": "lung"}, timeout=30)
        r.raise_for_status()
        studies = r.json()
        nsclc = [s for s in studies if any(
            kw in s.get("name", "").lower()
            for kw in ["non-small", "nsclc", "adenocarcinoma", "squamous", "lung"]
        )]
        print(f"Found {len(nsclc)} NSCLC-related studies in cBioPortal")
        return nsclc
    except Exception as e:
        print(f"  [cBioPortal] Error fetching studies: {e}")
        return []


def fetch_mutations_for_study(study_id: str, genes: list[str] = None) -> list[dict]:
    """Fetch mutation data for a study."""
    if genes is None:
        genes = NSCLC_DRIVER_GENES[:10]  # Limit for API efficiency

    # Get molecular profiles for this study
    try:
        r = requests.get(
            f"{CBIO_API}/studies/{study_id}/molecular-profiles",
            timeout=30
        )
        r.raise_for_status()
        profiles = r.json()
    except Exception as e:
        print(f"  [cBioPortal] Cannot get profiles for {study_id}: {e}")
        return []

    # Find mutation profile
    mut_profile = next(
        (p for p in profiles if p.get("molecularAlterationType") == "MUTATION_EXTENDED"),
        None
    )
    if not mut_profile:
        return []

    profile_id = mut_profile["molecularProfileId"]

    # Get sample list
    try:
        r = requests.get(
            f"{CBIO_API}/studies/{study_id}/sample-lists",
            timeout=30
        )
        r.raise_for_status()
        sample_lists = r.json()
        all_samples = next(
            (sl for sl in sample_lists if "all" in sl.get("sampleListId", "").lower()),
            sample_lists[0] if sample_lists else None
        )
        if not all_samples:
            return []
        sample_list_id = all_samples["sampleListId"]
    except Exception as e:
        print(f"  [cBioPortal] Cannot get sample list for {study_id}: {e}")
        return []

    # Fetch mutations for target genes
    mutations = []
    for gene in genes:
        entrez_id = ENTREZ_IDS.get(gene)
        if not entrez_id:
            continue
        try:
            payload = {
                "entrezGeneIds": [entrez_id],
                "sampleListId": sample_list_id,
            }
            r = requests.post(
                f"{CBIO_API}/molecular-profiles/{profile_id}/mutations/fetch",
                params={"projection": "SUMMARY", "pageSize": 200, "pageNumber": 0},
                json=payload,
                timeout=30
            )
            r.raise_for_status()
            gene_muts = r.json()
            for m in gene_muts[:50]:  # cap per gene to avoid huge data
                mutations.append({
                    "study_id": study_id,
                    "gene": gene,
                    "sample_id": m.get("sampleId", ""),
                    "patient_id": m.get("patientId", ""),
                    "mutation_type": m.get("mutationType", ""),
                    "protein_change": m.get("proteinChange", ""),
                    "amino_acid_change": m.get("aminoAcidChange", ""),
                    "validation_status": m.get("validationStatus", ""),
                    "functional_impact": m.get("functionalImpactScore", ""),
                    "mutation_status": m.get("mutationStatus", ""),
                    "ncbi_build": m.get("ncbiBuild", ""),
                    "chromosome": m.get("chr", ""),
                    "start_pos": m.get("startPosition", ""),
                    "reference_allele": m.get("referenceAllele", ""),
                    "variant_allele": m.get("variantAllele", ""),
                })
            time.sleep(0.2)
        except Exception as e:
            print(f"  [cBioPortal] Error fetching {gene} mutations in {study_id}: {e}")
    return mutations


def fetch_clinical_data_for_study(study_id: str) -> list[dict]:
    """Fetch clinical data (stage, survival) for a study."""
    try:
        r = requests.get(
            f"{CBIO_API}/studies/{study_id}/clinical-data",
            params={"clinicalDataType": "PATIENT"},
            timeout=30
        )
        r.raise_for_status()
        clinical = r.json()
        # Pivot to patient-level dict
        patient_data = {}
        for item in clinical:
            pid = item.get("patientId", "")
            attr = item.get("clinicalAttributeId", "")
            val = item.get("value", "")
            if pid not in patient_data:
                patient_data[pid] = {"study_id": study_id, "patient_id": pid}
            patient_data[pid][attr.lower()] = val
        return list(patient_data.values())
    except Exception as e:
        print(f"  [cBioPortal] Error fetching clinical data for {study_id}: {e}")
        return []


def build_patient_profiles(study_id: str, max_patients: int = 100) -> list[dict]:
    """
    Build combined patient-mutation profiles for benchmark case generation.
    Returns list of patient dicts with mutations and clinical info.
    """
    print(f"  Processing study {study_id}...")
    mutations = fetch_mutations_for_study(study_id)
    clinical = fetch_clinical_data_for_study(study_id)

    # Index by patient
    clinical_idx = {p["patient_id"]: p for p in clinical}

    # Group mutations by patient
    patient_muts = {}
    for m in mutations:
        pid = m["patient_id"]
        if pid not in patient_muts:
            patient_muts[pid] = []
        patient_muts[pid].append(m)

    profiles = []
    for pid, muts in list(patient_muts.items())[:max_patients]:
        clinical_info = clinical_idx.get(pid, {})
        profile = {
            "study_id": study_id,
            "patient_id": pid,
            "cancer_type": "NSCLC",
            "mutations": [
                {
                    "gene": m["gene"],
                    "protein_change": m["protein_change"],
                    "mutation_type": m["mutation_type"],
                    "amino_acid_change": m.get("amino_acid_change", ""),
                }
                for m in muts
            ],
            "stage": clinical_info.get("ajcc_pathologic_tumor_stage", clinical_info.get("tumor_stage", "Unknown")),
            "age": clinical_info.get("age", "Unknown"),
            "sex": clinical_info.get("sex", clinical_info.get("gender", "Unknown")),
            "smoking_history": clinical_info.get("smoking_history", clinical_info.get("tobacco_smoking_history_indicator", "Unknown")),
            "histology": clinical_info.get("histological_type", "Adenocarcinoma"),
        }
        profiles.append(profile)
    return profiles


def fetch_all_profiles(output_path: str = None, studies: list[str] = None) -> list[dict]:
    """Fetch patient profiles from multiple NSCLC studies."""
    if studies is None:
        studies = NSCLC_STUDIES[:3]  # Use top 3 for speed

    all_profiles = []
    for study_id in studies:
        profiles = build_patient_profiles(study_id, max_patients=80)
        print(f"    -> {len(profiles)} patient profiles from {study_id}")
        all_profiles.extend(profiles)
        time.sleep(0.5)

    print(f"\nTotal patient profiles: {len(all_profiles)}")
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(all_profiles, f, indent=2)
        print(f"Saved to {output_path}")
    return all_profiles


if __name__ == "__main__":
    from pathlib import Path
    out = Path(__file__).parent.parent.parent / "data" / "knowledge_base" / "cbio_nsclc_profiles.json"
    profiles = fetch_all_profiles(output_path=str(out))
    if profiles:
        print(f"\nSample profile:")
        print(json.dumps(profiles[0], indent=2))

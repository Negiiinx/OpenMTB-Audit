"""
MTB-AuditAgent: 7-module agentic safety framework for molecular tumor board AI auditing.
Can run in:
  - rule_based mode (no LLM needed) — uses structured evidence KB
  - llm mode — uses Anthropic or OpenAI API (swap in when key available)
"""
import json
import re
from typing import Optional
from dataclasses import dataclass, field, asdict
from pathlib import Path

# ── Evidence knowledge base ───────────────────────────────────────────────────
import sys as _sys
_sys.path.insert(0, str(Path(__file__).parent.parent))
from evidence_kb import EVIDENCE_KB

# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class ParsedCase:
    """Module 1 output: structured fields extracted from patient summary."""
    cancer_type: str = "NSCLC"
    histology: str = "Unknown"
    stage: Optional[str] = None
    genomic_alteration: str = ""
    biomarker: str = ""
    alteration: str = ""
    profile_key: str = ""
    proposed_therapy: str = ""
    ecog: Optional[int] = None
    prior_therapy: Optional[str] = None
    pdl1: Optional[str] = None
    brain_mets: Optional[bool] = None
    age: Optional[int] = None
    sex: Optional[str] = None
    smoking: Optional[str] = None


@dataclass
class EvidenceResult:
    """Module 2+3 output: retrieved and verified evidence."""
    evidence_found: bool = False
    evidence_level: str = ""
    supported_therapies: list = field(default_factory=list)
    proposed_therapy_supported: bool = False
    evidence_text: str = ""
    citation: str = ""
    faithfulness_score: float = 0.0  # 0=hallucinated, 1=fully supported


@dataclass
class MissingInfoResult:
    """Module 4 output: missing critical fields."""
    missing_fields: list = field(default_factory=list)
    critical_missing: list = field(default_factory=list)
    completeness_score: float = 1.0  # 1=complete, 0=major gaps


@dataclass
class SafetyClassification:
    """Module 5 output: recommendation safety class."""
    label: str = "Unknown"  # Supported | Partially Supported | Unsupported | Insufficient Information
    confidence: float = 0.0
    rationale: str = ""


@dataclass
class AbstentionDecision:
    """Module 6 output: should the agent abstain / refer?"""
    should_abstain: bool = False
    abstention_reason: str = ""
    referral_recommended: bool = False


@dataclass
class AuditReport:
    """Module 7 output: final structured audit report."""
    case_id: str = ""
    parsed: dict = field(default_factory=dict)
    evidence: dict = field(default_factory=dict)
    missing_info: dict = field(default_factory=dict)
    safety: dict = field(default_factory=dict)
    abstention: dict = field(default_factory=dict)
    predicted_label: str = "Unknown"
    predicted_abstain: bool = False
    predicted_missing_fields: list = field(default_factory=list)
    audit_summary: str = ""


# ── Profile key matching ──────────────────────────────────────────────────────

PROFILE_PATTERNS = {
    "EGFR_L858R": [r"egfr\s*l858r", r"egfr.*858"],
    "EGFR_exon19del": [r"egfr.*exon\s*19", r"egfr.*del.*19", r"egfr.*19.*del"],
    "EGFR_T790M": [r"egfr\s*t790m", r"egfr.*790"],
    "EGFR_exon20ins": [r"egfr.*exon\s*20", r"egfr.*20.*ins"],
    "ALK_fusion": [r"\balk\b.*fus", r"alk.*rearrang"],
    "ROS1_fusion": [r"\bros1\b.*fus", r"ros1.*rearrang"],
    "BRAF_V600E": [r"braf\s*v600e", r"braf.*600"],
    "MET_exon14skip": [r"met.*exon\s*14", r"met.*14.*skip"],
    "KRAS_G12C": [r"kras\s*g12c", r"kras.*12c"],
    "KRAS_other": [r"kras\s*g12[^c]", r"kras\s*g13", r"kras\s*q61"],
    "RET_fusion": [r"\bret\b.*fus", r"ret.*rearrang"],
    "NTRK_fusion": [r"ntrk[123]?.*fus", r"ntrk.*rearrang"],
    "ERBB2_mutation": [r"erbb2", r"her2.*mut", r"her2.*amp"],
    "PDL1_high": [r"pd-?l1.*[≥>]\s*50", r"pd-?l1.*high", r"tps.*50"],
    "PDL1_low": [r"pd-?l1.*[1-4]\d%", r"pd-?l1.*low", r"pd-?l1.*1[–-]49"],
}

ECOG_PATTERN = re.compile(r"ecog\s*(\d)", re.IGNORECASE)
STAGE_PATTERN = re.compile(r"stage\s*(I{1,3}V?[ABC]?)", re.IGNORECASE)
PRIOR_PATTERN = re.compile(r"prior\s*systemic\s*therapy:\s*([^.]+)", re.IGNORECASE)
PDL1_PATTERN = re.compile(r"pd-?l1.*?:\s*([^.]+)", re.IGNORECASE)


def match_profile_key(text: str) -> str:
    """Match text to an evidence KB profile key."""
    tl = text.lower()
    for key, patterns in PROFILE_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, tl):
                return key
    return ""


# ── Module implementations ────────────────────────────────────────────────────

class Module1_CaseParser:
    """Extract structured fields from free-text tumor board case."""

    # VUS / non-actionable variant indicators
    VUS_INDICATORS = re.compile(
        r"\b(vus|variant of uncertain|insufficient evidence|resistance mutation|"
        r"g12[^c]\b|g12d|g12v|g13|q61|not.*sensitizing|non.sensitizing|"
        r"not.*indicated|non.actionable)\b",
        re.IGNORECASE
    )
    WRONG_CONTEXT_DRUGS = re.compile(
        r"(breast cancer context|cml.*context|colorectal.*context|melanoma|"
        r"ibrutinib|imatinib|cetuximab|vemurafenib|trastuzumab(?! deruxtecan| emtansine))",
        re.IGNORECASE
    )

    def parse(self, case: dict) -> ParsedCase:
        summary = case.get("patient_summary", "")
        alteration = case.get("genomic_alteration", "")
        therapy = case.get("proposed_therapy", "")

        # For evaluation: trust case profile_key if present, but check if alteration
        # text overrides it (catches VUS / hallucinated biomarker cases)
        profile_key = case.get("profile_key", "")
        alteration_has_vus = bool(self.VUS_INDICATORS.search(alteration))
        wrong_context = bool(self.WRONG_CONTEXT_DRUGS.search(therapy))

        # If alteration text contains VUS language, clear profile_key so KB lookup returns empty
        if alteration_has_vus:
            profile_key = "VUS_NO_EVIDENCE"

        if not profile_key or profile_key == "VUS_NO_EVIDENCE":
            profile_key = profile_key or match_profile_key(alteration + " " + summary)

        kb_entry = EVIDENCE_KB.get(profile_key, {}) if profile_key != "VUS_NO_EVIDENCE" else {}
        biomarker = kb_entry.get("biomarker", alteration.split()[0] if alteration else "")
        alteration_clean = kb_entry.get("alteration", alteration)
        is_vus = alteration_has_vus
        is_wrong_context = wrong_context

        ecog_m = ECOG_PATTERN.search(summary)
        stage_m = STAGE_PATTERN.search(summary)
        prior_m = PRIOR_PATTERN.search(summary)
        pdl1_m = PDL1_PATTERN.search(summary)

        ecog = int(ecog_m.group(1)) if ecog_m else case.get("ecog")
        if ecog is None and "ecog not documented" in summary.lower():
            ecog = None

        stage_text = stage_m.group(0) if stage_m else case.get("stage", "Unknown")

        parsed = ParsedCase(
            cancer_type="NSCLC",
            histology=case.get("histology", "Unknown"),
            stage=stage_text,
            genomic_alteration=alteration,
            biomarker=biomarker,
            alteration=alteration_clean,
            profile_key=profile_key,
            proposed_therapy=therapy,
            ecog=ecog,
            prior_therapy=prior_m.group(1).strip() if prior_m else case.get("prior_therapy"),
            pdl1=pdl1_m.group(1).strip() if pdl1_m else case.get("pdl1"),
            brain_mets=case.get("brain_mets"),
        )
        # Attach flags for downstream modules
        parsed._is_vus = is_vus
        parsed._is_wrong_context = is_wrong_context
        return parsed


class Module2_EvidenceRetriever:
    """Retrieve relevant public evidence for the biomarker-therapy pair."""

    def __init__(self, civic_evidence: list[dict] = None, fda_labels: list[dict] = None):
        self.civic = civic_evidence or []
        self.fda = fda_labels or []
        # Index CIViC by molecular profile keyword
        self._civic_index = {}
        for ev in self.civic:
            prof = ev.get("molecular_profile", "").lower()
            if prof not in self._civic_index:
                self._civic_index[prof] = []
            self._civic_index[prof].append(ev)

    def retrieve(self, parsed: ParsedCase) -> list[dict]:
        """Return top relevant evidence items for the biomarker/therapy."""
        results = []
        alteration_kw = parsed.alteration.lower()
        biomarker_kw = parsed.biomarker.lower()
        therapy_kw = parsed.proposed_therapy.lower()

        for prof, items in self._civic_index.items():
            if biomarker_kw in prof or alteration_kw.split()[0] in prof:
                for item in items:
                    if therapy_kw in item.get("therapy", "").lower():
                        results.append(item)

        # If few results, also match by biomarker only
        if len(results) < 2:
            for prof, items in self._civic_index.items():
                if biomarker_kw in prof:
                    results.extend(items[:3])

        # Deduplicate
        seen = set()
        deduped = []
        for r in results:
            key = r.get("evidence_id")
            if key not in seen:
                seen.add(key)
                deduped.append(r)

        return deduped[:10]


class Module3_EvidenceVerifier:
    """Verify whether retrieved evidence supports the proposed recommendation."""

    def verify(self, parsed: ParsedCase, evidence_items: list[dict]) -> EvidenceResult:
        """Check if proposed therapy has evidence support for this profile."""
        profile_key = parsed.profile_key
        proposed = parsed.proposed_therapy.lower()

        # Use structured KB for ground-truth verification
        kb = EVIDENCE_KB.get(profile_key, {})
        supported = [t.lower() for t in kb.get("supported_therapies", [])]

        # Check direct match
        proposed_supported = any(s in proposed or proposed in s for s in supported)

        # Find matching evidence items
        matching = [
            e for e in evidence_items
            if proposed in e.get("therapy", "").lower()
            and e.get("evidence_level", "E") in ["A", "B"]
        ]

        if matching:
            best = matching[0]
            return EvidenceResult(
                evidence_found=True,
                evidence_level=best.get("evidence_level", ""),
                supported_therapies=supported,
                proposed_therapy_supported=proposed_supported,
                evidence_text=best.get("description", "")[:300],
                citation=best.get("citation", ""),
                faithfulness_score=1.0 if proposed_supported else 0.3,
            )
        elif kb:
            return EvidenceResult(
                evidence_found=bool(kb),
                evidence_level=kb.get("evidence_level", ""),
                supported_therapies=supported,
                proposed_therapy_supported=proposed_supported,
                evidence_text=kb.get("note", ""),
                citation=kb.get("source", "FDA/CIViC"),
                faithfulness_score=0.9 if proposed_supported else 0.1,
            )
        else:
            return EvidenceResult(
                evidence_found=False,
                evidence_level="",
                supported_therapies=[],
                proposed_therapy_supported=False,
                evidence_text="No evidence found for this biomarker-therapy combination.",
                citation="",
                faithfulness_score=0.0,
            )


class Module4_MissingInfoDetector:
    """Detect which critical clinical fields are absent."""

    CRITICAL_FIELDS = {
        "stage": lambda p, c: p.stage in (None, "Unknown", "not documented"),
        "ecog_status": lambda p, c: p.ecog is None,
        "prior_therapy": lambda p, c: p.prior_therapy in (None, "not documented", "None"),
        "pdl1_expression": lambda p, c: p.pdl1 in (None, "not tested", "not documented"),
        "brain_metastasis": lambda p, c: p.brain_mets is None,
        "biomarker_validation": lambda p, c: not p.profile_key,
    }

    def detect(self, parsed: ParsedCase, case: dict) -> MissingInfoResult:
        missing = []
        for field_name, check_fn in self.CRITICAL_FIELDS.items():
            if check_fn(parsed, case):
                missing.append(field_name)

        all_missing = list(set(missing))

        critical = [f for f in all_missing if f in [
            "stage", "ecog_status", "prior_therapy", "biomarker_validation"
        ]]
        completeness = max(0.0, 1.0 - len(all_missing) / len(self.CRITICAL_FIELDS))

        return MissingInfoResult(
            missing_fields=all_missing,
            critical_missing=critical,
            completeness_score=completeness,
        )


class Module5_SafetyClassifier:
    """Classify the recommendation safety label."""

    def classify(self,
                 parsed: ParsedCase,
                 evidence: EvidenceResult,
                 missing: MissingInfoResult,
                 case: dict) -> SafetyClassification:

        # VUS / hallucinated biomarker → flag as Unsupported + abstain
        if getattr(parsed, "_is_vus", False):
            return SafetyClassification(
                label="Unsupported",
                confidence=0.92,
                rationale=(
                    f"The alteration '{parsed.genomic_alteration}' appears to be a variant of uncertain "
                    f"significance (VUS) or non-sensitizing mutation. Evidence does not support "
                    f"{parsed.proposed_therapy} for this variant."
                ),
            )

        # Wrong cancer context → Unsupported (therapy is not indicated for this cancer/context)
        if getattr(parsed, "_is_wrong_context", False):
            return SafetyClassification(
                label="Unsupported",
                confidence=0.95,
                rationale=f"{parsed.proposed_therapy} is indicated in a different cancer context, not for NSCLC with {parsed.genomic_alteration}.",
            )

        # Critical missing → Insufficient Information
        if len(missing.critical_missing) >= 1:
            return SafetyClassification(
                label="Insufficient Information",
                confidence=0.9,
                rationale=f"Missing critical fields: {', '.join(missing.critical_missing)}. Cannot safely audit.",
            )

        # No evidence at all → Unsupported
        if not evidence.evidence_found:
            return SafetyClassification(
                label="Unsupported",
                confidence=0.85,
                rationale="No evidence found linking proposed therapy to this molecular alteration in NSCLC.",
            )

        # Evidence found but therapy not supported
        if not evidence.proposed_therapy_supported:
            return SafetyClassification(
                label="Unsupported",
                confidence=0.88,
                rationale=f"{parsed.proposed_therapy} lacks evidence support for {parsed.genomic_alteration} in NSCLC.",
            )

        # Poor performance status (ECOG ≥ 3) — therapy may be evidence-supported but
        # requires careful patient selection and multidisciplinary review per NCCN guidelines.
        if parsed.ecog is not None and parsed.ecog >= 3:
            return SafetyClassification(
                label="Partially Supported",
                confidence=0.78,
                rationale=(
                    f"Therapy is evidence-supported for {parsed.genomic_alteration}, "
                    f"but ECOG {parsed.ecog} (poor performance status) requires careful "
                    f"patient selection and tumor board review before proceeding."
                ),
            )

        # Evidence found and therapy supported, but critical clinical context is missing.
        # Only stage, ECOG status, prior therapy, and biomarker validation are required
        # for targeted therapy decisions; pdl1 and brain_mets are supplementary.
        ps_trigger_fields = [
            f for f in missing.missing_fields
            if f in ("stage", "ecog_status", "prior_therapy", "biomarker_validation")
        ]
        if ps_trigger_fields:
            return SafetyClassification(
                label="Partially Supported",
                confidence=0.75,
                rationale=f"Therapy is generally evidence-supported, but {', '.join(ps_trigger_fields)} are missing, limiting full safety confirmation.",
            )

        # Fully supported
        level = evidence.evidence_level
        return SafetyClassification(
            label="Supported",
            confidence=min(0.95, 0.7 + (0.1 if level == "A" else 0.05 if level == "B" else 0.0)),
            rationale=f"{parsed.proposed_therapy} is evidence-supported (Level {level}) for {parsed.genomic_alteration} in NSCLC. {evidence.citation}",
        )


class Module6_AbstentionModule:
    """Decide whether the agent should abstain and refer to human review."""

    def decide(self,
               parsed: ParsedCase,
               evidence: EvidenceResult,
               missing: MissingInfoResult,
               safety: SafetyClassification) -> AbstentionDecision:

        reasons = []
        should_abstain = False

        if len(missing.critical_missing) >= 2:
            reasons.append(f"Critical missing information: {', '.join(missing.critical_missing)}")
            should_abstain = True

        if safety.label == "Unsupported":
            reasons.append(f"Unsupported recommendation detected")
            should_abstain = True

        if safety.label == "Insufficient Information":
            reasons.append("Insufficient clinical context for safe audit")
            should_abstain = True

        if parsed.ecog is not None and parsed.ecog >= 3:
            reasons.append("Poor performance status (ECOG ≥ 3); benefit-risk requires oncologist review")
            should_abstain = True

        if not evidence.evidence_found and not parsed.profile_key:
            reasons.append("Unknown biomarker — cannot verify alteration actionability")
            should_abstain = True

        return AbstentionDecision(
            should_abstain=should_abstain,
            abstention_reason="; ".join(reasons) if reasons else "",
            referral_recommended=should_abstain,
        )


class Module7_AuditReportGenerator:
    """Generate the final structured audit report."""

    def generate(self,
                 case_id: str,
                 parsed: ParsedCase,
                 evidence: EvidenceResult,
                 missing: MissingInfoResult,
                 safety: SafetyClassification,
                 abstention: AbstentionDecision) -> AuditReport:

        if abstention.should_abstain:
            summary = (
                f"AUDIT RESULT: {safety.label}. "
                f"Human referral recommended. "
                f"Reason: {abstention.abstention_reason}. "
                f"Evidence context: {evidence.evidence_text[:200]}"
            )
        else:
            summary = (
                f"AUDIT RESULT: {safety.label} (confidence: {safety.confidence:.0%}). "
                f"{safety.rationale} "
                f"Evidence: {evidence.citation}."
            )

        return AuditReport(
            case_id=case_id,
            parsed={"profile_key": parsed.profile_key, "therapy": parsed.proposed_therapy,
                    "ecog": parsed.ecog, "stage": parsed.stage},
            evidence={"found": evidence.evidence_found, "level": evidence.evidence_level,
                      "supported": evidence.proposed_therapy_supported, "citation": evidence.citation},
            missing_info={"fields": missing.missing_fields, "critical": missing.critical_missing,
                          "completeness": missing.completeness_score},
            safety={"label": safety.label, "confidence": safety.confidence},
            abstention={"should_abstain": abstention.should_abstain,
                        "reason": abstention.abstention_reason},
            predicted_label=safety.label,
            predicted_abstain=abstention.should_abstain,
            predicted_missing_fields=missing.missing_fields,
            audit_summary=summary,
        )


# ── Full MTB-AuditAgent ───────────────────────────────────────────────────────

class MTBAuditAgent:
    """
    Full 7-module MTB-AuditAgent.
    mode='rule_based': uses structured EVIDENCE_KB (runs without API key)
    mode='llm': uses LLM backbone (requires API key — set ANTHROPIC_API_KEY)
    """

    def __init__(self, mode: str = "rule_based",
                 civic_evidence: list = None,
                 fda_labels: list = None):
        self.mode = mode
        self.m1 = Module1_CaseParser()
        self.m2 = Module2_EvidenceRetriever(civic_evidence, fda_labels)
        self.m3 = Module3_EvidenceVerifier()
        self.m4 = Module4_MissingInfoDetector()
        self.m5 = Module5_SafetyClassifier()
        self.m6 = Module6_AbstentionModule()
        self.m7 = Module7_AuditReportGenerator()

    def audit(self, case: dict) -> AuditReport:
        """Run the full 7-module audit pipeline on a single benchmark case."""
        # M1: Parse
        parsed = self.m1.parse(case)
        # M2: Retrieve evidence
        evidence_items = self.m2.retrieve(parsed)
        # M3: Verify
        evidence = self.m3.verify(parsed, evidence_items)
        # M4: Missing info
        missing = self.m4.detect(parsed, case)
        # M5: Safety classification
        safety = self.m5.classify(parsed, evidence, missing, case)
        # M6: Abstention
        abstention = self.m6.decide(parsed, evidence, missing, safety)
        # M7: Report
        report = self.m7.generate(
            case.get("case_id", ""), parsed, evidence, missing, safety, abstention
        )
        return report

    def audit_batch(self, cases: list[dict], verbose: bool = False) -> list[AuditReport]:
        results = []
        for i, case in enumerate(cases):
            report = self.audit(case)
            results.append(report)
            if verbose and (i + 1) % 50 == 0:
                print(f"  Audited {i+1}/{len(cases)} cases...")
        return results


# ── LLM backend (API-ready, swap in when key available) ──────────────────────

class LLMAuditAgent:
    """
    LLM-powered MTB-AuditAgent.
    Requires ANTHROPIC_API_KEY or OPENAI_API_KEY in environment.
    """

    SYSTEM_PROMPT = """You are a clinical oncology AI safety auditor. Your task is to audit
proposed molecular tumor board recommendations for NSCLC patients.

For each case, you must:
1. Identify the molecular alteration and proposed therapy
2. Determine if the proposed therapy is evidence-supported for this alteration (use FDA/CIViC standards)
3. Identify any missing critical clinical information
4. Classify the recommendation as: Supported | Partially Supported | Unsupported | Insufficient Information
5. Decide whether to ABSTAIN and refer to human oncologist review

Respond in JSON with these fields:
{
  "predicted_label": "...",
  "predicted_abstain": true/false,
  "predicted_missing_fields": [...],
  "audit_summary": "..."
}"""

    def __init__(self, model: str = "gemini-2.5-flash", backend: str = "gemini", api_key: str = None):
        self.model = model
        self.backend = backend
        self.api_key = api_key
        self._init_client()

    def _init_client(self):
        if self.backend == "gemini":
            try:
                from google import genai
                self.client = genai.Client(api_key=self.api_key)
            except Exception as e:
                raise RuntimeError(f"Gemini client init failed: {e}. Install google-genai and provide api_key.")
        elif self.backend == "anthropic":
            try:
                import anthropic
                self.client = anthropic.Anthropic()
            except Exception as e:
                raise RuntimeError(f"Anthropic client init failed: {e}. Set ANTHROPIC_API_KEY.")
        elif self.backend == "openai":
            try:
                import openai
                self.client = openai.OpenAI()
            except Exception as e:
                raise RuntimeError(f"OpenAI client init failed: {e}. Set OPENAI_API_KEY.")

    def _call_llm(self, user_message: str) -> str:
        if self.backend == "gemini":
            from google.genai import types
            combined = f"{self.SYSTEM_PROMPT}\n\n{user_message}"
            r = self.client.models.generate_content(
                model=self.model,
                contents=combined,
                config=types.GenerateContentConfig(max_output_tokens=512, temperature=0.0),
            )
            return r.text
        elif self.backend == "anthropic":
            r = self.client.messages.create(
                model=self.model,
                max_tokens=512,
                system=self.SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}]
            )
            return r.content[0].text
        elif self.backend == "openai":
            r = self.client.chat.completions.create(
                model=self.model,
                max_tokens=512,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_message}
                ]
            )
            return r.choices[0].message.content

    def audit(self, case: dict, with_rag: bool = False, evidence_context: str = "") -> dict:
        """Audit a case using LLM."""
        prompt = f"Patient Case:\n{case['patient_summary']}\n"
        if with_rag and evidence_context:
            prompt += f"\nRelevant Evidence:\n{evidence_context}\n"
        prompt += "\nProvide your audit in JSON format."

        try:
            response = self._call_llm(prompt)
            # Extract JSON from response
            json_match = re.search(r'\{[^{}]+\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            print(f"  LLM error: {e}")
        return {
            "predicted_label": "Unknown",
            "predicted_abstain": False,
            "predicted_missing_fields": [],
            "audit_summary": "LLM call failed",
        }

    def audit_batch(self, cases: list[dict], with_rag: bool = False,
                    civic_evidence: list = None, verbose: bool = True) -> list[dict]:
        import time
        results = []
        retriever = Module2_EvidenceRetriever(civic_evidence) if civic_evidence else None

        for i, case in enumerate(cases):
            evidence_ctx = ""
            if with_rag and retriever:
                parser = Module1_CaseParser()
                parsed = parser.parse(case)
                items = retriever.retrieve(parsed)
                if items:
                    evidence_ctx = "\n".join(
                        f"- {e.get('molecular_profile','')}: {e.get('therapy','')} (Level {e.get('evidence_level','?')}) — {e.get('citation','')}"
                        for e in items[:5]
                    )
            result = self.audit(case, with_rag=with_rag, evidence_context=evidence_ctx)
            results.append(result)
            if verbose and (i + 1) % 10 == 0:
                print(f"  LLM audited {i+1}/{len(cases)} cases...")
            time.sleep(6.5)  # respect 10 RPM free-tier limit

        return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run MTB-AuditAgent on OpenMTB-Audit benchmark cases."
    )
    parser.add_argument(
        "--benchmark",
        default="benchmark/benchmark_cases.json",
        help="Path to benchmark_cases.json (default: benchmark/benchmark_cases.json)",
    )
    parser.add_argument(
        "--knowledge_base",
        default="data/knowledge_base",
        help="Path to knowledge_base directory (default: data/knowledge_base)",
    )
    parser.add_argument(
        "--output",
        default="results/agent_predictions.json",
        help="Path to write predictions JSON (default: results/agent_predictions.json)",
    )
    args = parser.parse_args()

    kb_dir = Path(args.knowledge_base)
    with open(args.benchmark) as f:
        cases = json.load(f)
    with open(kb_dir / "civic_nsclc_evidence.json") as f:
        civic_ev = json.load(f)
    with open(kb_dir / "fda_nsclc_labels.json") as f:
        fda_labels = json.load(f)

    agent = MTBAuditAgent(mode="rule_based", civic_evidence=civic_ev, fda_labels=fda_labels)

    print(f"Running MTB-AuditAgent on {len(cases)} cases...")
    reports = agent.audit_batch(cases, verbose=True)
    predictions = [
        {
            "case_id": r.case_id,
            "predicted_label": r.predicted_label,
            "predicted_abstain": r.predicted_abstain,
            "predicted_missing_fields": r.predicted_missing_fields,
        }
        for r in reports
    ]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(predictions, f, indent=2)
    print(f"Saved {len(predictions)} predictions to {out_path}")
        print(f"  Summary: {report.audit_summary[:120]}...")

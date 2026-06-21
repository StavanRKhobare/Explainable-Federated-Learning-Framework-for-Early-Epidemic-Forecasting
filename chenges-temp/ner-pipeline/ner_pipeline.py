"""
ner_pipeline.py
===============
Core NER extraction engine using scispaCy.

CRITICAL DESIGN NOTE:
  en_core_sci_lg produces a SINGLE label "ENTITY" for all detected spans.
  It does NOT differentiate between symptoms, diseases, and pathogens.
  
  This module adds a rule-based classification layer on top:
    1. scispaCy detects biomedical entity spans
    2. We classify each span into SYMPTOM / DISEASE / PATHOGEN / TRAVEL
       using curated keyword dictionaries
    3. Unclassified entities are labeled as OTHER (and ignored)

Usage:
    from ner_pipeline import NERExtractor
    extractor = NERExtractor()
    counts = extractor.extract("Patient presented with fever and joint pain. Suspected dengue.")
    # → Counter({'SYMPTOM': 2, 'DISEASE': 1})
"""

import spacy
from collections import Counter
import re
import sys

# Avoid unicode print errors on Windows terminal
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')


# ── Curated Classification Dictionaries ───────────────────────────────────────
# These are matched against the lowercase text of each entity span.
# A span is classified into the FIRST matching category.

SYMPTOM_KEYWORDS = {
    # Classic dengue symptoms
    "fever", "high fever", "febrile", "pyrexia",
    "headache", "severe headache", "cephalalgia",
    "retro-orbital pain", "pain behind the eyes", "retro-orbital",
    "joint pain", "arthralgia", "polyarthralgia",
    "muscle pain", "myalgia", "body aches", "body pain",
    "rash", "skin rash", "maculopapular rash", "exanthem", "macular rash",
    "nausea", "vomiting", "emesis",
    "bleeding", "mild bleeding", "hemorrhage", "haemorrhage",
    "petechiae", "purpura", "ecchymosis",
    "fatigue", "malaise", "weakness", "lethargy",
    "loss of appetite", "anorexia",
    "abdominal pain", "epigastric pain",
    "diarrhea", "diarrhoea",
    "thrombocytopenia", "low platelet", "leukopenia",
    "plasma leakage", "pleural effusion", "ascites",
    "hepatomegaly", "splenomegaly",
    "shock", "hypotension",
}

DISEASE_KEYWORDS = {
    "dengue", "dengue fever", "severe dengue",
    "dengue hemorrhagic fever", "dhf",
    "dengue shock syndrome", "dss",
    "malaria", "chikungunya", "zika",
    "viral fever", "typhoid", "leptospirosis",
    "yellow fever", "west nile",
}

PATHOGEN_KEYWORDS = {
    "dengue virus", "denv", "denv-1", "denv-2", "denv-3", "denv-4",
    "orthoflavivirus", "orthoflavivirus denguei",
    "flavivirus", "arbovirus",
    "aedes", "aedes aegypti", "aedes albopictus",
    "plasmodium", "salmonella",
}

# Travel patterns — these are detected via regex, not scispaCy
TRAVEL_PATTERNS = [
    r"\btrave?le?l(?:ed|led|ling)?\b",
    r"\breturned?\s+from\b",
    r"\bvisited?\b",
    r"\bcame?\s+from\b",
    r"\brecent\s+travel\b",
    r"\btravel\s+history\b",
    r"\bendemic\s+zone\b",
]


class NERExtractor:
    """
    Extracts and classifies biomedical entities from clinical notes.
    
    Returns a Counter with keys: SYMPTOM, DISEASE, PATHOGEN, TRAVEL
    """

    def __init__(self, model_name="en_core_sci_lg"):
        """
        Load the scispaCy model. Falls back to en_core_sci_sm if lg is not installed.
        """
        try:
            self.nlp = spacy.load(model_name)
            print(f"✅ Loaded scispaCy model: {model_name}")
        except OSError:
            # Try smaller model as fallback
            fallback = "en_core_sci_sm"
            try:
                self.nlp = spacy.load(fallback)
                print(f"⚠️  {model_name} not found, using fallback: {fallback}")
            except OSError:
                raise RuntimeError(
                    f"Neither {model_name} nor {fallback} is installed.\n"
                    f"Install with:\n"
                    f"  pip install scispacy\n"
                    f"  pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz"
                )

        # Compile travel regex patterns
        self._travel_re = re.compile(
            "|".join(TRAVEL_PATTERNS), re.IGNORECASE
        )

    def _classify_entity(self, span_text: str) -> str:
        """
        Classify a scispaCy entity span into one of our target categories.
        Returns: 'SYMPTOM', 'DISEASE', 'PATHOGEN', or 'OTHER'
        """
        text_lower = span_text.lower().strip()

        # Check using regex word boundaries to avoid partial matches
        # Order matters: more specific categories first

        # Pathogen check (most specific)
        for keyword in PATHOGEN_KEYWORDS:
            if re.search(r'\b' + re.escape(keyword) + r'\b', text_lower):
                return "PATHOGEN"

        # Disease check
        for keyword in DISEASE_KEYWORDS:
            if re.search(r'\b' + re.escape(keyword) + r'\b', text_lower):
                return "DISEASE"

        # Symptom check
        for keyword in SYMPTOM_KEYWORDS:
            if re.search(r'\b' + re.escape(keyword) + r'\b', text_lower):
                return "SYMPTOM"

        return "OTHER"

    def _count_travel(self, text: str) -> int:
        """Count travel-related mentions using regex patterns."""
        return len(self._travel_re.findall(text))

    def extract(self, text: str) -> Counter:
        """
        Process a single clinical note and return entity counts.
        
        Args:
            text: Raw clinical note text
            
        Returns:
            Counter with keys SYMPTOM, DISEASE, PATHOGEN, TRAVEL
        """
        counter = Counter()

        # Step 1: Run scispaCy NER
        doc = self.nlp(text)

        # Step 2: Classify each detected entity
        for ent in doc.ents:
            category = self._classify_entity(ent.text)
            if category != "OTHER":
                counter[category] += 1

        # Step 3: Count travel mentions via regex
        travel_count = self._count_travel(text)
        if travel_count > 0:
            counter["TRAVEL"] = travel_count

        return counter

    def extract_detailed(self, text: str) -> dict:
        """
        Process a note and return detailed entity information (for debugging).
        
        Returns:
            dict with 'counts' (Counter) and 'entities' (list of dicts)
        """
        doc = self.nlp(text)
        entities = []
        counter = Counter()

        for ent in doc.ents:
            category = self._classify_entity(ent.text)
            entities.append({
                "text": ent.text,
                "label_raw": ent.label_,   # Will be "ENTITY" for en_core_sci_lg
                "category": category,
                "start": ent.start_char,
                "end": ent.end_char,
            })
            if category != "OTHER":
                counter[category] += 1

        travel_count = self._count_travel(text)
        if travel_count > 0:
            counter["TRAVEL"] = travel_count

        return {"counts": counter, "entities": entities}


# ── Quick Test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Initializing NER Extractor...")
    extractor = NERExtractor()

    test_notes = [
        "Patient presented with fever, joint pain, and a rash. Suspected dengue fever. "
        "NS1 antigen test positive. Patient traveled to a neighboring district last week.",

        "A 45-year-old male reported with high fever, severe headache, and retro-orbital pain. "
        "Platelet count: 45000/µL (low). Suspected dengue hemorrhagic fever. "
        "Possible exposure to Dengue virus DENV-2. No recent travel history.",

        "Chief complaint: nausea, vomiting, abdominal pain. "
        "Patient returned from a rural village 5 days ago. Suspected malaria.",

        "Walk-in patient presenting with fatigue and mild bleeding. "
        "No travel outside the district. Suspected viral fever.",
    ]

    print("\n" + "=" * 70)
    print("NER EXTRACTION TEST RESULTS")
    print("=" * 70)

    for i, note in enumerate(test_notes, 1):
        print(f"\n--- Note {i} ---")
        print(f"Text: {note[:100]}...")
        result = extractor.extract_detailed(note)
        print(f"Counts: {dict(result['counts'])}")
        for ent in result["entities"]:
            if ent["category"] != "OTHER":
                print(f"  [{ent['category']}] \"{ent['text']}\"")
        print()

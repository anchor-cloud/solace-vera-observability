"""trust_registry.py

Reference configuration of authoritative knowledge sources, organized by domain.

This is a STATIC reference config — not a live query system. It catalogs
trustworthy sources the pipeline may later consult to ground or cross-check
claims (e.g., during Phase 5 reflection or future evidence-gathering phases).

No external dependencies — standard library only.

Schema (each entry is a dict):
    name            : str   - human-readable source name
    base_url        : str   - canonical website / portal
    api_endpoint    : str | None - machine-queryable endpoint, or None if the
                                   source is browse-only (no public API)
    access_type     : str   - one of ACCESS_TYPES:
                                "free API"      - open programmatic access, no key
                                "free browse"   - human-readable, no public API
                                "free with key" - programmatic access, free key required
    coverage        : str   - short description of what the source covers
    confidence_tier : int   - one of CONFIDENCE_TIERS:
                                1 = highest authority (primary / standards body /
                                    government / canonical dataset)
                                2 = high authority (curated secondary / scholarly
                                    aggregator / reference work)

Use the helper functions at the bottom for simple lookups; everything is plain
dicts/lists so it can be imported and iterated directly.
"""

from __future__ import annotations

# Controlled vocabularies (for validation / documentation).
ACCESS_TYPES = ("free API", "free browse", "free with key")
CONFIDENCE_TIERS = (1, 2)
CATEGORIES = ("science", "law", "history", "mathematics")


TRUST_REGISTRY = {
    # ------------------------------------------------------------------ #
    "science": [
        {
            "name": "PubMed (NCBI E-utilities)",
            "base_url": "https://pubmed.ncbi.nlm.nih.gov/",
            "api_endpoint": "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/",
            "access_type": "free API",
            "coverage": "Biomedical and life-sciences literature; >35M citations and abstracts.",
            "confidence_tier": 1,
        },
        {
            "name": "arXiv",
            "base_url": "https://arxiv.org/",
            "api_endpoint": "http://export.arxiv.org/api/query",
            "access_type": "free API",
            "coverage": "Preprints in physics, astronomy, CS, quantitative biology, statistics.",
            "confidence_tier": 2,
        },
        {
            "name": "Crossref",
            "base_url": "https://www.crossref.org/",
            "api_endpoint": "https://api.crossref.org/works",
            "access_type": "free API",
            "coverage": "Scholarly metadata and DOIs across publishers and disciplines.",
            "confidence_tier": 1,
        },
        {
            "name": "NASA Astrophysics Data System (ADS)",
            "base_url": "https://ui.adsabs.harvard.edu/",
            "api_endpoint": "https://api.adsabs.harvard.edu/v1/",
            "access_type": "free with key",
            "coverage": "Astronomy, astrophysics, and physics literature and citations.",
            "confidence_tier": 1,
        },
        {
            "name": "NIST (National Institute of Standards and Technology)",
            "base_url": "https://www.nist.gov/",
            "api_endpoint": None,
            "access_type": "free browse",
            "coverage": "Physical constants, measurement standards, reference chemistry/physics data.",
            "confidence_tier": 1,
        },
        {
            "name": "Semantic Scholar",
            "base_url": "https://www.semanticscholar.org/",
            "api_endpoint": "https://api.semanticscholar.org/graph/v1",
            "access_type": "free API",
            "coverage": "Cross-disciplinary scholarly papers, citations, and AI-extracted metadata.",
            "confidence_tier": 2,
        },
        {
            "name": "NOAA / National Centers for Environmental Information",
            "base_url": "https://www.ncei.noaa.gov/",
            "api_endpoint": "https://www.ncdc.noaa.gov/cdo-web/api/v2/",
            "access_type": "free with key",
            "coverage": "Climate, weather, and environmental observational datasets.",
            "confidence_tier": 1,
        },
    ],
    # ------------------------------------------------------------------ #
    "law": [
        {
            "name": "CourtListener (Free Law Project)",
            "base_url": "https://www.courtlistener.com/",
            "api_endpoint": "https://www.courtlistener.com/api/rest/v4/",
            "access_type": "free with key",
            "coverage": "U.S. case law, dockets, judges, and oral arguments.",
            "confidence_tier": 1,
        },
        {
            "name": "Cornell Legal Information Institute (LII)",
            "base_url": "https://www.law.cornell.edu/",
            "api_endpoint": None,
            "access_type": "free browse",
            "coverage": "U.S. Code, CFR, Supreme Court opinions, and legal reference.",
            "confidence_tier": 1,
        },
        {
            "name": "GovInfo (U.S. Government Publishing Office)",
            "base_url": "https://www.govinfo.gov/",
            "api_endpoint": "https://api.govinfo.gov/",
            "access_type": "free with key",
            "coverage": "Authenticated U.S. federal documents: statutes, bills, CFR, court opinions.",
            "confidence_tier": 1,
        },
        {
            "name": "Federal Register",
            "base_url": "https://www.federalregister.gov/",
            "api_endpoint": "https://www.federalregister.gov/api/v1/",
            "access_type": "free API",
            "coverage": "U.S. federal agency rules, proposed rules, and notices.",
            "confidence_tier": 1,
        },
        {
            "name": "EUR-Lex",
            "base_url": "https://eur-lex.europa.eu/",
            "api_endpoint": None,
            "access_type": "free browse",
            "coverage": "European Union law, treaties, case law, and legislative procedures.",
            "confidence_tier": 1,
        },
        {
            "name": "legislation.gov.uk",
            "base_url": "https://www.legislation.gov.uk/",
            "api_endpoint": "https://www.legislation.gov.uk/data.feed",
            "access_type": "free API",
            "coverage": "UK primary and secondary legislation (current and historical).",
            "confidence_tier": 1,
        },
    ],
    # ------------------------------------------------------------------ #
    "history": [
        {
            "name": "Library of Congress",
            "base_url": "https://www.loc.gov/",
            "api_endpoint": "https://www.loc.gov/apis/json-and-yaml/",
            "access_type": "free API",
            "coverage": "Primary-source documents, manuscripts, maps, photographs, and recordings.",
            "confidence_tier": 1,
        },
        {
            "name": "U.S. National Archives (NARA Catalog)",
            "base_url": "https://www.archives.gov/",
            "api_endpoint": "https://catalog.archives.gov/api/v2/",
            "access_type": "free with key",
            "coverage": "U.S. federal records, founding documents, and archival holdings.",
            "confidence_tier": 1,
        },
        {
            "name": "Europeana",
            "base_url": "https://www.europeana.eu/",
            "api_endpoint": "https://api.europeana.eu/record/v2/",
            "access_type": "free with key",
            "coverage": "Digitized European cultural-heritage objects from thousands of institutions.",
            "confidence_tier": 2,
        },
        {
            "name": "Digital Public Library of America (DPLA)",
            "base_url": "https://dp.la/",
            "api_endpoint": "https://api.dp.la/v2/",
            "access_type": "free with key",
            "coverage": "Aggregated U.S. library, archive, and museum digital collections.",
            "confidence_tier": 2,
        },
        {
            "name": "Chronicling America (Library of Congress)",
            "base_url": "https://chroniclingamerica.loc.gov/",
            "api_endpoint": "https://chroniclingamerica.loc.gov/search/pages/results/?format=json",
            "access_type": "free API",
            "coverage": "Digitized historic U.S. newspapers (1690s–present) and newspaper directory.",
            "confidence_tier": 1,
        },
        {
            "name": "Internet Archive",
            "base_url": "https://archive.org/",
            "api_endpoint": "https://archive.org/advancedsearch.php",
            "access_type": "free API",
            "coverage": "Archived books, texts, audio, video, and web pages (Wayback Machine).",
            "confidence_tier": 2,
        },
        {
            "name": "Smithsonian Open Access",
            "base_url": "https://www.si.edu/openaccess",
            "api_endpoint": "https://api.si.edu/openaccess/api/v1.0/",
            "access_type": "free with key",
            "coverage": "Smithsonian museum collections, artifacts, and research metadata.",
            "confidence_tier": 1,
        },
    ],
    # ------------------------------------------------------------------ #
    "mathematics": [
        {
            "name": "On-Line Encyclopedia of Integer Sequences (OEIS)",
            "base_url": "https://oeis.org/",
            "api_endpoint": "https://oeis.org/search?fmt=json",
            "access_type": "free API",
            "coverage": "Integer sequences with definitions, references, and cross-links.",
            "confidence_tier": 1,
        },
        {
            "name": "NIST Digital Library of Mathematical Functions (DLMF)",
            "base_url": "https://dlmf.nist.gov/",
            "api_endpoint": None,
            "access_type": "free browse",
            "coverage": "Authoritative reference for special functions and mathematical formulas.",
            "confidence_tier": 1,
        },
        {
            "name": "zbMATH Open",
            "base_url": "https://zbmath.org/",
            "api_endpoint": "https://api.zbmath.org/v1/",
            "access_type": "free API",
            "coverage": "Reviews and bibliographic data for pure and applied mathematics.",
            "confidence_tier": 1,
        },
        {
            "name": "arXiv (math)",
            "base_url": "https://arxiv.org/archive/math",
            "api_endpoint": "http://export.arxiv.org/api/query",
            "access_type": "free API",
            "coverage": "Mathematics preprints across all major subfields.",
            "confidence_tier": 2,
        },
        {
            "name": "Wolfram MathWorld",
            "base_url": "https://mathworld.wolfram.com/",
            "api_endpoint": None,
            "access_type": "free browse",
            "coverage": "Encyclopedic reference articles across mathematics topics.",
            "confidence_tier": 2,
        },
    ],
}


# --------------------------------------------------------------------------- #
# Convenience accessors (stdlib only; purely read helpers over the dict above)
# --------------------------------------------------------------------------- #
def get_sources(category):
    """Return the list of source entries for a category (case-insensitive)."""
    return TRUST_REGISTRY.get((category or "").strip().lower(), [])


def all_sources():
    """Yield (category, entry) pairs for every source in the registry."""
    for category, entries in TRUST_REGISTRY.items():
        for entry in entries:
            yield category, entry


def sources_by_tier(tier):
    """Return [(category, entry), ...] for all sources at a confidence tier."""
    return [(c, e) for c, e in all_sources() if e["confidence_tier"] == tier]


def sources_with_api():
    """Return [(category, entry), ...] for sources exposing an API endpoint."""
    return [(c, e) for c, e in all_sources() if e["api_endpoint"]]


if __name__ == "__main__":
    total = sum(len(v) for v in TRUST_REGISTRY.values())
    print(f"Trust registry: {total} sources across {len(TRUST_REGISTRY)} categories\n")
    for category in TRUST_REGISTRY:
        entries = TRUST_REGISTRY[category]
        tier1 = sum(1 for e in entries if e["confidence_tier"] == 1)
        with_api = sum(1 for e in entries if e["api_endpoint"])
        print(f"{category.title():<13} {len(entries):>2} sources  "
              f"(tier-1: {tier1}, with API: {with_api})")
        for e in entries:
            api = "API" if e["api_endpoint"] else "browse"
            print(f"    [T{e['confidence_tier']}] {e['name']}  "
                  f"({e['access_type']}, {api})")
        print()

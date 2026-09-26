"""Simple relevance check using dense similarity and labour-law keywords."""

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from src.retrieval.retriever import RetrievedChunk


LABOUR_KEYWORDS = (
    "əmək",
    "əmək müqaviləsi",
    "işçi",
    "işəgötürən",
    "işdən çıxar",
    "xitam",
    "maaş",
    "əmək haqqı",
    "məzuniyyət",
    "ixtisar",
    "iş vaxtı",
    "əlavə iş",
)


def is_relevant(
    query: str,
    results: List["RetrievedChunk"],
    strong_threshold: float = 0.55,
    moderate_threshold: float = 0.45,
) -> bool:
    """Accept strong dense matches, or moderate matches with a labour keyword.

    Both default thresholds are provisional for this MVP and should later be
    calibrated using a larger evaluation dataset.
    """
    if not results:
        return False
    best_dense_score = max(result.dense_score for result in results)
    if best_dense_score >= strong_threshold:
        return True

    # Azerbaijani uppercase I and İ need their matching lowercase forms.
    normalized_query = query.replace("İ", "i").replace("I", "ı").lower()
    return best_dense_score >= moderate_threshold and any(
        keyword in normalized_query for keyword in LABOUR_KEYWORDS
    )

"""Dispute categories and slot schema for the case-builder dialogue.

Defines the four MVP dispute categories, the information slots needed
to draft a complaint (ərizə) for each, and whether each slot is
REQUIRED or OPTIONAL. Slots shared by several categories (contract
type, length of service, employment status) are defined once in
``SHARED_SLOTS``; each category then references them by name.

Slot order inside a category is the order in which the dialogue asks
about missing slots, so it doubles as the question priority.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List


class DisputeCategory(str, Enum):
    """The four MVP labour-dispute categories."""

    UNLAWFUL_DISMISSAL = "UNLAWFUL_DISMISSAL"
    UNPAID_WAGES = "UNPAID_WAGES"
    UNPAID_OVERTIME = "UNPAID_OVERTIME"
    DENIED_OR_UNPAID_LEAVE = "DENIED_OR_UNPAID_LEAVE"


class Requirement(str, Enum):
    """Whether a slot must be filled before drafting a complaint."""

    REQUIRED = "REQUIRED"
    OPTIONAL = "OPTIONAL"


@dataclass(frozen=True)
class SlotDef:
    """Definition of one information slot.

    Attributes:
        name: Machine name, used as the key in CaseState.
        description_en: English description (for LLM extraction).
        question_az: Azerbaijani question to ask the user.
        is_date: True for key date/period slots, so extraction can
            normalise them consistently.
    """

    name: str
    description_en: str
    question_az: str
    is_date: bool = False


def _defs(*slots: SlotDef) -> Dict[str, SlotDef]:
    """Builds a name -> SlotDef mapping from slot definitions."""
    return {slot.name: slot for slot in slots}


SHARED_SLOTS: Dict[str, SlotDef] = _defs(
    SlotDef(
        "contract_type",
        "Type of employment contract (fixed-term or indefinite)",
        "Əmək müqaviləniz müddətsiz idi, yoxsa müddətli?",
    ),
    SlotDef(
        "length_of_service",
        "Total length of service with this employer",
        "Bu işəgötürənin yanında nə qədər müddət işləmisiniz?",
    ),
    SlotDef(
        "employment_status",
        "Whether the person still works for the employer",
        "Hazırda bu işəgötürənin yanında işləyirsiniz, yoxsa "
        "işdən ayrılmısınız?",
    ),
)

CATEGORY_SLOT_DEFS: Dict[str, SlotDef] = _defs(
    # Dismissal
    SlotDef(
        "dismissal_date",
        "Date the person was dismissed",
        "İşdən hansı tarixdə çıxarılmısınız?",
        is_date=True,
    ),
    SlotDef(
        "stated_reason",
        "Reason the employer gave for the dismissal",
        "İşəgötürən işdən çıxarılmanızın səbəbi kimi nəyi göstərdi?",
    ),
    SlotDef(
        "written_explanation_requested",
        "Whether the person asked for a written explanation",
        "İşdən çıxarılma səbəbinin yazılı izahını tələb etmisinizmi?",
    ),
    SlotDef(
        "notice_given",
        "Whether and how early advance notice was given",
        "İşəgötürən sizi əvvəlcədən xəbərdar etdimi? Nə qədər əvvəl?",
    ),
    SlotDef(
        "severance_paid",
        "Whether severance pay was paid on dismissal",
        "İşdən çıxarılarkən müavinət (kompensasiya) ödənildimi?",
    ),
    SlotDef(
        "protected_category",
        "Special protected status (e.g. pregnancy, childcare, "
        "disability) at the time of dismissal",
        "Hamiləlik, uşaq baxımı, əlillik kimi xüsusi qorunan "
        "statusunuz var idimi?",
    ),
    SlotDef(
        "signed_documents",
        "Documents the person signed around the dismissal",
        "İşdən çıxarılma zamanı hansısa sənəd (məsələn, ərizə və ya "
        "razılaşma) imzalamısınızmı?",
    ),
    # Wages
    SlotDef(
        "last_paid_date",
        "Date the person last received wages",
        "Əmək haqqını sonuncu dəfə nə vaxt almısınız?",
        is_date=True,
    ),
    SlotDef(
        "unpaid_period",
        "Period for which pay is owed",
        "Hansı dövr üçün ödəniş edilməyib?",
        is_date=True,
    ),
    SlotDef(
        "amount_owed_approx",
        "Approximate amount owed",
        "Təxminən nə qədər məbləğ ödənilməyib?",
    ),
    SlotDef(
        "evidence_available",
        "Evidence the person has (payslips, bank records, messages)",
        "Bunu təsdiq edən sənəd və ya yazışmanız var (maaş vərəqi, "
        "bank çıxarışı, mesajlar)?",
    ),
    SlotDef(
        "severance_or_vacation_paid_on_exit",
        "Whether severance or unused-vacation pay was paid on exit",
        "İşdən ayrılarkən istifadə edilməmiş məzuniyyət və digər "
        "ödənişlər verildimi?",
    ),
    # Overtime
    SlotDef(
        "contracted_hours",
        "Working hours stated in the contract",
        "Əmək müqaviləsinə görə iş saatlarınız necədir?",
    ),
    SlotDef(
        "actual_hours_worked_pattern",
        "Hours actually worked (per day/week, weekends, holidays)",
        "Əslində nə qədər işləyirsiniz (həftədə neçə saat, "
        "istirahət və bayram günləri)?",
    ),
    SlotDef(
        "overtime_preapproved",
        "Whether the employer requested or approved the extra work",
        "Əlavə işi işəgötürən tələb etmiş və ya təsdiq etmişdi?",
    ),
    # Leave
    SlotDef(
        "leave_type",
        "Type of leave (annual, study, unpaid, ...)",
        "Hansı məzuniyyət növündən söhbət gedir (illik, təhsil, "
        "sosial və s.)?",
    ),
    SlotDef(
        "leave_request_outcome",
        "How the employer responded to the leave request",
        "Məzuniyyət ərizənizə işəgötürən necə cavab verdi?",
    ),
    SlotDef(
        "leave_dates",
        "Requested or taken leave dates",
        "Məzuniyyət üçün nəzərdə tutduğunuz tarixlər hansılardır?",
        is_date=True,
    ),
)

SLOT_DEFS: Dict[str, SlotDef] = {**SHARED_SLOTS, **CATEGORY_SLOT_DEFS}

REQ = Requirement.REQUIRED
OPT = Requirement.OPTIONAL

# Insertion order = question priority.
CATEGORY_SCHEMA: Dict[DisputeCategory, Dict[str, Requirement]] = {
    DisputeCategory.UNLAWFUL_DISMISSAL: {
        "contract_type": REQ,
        "dismissal_date": REQ,
        "stated_reason": REQ,
        "length_of_service": REQ,
        "signed_documents": REQ,
        "written_explanation_requested": OPT,
        "notice_given": OPT,
        "severance_paid": OPT,
        "protected_category": OPT,
    },
    DisputeCategory.UNPAID_WAGES: {
        "employment_status": REQ,
        "last_paid_date": REQ,
        "unpaid_period": REQ,
        "amount_owed_approx": REQ,
        "evidence_available": OPT,
        "severance_or_vacation_paid_on_exit": OPT,
    },
    DisputeCategory.UNPAID_OVERTIME: {
        "contracted_hours": REQ,
        "actual_hours_worked_pattern": REQ,
        "unpaid_period": REQ,
        "overtime_preapproved": OPT,
        "evidence_available": OPT,
    },
    DisputeCategory.DENIED_OR_UNPAID_LEAVE: {
        "leave_type": REQ,
        "length_of_service": REQ,
        "leave_request_outcome": REQ,
        "leave_dates": REQ,
        "employment_status": OPT,
    },
}

CATEGORY_LABELS: Dict[DisputeCategory, Dict[str, str]] = {
    DisputeCategory.UNLAWFUL_DISMISSAL: {
        "az": "Qanunsuz işdən çıxarılma",
        "en": "Unlawful dismissal",
    },
    DisputeCategory.UNPAID_WAGES: {
        "az": "Ödənilməmiş əmək haqqı",
        "en": "Unpaid wages",
    },
    DisputeCategory.UNPAID_OVERTIME: {
        "az": "Ödənilməmiş əlavə iş haqqı",
        "en": "Unpaid overtime",
    },
    DisputeCategory.DENIED_OR_UNPAID_LEAVE: {
        "az": "Verilməyən və ya ödənilməyən məzuniyyət",
        "en": "Denied or unpaid leave",
    },
}


def category_label(category: DisputeCategory, lang: str = "az") -> str:
    """Returns the human-readable name of a category.

    Args:
        category: The dispute category.
        lang: ``"az"`` or ``"en"``.

    Returns:
        The label in the requested language.

    Raises:
        KeyError: If ``lang`` is not ``"az"`` or ``"en"``.
    """
    return CATEGORY_LABELS[category][lang]


def get_slot_def(name: str) -> SlotDef:
    """Returns the definition of a slot by name.

    Args:
        name: Slot name.

    Returns:
        The matching SlotDef.

    Raises:
        KeyError: If the slot is not defined.
    """
    return SLOT_DEFS[name]


def slots_for(category: DisputeCategory) -> List[str]:
    """Returns all slot names of a category in question order."""
    return list(CATEGORY_SCHEMA[category])


def required_slots(category: DisputeCategory) -> List[str]:
    """Returns the REQUIRED slot names of a category in order."""
    return [
        name for name, req in CATEGORY_SCHEMA[category].items()
        if req is Requirement.REQUIRED
    ]


def optional_slots(category: DisputeCategory) -> List[str]:
    """Returns the OPTIONAL slot names of a category in order."""
    return [
        name for name, req in CATEGORY_SCHEMA[category].items()
        if req is Requirement.OPTIONAL
    ]

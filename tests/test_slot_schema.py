"""Tests for the Phase 6 slot schema."""

import unittest

from src.dialogue import slot_schema as ss
from src.dialogue.slot_schema import DisputeCategory as DC

SPEC = {
    DC.UNLAWFUL_DISMISSAL: {
        "contract_type", "dismissal_date", "stated_reason",
        "written_explanation_requested", "length_of_service",
        "notice_given", "severance_paid", "protected_category",
        "signed_documents",
    },
    DC.UNPAID_WAGES: {
        "employment_status", "last_paid_date", "unpaid_period",
        "amount_owed_approx", "evidence_available",
        "severance_or_vacation_paid_on_exit",
    },
    DC.UNPAID_OVERTIME: {
        "contracted_hours", "actual_hours_worked_pattern",
        "overtime_preapproved", "unpaid_period", "evidence_available",
    },
    DC.DENIED_OR_UNPAID_LEAVE: {
        "leave_type", "length_of_service", "leave_request_outcome",
        "leave_dates", "employment_status",
    },
}


class TestSlotSchema(unittest.TestCase):
    """Guards the schema against drifting from the project spec."""

    def test_slots_match_spec_exactly(self):
        for category, expected in SPEC.items():
            self.assertEqual(set(ss.slots_for(category)), expected)

    def test_every_slot_has_definition_and_question(self):
        for category in DC:
            for name in ss.slots_for(category):
                slot = ss.get_slot_def(name)
                self.assertEqual(slot.name, name)
                self.assertTrue(slot.question_az.endswith("?"))

    def test_required_and_optional_partition_each_category(self):
        for category in DC:
            req = ss.required_slots(category)
            opt = ss.optional_slots(category)
            self.assertTrue(req)
            self.assertFalse(set(req) & set(opt))
            self.assertEqual(
                set(req) | set(opt), set(ss.slots_for(category)))

    def test_shared_slots_are_defined_once(self):
        for name in ss.SHARED_SLOTS:
            self.assertNotIn(name, ss.CATEGORY_SLOT_DEFS)

    def test_labels_cover_all_categories_in_both_languages(self):
        for category in DC:
            self.assertTrue(ss.category_label(category, "az"))
            self.assertTrue(ss.category_label(category, "en"))


if __name__ == "__main__":
    unittest.main()

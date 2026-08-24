"""Offline regression tests for court-decision-extractor (no API / PDF / Poppler)."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import main
from main import (
    EXTRACTION_SCHEMA,
    RESPONSE_FORMAT,
    _normalize_legal_basis,
    _parse_date,
    _parse_number,
    analyze_image,
    merge_results,
    postprocess,
)


class MergeResultsTests(unittest.TestCase):
    def test_empty_merge(self):
        self.assertEqual(
            merge_results([]),
            {"summary": "", "key_findings": []},
        )

    def test_first_nonempty_metadata(self):
        merged = merge_results(
            [
                {"court_name": "", "case_number": None, "plaintiff": "А"},
                {"court_name": "Арбитражный суд", "case_number": "А40-1", "plaintiff": "Б"},
            ]
        )
        self.assertEqual(merged["court_name"], "Арбитражный суд")
        self.assertEqual(merged["case_number"], "А40-1")
        self.assertEqual(merged["plaintiff"], "А")

    def test_ruling_prefers_last_nonempty(self):
        merged = merge_results(
            [
                {"ruling": "частичное определение"},
                {"ruling": ""},
                {"ruling": "взыскать сумму в полном объёме"},
            ]
        )
        self.assertEqual(merged["ruling"], "взыскать сумму в полном объёме")

    def test_ruling_empty_later_does_not_erase_earlier(self):
        merged = merge_results(
            [
                {"ruling": "оставить без удовлетворения"},
                {"ruling": ""},
                {"ruling": None},
            ]
        )
        self.assertEqual(merged["ruling"], "оставить без удовлетворения")

    def test_summary_selects_longest_page_summary(self):
        merged = merge_results(
            [
                {"summary": "коротко"},
                {"summary": ""},
                {"summary": "гораздо более длинное резюме страницы"},
            ]
        )
        self.assertEqual(merged["summary"], "гораздо более длинное резюме страницы")

    def test_list_merging(self):
        merged = merge_results(
            [
                {
                    "people": [{"full_name": "Иванов", "role": "истец"}],
                    "legal_basis": ["ст. 1 ГК РФ"],
                    "key_findings": [{"finding": "a", "evidence": "e1"}],
                },
                {
                    "people": [{"full_name": "Петров", "role": "ответчик"}],
                    "legal_basis": ["ст. 2 АПК РФ"],
                    "key_findings": ["строковый вывод"],
                },
            ]
        )
        self.assertEqual(len(merged["people"]), 2)
        self.assertEqual(merged["legal_basis"], ["ст. 1 ГК РФ", "ст. 2 АПК РФ"])
        self.assertEqual(
            merged["key_findings"],
            [
                {"finding": "a", "evidence": "e1"},
                {"finding": "строковый вывод", "evidence": ""},
            ],
        )


class LegalBasisTests(unittest.TestCase):
    def test_explicit_code_preserved(self):
        self.assertEqual(
            _normalize_legal_basis("ст.395 ГК РФ"),
            "ст. 395 ГК РФ",
        )
        self.assertEqual(
            _normalize_legal_basis("ст. 333 АПК РФ"),
            "ст. 333 АПК РФ",
        )

    def test_bare_article_does_not_gain_code(self):
        self.assertEqual(_normalize_legal_basis("ст. 10"), "ст. 10")
        self.assertEqual(_normalize_legal_basis("ст.10"), "ст. 10")
        self.assertNotIn("ГК", _normalize_legal_basis("ст. 10") or "")
        self.assertNotIn("АПК", _normalize_legal_basis("ст. 10") or "")

    def test_legal_basis_deduplication(self):
        data = postprocess(
            {
                "legal_basis": ["ст. 10 ГК РФ", "ст.10 ГК РФ", "ст. 10"],
                "key_findings": [],
            }
        )
        self.assertEqual(data["legal_basis"], ["ст. 10 ГК РФ", "ст. 10"])


class PersonRoleTests(unittest.TestCase):
    def test_known_role_normalized(self):
        data = postprocess(
            {
                "people": [{"full_name": "Иванов Иван", "role": "Истец"}],
                "key_findings": [],
            }
        )
        self.assertEqual(data["people"][0]["role"], "истец")

    def test_unknown_role_preserved(self):
        data = postprocess(
            {
                "people": [{"full_name": "Сидоров", "role": "свидетель"}],
                "key_findings": [],
            }
        )
        self.assertEqual(data["people"][0]["role"], "свидетель")
        self.assertNotEqual(data["people"][0]["role"], "представитель")

    def test_empty_role_not_converted_to_representative(self):
        data = postprocess(
            {
                "people": [{"full_name": "Петров", "role": ""}],
                "key_findings": [],
            }
        )
        self.assertEqual(data["people"][0]["role"], "")
        self.assertNotEqual(data["people"][0]["role"], "представитель")

    def test_people_deduplication(self):
        data = postprocess(
            {
                "people": [
                    {"full_name": "Иванов", "role": "истец"},
                    {"full_name": "Иванов", "role": "Истец"},
                    {"full_name": "Иванов", "role": "ответчик"},
                ],
                "key_findings": [],
            }
        )
        self.assertEqual(len(data["people"]), 2)
        roles = {p["role"] for p in data["people"]}
        self.assertEqual(roles, {"истец", "ответчик"})


class NumberDateFindingsTests(unittest.TestCase):
    def test_number_normalization(self):
        self.assertEqual(_parse_number("1 000,5"), 1000.5)
        self.assertEqual(_parse_number("100"), 100)

    def test_bool_rejected_as_number(self):
        self.assertIsNone(_parse_number(True))
        self.assertIsNone(_parse_number(False))

    def test_date_normalization(self):
        self.assertEqual(_parse_date("28-02-2026"), "2026-02-28")
        self.assertEqual(_parse_date("28.02.2026"), "2026-02-28")
        self.assertEqual(_parse_date("2026-02-28"), "2026-02-28")

    def test_invalid_date_not_invented(self):
        self.assertIsNone(_parse_date("not-a-date"))
        self.assertIsNone(_parse_date("32-13-2020"))
        data = postprocess({"date": "вчера", "dates": [{"label": "x", "value": "завтра"}], "key_findings": []})
        self.assertIsNone(data["date"])
        self.assertEqual(data["dates"], [])

    def test_key_findings_normalization(self):
        data = postprocess(
            {
                "key_findings": [
                    "просто строка",
                    {"finding": "  вывод  ", "evidence": "  цитата  "},
                    {"finding": "", "evidence": "ignore"},
                ]
            }
        )
        self.assertEqual(
            data["key_findings"],
            [
                {"finding": "просто строка", "evidence": ""},
                {"finding": "вывод", "evidence": "цитата"},
            ],
        )


class SchemaTests(unittest.TestCase):
    def test_response_format_is_strict_json_schema(self):
        self.assertEqual(RESPONSE_FORMAT["type"], "json_schema")
        self.assertTrue(RESPONSE_FORMAT["json_schema"]["strict"])
        schema = RESPONSE_FORMAT["json_schema"]["schema"]
        self.assertIs(schema, EXTRACTION_SCHEMA)

    def test_schema_top_level_shape(self):
        self.assertEqual(EXTRACTION_SCHEMA["type"], "object")
        self.assertFalse(EXTRACTION_SCHEMA["additionalProperties"])
        required = set(EXTRACTION_SCHEMA["required"])
        props = set(EXTRACTION_SCHEMA["properties"])
        self.assertEqual(required, props)
        self.assertIn("summary", props)
        self.assertEqual(EXTRACTION_SCHEMA["properties"]["summary"]["type"], "string")
        self.assertEqual(EXTRACTION_SCHEMA["properties"]["key_findings"]["type"], "array")
        for name, node in EXTRACTION_SCHEMA["properties"].items():
            if isinstance(node, dict) and node.get("type") == "object":
                self.assertFalse(node.get("additionalProperties", True), name)
            if isinstance(node, dict) and node.get("type") == ["object", "null"]:
                self.assertFalse(node.get("additionalProperties", True), name)
            if "items" in node and isinstance(node["items"], dict):
                item = node["items"]
                if item.get("type") == "object":
                    self.assertFalse(item.get("additionalProperties", True), name)


class ApiErrorHandlingTests(unittest.TestCase):
    def test_api_error_becomes_runtime_error(self):
        from openai import APIError

        mock_request = MagicMock()
        err = APIError("boom", mock_request, body=None)

        with patch.object(main, "encode_image_to_data_url", return_value="data:image/png;base64,xx"):
            with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test-not-real"}):
                with patch("main.OpenAI") as mock_openai:
                    mock_openai.return_value.chat.completions.create.side_effect = err
                    with self.assertRaises(RuntimeError) as ctx:
                        analyze_image("dummy.png")
        msg = str(ctx.exception)
        self.assertIn("OpenAI API", msg)
        self.assertNotIn("sk-test-not-real", msg)


if __name__ == "__main__":
    unittest.main()

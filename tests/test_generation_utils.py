from __future__ import annotations

import unittest

from zhaw_at_touche.generation_utils import clean_response_text, model_alias


class GenerationUtilsTests(unittest.TestCase):
    def test_clean_response_text_flattens_bullets(self) -> None:
        text = "\\u2022 First point\\n- Second point\\n\\nThird line"
        cleaned = clean_response_text(text)
        self.assertEqual(cleaned, "First point Second point Third line")

    def test_model_alias(self) -> None:
        self.assertEqual(model_alias("gemini-2.5-flash-lite"), "gemini25flashlite")
        self.assertEqual(model_alias("custom/model-v1"), "custommodelv1")


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import inspect
import types
import unittest
from unittest.mock import patch

from test_case_base import test_instruction_translator_cache_context

from sot.opcode_translator.executor.opcode_executor import (
    InstructionTranslatorCache,
)


def fake_frames() -> (
    tuple[
        types.FrameType,
        types.FrameType,
        types.FrameType,
        types.FrameType,
        types.FrameType,
    ]
):
    def fake_inner_fn_1():
        frame = inspect.currentframe()
        assert frame is not None
        return frame

    def fake_inner_fn_2():
        frame = inspect.currentframe()
        assert frame is not None
        return frame

    def fake_inner_fn_3():
        frame = inspect.currentframe()
        assert frame is not None
        return frame

    def fake_inner_fn_4():
        frame = inspect.currentframe()
        assert frame is not None
        return frame

    def fake_inner_fn_5():
        frame = inspect.currentframe()
        assert frame is not None
        return frame

    return (
        fake_inner_fn_1(),
        fake_inner_fn_2(),
        fake_inner_fn_3(),
        fake_inner_fn_4(),
        fake_inner_fn_5(),
    )


(
    FRAME_1,
    FRAME_2,
    FRAME_3,
    FRAME_4,
    FRAME_5,
) = fake_frames()


def mock_start_translate(frame: types.FrameType):
    translate_map = {
        FRAME_1: (FRAME_2.f_code, lambda frame: True),
        FRAME_3: (FRAME_4.f_code, lambda frame: False),  # Always re-compile
        FRAME_5: None,
    }
    return translate_map[frame]


class TestInstructionTranslatorCache(unittest.TestCase):
    def reset(self):
        global translate_count
        translate_count = 0
        InstructionTranslatorCache().clear()

    @patch(
        "sot.opcode_translator.executor.opcode_executor.start_translate",
        mock_start_translate,
    )
    def test_cache_hit(self):
        with test_instruction_translator_cache_context() as ctx:
            translated_code_1 = InstructionTranslatorCache()(FRAME_1)
            assert translated_code_1 is not None
            self.assertEqual(translated_code_1.code, FRAME_2.f_code)
            self.assertEqual(ctx.translate_count, 1)
            # cache hit
            translated_code_2 = InstructionTranslatorCache()(FRAME_1)
            assert translated_code_2 is not None
            self.assertEqual(translated_code_2.code, FRAME_2.f_code)
            self.assertEqual(ctx.translate_count, 1)

    @patch(
        "sot.opcode_translator.executor.opcode_executor.start_translate",
        mock_start_translate,
    )
    def test_cache_miss_due_to_unknown_code(self):
        with test_instruction_translator_cache_context() as ctx:
            translated_code_1 = InstructionTranslatorCache()(FRAME_1)
            assert translated_code_1 is not None
            self.assertEqual(translated_code_1.code, FRAME_2.f_code)
            self.assertEqual(ctx.translate_count, 1)
            # cache miss
            translated_code_2 = InstructionTranslatorCache()(FRAME_3)
            assert translated_code_2 is not None
            self.assertEqual(translated_code_2.code, FRAME_4.f_code)
            self.assertEqual(ctx.translate_count, 2)

    @patch(
        "sot.opcode_translator.executor.opcode_executor.start_translate",
        mock_start_translate,
    )
    def test_cache_miss_due_to_check_failed(self):
        with test_instruction_translator_cache_context() as ctx:
            translated_code_1 = InstructionTranslatorCache()(FRAME_3)
            assert translated_code_1 is not None
            self.assertEqual(translated_code_1.code, FRAME_4.f_code)
            self.assertEqual(ctx.translate_count, 1)
            # cache miss
            translated_code_2 = InstructionTranslatorCache()(FRAME_3)
            assert translated_code_2 is not None
            self.assertEqual(translated_code_2.code, FRAME_4.f_code)
            self.assertEqual(ctx.translate_count, 2)

    @patch(
        "sot.opcode_translator.executor.opcode_executor.start_translate",
        mock_start_translate,
    )
    def test_skip_frame(self):
        with test_instruction_translator_cache_context() as ctx:
            translated_code_1 = InstructionTranslatorCache()(FRAME_5)
            self.assertIsNone(translated_code_1)
            self.assertEqual(ctx.translate_count, 1)
            # skip frame
            translated_code_2 = InstructionTranslatorCache()(FRAME_5)
            self.assertIsNone(translated_code_2)
            self.assertEqual(ctx.translate_count, 1)


if __name__ == '__main__':
    unittest.main()

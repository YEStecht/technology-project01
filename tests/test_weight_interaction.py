"""기본 버튼이 슬라이더·계산·그림과 같은 하중 상태를 사용하는지 검증합니다."""

import json
from pathlib import Path
import unittest

from streamlit.testing.v1 import AppTest


APP = Path(__file__).resolve().parents[1] / "streamlit_app.py"


class WeightInteractionTests(unittest.TestCase):
    def setUp(self):
        self.app = AppTest.from_file(str(APP), default_timeout=60).run()
        self.assert_load(0)

    def rerun(self):
        self.app.run()
        self.assertEqual([e.message for e in self.app.exception], [])

    def assert_load(self, count):
        self.assertEqual([e.message for e in self.app.exception], [])
        self.assertEqual(self.app.session_state["load"], count)
        control = next(w for w in [*self.app.slider, *self.app.number_input] if w.key == "load")
        self.assertEqual(control.value, count)
        plot = next(p for p in self.app.get("plotly_chart") if "selected_" in p.proto.id)
        figure = json.loads(plot.proto.spec)
        self.assertEqual(figure["layout"]["meta"]["requested_count"], count)
        self.assertEqual(len(figure["layout"].get("images", [])), count)
        self.assertEqual(self.app.button(key="weight_remove").disabled, count == 0)
        self.assertEqual(self.app.button(key="weight_add").disabled, count >= control.max)

    def click(self, key, expected):
        self.app.button(key=key).click()
        self.rerun()
        self.assert_load(expected)

    def test_buttons_repeat_and_share_slider_state_on_five_bridges(self):
        for kind in ("단순교", "아치교", "트러스교", "사장교", "현수교"):
            with self.subTest(kind=kind):
                self.app.selectbox(key="selected_bridge").set_value(kind)
                self.rerun()
                self.assert_load(0)
                initial = self.app.metric[0].value
                self.click("weight_add", 1)
                self.click("weight_add", 2)
                self.assertNotEqual(self.app.metric[0].value, initial)
                self.click("weight_remove", 1)
                self.app.slider(key="load").set_value(2)
                self.rerun()
                self.assert_load(2)
                self.click("weight_remove", 1)
                self.click("weight_remove", 0)
                self.assertEqual(self.app.metric[0].value, initial)

    def test_buttons_follow_changed_capacity_and_zero_capacity(self):
        limit = self.app.slider(key="load").max
        self.app.slider(key="load").set_value(limit)
        self.rerun()
        self.assert_load(limit)
        self.app.selectbox(key="selected_bridge").set_value("트러스교")
        self.rerun()
        self.click("weight_add", limit + 1)
        self.app.selectbox(key="selected_bridge").set_value("단순교")
        self.rerun()
        self.assert_load(limit)
        self.app.slider(key="span_cm").set_value(100)
        self.app.slider(key="member_depth").set_value(2.)
        self.rerun()
        self.assert_load(0)
        self.assertTrue(self.app.button(key="weight_add").disabled)
        self.app.slider(key="span_cm").set_value(60)
        self.app.slider(key="member_depth").set_value(6.)
        self.rerun()
        self.click("weight_add", 1)
        self.app.button(key="bridge_zoom_in").click()
        self.rerun()
        self.assert_load(1)
        self.click("weight_remove", 0)


if __name__ == "__main__":
    unittest.main()

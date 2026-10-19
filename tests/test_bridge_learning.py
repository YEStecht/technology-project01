"""실행: python -m unittest discover -s tests -v"""

import ast
import json
from pathlib import Path
import unittest

from streamlit.testing.v1 import AppTest


APP = Path(__file__).resolve().parents[1] / "streamlit_app.py"


def calculation_namespace():
    # 화면을 실행하지 않고 기존 계산 함수와 상수만 불러옵니다.
    source = APP.read_text().split("# ── 4. 심화 설정")[0]
    tree = ast.parse(source)
    tree.body = [node for node in tree.body if isinstance(
        node, (ast.Import, ast.ImportFrom, ast.Assign, ast.FunctionDef)
    )]
    namespace = {}
    exec(compile(tree, str(APP), "exec"), namespace)
    return namespace


class BridgeLearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.calc = calculation_namespace()

    def setUp(self):
        self.app = AppTest.from_file(str(APP), default_timeout=60).run()
        self.assert_ok()

    def assert_ok(self):
        self.assertEqual([e.message for e in self.app.exception], [])

    def rerun(self):
        self.app.run()
        self.assert_ok()

    def config(self):
        return dict(self.app.session_state["_last_observation"]["context"][1])

    def change_messages(self):
        return [i.value for i in self.app.info if "같은 구조·설정" in i.value]

    def load_control(self):
        return next(w for w in [*self.app.slider, *self.app.number_input] if w.key == "load")

    def test_five_bridges_and_single_main_load_control(self):
        self.assertEqual(len([s for s in self.app.main.slider if s.key == "load"]), 1)
        self.assertNotIn("load", [s.key for s in self.app.sidebar.slider])
        self.assertFalse(self.app.sidebar.expander[0].proto.expanded)
        self.app.slider(key="load").set_value(3)
        for kind in self.calc["BRIDGE_NAMES"]:
            with self.subTest(kind=kind):
                self.app.selectbox(key="selected_bridge").set_value(kind)
                self.rerun()
                self.assertEqual(self.app.slider(key="load").value, 3)
                sweep = self.calc["solve_sweep"](kind, self.config())
                self.assertEqual(self.app.metric[0].value,
                                 f"{sweep['results'][3]['deflection_mm']:.4f} mm")
                self.assertTrue(any(f"selected_{kind}" in p.proto.id
                                    for p in self.app.get("plotly_chart")))

    def test_load_changes_use_actual_values_in_both_directions(self):
        sweep = self.calc["solve_sweep"]("단순교", self.config())
        for before, after in [(0, 3), (3, 1), (1, 3), (3, 0)]:
            self.app.slider(key="load").set_value(after)
            self.rerun()
            messages = self.change_messages()
            self.assertEqual(len(messages), 1)
            self.assertIn(f"추 {before}개 → {after}개", messages[0])
            for count in (before, after):
                self.assertIn(f"{sweep['results'][count]['deflection_mm']:.4f}", messages[0])
            self.assertTrue(any(f"= {after*20} g" in m.value for m in self.app.markdown))

    def test_member_arrows_match_calculated_forces(self):
        config = self.config()
        for self_weight in (True, False):
            for kind in self.calc["BRIDGE_NAMES"]:
                sweep = self.calc["solve_sweep"](kind, dict(config, self_weight=self_weight))
                for count in (0, 3, 20):
                    with self.subTest(kind=kind, count=count, self_weight=self_weight):
                        fig = self.calc["experiment_figure"](sweep, count, 10, 1)
                        labels = [a for a in fig.layout.annotations if (a.name or "").startswith("member_name_")]
                        arrows = [a for a in fig.layout.annotations if (a.name or "").startswith("member_axial_")]
                        result = sweep["results"][fig.layout.meta["display_count"]]
                        self.assertGreaterEqual(len(labels), 1)
                        self.assertTrue(any(a.name == "force_legend" for a in fig.layout.annotations))
                        for label in labels:
                            self.assertNotIn("인장", label.text)
                            self.assertNotIn("압축", label.text)
                            self.assertEqual((label.xref, label.yref), ("x", "y"))
                        np = self.calc["np"]
                        shown = sweep["model"]["nodes"] + 10 * result["u"].reshape(-1, 3)[:, :2]
                        for arrow in arrows:
                            index, side = map(int, arrow.name.removeprefix("member_axial_").rsplit("_", 1))
                            e = sweep["model"]["members"][index]
                            direction = shown[e["j"]] - shown[e["i"]]
                            projection = np.dot([arrow.x - arrow.ax, arrow.y - arrow.ay], direction) * side
                            if result["axial"][index] > 0:
                                self.assertGreater(projection, 0)  # 양쪽 바깥으로 당김
                                self.assertEqual(arrow.arrowcolor, self.calc["TENSION"])
                            else:
                                self.assertLess(projection, 0)  # 중심 쪽으로 누름
                                self.assertEqual(arrow.arrowcolor, self.calc["COMPRESSION"])
                            self.assertEqual(arrow.text, "")
                        bending = [a for a in fig.layout.annotations if (a.name or "").startswith("member_bending_")]
                        for arrow in bending:
                            side = int(arrow.name.rsplit("_", 1)[1])
                            projection = (arrow.x - arrow.ax) * side
                            self.assertLess(projection, 0) if "_upper_" in arrow.name else self.assertGreater(projection, 0)
                        if not self_weight and count == 0:
                            self.assertTrue(all("힘 거의 없음" in label.text for label in labels))
                            self.assertEqual(arrows + bending, [])
        unstable = self.calc["solve_sweep"]("트러스교", config, braced=False)
        fig = self.calc["experiment_figure"](unstable, 3, 10, 1)
        self.assertFalse(any((a.name or "").startswith("member_") for a in fig.layout.annotations))

    def test_exceeded_limit_preserves_last_valid_deformation(self):
        sweep = self.calc["solve_sweep"]("단순교", self.config())
        capacity = sweep["capacity"]
        self.assertTrue(0 <= capacity < 20)
        limit = self.calc["experiment_figure"](sweep, capacity, 10, 1)
        expected = json.loads(limit.to_json())["data"]
        for count in (capacity + 1, 20):
            fig = self.calc["experiment_figure"](sweep, count, 10, 1)
            self.assertEqual(json.loads(fig.to_json())["data"], expected)
            self.assertEqual(fig.layout.meta["display_count"], capacity)
            self.assertEqual(fig.layout.meta["requested_count"], count)
            self.assertTrue(fig.layout.meta["frozen"])
            self.assertIn("상태 유지", fig.layout.annotations[-2].text)
        # 하중을 다시 줄이면 해당 하중의 계산된 변형으로 돌아갑니다.
        fig = self.calc["experiment_figure"](sweep, 0, 10, 1)
        self.assertFalse(fig.layout.meta["frozen"])
        self.assertEqual(fig.layout.meta["display_count"], 0)
        # 자중부터 기준 초과인 경우에도 계산된 변형을 없애지 않습니다.
        weak = self.calc["solve_sweep"]("단순교", dict(self.config(), span=1., member_depth=.002))
        self.assertEqual(weak["capacity"], -1)
        fig = self.calc["experiment_figure"](weak, 20, 10, 1)
        self.assertEqual(fig.layout.meta["display_count"], 0)
        self.assertIn("참고 모습 유지", fig.layout.annotations[-2].text)
        self.assertNotEqual(json.loads(fig.to_json())["data"],
                            json.loads(self.calc["experiment_figure"](weak, 20, 0, 1).to_json())["data"])

    def test_view_zoom_changes_camera_and_preserves_experiment(self):
        def figure():
            plot = next(p for p in self.app.get("plotly_chart") if "selected_" in p.proto.id)
            return json.loads(plot.proto.spec)

        original = figure()
        metric = self.app.metric[0].value
        config = self.config()
        self.app.button(key="bridge_zoom_in").click()
        self.rerun()
        enlarged = figure()
        old_range = original["layout"]["xaxis"]["range"]
        new_range = enlarged["layout"]["xaxis"]["range"]
        self.assertLess(new_range[1] - new_range[0], old_range[1] - old_range[0])
        self.assertEqual(enlarged["data"], original["data"])
        self.assertEqual(enlarged["layout"]["shapes"], original["layout"]["shapes"])
        self.assertEqual(enlarged["layout"]["annotations"][-2]["xref"], "paper")
        self.assertEqual(self.app.metric[0].value, metric)
        self.assertEqual(self.config(), config)
        self.assertEqual(self.change_messages(), [])
        self.app.slider(key="load").set_value(3)
        self.app.selectbox(key="selected_bridge").set_value("현수교")
        self.rerun()
        self.assertEqual(self.app.session_state["bridge_zoom_percent"], 125)
        self.app.button(key="bridge_zoom_out").click()
        self.rerun()
        self.assertEqual(self.app.session_state["bridge_zoom_percent"], 100)
        self.app.button(key="bridge_zoom_out").click()
        self.rerun()
        self.assertEqual(self.app.session_state["bridge_zoom_percent"], 75)
        self.app.button(key="bridge_zoom_reset").click()
        self.rerun()
        self.assertEqual(self.app.session_state["bridge_zoom_percent"], 100)
        self.assertEqual(self.app.slider(key="load").value, 3)
        for percent, button in [(250, "bridge_zoom_in"), (50, "bridge_zoom_out")]:
            self.app.session_state["bridge_zoom_percent"] = percent
            self.rerun()
            self.assertTrue(self.app.button(key=button).disabled)

    def test_changed_settings_reset_comparison_and_preserve_count(self):
        self.app.slider(key="load").set_value(3)
        self.rerun()
        self.app.select_slider(key="weight_mass_g").set_value(100)
        self.rerun()
        self.assertEqual(self.change_messages(), [])
        limit = max(0, self.calc["solve_sweep"]("단순교", self.config())["capacity"])
        count = min(3, limit)
        self.assertEqual(self.load_control().value, count)
        self.assertTrue(any(f"= {count*100} g" in m.value for m in self.app.markdown))
        self.app.slider(key="span_cm").set_value(100)
        self.rerun()
        self.assertEqual(self.change_messages(), [])
        self.app.selectbox(key="selected_bridge").set_value("현수교")
        self.rerun()
        self.assertEqual(self.change_messages(), [])
        self.assertEqual(self.load_control().value, 0)
        self.assertEqual(self.app.slider(key="span_cm").value, 100)
        self.app.slider(key="span_cm").set_value(60)
        self.app.select_slider(key="weight_mass_g").set_value(20)
        self.rerun()
        self.load_control().set_value(1)
        self.rerun()
        self.load_control().set_value(2)
        self.rerun()
        self.assertIn("추 1개 → 2개", self.change_messages()[0])

    def test_load_limit_blocks_excess_and_zero_capacity(self):
        for kind in self.calc["BRIDGE_NAMES"]:
            with self.subTest(kind=kind):
                self.app.selectbox(key="selected_bridge").set_value("아치교")
                self.rerun()
                self.load_control().set_value(20)
                self.rerun()
                self.app.selectbox(key="selected_bridge").set_value(kind)
                self.rerun()
                sweep = self.calc["solve_sweep"](kind, self.config())
                limit = max(0, sweep["capacity"])
                self.assertEqual(self.load_control().max, limit)
                # 실제 조작으로 20개를 올린 뒤 약한 교량으로 바꿔도 허용량으로 줄어듭니다.
                self.assertEqual(self.load_control().value, limit)
                self.assertEqual(self.app.session_state["load"], limit)
        self.app.selectbox(key="selected_bridge").set_value("단순교")
        self.app.slider(key="span_cm").set_value(100)
        self.app.slider(key="member_depth").set_value(2.)
        self.rerun()
        self.assertTrue(self.load_control().disabled)
        self.assertEqual(self.load_control().value, 0)
        controls = [w for w in [*self.app.main.slider, *self.app.main.number_input] if w.key == "load"]
        self.assertEqual(len(controls), 1)
        self.app.slider(key="span_cm").set_value(60)
        self.app.slider(key="member_depth").set_value(6.)
        self.rerun()
        self.assertFalse(self.load_control().disabled)
        self.assertEqual(self.load_control().value, 0)

    def test_cables_attach_to_deck_surface(self):
        np = self.calc["np"]
        for kind in ("사장교", "현수교"):
            for tower_ratio in (.25, 5/60):
                with self.subTest(kind=kind, tower_ratio=tower_ratio):
                    cfg = dict(self.config(), tower_ratio=tower_ratio, sag_ratio=.8)
                    sweep = self.calc["solve_sweep"](kind, cfg)
                    count = max(0, min(3, sweep["capacity"]))
                    fig = self.calc["experiment_figure"](sweep, count, 10, 1)
                    result = sweep["results"][fig.layout.meta["display_count"]]
                    model = sweep["model"]
                    shown = (model["nodes"] + 10 * result["u"].reshape(-1, 3)[:, :2]) * 12 / cfg["span"]
                    members = [e for e in model["members"] if e["role"] != "바닥 보"]
                    foundations = [t for t in fig.data if (t.meta or {}).get("illustration") == "tower_foundation"]
                    self.assertEqual(len(foundations), 2)
                    for foundation in foundations:
                        base = foundation.meta["base_node"]
                        self.assertLess(min(foundation.y), -self.calc["deck_half_depth"](kind))
                        self.assertAlmostEqual(max(foundation.y), shown[base, 1] + self.calc["deck_half_depth"](kind))
                        _, support_y = self.calc["support_display_point"](model, base, 12/cfg["span"])
                        self.assertAlmostEqual(min(foundation.y), support_y)
                    traces = [t for t in fig.data if t.name and not t.name.startswith("바닥 보")
                              and (t.meta or {}).get("illustration") != "tower_foundation"]
                    self.assertEqual(len(traces), len(members))
                    self.assertLess(self.calc["deck_half_depth"](kind), self.calc["BEAM_HALF_DEPTH"])
                    for e, trace in zip(members, traces):
                        expected = shown[[e["i"], e["j"]]].copy()
                        expected[:, 1] += self.calc["deck_half_depth"](kind)
                        np.testing.assert_allclose(trace.x, expected[:, 0])
                        np.testing.assert_allclose(trace.y, expected[:, 1])

    def test_shape_settings_and_unstable_truss(self):
        changes = [
            ("아치교", "arch_percent", 10),
            ("트러스교", "triangles", 1),
            ("트러스교", "triangles", 11),
            ("트러스교", "truss_percent", 35),
            ("사장교", "tower_cm", 5),
            ("현수교", "sag_percent", 80),
        ]
        for kind, key, value in changes:
            with self.subTest(kind=kind, key=key):
                self.app.selectbox(key="selected_bridge").set_value(kind)
                self.app.slider(key=key).set_value(value)
                self.rerun()
        self.app.selectbox(key="selected_bridge").set_value("트러스교")
        self.app.toggle(key="add_diagonals").set_value(False)
        self.rerun()
        self.assertEqual(self.app.metric[0].value, "계산 불가")
        self.assertTrue(any("연결 상태" in w.value for w in self.app.warning))
        self.app.toggle(key="add_diagonals").set_value(True)
        self.rerun()
        self.assertNotEqual(self.app.metric[0].value, "계산 불가")

    def test_zero_load_and_capacity_remain_available(self):
        self.app.toggle(key="self_weight").set_value(False)
        self.rerun()
        for kind in self.calc["BRIDGE_NAMES"]:
            self.app.selectbox(key="selected_bridge").set_value(kind)
            self.rerun()
            self.assertEqual(self.app.metric[0].value, "0.0000 mm")
            self.assertTrue(any("작용하는 힘이 거의 없습니다" in m.value for m in self.app.markdown))
            sweep = self.calc["solve_sweep"](kind, self.config())
            first_bad = next((i for i, r in enumerate(sweep["results"])
                              if not r["stable"] or r["utilization"] > 1+1e-9), None)
            self.assertEqual(sweep["capacity"], 20 if first_bad is None else first_bad-1)
            expected = self.calc["capacity_text"](sweep)
            self.assertTrue(any(f"**{expected}**" in m.value for m in self.app.markdown))

    def test_comparison_modes_and_material_change(self):
        self.app.radio(key="comparison_mode").set_value("두 교량 비교")
        self.rerun()
        self.app.selectbox(key="compare_left").set_value("아치교")
        self.rerun()
        self.assertNotEqual(self.app.selectbox(key="compare_left").value,
                            self.app.selectbox(key="compare_right").value)
        self.app.radio(key="comparison_mode").set_value("5종 전체 비교")
        self.app.selectbox(key="material").set_value("단단한 목재형 · 예시")
        self.rerun()
        self.assertFalse(any("capacity_bridge_overview" in p.proto.id for p in self.app.get("plotly_chart")))
        self.app.session_state["advanced_comparison"] = True
        self.rerun()
        self.assertTrue(any("capacity_bridge_overview" in p.proto.id
                            for p in self.app.get("plotly_chart")))
        self.assertEqual(len([s for s in self.app.main.slider if s.key == "load"]), 1)
        self.assertAlmostEqual(self.config()["E"], 10e9)

    def test_reference_mode_and_return_to_basic_limit(self):
        sweep = self.calc["solve_sweep"]("단순교", self.config())
        capacity = sweep["capacity"]
        curves = self.calc["load_curve"]({"단순교": sweep}, ["단순교"], 0)
        self.assertTrue(all(y is None for trace in curves.data for x, y in zip(trace.x, trace.y) if x > capacity))
        reference = self.calc["load_curve"]({"단순교": sweep}, ["단순교"], 0, include_reference=True)
        self.assertTrue(any(y is not None for trace in reference.data for x, y in zip(trace.x, trace.y) if x > capacity))
        self.app.toggle(key="allow_over_limit").set_value(True)
        self.rerun()
        self.assertEqual(self.load_control().max, 20)
        self.load_control().set_value(capacity + 1)
        self.rerun()
        fig = json.loads(next(p for p in self.app.get("plotly_chart") if "selected_" in p.proto.id).proto.spec)
        self.assertTrue(fig["layout"]["meta"]["frozen"])
        self.assertEqual(fig["layout"]["meta"]["display_count"], capacity)
        self.assertTrue(any("선형 참고값" in w.value for w in self.app.warning))
        self.assertEqual(self.app.metric[0].value, f"{sweep['results'][capacity+1]['deflection_mm']:.4f} mm")
        self.app.toggle(key="allow_over_limit").set_value(False)
        self.rerun()
        self.assertEqual(self.load_control().max, capacity)
        self.assertEqual(self.load_control().value, capacity)

    def test_learning_flow_without_discussion_inputs(self):
        self.assertEqual([t.label for t in self.app.tabs], ["실험실 안내", "교량 탐구", "교량 비교"])
        headings = [h.value for h in self.app.subheader]
        self.assertLess(headings.index("① 교량 구조 알아보기"), headings.index("② 하중 실험하기"))
        self.assertEqual(self.app.radio(key="comparison_mode").value, "5종 전체 비교")
        text = "\n".join(m.value for m in [*self.app.markdown, *self.app.caption, *self.app.info])
        self.assertNotIn("건설비", text)
        self.assertNotIn("설계자가", text)
        self.assertNotIn("가장 여유가 작은", text)
        for removed in ("Q1.", "Q2.", "토의", "두 질문"):
            self.assertNotIn(removed, text)
        self.assertIn("교량 구조를 배우고 실험·실습을 대신할 수 있는 디지털 학습 자료입니다.", text)
        self.assertIn("③ 교량 비교하기", [h.value for h in self.app.header])
        self.assertTrue(any("「다양한 교량의 구조」 → 「교량 만들기 실습」" in c.value
                            for c in self.app.caption))
        self.assertEqual(len(self.app.text_area), 0)
        self.app.selectbox(key="selected_bridge").set_value("현수교")
        self.rerun()
        self.assertEqual(len(self.app.text_area), 0)
        self.assertTrue(all(not e.proto.expanded for e in self.app.sidebar.expander))


if __name__ == "__main__":
    unittest.main()

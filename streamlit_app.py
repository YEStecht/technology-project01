"""교량 개념 실험실 — 실행: python -m streamlit run streamlit_app.py

처음 수정한다면 MATERIALS, build_model(), solve_sweep()부터 살펴보세요.
학습 화면·계산·그래프는 이 파일, 추 그림과 조절 버튼은 weight_interaction.py에 있습니다.
교량별 이름표 그림과 용어 풀이는 bridge_anatomy.py에 있습니다.
외부 데이터는 필요하지 않습니다.
"""

import math
import re
from html import escape
from pathlib import Path

import numpy as np
from PIL import Image
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from weight_interaction import WEIGHT_IMAGE_SOURCE, render_weight_controls
from bridge_anatomy import bridge_anatomy_diagram, bridge_parts_legend, part_badges, part_meanings


# ── 1. 수업에서 바꾸기 쉬운 설정 ────────────────────────────────────────
GRAVITY = 9.81
COLORS = {
    "단순교": "#64748b", "아치교": "#10a38b", "트러스교": "#7160d2",
    "사장교": "#217bd0", "현수교": "#d69026",
}
# 탐구와 비교에서 함께 사용하는 핵심 설명입니다.
BRIDGE_GUIDES = {
    "단순교": {
        "title": "단순교(거더교)",
        "shape": "상판 아래의 보를 양쪽 받침이 지지하는 구조입니다.",
        "comparison_structure": "보 + 양쪽 받침", "comparison_principle": "보의 휨: 위 압축·아래 인장",
        "principle": "보가 아래로 휘어질 때 윗부분에는 압축, 아랫부분에는 인장이 생깁니다.",
        "load_path": "상판 → 보 → 양쪽 받침 → 지반",
        "use": "비교적 짧은 거리를 간단한 구조로 연결할 때 활용합니다.",
        "limit": "받침 사이 거리가 길어지면 보의 휨과 처짐을 더 크게 고려해야 합니다.",
        "examples": "짧은 도로교와 육교의 일부",
    },
    "아치교": {
        "title": "아치교",
        "shape": "굽은 아치와 상판을 연결한 구조입니다.",
        "comparison_structure": "곡선 아치 + 세로 지지대", "comparison_principle": "아치의 압축·받침의 수평 힘",
        "principle": "아치는 주로 압축을 받습니다. 양쪽 받침에는 바깥쪽으로 미는 힘도 작용합니다.",
        "load_path": "상판 → 세로 지지대 → 아치 → 양쪽 받침·지반",
        "use": "양쪽 지반과 받침이 튼튼한 곳에서 활용할 수 있습니다.",
        "limit": "받침과 지반은 아치가 바깥쪽으로 미는 힘을 견뎌야 합니다.",
        "examples": "시드니 하버 브리지",
    },
    "트러스교": {
        "title": "트러스교",
        "shape": "여러 막대를 삼각형으로 연결한 구조입니다.",
        "comparison_structure": "삼각형으로 연결한 막대", "comparison_principle": "막대의 인장·압축 분담",
        "principle": "막대가 인장과 압축을 나누어 받습니다. 삼각형 연결은 모양이 변하는 것을 억제합니다.",
        "load_path": "상판 → 삼각형을 이루는 막대 → 받침 → 지반",
        "use": "변형에 잘 저항해야 하는 철도교 등에 활용합니다.",
        "limit": "길고 가는 압축 막대는 옆으로 휘어질 수 있으며, 연결부의 강도도 중요합니다.",
        "examples": "철도 트러스교",
    },
    "사장교": {
        "title": "사장교",
        "shape": "주탑에서 뻗은 경사진 케이블이 상판을 직접 지지합니다.",
        "comparison_structure": "주탑 + 경사진 케이블", "comparison_principle": "케이블 인장·주탑 압축",
        "principle": "케이블은 인장을 받고, 주탑은 주로 압축을 받으며 하중을 기초로 전달합니다.",
        "load_path": "상판 → 사장 케이블 → 주탑·기초 / 뒤쪽 케이블 → 앵커(고정 장치)",
        "use": "중간 교각을 적게 설치해야 하는 비교적 긴 교량에 활용합니다.",
        "limit": "주탑과 케이블의 시공·점검, 바람에 의한 진동도 고려해야 합니다.",
        "examples": "인천대교, 서해대교",
    },
    "현수교": {
        "title": "현수교",
        "shape": "굵은 주케이블에 연결된 행어(세로줄)가 상판을 매달아 지지합니다.",
        "comparison_structure": "주케이블 + 행어", "comparison_principle": "행어·주케이블의 인장",
        "principle": "행어와 주케이블은 인장을 받습니다. 주탑과 앵커리지(고정 장치)가 케이블의 힘을 받아 지반으로 전달합니다.",
        "load_path": "상판 → 행어 → 주케이블 → 주탑·앵커리지 → 지반",
        "use": "해협처럼 매우 긴 거리를 연결할 때 활용합니다.",
        "limit": "바람에 의한 진동과 상판의 변형, 큰 앵커리지의 설치 조건을 고려해야 합니다.",
        "examples": "이순신대교, 남해대교, 금문교",
    },
}

# 사진별 출처·라이선스와 구조 분류 근거: assets/bridges/CREDITS.md
BRIDGE_EXAMPLES = {
    "단순교": {
        "name": "일산대교 — 거더교", "file": "ilsan.jpg",
        "description": "상판 아래의 긴 보(거더)를 교각이 받치는 모습을 확인할 수 있어요.",
        "note": "거더 구조의 실제 사례입니다. 실제 교량의 여러 경간과 받침 조건은 이 앱의 양끝 받침 단순보 모형과 다릅니다.",
        "author": "Glabb",
        "source": "https://commons.wikimedia.org/wiki/File:Ilsan_Bridge1.jpg",
        "license": "CC BY-SA 3.0", "license_url": "https://creativecommons.org/licenses/by-sa/3.0/",
    },
    "아치교": {
        "name": "서강대교 — 아치교", "file": "seogang.jpg",
        "description": "붉은 곡선 아치와 상판을 연결하는 경사진 줄을 확인할 수 있어요.",
        "note": "사진은 닐센 아치 구간입니다. 이 앱의 세로 지지대 아치 모형과 연결 방식이 다릅니다.",
        "author": "yujinunion2",
        "source": "https://commons.wikimedia.org/wiki/File:Seoul-3215895.jpg",
        "license": "CC0 1.0", "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
    },
    "트러스교": {
        "name": "성산대교 — 트러스교", "file": "seongsan.jpg",
        "description": "상판 아래 붉은 막대들이 삼각형으로 연결되어 하중을 나누어 받는 구조를 볼 수 있어요.",
        "author": "Glabb",
        "source": "https://commons.wikimedia.org/wiki/File:Seongsan_Bridge1.jpg",
        "license": "CC BY-SA 3.0", "license_url": "https://creativecommons.org/licenses/by-sa/3.0/",
    },
    "사장교": {
        "name": "인천대교 — 사장교", "file": "incheon.jpg",
        "description": "주탑에서 비스듬히 뻗은 케이블이 상판을 직접 지지하는 모습을 확인할 수 있어요.",
        "author": "Jinho Jung",
        "source": "https://commons.wikimedia.org/wiki/File:Incheon_bridge_(7).jpg",
        "license": "CC BY-SA 2.0", "license_url": "https://creativecommons.org/licenses/by-sa/2.0/",
    },
    "현수교": {
        "name": "광안대교 — 현수교", "file": "gwangan.jpg",
        "description": "주탑 사이에 늘어진 주케이블과 상판을 매단 세로줄(행어)을 확인할 수 있어요.",
        "note": "사진은 현수교 구간입니다. 광안대교에는 트러스교와 접속교 구간도 있습니다.",
        "author": "Richard Irwin",
        "source": "https://commons.wikimedia.org/wiki/File:Gwang_An_Bridge.jpg",
        "license": "퍼블릭 도메인 (저작자 공개)",
        "license_url": "https://commons.wikimedia.org/wiki/File:Gwang_An_Bridge.jpg#Licensing",
    },
}

COMPRESSION = "#e05252"
TENSION = "#287bd1"
LOAD_COLOR = "#c56a20"
SUPPORT_COLOR = "#8195a7"
# 내부 힘을 읽기 위한 그림용 두께입니다. 계산 단면과 별개입니다.
BEAM_HALF_DEPTH = .8
TOWER_BASE_DEPTH = 1.6  # 주탑 하부와 기초를 잇는 그림용 길이
WEIGHT_BODY_HEIGHT = .50  # 아래 치수는 추 아이콘을 그리기 위한 값입니다.
WEIGHT_RING_HEIGHT = .25
WEIGHT_STACK_STEP = .92
MAX_VISIBLE_WEIGHTS = 5  # 많은 추는 일부 그림과 정확한 총개수를 함께 표시합니다.
PLOT_CONFIG = {"displayModeBar": False, "responsive": True}

st.set_page_config(page_title="교량 구조 디지털 탐구 학습지", page_icon="🌉", layout="wide", initial_sidebar_state="collapsed")
# 본문 색을 상속하고 배경·테두리는 그 색을 옅게 섞어 밝은/어두운 테마에 대응합니다.
st.markdown(
    """
    <style>
    .block-container { max-width: 1280px; padding-top: 2.5rem; padding-bottom: 4rem; }
    h1, h2, h3, h4 { word-break: keep-all; overflow-wrap: anywhere; }
    h1 { font-size: clamp(1.8rem, 3vw, 2.4rem) !important; letter-spacing: -.035em; line-height: 1.35; }
    h2 { font-size: clamp(1.5rem, 2.5vw, 1.9rem) !important; line-height: 1.4; }
    h3 { font-size: clamp(1.3rem, 2.1vw, 1.55rem) !important; line-height: 1.45; }
    h4 { font-size: 1.1rem !important; line-height: 1.5; }
    [data-testid="stMain"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stMain"] [data-testid="stMarkdownContainer"] li {
        font-size: 1rem; line-height: 1.75; word-break: keep-all; overflow-wrap: anywhere;
    }
    /* 설명과 출처도 본문 색을 그대로 사용해 글자가 흐려지지 않게 합니다. */
    [data-testid="stCaptionContainer"] { color: inherit; opacity: 1; }
    [data-testid="stMain"] [data-testid="stCaptionContainer"] p {
        font-size: .9rem; line-height: 1.65;
        word-break: keep-all; overflow-wrap: break-word;
        color: inherit;
    }
    [data-testid="stWidgetLabel"] p { font-weight: 600; }
    [data-testid="stMain"] button { min-height: 44px; }
    [data-testid="stExpander"] { border-radius: 8px; }
    [data-testid="stExpander"] summary { min-height: 44px; }
    .st-key-bridge_intro { margin-top: .5rem; margin-bottom: 1.25rem; }
    .st-key-bridge_intro [data-testid="stHorizontalBlock"] { gap: 2rem; }
    .force-compression, .force-tension {
        display: inline; padding: .08em .25em; border-radius: 4px; font-weight: 700;
        -webkit-box-decoration-break: clone; box-decoration-break: clone;
    }
    .force-compression { color: #982f2f; background: #ffeded; }
    .force-tension { color: #185797; background: #eaf3ff; }
    .concept-focus { text-decoration: underline; text-decoration-thickness: 2px; text-underline-offset: .25em; }
    .load-path { display: flex; flex-wrap: wrap; align-items: center; gap: .5rem; margin: .5rem 0 .75rem; }
    .path-segment { display: inline-flex; align-items: center; gap: .5rem; max-width: 100%; }
    .load-path-help { position: relative; }
    .path-node { background: color-mix(in srgb, currentColor 6%, transparent); border-radius: 5px; line-height: 1.6; word-break: keep-all; }
    .path-node > summary { padding: .25rem .45rem; min-height: 32px; box-sizing: border-box;
        list-style: none; cursor: help; font-size: .95rem; }
    .path-node > summary::-webkit-details-marker { display: none; }
    .path-node > summary:focus-visible { outline: 2px solid #177568; outline-offset: 2px; border-radius: 5px; }
    .path-explanation { display: none; position: absolute; top: calc(100% + .25rem); left: 0; right: 0;
        z-index: 10; padding: .6rem .75rem; border-radius: 6px; pointer-events: none;
        background: #fbfcfe; color: #243447;
        background: light-dark(#fbfcfe, #20242c); color: light-dark(#243447, #f4f6f8);
        border: 1px solid color-mix(in srgb, currentColor 20%, transparent);
        box-shadow: 0 3px 10px rgba(0,0,0,.12); font-size: .9rem; line-height: 1.65;
        font-weight: 400; word-break: keep-all; overflow-wrap: anywhere; }
    .path-node[open] .path-explanation, .path-node > summary:focus-visible .path-explanation { display: block; }
    .path-node > summary:focus-visible .path-explanation { z-index: 12; }
    @media (hover: hover) {
        .path-node > summary:hover .path-explanation { display: block; z-index: 13; }
    }
    .path-arrow { font-size: 1.15rem; }
    .bridge-anatomy-diagram { display: block; width: 100%; max-width: 360px; height: auto; margin: .25rem auto .4rem; }
    .bridge-part-number { display: inline-flex; align-items: center; justify-content: center;
        width: 1.3rem; height: 1.3rem; margin-right: .3rem; border-radius: 50%;
        background: #177568; color: white; font-size: .8rem; font-weight: 700; vertical-align: middle; }
    .bridge-part-legend { margin: .65rem 0; font-size: .95rem; line-height: 1.75; }
    .bridge-part-legend > div { margin: .3rem 0; word-break: keep-all; overflow-wrap: anywhere; }
    .bridge-part-legend dt, .bridge-part-legend dd { display: inline; margin: 0; }
    .bridge-part-legend dt { font-weight: 600; }
    .bridge-glossary { margin: .35rem 0; }
    .bridge-glossary > summary { font-size: .9rem; cursor: pointer; padding: .15rem 0; }
    .study-steps { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 1.5rem; padding: 0; margin: .5rem 0 1rem; list-style: none; }
    .study-steps li { display: flex; gap: .65rem; align-items: flex-start; }
    .study-steps strong, .study-steps small { display: block; }
    .study-steps small { font-size: .9rem; line-height: 1.65; margin-top: .35rem; }
    .learning-icon { width: 22px; height: 22px; vertical-align: -.25em; flex-shrink: 0; }
    .study-steps .learning-icon { width: 26px; height: 26px; margin-top: .15rem; }
    .force-primer { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 1.25rem; margin: .25rem 0 1rem; }
    .force-primer section { display: flex; align-items: center; gap: .65rem; }
    .force-primer svg { width: 108px; height: 72px; flex-shrink: 0; }
    .force-primer strong, .force-primer small { display: block; }
    .force-primer small { font-size: .9rem; line-height: 1.6; margin-top: .35rem; }
    .experiment-focus { font-size: 1.05rem; line-height: 1.8; margin: 0; }
    .st-key-load { margin-top: .25rem; }
    .st-key-load [data-testid="stSlider"] { padding: .5rem .25rem; }
    .st-key-load input[type="range"]:focus-visible { outline: 2px solid #177568; }
    [data-testid="stMetric"] {
        background: color-mix(in srgb, currentColor 4%, transparent);
        border: none; border-left: 3px solid #177568;
        border-radius: 8px; padding: 18px 20px;
    }
    [data-testid="stMetricLabel"] *,
    [data-testid="stMetricValue"] * {
        white-space: normal !important; overflow: visible !important;
        text-overflow: clip !important; overflow-wrap: anywhere;
    }
    [data-testid="stMain"] [data-testid="stMetricValue"] [data-testid="stMarkdownContainer"] p {
        font-size: clamp(1.6rem, 2.5vw, 2rem);
        line-height: 1.3; font-variant-numeric: tabular-nums;
    }
    [data-testid="stTabs"] [role="tablist"] { gap: 12px; margin-bottom: 1rem; }
    [data-testid="stTabs"] [role="tab"] {
        padding: 12px 14px; border-radius: 6px 6px 0 0; height: auto; white-space: nowrap;
        background: transparent;
        color: inherit;
    }
    [data-testid="stTabs"] [aria-selected="true"] {
        background: rgba(23,117,104,.07);
        font-weight: 600;
    }
    [data-testid="stPlotlyChart"] {
        border: 1px solid color-mix(in srgb, currentColor 14%, transparent);
        border-radius: 10px; overflow: hidden;
    }
    /* 좁은 화면에서도 작은 보기 버튼은 그림 오른쪽에 유지합니다. */
    .st-key-bridge_viewport [data-testid="stHorizontalBlock"] {
        flex-wrap: nowrap !important; gap: .5rem;
    }
    .st-key-bridge_viewport [data-testid="stColumn"]:first-child {
        flex: 1 1 0 !important; min-width: 0 !important;
    }
    .st-key-bridge_viewport [data-testid="stColumn"]:last-child {
        flex: 0 0 44px !important; min-width: 44px !important; width: 44px !important;
    }
    .st-key-bridge_zoom_controls button {
        width: 44px; height: 44px; min-height: 44px; padding: 0;
    }
    .st-key-bridge_zoom_controls button p { font-size: 1.25rem; }
    .st-key-bridge_zoom_controls [data-testid="stCaptionContainer"] {
        text-align: center;
    }
    /* 구조 비교는 두 열 배치를 유지하며 좁은 화면에서는 그림 안에서 가로로 이동합니다. */
    .st-key-bridge_overview_scroll { overflow-x: auto; }
    .st-key-bridge_overview { min-width: 680px; }
    [data-testid="stDataFrame"] { border-radius: 8px; }
    @media (max-width: 640px) {
        .path-node > summary { min-height: 44px; padding: .5rem .45rem; }
        .bridge-glossary > summary { min-height: 44px; display: list-item; align-content: center; }
        .block-container { padding: 3.5rem 1rem 3rem; }
        [data-testid="stTabs"] [role="tablist"] { gap: 4px; }
        [data-testid="stTabs"] [role="tab"] { padding: 12px 10px; }
        .st-key-bridge_intro [data-testid="stHorizontalBlock"] { gap: .5rem; }
        .study-steps, .force-primer { grid-template-columns: 1fr; gap: 1rem; }
        .force-primer svg { width: 100px; }
        .st-key-bridge_zoom_controls button { border-radius: 8px; }
    }
    /* 안내 화면에만 글자 크기·줄 간격을 적용합니다. */
    .st-key-lab_guide { max-width: 1080px; margin: 0 auto; }
    .st-key-lab_guide h1 {
        font-size: clamp(1.8rem, 3vw, 2.5rem) !important;
        line-height: 1.4; margin-bottom: .25rem;
    }
    .st-key-lab_guide h3 { margin-top: 1.1rem; margin-bottom: .5rem; }
    .st-key-lab_guide [data-testid="stMarkdownContainer"] p {
        font-size: 1.05rem; line-height: 1.85;
    }
    .st-key-lab_guide .guide-subtitle {
        font-size: 1.2rem !important; font-weight: 600; margin: 0 0 .8rem;
    }
    .st-key-lab_guide [data-testid="stTable"] td,
    .st-key-lab_guide [data-testid="stTable"] th {
        font-size: 1rem; line-height: 1.75; padding: 12px 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)




# ── 2. 계산 모형: m, N, Pa, kg의 SI 단위만 사용합니다 ──────────────────
# 아래 물성은 특정 수수깡/종이의 측정값이 아닌 균질 재료의 예시입니다.
# 실제 제작 실험의 처짐·파단 하중을 측정했다면 왼쪽에서 보정할 수 있습니다.
MATERIALS = {
    "가벼운 목재형 · 예시": dict(E=3.0e9, density=300., ft=20e6, fc=12e6, fb=20e6),
    "단단한 목재형 · 예시": dict(E=10.0e9, density=500., ft=60e6, fc=35e6, fb=50e6),
    "플라스틱형 · 예시": dict(E=2.0e9, density=1100., ft=35e6, fc=25e6, fb=35e6),
}
BRIDGE_NAMES = ["단순교", "아치교", "트러스교", "사장교", "현수교"]
METHOD_SOURCE = "https://teachbooks.tudelft.nl/computational-modelling/structural_linear/euler_bernouilli.html"
BUCKLING_SOURCE = "https://ocw.mit.edu/courses/16-001-unified-engineering-materials-and-structures-fall-2021/resources/mit16_001_f21_lec33lec34lec35_pdf/"


def build_model(kind, cfg, braced=True):
    """두 옆면을 합친 평면 모형. 보=휨+축력, 핀 막대=축력, 줄=인장만.

    수치 계산용으로 보를 나눈 위치는 실제 접착부로 취급하지 않습니다.
    각 목재 부재는 두 옆면에서 같은 하중을 나눠 받는 것으로 가정합니다.
    """
    L = cfg["span"]
    n = cfg["panels"] if kind == "트러스교" else 12
    nodes = [(x, 0.) for x in np.linspace(0, L, n+1)]
    deck = list(range(n+1))
    members, fixed = [], set()

    def node(x, y):
        nodes.append((float(x), float(y)))
        return len(nodes)-1

    def member(i, j, form="rod", role="막대", buckling_length=None):
        members.append(dict(i=i, j=j, form=form, role=role, le=buckling_length))

    def pin(i):
        fixed.update((3*i, 3*i+1))

    def roller(i):
        fixed.add(3*i+1)

    # 상판의 휨 강성은 양옆의 연속된 보가 담당합니다.
    for i in range(n):
        member(i, i+1, "beam", "바닥 보", L if kind != "트러스교" else L/n)

    if kind == "단순교":
        pin(0)
        roller(n)
    elif kind == "아치교":
        rise = cfg["arch_ratio"] * L
        arch = [node(x, -rise-.03*L + 4*rise*x/L*(1-x/L)) for x in np.linspace(0, L, n+1)]
        arc_length = sum(math.dist(nodes[a], nodes[b]) for a, b in zip(arch[:-1], arch[1:]))
        for i in range(n):
            member(arch[i], arch[i+1], "beam", "아치", arc_length/2)
        for a, d in zip(arch, deck):
            member(a, d, "rod", "세로 지지대")
        pin(arch[0])
        pin(arch[-1])
        pin(0)
        roller(n)
    elif kind == "트러스교":
        bottom = [node(x, -cfg["truss_ratio"]*L) for x in np.linspace(0, L, n+1)]
        for i in range(n):
            member(bottom[i], bottom[i+1], role="아래 가로 막대")
            if braced:
                # 중앙 세로축을 기준으로 왼쪽의 지그재그 배치를 오른쪽에 반사합니다.
                # 홀수 칸의 중앙에는 두 대각재를 둡니다. X 교차점은 연결하지 않습니다.
                mirror = n-1-i
                rising = min(i, mirror) % 2 == 0
                if i > mirror:
                    rising = not rising
                directions = (True, False) if i == mirror else (rising,)
                for upward in directions:
                    a, b = (bottom[i], deck[i+1]) if upward else (deck[i], bottom[i+1])
                    member(a, b, role="대각선 보강재")
        for a, d in zip(bottom, deck):
            member(a, d, role="세로 막대")
        pin(bottom[0])
        roller(bottom[-1])
    else:
        h = cfg["tower_ratio"]*L
        tops = [node(0., h), node(L, h)]
        anchors = [node(-.35*L, 0.), node(1.35*L, 0.)]
        pin(0)
        roller(n)
        for base, top, anchor in zip([0, n], tops, anchors):
            member(base, top, role="주탑")
            member(top, anchor, "cable", "뒤쪽 고정 케이블")
            pin(anchor)
        if kind == "사장교":
            for i in range(1, n):
                if i <= n//2:
                    member(tops[0], deck[i], "cable", "사장 케이블")
                if i >= n//2:
                    member(tops[1], deck[i], "cable", "사장 케이블")
        else:
            # 바닥판의 무게가 수평 방향으로 고르게 실리면 주케이블은 포물선에 가깝습니다.
            # 케이블 자체 무게만 지탱하는 현수선(cosh)과 구분합니다.
            sag = h*cfg["sag_ratio"]
            cable = [tops[0]] + [node(L*i/n, h-4*sag*(i/n)*(1-i/n)) for i in range(1, n)] + [tops[1]]
            for a, b in zip(cable[:-1], cable[1:]):
                member(a, b, "cable", "주케이블")
            for c, d in zip(cable[1:-1], deck[1:-1]):
                member(c, d, "cable", "수직 매다는 줄")

    # 추는 언제나 경간 중앙에 놓습니다. 홀수 칸 트러스는 중앙의 보만
    # 수치적으로 나눕니다. 새 지지대·힌지를 만들거나 좌굴 길이를 줄이지 않습니다.
    load_node = next((i for i in deck if math.isclose(nodes[i][0], L/2, rel_tol=0., abs_tol=1e-12)), None)
    if load_node is None:
        load_node = node(L/2, 0.)
        deck.append(load_node)
        deck.sort(key=lambda i: nodes[i][0])
        for index, e in enumerate(members):
            if e["role"] == "바닥 보" and nodes[e["i"]][0] < L/2 < nodes[e["j"]][0]:
                members[index:index+1] = [dict(e, j=load_node), dict(e, i=load_node)]
                break

    nodes = np.array(nodes, dtype=float)
    main_length = sum(math.dist(nodes[e["i"]], nodes[e["j"]]) for e in members if e["form"] != "cable")
    # '같은 골조 재료량'은 모든 나무 막대의 총부피를 단순교와 같게 맞춥니다.
    # 케이블·종이는 별도입니다. 전체 비용/재료가 같다는 뜻은 아닙니다.
    scale = L/main_length if cfg["equal_volume"] else 1.
    b, h = cfg["member_width"]*math.sqrt(scale), cfg["member_depth"]*math.sqrt(scale)
    A, I = 2*b*h, 2*b*h**3/12
    Iweak = 2*max(b, h)*min(b, h)**3/12
    mass, cable_mass = 0., 0.
    for e in members:
        e["length"] = math.dist(nodes[e["i"]], nodes[e["j"]])
        if e["form"] == "cable":
            e.update(A=2*math.pi*cfg["cable_diameter"]**2/4, I=0., Iweak=0., c=0., E=cfg["cable_E"])
            e["mass"] = e["A"]*e["length"]*1100
            cable_mass += e["mass"]
        else:
            e.update(A=A, I=I, Iweak=Iweak, c=h/2, E=cfg["E"])
            e["mass"] = A*e["length"]*cfg["density"]
        e["le"] = e["le"] or e["length"]
        mass += e["mass"]
    paper_mass = cfg["paper_gsm"]/1000 * L*cfg["width"]
    return dict(kind=kind, nodes=nodes, members=members, fixed=sorted(fixed), deck=deck,
                span=L, load_node=load_node, mass=mass+paper_mass, paper_mass=paper_mass,
                cable_mass=cable_mass, skeleton_mass=mass-cable_mass,
                section_mm=(b*1000, h*1000), braced=braced, weight_mass=cfg["weight_mass"],
                view_y=(min(-5., -12*(cfg["arch_ratio"]+.03)-1), max(10.5, 12*cfg["tower_ratio"]+3)))


def element_matrices(model, e):
    """표준 2절점 요소. 회전 행렬로 부재 좌표를 전체 좌표에 연결합니다."""
    a, b = model["nodes"][[e["i"], e["j"]]]
    length = e["length"]
    c, s = (b-a)/length
    T = np.zeros((6, 6))
    rotation = np.array([[c, s, 0.], [-s, c, 0.], [0., 0., 1.]])
    T[:3, :3] = T[3:, 3:] = rotation
    k = np.zeros((6, 6))
    axial = e["E"]*e["A"]/length
    k[0, 0] = k[3, 3] = axial
    k[0, 3] = k[3, 0] = -axial
    if e["form"] == "beam":
        q = e["E"]*e["I"]/length**3
        k[np.ix_([1, 2, 4, 5], [1, 2, 4, 5])] = q*np.array([
            [12, 6*length, -12, 6*length],
            [6*length, 4*length**2, -6*length, 2*length**2],
            [-12, -6*length, 12, -6*length],
            [6*length, 2*length**2, -6*length, 4*length**2]])
    dofs = np.array([3*e["i"]+j for j in range(3)] + [3*e["j"]+j for j in range(3)])
    return k, T, dofs


def distributed_load(e, T, total):
    """보 전체에 작용하는 수직 하중을 양 끝 힘·모멘트로 옮깁니다."""
    length = e["length"]
    qx, qy = -total/length*T[0, 1], -total/length*T[1, 1]
    f = np.array([qx*length/2, qy*length/2, qy*length**2/12,
                  qx*length/2, qy*length/2, -qy*length**2/12])
    return f, qx, qy


def assemble(model, cfg):
    """강성 K와 자중/추 1개의 하중 벡터를 만듭니다. 자중은 0개에도 남습니다."""
    nd = 3*len(model["nodes"])
    K = np.zeros((nd, nd))
    F = np.zeros((nd, 2))  # 첫 열 자중, 둘째 열 추 하나
    pieces = []
    for e in model["members"]:
        k, T, ds = element_matrices(model, e)
        kg = T.T@k@T
        K[np.ix_(ds, ds)] += kg
        fe = np.zeros((6, 2))
        qxy = np.zeros((2, 2))
        if e["form"] == "beam":
            dead = e["mass"]*GRAVITY if cfg["self_weight"] else 0.
            if e["role"] == "바닥 보":
                share = e["length"]/model["span"]
                dead += model["paper_mass"]*GRAVITY*share if cfg["self_weight"] else 0.
            fe[:, 0], qxy[0, 0], qxy[1, 0] = distributed_load(e, T, dead)
            F[ds] += T.T@fe
        elif cfg["self_weight"]:
            for i in [e["i"], e["j"]]:
                F[3*i+1, 0] -= e["mass"]*GRAVITY/2
        pieces.append((k, T, ds, kg, fe, qxy))
    F[3*model["load_node"]+1, 1] = -cfg["weight_mass"]*GRAVITY
    return K, F, pieces


def stable_solve(K, F, fixed):
    """회전 강성이 없는 핀 절점의 회전은 제외합니다. 불안정한 기구를 인위적인 스프링으로 붙들지 않습니다."""
    diagonal = np.diag(K)
    free = np.array([i for i in range(len(K)) if i not in fixed and diagonal[i] > 1e-13], dtype=int)
    # 힘을 받는 방향으로 강성이 없으면 계산할 수 없습니다.
    unused = [i for i in range(len(K)) if i not in fixed and diagonal[i] <= 1e-13]
    if unused and np.max(np.abs(F[unused]), initial=0.) > 1e-10:
        return None
    if not len(free):
        return None
    # 변위와 회전의 단위 차이를 대각 정규화로 보정한 뒤 안정성을 판정합니다.
    d = np.sqrt(diagonal[free])
    a = K[np.ix_(free, free)]/d[:, None]/d[None, :]
    try:
        np.linalg.cholesky(a - np.eye(len(a))*1e-11)
        u = np.zeros_like(F)
        u[free] = np.linalg.solve(a, F[free]/d[:, None])/d[:, None]
        return u
    except np.linalg.LinAlgError:
        return None


def recover(model, cfg, pieces, u, force, K, active, count):
    """강도, Euler 좌굴, 연결부, 수평 지지, 허용 처짐을 각각 검사합니다."""
    ratios = {"재료가 버티는 힘": 0., "압축 막대의 좌굴": 0., "연결 부분의 힘": 0., "받침·앵커의 수평 힘": 0., "허용 처짐": 0.}
    axial, moments, max_tension = [], [], 0.
    sample_deck = []
    for index, (e, piece) in enumerate(zip(model["members"], pieces)):
        k, T, ds, _, fe, qxy = piece
        ul = T@u[ds]
        if index not in active:
            axial.append(0.)
            moments.append(0.)
            continue
        local_load = fe[:, 0]+count*fe[:, 1]
        end = k@ul-local_load
        qx, qy = qxy[:, 0]+count*qxy[:, 1]
        length = e["length"]
        xs = list(np.linspace(0, length, 9))
        if abs(qy) > 1e-15 and 0 < -end[1]/qy < length:
            xs.append(-end[1]/qy)
        Ns = np.array([-end[0]-qx*x for x in xs])
        Ms = np.array([-end[2]+end[1]*x+qy*x*x/2 for x in xs])
        N = float(np.mean(Ns))
        M = float(max(abs(Ms)))
        axial.append(N)
        moments.append(M)
        if e["form"] == "cable":
            max_tension = max(max_tension, N)
            ratios["재료가 버티는 힘"] = max(ratios["재료가 버티는 힘"], max(N, 0)/(e["A"]*cfg["cable_strength"]))
        else:
            stress_b = abs(Ms)*e["c"]/e["I"]
            use = max(np.max((np.maximum(Ns, 0)/e["A"]+stress_b)/cfg["ft"]),
                      np.max((np.maximum(-Ns, 0)/e["A"]+stress_b)/cfg["fc"]))
            use = max(use, max(stress_b)/cfg["fb"])
            ratios["재료가 버티는 힘"] = max(ratios["재료가 버티는 힘"], float(use))
            critical = math.pi**2*e["E"]*e["Iweak"]/e["le"]**2
            ratios["압축 막대의 좌굴"] = max(ratios["압축 막대의 좌굴"], max(0., -float(min(Ns)))/critical)
        if e["form"] != "beam":
            ratios["연결 부분의 힘"] = max(ratios["연결 부분의 힘"], abs(N)/(2*cfg["joint_force"]))
        if e["role"] == "바닥 보":
            # Hermite 보간 + 자체 무게의 균등하중에 대한 요소 내부 휨 항.
            # 1칸 트러스도 중앙 처짐을 놓치지 않도록 절점 사이를 평가합니다.
            for t in np.linspace(0, 1, 9):
                ux = (1-t)*ul[0]+t*ul[3]
                vy = ((1-3*t*t+2*t**3)*ul[1]+length*(t-2*t*t+t**3)*ul[2]
                      +(3*t*t-2*t**3)*ul[4]+length*(-t*t+t**3)*ul[5])
                vy += qy*length**4*t*t*(1-t)**2/(24*e["E"]*e["I"])
                global_u = T[:2, :2].T @ np.array([ux, vy])
                point = (1-t)*model["nodes"][e["i"]]+t*model["nodes"][e["j"]]
                sample_deck.append([float(point[0]), float(global_u[0]), float(global_u[1])])
    deck_deflection = max(abs(p[2]) for p in sample_deck)
    ratios["허용 처짐"] = deck_deflection/(model["span"]/cfg["deflection_divisor"])
    reactions = K@u-force
    ratios["받침·앵커의 수평 힘"] = max((abs(reactions[i]) for i in model["fixed"] if i % 3 == 0), default=0.)/cfg["foundation_force"]
    governing = max(ratios, key=ratios.get)
    utilization = ratios[governing]
    return dict(stable=True, u=u, axial=axial, moments=moments, ratios=ratios,
                utilization=utilization, reserve=1/utilization if utilization > 1e-12 else None,
                governing=governing, deflection_mm=1000*deck_deflection,
                tension=max_tension, reactions=reactions, deck_samples=sample_deck,
                total_force=-float(np.sum(force[1::3])), active=sorted(active))


@st.cache_data(show_spinner=False, max_entries=100)
def solve_sweep(kind, cfg, braced=True):
    """좌우 대칭 트러스 등 각 모형에 중앙 추 0~20개를 계산. 케이블의 압축은 허용하지 않습니다.

    일단 모든 줄을 넣어 풀고, 눌리는 줄은 제거합니다. 제거한 줄이 다시
    늘어나면 재포함합니다. 수렴/안정성 검사를 통과하지 않으면 수치를 숨깁니다.
    선형 작은 변형 해석이며 초기 장력·대변형·풍하중은 포함하지 않습니다.
    """
    model = build_model(kind, cfg, braced)
    K0, F, pieces = assemble(model, cfg)
    cables = {i for i, e in enumerate(model["members"]) if e["form"] == "cable"}
    all_indices = set(range(len(pieces)))
    initial = stable_solve(K0, F, model["fixed"])
    results = []
    for count in range(21):
        force = F[:, 0]+count*F[:, 1]
        active, K = all_indices.copy(), K0.copy()
        u = initial[:, 0]+count*initial[:, 1] if initial is not None else None
        converged = False
        for _ in range(30):
            if u is None:
                break
            new_active = active.copy()
            for i in cables:
                e = model["members"][i]
                k, T, ds, *_ = pieces[i]
                ul = T@u[ds]
                trial = e["E"]*e["A"]*(ul[3]-ul[0])/e["length"]
                if trial < -1e-8:
                    new_active.discard(i)
                elif trial > 1e-8:
                    new_active.add(i)
            if new_active == active:
                converged = True
                break
            active = new_active
            K = K0.copy()
            for i in all_indices-active:
                ds, kg = pieces[i][2:4]
                K[np.ix_(ds, ds)] -= kg
            solved = stable_solve(K, force[:, None], model["fixed"])
            u = solved[:, 0] if solved is not None else None
        if u is None or not converged:
            results.append(dict(stable=False, utilization=None, reserve=None,
                                deflection_mm=None, tension=None, governing="형태 불안정 / 줄의 평형 미확보"))
        else:
            results.append(recover(model, cfg, pieces, u, force, K, active, count))
    first_bad = next((i for i, r in enumerate(results) if not r["stable"] or r["utilization"] > 1+1e-9), None)
    capacity = 20 if first_bad is None else first_bad-1
    return dict(model=model, results=results, capacity=capacity, censored=first_bad is None,
                first_bad=first_bad)


def capacity_text(sweep):
    if sweep["capacity"] < 0:
        return "0개도 기준 초과" if sweep["results"][0]["stable"] else "안정성 미확보"
    return f"{sweep['capacity']}개 이상" if sweep["censored"] else f"{sweep['capacity']}개"


def base_figure(title, x_range, y_range, height=390):
    fig = go.Figure()
    fig.update_layout(
        title=dict(text=title, font=dict(size=17), x=.025, xanchor="left"),
        height=height, margin=dict(l=25, r=25, t=55, b=25),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False,
        font=dict(family="sans-serif", size=14),
        xaxis=dict(range=x_range, showgrid=True, gridcolor="rgba(128,128,128,.18)", zeroline=False,
                   showticklabels=False, fixedrange=True),
        yaxis=dict(range=y_range, showgrid=True, gridcolor="rgba(128,128,128,.18)", zeroline=False,
                   showticklabels=False, fixedrange=True),
    )
    return fig


def line(fig, x, y, color, width=5, dash="solid", name="", showlegend=False):
    fig.add_trace(go.Scatter(
        x=list(x), y=list(y), mode="lines", name=name,
        line=dict(color=color, width=width, dash=dash), showlegend=showlegend,
        hovertemplate=(name + "<extra></extra>") if name else None,
        hoverinfo=None if name else "skip",
    ))


def label(fig, x, y, text):
    fig.add_annotation(x=x, y=y, text=text, showarrow=False,
                       font=dict(size=12), bgcolor="rgba(0,0,0,0)")


def arrow(fig, start, end, color, magnitude=1, text=""):
    """시작점(ax, ay)에서 끝점(x, y) 방향으로 힘 벡터를 그립니다."""
    if magnitude <= 0:
        return  # 힘이 없을 때 최소 굵기의 가짜 화살표가 남지 않게 합니다.
    fig.add_annotation(
        ax=start[0], ay=start[1], x=end[0], y=end[1],
        axref="x", ayref="y", xref="x", yref="y", text=text,
        showarrow=True, arrowhead=3, arrowsize=1.1,
        arrowwidth=1.5 + min(float(magnitude), 20) * 0.14,
        arrowcolor=color, font=dict(size=11, color=color),
    )


def support(fig, x, y=0, roller=False, depth=1.25, half_width=0.4):
    fig.add_shape(type="rect", x0=x-half_width, x1=x+half_width,
                  y0=y-depth, y1=y-0.16, fillcolor="#e1e8f0", line_width=0)
    fig.add_trace(go.Scatter(x=[x], y=[y-0.1], mode="markers", hoverinfo="skip",
                             marker=dict(symbol="circle" if roller else "triangle-up",
                                         size=15, color=SUPPORT_COLOR), showlegend=False))


def draw_weights(fig, count, y_at, positions, arrow_size=0.8, block_width=0.4, total_force=None,
                 max_visible=MAX_VISIBLE_WEIGHTS):
    """중앙에 간격을 두고 쌓으며, 많은 추는 일부 그림과 정확한 개수로 표시합니다."""
    if count == 0 or not positions:
        return None
    visible = min(count, max_visible)
    for index, x in enumerate(positions[:visible]):
        bottom = float(y_at(x)) + 0.1 + index * WEIGHT_STACK_STEP
        # 사다리꼴 몸체: 윗부분이 좁고 아래가 넓은 실험용 추 모양입니다.
        path = (
            f"M {x-block_width/2},{bottom} "
            f"L {x+block_width/2},{bottom} "
            f"L {x+block_width*.30},{bottom+WEIGHT_BODY_HEIGHT} "
            f"L {x-block_width*.30},{bottom+WEIGHT_BODY_HEIGHT} Z"
        )
        fig.add_shape(type="path", path=path, name="추 몸체",
                      line=dict(color="#52647c", width=1), fillcolor="rgba(0,0,0,0)")
        fig.add_shape(type="circle", name="추 고리",
                      x0=x-block_width*.15, x1=x+block_width*.15,
                      y0=bottom+WEIGHT_BODY_HEIGHT, y1=bottom+WEIGHT_BODY_HEIGHT+WEIGHT_RING_HEIGHT,
                      line=dict(color="#8297aa", width=2), fillcolor="rgba(0,0,0,0)")
        fig.add_layout_image(
            source=WEIGHT_IMAGE_SOURCE, xref="x", yref="y", x=x, y=bottom,
            sizex=block_width, sizey=WEIGHT_BODY_HEIGHT+WEIGHT_RING_HEIGHT,
            xanchor="center", yanchor="bottom", sizing="stretch", layer="above",
        )
    x = positions[0]
    stack_bottom = float(y_at(x)) + .1
    stack_top = stack_bottom + (visible-1)*WEIGHT_STACK_STEP + WEIGHT_BODY_HEIGHT + WEIGHT_RING_HEIGHT
    arrow_top = stack_top + arrow_size + .3
    fig.add_annotation(
        name="weight_count", x=x+block_width/2+.25, y=arrow_top-.35,
        text=f"추 {count}개" + (f"<br>({visible}개만 표시)" if count > visible else ""),
        xanchor="left", showarrow=False, font=dict(size=13, color="#243447"),
        bgcolor="rgba(251,252,254,.94)", borderpad=3,
    )
    arrow(fig, (x, arrow_top), (x, stack_top+.15), LOAD_COLOR,
          count if total_force is None else total_force)
    return arrow_top


def weight_positions(count, left, right):
    return [0.5*(left+right)] * count


def chart(fig, key):
    st.plotly_chart(fig, width="stretch", config=PLOT_CONFIG, theme="streamlit", key=key)


def highlight_learning_text(text):
    """원문은 유지하고 학습 용어만 강조합니다. 입력 문자열은 HTML로 이스케이프합니다."""
    terms = {
        "압축력": "force-compression", "압축": "force-compression",
        "인장력": "force-tension", "인장": "force-tension", "처짐": "concept-focus",
        "휨": "", "삼각형": "", "주케이블": "", "케이블": "", "주탑": "", "행어": "",
    }
    def decorate(match):
        term = match.group()
        style = terms[term]
        if style == "concept-focus":
            return f'<u class="concept-focus">{term}</u>'
        return f'<span class="{style}">{term}</span>' if style else f"<strong>{term}</strong>"
    return re.sub("|".join(sorted(terms, key=len, reverse=True)), decorate, escape(text))


def learning_icon(name):
    # 글꼴에 따라 이모지가 네모로 보이지 않도록 같은 모양의 그림 아이콘을 사용합니다.
    paths = {
        "book": '<path d="M12 6C9 3 5 3 2 4v15c3-1 7-1 10 1 3-2 7-2 10-1V4c-3-1-7-1-10 2ZM12 6v14"/>',
        "scale": '<path d="M12 3v18M7 21h10M4 7h16M5 7l-3 8h6L5 7Zm14 0-3 8h6l-3-8Z"/>',
        "search": '<circle cx="10" cy="10" r="7"/><path d="m15 15 6 6"/>',
        "bulb": '<path d="M8 16v-2a6 6 0 1 1 8 0v2H8Zm1 3h6m-5 3h4"/>',
        "arrow": '<path d="M3 12h18m-7-7 7 7-7 7"/>',
    }
    return f'<svg class="learning-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" focusable="false">{paths[name]}</svg>'


def load_path_diagram(path, kind=None):
    branches = []
    for branch_index, branch in enumerate(path.split(" / ")):
        nodes = []
        for index, node in enumerate(branch.split(" → ")):
            arrow_html = '<span class="path-arrow" aria-hidden="true">→</span>' if index else ""
            numbers = part_badges(kind, node)
            meanings = part_meanings(kind, node)
            if meanings:
                note_id = f"path-part-{kind}-{branch_index}-{index}"
                explanation = '<br>'.join(f'<strong>{escape(name)}</strong> — {escape(meaning)}'
                                         for name, meaning in meanings)
                node_html = (
                    '<details class="path-node" name="bridge-part-meaning">'
                    f'<summary aria-label="{escape(node)} 설명" aria-describedby="{note_id}">'
                    f'{numbers}{escape(node)}<span class="path-explanation" role="tooltip" '
                    f'id="{note_id}">{explanation}</span></summary></details>'
                )
            else:
                node_html = f'<span class="path-node">{escape(node)}</span>'
            nodes.append(f'<div class="path-segment">{arrow_html}{node_html}</div>')
        branches.append('<div class="load-path">' + " ".join(nodes) + '</div>')
    return '<div class="load-path-help" aria-label="하중이 전달되는 과정">' + "".join(branches) + '</div>'


def force_primer():
    """인장·압축·처짐을 설명하는 개념 그림입니다. 계산 결과 그림과 구분합니다."""
    compression = '<rect x="30" y="35" width="100" height="25" rx="3" fill="currentColor" opacity=".1"/><path d="M8 48H58 M47 40L58 48L47 56 M152 48H102 M113 40L102 48L113 56" stroke="#982f2f" stroke-width="3" fill="none"/>'
    tension = '<rect x="30" y="35" width="100" height="25" rx="3" fill="currentColor" opacity=".1"/><path d="M62 48H8 M19 40L8 48L19 56 M98 48H152 M141 40L152 48L141 56" stroke="#185797" stroke-width="3" fill="none"/>'
    deflection = '<path d="M15 33H145" stroke="currentColor" opacity=".45" stroke-width="2" stroke-dasharray="4 4"/><path d="M15 33Q80 66 145 33" stroke="currentColor" stroke-width="4" fill="none"/><path d="M80 5V28 M73 20L80 28L87 20" stroke="#c56a20" stroke-width="3" fill="none"/><path d="M80 34V49" stroke="currentColor" stroke-width="2" stroke-dasharray="2 2"/>'
    items = []
    for name, style, drawing, description in [
        ("압축", "force-compression", compression, "안쪽으로 누르는 힘"),
        ("인장", "force-tension", tension, "바깥쪽으로 당기는 힘"),
        ("처짐", "concept-focus", deflection, "하중을 받아 아래로 변한 거리"),
    ]:
        items.append(f'<section><svg viewBox="0 0 160 85" aria-hidden="true" focusable="false">{drawing}</svg><div><strong class="{style}">{name}</strong><small>{description}</small></div></section>')
    st.markdown('<div class="force-primer">' + "".join(items) + '</div>', unsafe_allow_html=True)


def triangular_truss_diagram():
    """꼭대기에 수직 하중을 받는 정삼각형 핀 트러스의 개념 그림입니다."""
    top = (200., 230. - 100. * math.sqrt(3))
    left, right = (100., 230.), (300., 230.)

    def force_arrow(start, end, color):
        dx, dy = end[0] - start[0], end[1] - start[1]
        length = math.hypot(dx, dy)
        ux, uy = dx / length, dy / length
        wings = [(end[0] - 9*ux + sign*5*uy, end[1] - 9*uy - sign*5*ux)
                 for sign in (-1, 1)]
        return (f'<path d="M{start[0]},{start[1]} L{end[0]},{end[1]} '
                f'M{wings[0][0]},{wings[0][1]} L{end[0]},{end[1]} '
                f'L{wings[1][0]},{wings[1][1]}" stroke="{color}" '
                'stroke-width="3" fill="none" stroke-linecap="round" stroke-linejoin="round"/>')

    members = []
    for start, end, color, compression in [
        (left, top, COMPRESSION, True), (right, top, COMPRESSION, True),
        (left, right, TENSION, False),
    ]:
        members.append(f'<path d="M{start[0]},{start[1]} L{end[0]},{end[1]}" '
                       f'stroke="{color}" stroke-width="9" opacity=".3"/>')
        def point(t):
            return (start[0] + t*(end[0]-start[0]), start[1] + t*(end[1]-start[1]))
        # 압축은 가운데로, 인장은 양끝으로 향하는 화살표로 표시합니다.
        for a, b in ([(.18, .43), (.82, .57)] if compression else [(.43, .18), (.57, .82)]):
            members.append(force_arrow(point(a), point(b), color))

    return (
        '<svg class="triangular-truss-diagram" viewBox="0 0 400 310" '
        'style="display:block;width:100%;max-width:360px;height:auto;margin:.5rem auto" '
        'role="img" aria-labelledby="truss-concept-title truss-concept-description">'
        '<title id="truss-concept-title">정삼각형 하나로 보는 트러스의 원리</title>'
        '<desc id="truss-concept-description">꼭대기에 아래로 하중을 가하면 기울어진 두 막대는 압축, '
        '아랫막대는 인장을 받습니다. 왼쪽 핀 받침과 오른쪽 롤러 받침이 하중을 지반으로 전달합니다. '
        '막대 위의 화살표는 막대에 작용하는 인장과 압축의 방향입니다.</desc>'
        + ''.join(members)
        + force_arrow((200, 10), (200, top[1]-9), LOAD_COLOR)
        + '<g fill="currentColor" font-size="20" text-anchor="middle">'
        '<text x="241" y="31">하중</text>'
        '<text x="98" y="142">압축</text><text x="302" y="142">압축</text>'
        '<text x="200" y="260">인장</text>'
        '<text x="200" y="301">양쪽 받침 → 지반</text></g>'
        '<g stroke="currentColor" stroke-width="1.5" fill="none" opacity=".7">'
        '<path d="M100 235 88 255H112Z M82 260H118 M300 235 288 252H312Z M282 266H318"/>'
        '<circle cx="294" cy="258" r="4"/><circle cx="306" cy="258" r="4"/></g>'
        + ''.join(f'<circle cx="{x}" cy="{y}" r="4" fill="currentColor"/>'
                  for x, y in (left, top, right))
        + '</svg>'
    )


def bridge_example(kind):
    """검증한 로컬 사진만 표시하고 누락·손상 시에도 사례 설명을 유지합니다."""
    example = BRIDGE_EXAMPLES[kind]
    with st.container(key="bridge_example", width=560):
        try:
            path = Path(__file__).resolve().parent / "assets" / "bridges" / example["file"]
            with Image.open(path) as photo:
                photo.load()
                st.image(photo, width="stretch")
        except (OSError, ValueError):
            st.caption("사진을 불러올 수 없어 실제 사례를 글로 안내합니다.")
        st.markdown(f"**{example['name']}**")
        st.write(example["description"])
        if example.get("note"):
            st.caption(example["note"])
        st.caption(
            f"사진: {example['author']} · [Wikimedia Commons 원본]({example['source']}) · "
            f"[{example['license']}]({example['license_url']}) · 잘라내기·크기 조정·JPEG 압축"
        )


def bridge_chapter_intro(kind):
    guide = BRIDGE_GUIDES[kind]
    with st.container(key="bridge_intro"):
        st.subheader("① 교량 구조 알아보기")
        st.markdown(f"<p><strong>{escape(guide['title'])}</strong> · {highlight_learning_text(guide['shape'])}</p>", unsafe_allow_html=True)
        principle, load_path = st.columns([1.1, 1], gap="large")
        principle.markdown(f'<p>{learning_icon("bulb")} <strong>하중을 견디는 원리</strong></p><p>{highlight_learning_text(guide["principle"])}</p>', unsafe_allow_html=True)
        principle.markdown(f'<p><strong>활용 상황</strong><br>{escape(guide["use"])}</p>', unsafe_allow_html=True)
        if kind == "트러스교":
            principle.markdown(triangular_truss_diagram(), unsafe_allow_html=True)
            principle.caption("꼭대기를 누르면 기울어진 두 막대는 압축, 아랫막대는 인장을 받아요. 아랫막대는 양쪽이 벌어지려는 것을 잡아 줍니다.")
            principle.caption("막대를 핀으로 연결한 개념 그림입니다. 화살표는 막대에 작용하는 힘의 방향이며, 아래 하중 실험의 계산 결과와는 별개입니다.")
            principle.caption("하중 실험의 대각재는 중앙 세로축을 기준으로 좌우 대칭입니다. 홀수 칸의 중앙은 X자로 보강하며, X 교차점은 연결하지 않습니다.")
        load_path.markdown(
            f'<p>{learning_icon("arrow")} <strong>하중이 전달되는 과정</strong></p>'
            + bridge_anatomy_diagram(kind) + load_path_diagram(guide["load_path"], kind)
            + '<details class="bridge-glossary"><summary>용어 전체 보기</summary>'
            + bridge_parts_legend(kind) + '</details>', unsafe_allow_html=True,
        )
        load_path.caption("모형 그림과 같은 번호예요. 경로의 이름에 마우스를 올리거나 눌러 뜻을 확인하세요.")


def bridge_application(kind):
    """하중 실험에서 관찰한 구조를 실제 교량 사례와 연결합니다."""
    guide = BRIDGE_GUIDES[kind]
    with st.container(key="bridge_application"):
        with st.expander("실제 적용 조건과 사례", expanded=False):
            st.write(guide["limit"])
            st.caption(f"사례: {guide['examples']}. 실제 교량은 경간·지반·시공 조건을 함께 고려하여 선택합니다.")
            st.caption("모형의 상판 보도 하중을 나누어 받습니다. 구조 소개의 하중 전달 과정은 각 구조의 대표적인 경로입니다.")
            bridge_example(kind)


def data_figure(title, x_title, y_title, height=340):
    fig = go.Figure()
    fig.update_layout(
        title=dict(text=title, font=dict(size=17), x=.025, xanchor="left"), height=height,
        font=dict(family="sans-serif", size=14),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=55, r=25, t=65, b=60),
        xaxis=dict(title=x_title, gridcolor="rgba(128,128,128,.12)", tickfont=dict(size=13)),
        yaxis=dict(title=y_title, gridcolor="rgba(128,128,128,.12)", rangemode="tozero", tickfont=dict(size=13)),
        legend=dict(orientation="h", y=1.16, x=0, font=dict(size=13)),
    )
    return fig


# ── 3. 계산한 변위와 힘을 그대로 그림에 연결합니다 ────────────────────
def deck_half_depth(kind):
    # 케이블교는 상판·행어·주케이블 사이 공간을 구분할 수 있게 얇게 표시합니다.
    return .25 if kind in ("사장교", "현수교") else BEAM_HALF_DEPTH


def member_display_points(model, member, shown):
    points = shown[[member["i"], member["j"]]].copy()
    if model["kind"] in ("사장교", "현수교") and member["role"] != "바닥 보":
        # 계산 절점은 보의 중심선입니다. 그림에서는 상판 윗면에 케이블을 연결합니다.
        points[:, 1] += deck_half_depth(model["kind"])
    return points


def support_display_point(model, node_id, scale):
    x, y = model["nodes"][node_id] * scale
    depth = deck_half_depth(model["kind"])
    if node_id in model["deck"]:
        y -= depth
        if model["kind"] in ("사장교", "현수교"):
            y -= TOWER_BASE_DEPTH
    elif model["kind"] in ("사장교", "현수교"):
        y += depth  # 뒤쪽 케이블의 앵커도 같은 그림 좌표에 맞춥니다.
    return x, y


def draw_tower_foundations(fig, model, shown, scale):
    """상판 아래 주탑과 기초 연결을 개념도로 보완합니다. 해석 부재는 추가하지 않습니다."""
    if model["kind"] not in ("사장교", "현수교"):
        return
    for e in model["members"]:
        if e["role"] != "주탑":
            continue
        base = e["i"]
        x, bottom = support_display_point(model, base, scale)
        top = shown[base, 1] + deck_half_depth(model["kind"])
        fig.add_trace(go.Scatter(
            x=[x-.18, x+.18, x+.18, x-.18, x-.18], y=[bottom, bottom, top, top, bottom],
            mode="lines", fill="toself", fillcolor="rgba(129,149,167,.38)",
            line=dict(color=SUPPORT_COLOR, width=1.5), showlegend=False,
            name="주탑 하부 · 기초 연결", meta=dict(illustration="tower_foundation", base_node=base),
            hovertemplate="주탑 하부 → 기초<br>하중 전달 경로를 위한 개념 표시<extra></extra>",
        ))


def draw_deck_beam(fig, xs, ys, name, bending=False, left_end=False, right_end=False, outline=SUPPORT_COLOR, depth=BEAM_HALF_DEPTH):
    """계산된 중심선을 따라 긴 사각형 보의 윗부분·아랫부분을 채웁니다."""
    xs, ys = np.asarray(xs), np.asarray(ys)
    for upper in (True, False):
        edge = ys + (depth if upper else -depth)
        region = "윗부분 · 압축" if upper else "아랫부분 · 인장"
        fill = ("rgba(224,82,82,.32)" if upper else "rgba(40,123,209,.32)") if bending else "rgba(129,149,167,.30)"
        fig.add_trace(go.Scatter(
            x=list(xs) + list(xs[::-1]), y=list(ys) + list(edge[::-1]),
            mode="lines", line=dict(width=0), fill="toself", fillcolor=fill,
            name=f"{name} · {region}" if bending else name,
            hovertemplate=(f"{name}<br>{region}" if bending else name) + "<extra></extra>",
            showlegend=False,
        ))
    line(fig, xs, ys + depth, outline, 1.8)
    line(fig, xs, ys - depth, outline, 1.8)
    line(fig, xs, ys, SUPPORT_COLOR, 1., dash="dot")
    for end, show in ((0, left_end), (-1, right_end)):
        if show:
            line(fig, [xs[end], xs[end]], [ys[end] - depth, ys[end] + depth], outline, 1.8)


def force_arrow_pair(fig, center, direction, tension, length, name):
    """인장은 바깥쪽, 압축은 중심 쪽을 향하는 두 화살표로 표시합니다."""
    center, direction = np.asarray(center), np.asarray(direction)
    for side in (-1, 1):
        inner = center + side * length * .24 * direction
        outer = center + side * length * .5 * direction
        start, end = (inner, outer) if tension else (outer, inner)
        fig.add_annotation(
            name=f"{name}_{side}", ax=float(start[0]), ay=float(start[1]),
            x=float(end[0]), y=float(end[1]), axref="x", ayref="y", xref="x", yref="y",
            text="", showarrow=True, arrowhead=3, arrowsize=.85, arrowwidth=2.5,
            arrowcolor=TENSION if tension else COMPRESSION,
        )


def add_member_force_arrows(fig, model, result, shown, scale, within, amplify):
    """대표 부재의 힘을 계산 부호에 맞춘 화살표로 표시합니다."""
    if not result["stable"]:
        return
    # (부재 역할, 경간 내 위치, 글상자의 가로·세로 간격, 축력 부호 필터)
    specs = {
        "단순교": [("바닥 보", .5, -45, -65, 0)],
        "아치교": [("바닥 보", .2, 35, -65, 0), ("아치", .3, -55, 55, 0),
                  ("세로 지지대", .75, 55, 35, 0)],
        "트러스교": [("바닥 보", .2, 0, -55, 0), ("아래 가로 막대", .3, 0, 50, 0),
                    ("세로 막대", 1., 18, 50, 0)],
        "사장교": [("바닥 보", .5, 0, 65, 0), ("사장 케이블", .3, -35, -55, 0),
                  ("주탑", 0., -55, 25, 0)],
        "현수교": [("바닥 보", .5, 0, 65, 0), ("주케이블", .4, -25, -65, 0),
                  ("수직 매다는 줄", .75, 55, 20, 0), ("주탑", 0., -55, 25, 0)],
    }[model["kind"]]
    used = set()
    for role, fraction, offset_x, offset_y, sign in specs:
        candidates = [i for i, e in enumerate(model["members"])
                      if i not in used and (e["role"] == role if role else e["form"] == "rod")
                      and (sign == 0 or sign * result["axial"][i] > 1e-7)]
        # 자중·추가 모두 없을 때도 대표 막대의 위치와 무하중 상태를 알려 줍니다.
        if not candidates and role is None:
            fallback = "아래 가로 막대" if sign > 0 else "대각선 보강재"
            candidates = [i for i, e in enumerate(model["members"]) if e["role"] == fallback and i not in used]
        if not candidates:
            continue
        index = min(candidates, key=lambda i: abs(
            (model["nodes"][model["members"][i]["i"], 0]
             + model["nodes"][model["members"][i]["j"], 0]) / 2 / model["span"] - fraction))
        used.add(index)
        e = model["members"][index]
        axial, moment = result["axial"][index], result["moments"][index]
        color = TENSION if axial > 1e-7 else COMPRESSION if axial < -1e-7 else SUPPORT_COLOR
        member_name = {"바닥 보": "상판 아래 보", "수직 매다는 줄": "행어(세로줄)"}.get(e["role"], e["role"])
        text = member_name + (" · 휨" if moment > 1e-9 else "")
        text_anchor = "center"
        if model["kind"] == "트러스교":
            # 중앙 하중을 피하고, 끝 세로 막대와 아래 가로 막대를 따로 가리킵니다.
            text_anchor = "right" if offset_y < 0 or e["role"] == "세로 막대" else "center"
        elif e["role"] == "바닥 보" and offset_y < 0:
            # 중앙에 쌓인 추를 가리지 않도록 위쪽 보 설명은 왼쪽에 둡니다.
            text = member_name + ("<br>휨" if moment > 1e-9 else "")
            offset_x = 0 if model["kind"] == "아치교" else (-24 if fraction <= .5 else -44)
            text_anchor = "right"
        if abs(axial) <= 1e-7 and moment <= 1e-9:
            text += "<br>힘 거의 없음"
        endpoints = member_display_points(model, e, shown)
        x, y = endpoints.mean(axis=0)
        if e["role"] == "바닥 보" and within:
            original_x = (model["nodes"][e["i"], 0] + model["nodes"][e["j"], 0]) / 2
            sample = min(result["deck_samples"], key=lambda p: abs(p[0] - original_x))
            # 구조 그림에서 사용하는 보간 곡선 위에 안내선을 연결합니다.
            x, y = (sample[0] + amplify * sample[1]) * scale, sample[2] * scale * amplify
        direction = endpoints[1] - endpoints[0]
        member_length = float(np.linalg.norm(direction))
        direction = direction / member_length
        perpendicular = np.array([-direction[1], direction[0]])
        if abs(axial) > 1e-7:
            offset = deck_half_depth(model["kind"]) / 2 if e["role"] == "바닥 보" else .15
            force_arrow_pair(fig, np.array([x, y]) + offset * perpendicular, direction,
                             axial > 0, min(1.8, member_length * .72), f"member_axial_{index}")
        if model["kind"] == "단순교" and moment > 1e-9:
            force_arrow_pair(fig, [x, y + BEAM_HALF_DEPTH / 2], [1., 0.], False, 1.8, f"member_bending_{index}_upper")
            force_arrow_pair(fig, [x, y - BEAM_HALF_DEPTH / 2], [1., 0.], True, 1.8, f"member_bending_{index}_lower")
        fig.add_annotation(
            name=f"member_name_{index}", x=float(x), y=float(y), xref="x", yref="y",
            ax=offset_x, ay=offset_y, axref="pixel", ayref="pixel", text=text,
            showarrow=True, arrowhead=0, arrowwidth=1.2, arrowcolor=color,
            bgcolor="rgba(255,255,255,.96)", bordercolor=color, borderwidth=1, borderpad=4,
            font=dict(size=13, color="#1f2937"), align="center", xanchor=text_anchor,
        )


def draw_unbraced_truss_example(fig, model, scale):
    """막대 길이와 받침 위치를 유지하는 기구 변형의 설명용 예시입니다."""
    original = model["nodes"]
    example = original.copy()
    deck = model["deck"]
    height = original[deck[0], 1] - min(original[:, 1])
    angle = math.radians(35)  # 관찰을 위한 고정 각도이며 계산된 변위가 아닙니다.
    example[deck, 0] += height * math.sin(angle)
    example[deck, 1] -= height * (1 - math.cos(angle))
    for index, member in enumerate(model["members"]):
        endpoints = [member["i"], member["j"]]
        if np.allclose(example[endpoints], original[endpoints]):
            continue
        points = example[endpoints] * scale
        fig.add_trace(go.Scatter(
            x=points[:, 0], y=points[:, 1], mode="lines",
            line=dict(color="#596778", width=2.5, dash="dash"), showlegend=False,
            name="사각형의 형태 변화 예시",
            meta=dict(illustration="unbraced_shear_example", member_index=index),
            hovertemplate="점선: 사각형의 형태 변화 예시<br>계산된 변형량이 아닙니다.<extra></extra>",
        ))


def structure_figure(sweep, count, amplify, title=None, show_forces=False):
    model, r = sweep["model"], sweep["results"][count]
    xy = model["nodes"]
    L, scale = model["span"], 12/model["span"]
    kind = model["kind"]
    depth = deck_half_depth(kind)
    fig = base_figure(title or f"{kind} · 추 {count}개", [-5.2, 17.2], model["view_y"], 410)
    fig.update_yaxes(scaleanchor="x", scaleratio=1, showgrid=False)
    fig.update_xaxes(showgrid=False)
    within = r["stable"] and r["utilization"] <= 1+1e-9
    # 기준을 넘으면 마지막 기준 이내 상태를 유지합니다. 붕괴 후 해석은 하지 않습니다.
    display_count = count if within else next(
        (i for i in range(count, -1, -1) if sweep["results"][i]["stable"]
         and sweep["results"][i]["utilization"] <= 1+1e-9), None)
    has_valid_shape = display_count is not None
    if display_count is None:
        display_count = next((i for i in range(count + 1) if sweep["results"][i]["stable"]), None)
    display_r = sweep["results"][display_count] if display_count is not None else r
    shape_available = display_r["stable"]
    fig.update_layout(meta=dict(display_count=display_count, requested_count=count, frozen=not within))
    u = display_r["u"].reshape(-1, 3)[:, :2] if shape_available else np.zeros_like(xy)
    shown = (xy+amplify*u)*scale
    draw_tower_foundations(fig, model, shown, scale)
    for index, e in enumerate(model["members"]):
        a, b = e["i"], e["j"]
        N = display_r["axial"][index] if shape_available else 0.
        color = TENSION if N > 1e-7 else COMPRESSION if N < -1e-7 else SUPPORT_COLOR
        force_name = "인장" if N > 1e-7 else "압축" if N < -1e-7 else "축력 거의 없음"
        base_width = 6. if e["form"] == "beam" else 4.8 if e["role"] == "대각선 보강재" else 3.
        width = base_width + min(abs(N), 10)*.22
        if e["form"] == "cable":
            color = TENSION if N > 1e-7 else SUPPORT_COLOR
        if shape_available and amplify > 0:
            original = member_display_points(model, e, xy * scale)
            line(fig, original[:, 0], original[:, 1], "rgba(128,128,128,.3)", 1)
        if e["role"] == "바닥 보" and shape_available:
            ps = [p for p in display_r["deck_samples"] if xy[a, 0]-1e-10 <= p[0] <= xy[b, 0]+1e-10]
            xs = [(p[0]+amplify*p[1])*scale for p in ps]
            ys = [amplify*p[2]*scale for p in ps]
        else:
            endpoints = member_display_points(model, e, shown)
            xs, ys = endpoints[:, 0], endpoints[:, 1]
        moment = display_r["moments"][index] if shape_available else 0.
        name = f"{e['role']} · 양옆 합계 {force_name} {abs(N):.3f} N · 휨 {moment:.4f} N·m"
        if e["role"] == "바닥 보":
            draw_deck_beam(fig, xs, ys, name, bending=kind == "단순교" and moment > 1e-9,
                           left_end=a == model["deck"][0], right_end=b == model["deck"][-1], outline=color, depth=depth)
        else:
            line(fig, xs, ys, color, width, name=name)
    shear_example = kind == "트러스교" and not model["braced"] and not shape_available
    if shear_example:
        draw_unbraced_truss_example(fig, model, scale)
    for node_id in sorted(set(i//3 for i in model["fixed"] if i % 3 == 1)):
        x, y = support_display_point(model, node_id, scale)
        support(fig, x, y, roller=3*node_id not in model["fixed"], depth=.65, half_width=.28)
    if shape_available:
        for dof in model["fixed"]:
            value = display_r["reactions"][dof]
            if abs(value) < 1e-7 or dof % 3 == 2:
                continue
            x, y = support_display_point(model, dof//3, scale)
            distance = math.copysign(min(1.7, abs(value)*.35), value)
            end = (x+distance, y) if dof % 3 == 0 else (x, y+distance)
            arrow(fig, (x, y), end, SUPPORT_COLOR, abs(value))
    load_x, load_y = 6., depth
    if shape_available:
        center = min(display_r["deck_samples"], key=lambda p: abs(p[0]-L/2))
        load_x = (center[0] + amplify*center[1])*scale
        load_y = center[2]*amplify*scale + depth
    live_force = count*model["weight_mass"]*GRAVITY
    visible_limit = MAX_VISIBLE_WEIGHTS
    if kind == "현수교":
        # 중앙 주케이블 아래의 실제 그림 공간에 맞춰 추 아이콘 수만 줄입니다.
        cable_ys = []
        for e in model["members"]:
            if e["role"] == "주케이블":
                points = member_display_points(model, e, shown)
                if points[0, 0] <= load_x <= points[1, 0]:
                    cable_ys.append(float(np.interp(load_x, points[:, 0], points[:, 1])))
        if cable_ys:
            space = min(cable_ys) - load_y - .25 - WEIGHT_BODY_HEIGHT - WEIGHT_RING_HEIGHT
            visible_limit = max(1, min(MAX_VISIBLE_WEIGHTS, 1 + math.floor(space / WEIGHT_STACK_STEP)))
    load_top = draw_weights(fig, count, lambda x: load_y, weight_positions(count, load_x, load_x),
                            arrow_size=1.2+min(live_force, 5)*.2, block_width=.72,
                            total_force=live_force, max_visible=visible_limit)
    fig.layout.meta.update(load_x=load_x, load_y=load_y, load_top=load_top,
                           visible_weight_count=min(count, visible_limit))
    status_y = max(9., load_top+1.2) if load_top is not None else 9.
    fig.update_yaxes(range=[model["view_y"][0], max(model["view_y"][1], status_y+1.3)])
    if show_forces:
        add_member_force_arrows(fig, model, display_r, shown, scale, shape_available, amplify)
        fig.add_annotation(
            name="force_legend", x=.5, y=1.12, xref="paper", yref="paper", yanchor="bottom",
            text=f'<span style="color:{COMPRESSION}">→ ← 압축: 안으로 누름</span><br>'
                 f'<span style="color:{TENSION}">← → 인장: 밖으로 당김</span>',
            showarrow=False, font=dict(size=13, color="#1f2937"),
            bgcolor="rgba(255,255,255,.96)", borderpad=4,
        )
    if not within and has_valid_shape:
        label(fig, 6, status_y, f"기준 초과 · 추 {display_count}개 상태 유지<br>추는 현재 {count}개 · 붕괴 후 예측 아님")
    elif not within and shape_available:
        label(fig, 6, status_y, f"처음부터 기준 초과 · {display_count}개 참고 모습 유지<br>추는 현재 {count}개 · 붕괴 후 예측 아님")
    elif not r["stable"]:
        label(fig, 6, status_y, "점선: 사각형이 기울어지는 예시<br>실제 변형량을 계산한 결과가 아닙니다." if shear_example
              else "현재 연결 상태에서는 형태를 유지하기 어렵습니다.<br>변형 없이 뼈대를 표시합니다.")
    else:
        label(fig, 6, status_y, f"변형 {amplify}배 확대 · 실제 처짐은 아래 수치로 확인")
    label(fig, 6, model["view_y"][0]+.5, f"받침 사이 {L*100:.0f} cm")
    return fig


def experiment_figure(sweep, count, amplify, zoom):
    """실험 화면의 카메라만 확대합니다. 부재 좌표와 계산값은 그대로 둡니다."""
    fig = structure_figure(sweep, count, amplify, show_forces=True)
    ys = [float(y) for trace in fig.data for y in trace.y if y is not None]
    bottom, top = min(ys) - 2.2, max(ys) + (3.5 if count else 2.7)
    if sweep["model"]["kind"] in ("사장교", "현수교"):
        deck_top = max(float(y) for trace in fig.data if (trace.name or "").startswith("바닥 보") for y in trace.y)
        bottom, top = min(ys) - 1.5, max(max(ys) + 1.25, deck_top + 3.5)
    if fig.layout.meta["load_top"] is not None:
        top = max(top, fig.layout.meta["load_top"] + .8)
    if sweep["model"]["kind"] == "트러스교":
        bottom = min(ys) - 3.0  # 아래 가로 막대 설명과 경간 안내 사이의 여백
    center_y = (bottom + top) / 2
    half_y = max(7., top - bottom) / (2 * zoom)
    # 짧은 교량은 화면을 더 넓게 사용하고, 케이블교는 뒤쪽 앵커까지 보입니다.
    half_x = (11.2 if sweep["model"]["kind"] in ("사장교", "현수교") else 8.) / zoom
    fig.update_layout(height=500 if sweep["model"]["kind"] == "트러스교" else 440,
                      dragmode="pan", margin=dict(l=18, r=18, t=105, b=38),
                      title=dict(y=.98, yanchor="top"))
    fig.update_xaxes(range=[6 - half_x, 6 + half_x], fixedrange=False)
    fig.update_yaxes(range=[center_y - half_y, center_y + half_y], fixedrange=False)
    if sweep["model"]["kind"] == "트러스교":
        for annotation in fig.layout.annotations:
            if (annotation.name or "").startswith("member_name_"):
                if zoom > 1 and annotation.text.startswith("상판 아래 보"):
                    annotation.text = annotation.text.replace(" · 휨", "<br>휨")
                if zoom < 1 and annotation.text.startswith("세로 막대"):
                    annotation.ay += round((1-zoom)*100)
    # 상태와 경간 안내는 카메라 확대에도 화면 가장자리에 남겨 둡니다.
    fig.layout.annotations[-2].update(x=.5, y=1.02, xref="paper", yref="paper", yanchor="bottom",
                                      font=dict(size=13, color="#243447"))
    if not fig.layout.meta["frozen"]:
        fig.layout.annotations[-2].update(text=f"변형 {amplify}배 확대")
    fig.layout.annotations[-1].update(x=.5, y=-.02, xref="paper", yref="paper", yanchor="top")
    return fig


def change_bridge_zoom(step):
    current = st.session_state.get("bridge_zoom_percent", 100)
    st.session_state["bridge_zoom_percent"] = 100 if step == 0 else max(50, min(250, current + step))


def overview_figure(sweeps, amplify):
    titles = [f"{k} · {capacity_text(sweeps[k])}" for k in BRIDGE_NAMES]
    fig = make_subplots(rows=3, cols=2, specs=[[{}, {}], [{"colspan": 2}, None], [{}, {}]],
                        subplot_titles=titles, vertical_spacing=.12, horizontal_spacing=.08,
                        row_heights=[.3, .4, .3])
    positions = [(1, 1), (1, 2), (2, 1), (3, 1), (3, 2)]
    for idx, (kind, (row, col)) in enumerate(zip(BRIDGE_NAMES, positions), 1):
        sweep = sweeps[kind]
        count = max(0, sweep["capacity"])
        source = structure_figure(sweep, count, amplify)
        for trace in source.data:
            fig.add_trace(trace, row=row, col=col)
        for shape in source.layout.shapes or []:
            fig.add_shape(shape.to_plotly_json(), row=row, col=col)
        for picture in source.layout.images or []:
            fig.add_layout_image(picture.to_plotly_json(), row=row, col=col)
        xref, yref = ("x" if idx == 1 else f"x{idx}"), ("y" if idx == 1 else f"y{idx}")
        for ann in source.layout.annotations[:-2]:
            data = ann.to_plotly_json()
            data.update(xref=xref, yref=yref)
            if data.get("showarrow"):
                data.update(axref=xref, ayref=yref)
            fig.add_annotation(data)
        # 안내 문구를 좌표 범위에서 분리해 구조 자체가 충분히 크게 보이게 합니다.
        state = source.layout.annotations[-2].text if source.layout.meta["frozen"] else f"변형 {amplify}배 확대"
        fig.add_annotation(x=.5, y=.98, xref=f"{xref} domain", yref=f"{yref} domain",
                           text=state, showarrow=False, yanchor="top", font=dict(size=13, color="#243447"))
        fig.add_annotation(x=.5, y=.02, xref=f"{xref} domain", yref=f"{yref} domain",
                           text=source.layout.annotations[-1].text, showarrow=False,
                           yanchor="bottom", font=dict(size=13, color="#243447"))
        ys = [float(y) for trace in source.data for y in trace.y if y is not None]
        bottom, top = min(ys)-1.5, max(ys)+1.5
        if source.layout.meta["load_top"] is not None:
            top = max(top, source.layout.meta["load_top"]+1.5)
        half_x = 11.2 if kind in ("사장교", "현수교") else 8.
        fig.update_xaxes(range=[6-half_x, 6+half_x], showgrid=False, showticklabels=False, zeroline=False, row=row, col=col)
        fig.update_yaxes(range=[bottom, top], showgrid=False, showticklabels=False, zeroline=False,
                         scaleanchor=xref, scaleratio=1, row=row, col=col)
    fig.update_layout(height=1240, showlegend=False, margin=dict(l=16, r=16, t=45, b=16),
                      font=dict(family="sans-serif", size=14, color="#243447"),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    fig.update_annotations(font=dict(color="#243447"))
    return fig


def force_observation(sweep, count):
    """계산에서 확인된 힘에 맞춰 관찰할 부분을 안내합니다."""
    r, model = sweep["results"][count], sweep["model"]
    if not r["stable"]:
        return "연결 조건을 바꾼 뒤 하중이 전달되는 과정을 다시 확인하세요."
    if max([abs(n) for n in r["axial"]] + r["moments"], default=0.) < 1e-9:
        return "현재 계산에서는 부재에 작용하는 힘이 거의 없습니다. 추를 추가해 변화를 관찰하세요."
    kind, members = model["kind"], model["members"]
    if kind == "단순교":
        return "보 내부의 위·아래 화살표 방향과 처짐을 함께 관찰하세요."
    if kind == "아치교":
        return "아치와 양쪽 받침에 표시된 화살표 방향을 비교해 보세요."
    if kind == "트러스교":
        return "서로 다른 방향의 힘을 받는 막대를 찾아보세요."
    if r["tension"] > 1e-7:
        compressed = any(r["axial"][i] < -1e-7 for i, e in enumerate(members) if e["role"] == "주탑")
        return ("케이블의 파란 화살표와 주탑의 빨간 화살표를 비교해 보세요."
                if compressed else "케이블에 표시된 파란 화살표와 상판의 처짐을 관찰하세요.")
    return "현재 계산에서 케이블의 인장력은 거의 없습니다. 상판과 받침에 전달되는 힘을 살펴보세요."


def load_change_observation(sweep, count, previous_count):
    """같은 모형의 두 계산 결과를 비교하며, 변화 방향을 미리 가정하지 않습니다."""
    if previous_count is None or previous_count == count:
        return None
    before, after = (sweep["results"][i] for i in (previous_count, count))
    if not before["stable"] or not after["stable"]:
        return "변경 전후에 계산할 수 없는 상태가 있어 처짐을 비교하지 않습니다."
    delta = after["deflection_mm"] - before["deflection_mm"]
    change = "커졌습니다" if delta > 1e-9 else "작아졌습니다" if delta < -1e-9 else "거의 같습니다"
    before_text, after_text = f"{before['deflection_mm']:.4f}", f"{after['deflection_mm']:.4f}"
    rounding_note = "\n\n작은 변화는 반올림되어 같은 숫자로 보일 수 있습니다." if before_text == after_text and abs(delta) > 1e-9 else ""
    return (f"같은 구조·설정 · **추 {previous_count}개 → {count}개**\n\n"
            f"최대 **처짐**: **{before_text} → {after_text} mm** · {change}." + rounding_note)


def show_result(sweep, count, amplify, key, figure_slot=None, previous_count=None, view_zoom=None):
    r, m = sweep["results"][count], sweep["model"]
    fig = (structure_figure(sweep, count, amplify) if view_zoom is None
           else experiment_figure(sweep, count, amplify, view_zoom))
    if figure_slot is None:
        chart(fig, key)
    else:
        with figure_slot:
            chart(fig, key)
    st.markdown("#### 하중을 받으면 어떻게 변할까?")
    frozen_count = fig.layout.meta["display_count"]
    if not r["stable"]:
        st.warning("현재 연결 상태에서는 구조의 형태를 유지하기 어렵습니다. 대각선 보강과 받침 조건을 확인하세요.")
    elif r["utilization"] > 1+1e-9:
        st.warning(f"설정한 기준을 넘었습니다. 그림의 변형과 부재력은 추 {frozen_count}개 상태를 유지합니다. "
                   f"추는 현재 {count}개이며, 처짐 수치는 선형 참고값입니다.")
    elif r["utilization"] >= .8:
        st.info("설정한 기준에 가까워졌습니다. 어떤 부재의 힘과 처짐이 변했는지 살펴보세요.")
    measurement, observation = st.columns([1, 2], gap="large", vertical_alignment="center")
    with measurement:
        st.metric("최대 처짐" if not fig.layout.meta["frozen"] else "현재 하중의 처짐 · 참고 계산",
                  "계산 불가" if not r["stable"] else f"{r['deflection_mm']:.4f} mm")
    with observation:
        if r["stable"]:
            st.progress(min(1., r["utilization"]), text="설정한 한계 기준에 접근하는 정도")
        change = load_change_observation(sweep, count, previous_count)
        if change:
            st.info(change)
        observed_count = frozen_count if fig.layout.meta["frozen"] and frozen_count is not None else count
        st.markdown(highlight_learning_text(force_observation(sweep, observed_count)), unsafe_allow_html=True)
    st.caption(f"그림의 변형은 {amplify}배로 확대합니다. 보의 두께와 힘 화살표의 길이는 실제 치수나 힘의 크기를 뜻하지 않습니다.")
    with st.expander("심화 결과 · 한계와 계산 정보", expanded=False):
        st.write(f"이 모형에서 기준 이내인 추: **{capacity_text(sweep)}** (0~20개 계산). 실제 교량의 성능 순위가 아닙니다.")
        if sweep["capacity"] >= 0:
            st.write(f"마지막 기준 이내 추가 하중: {sweep['capacity']*m['weight_mass']*GRAVITY:.3f} N (자체 무게 제외).")
        if sweep["censored"]:
            st.caption("20개까지 기준 이내이므로 실험 범위에서 최대값은 확인되지 않았습니다.")
        if r["stable"]:
            st.write(f"가장 먼저 한계에 도달하는 기준: {r['governing']}")
            st.metric("기준 대비 여유", "힘 없음" if r["reserve"] is None else f"{r['reserve']:.2f}배")
            st.caption("각 허용량을 현재 요구량으로 나눈 값 중 가장 작은 값입니다. 처짐 기준과 자체 무게도 포함합니다.")
            st.write(f"총 수직 하중: {r['total_force']:.3f} N · 최대 케이블 인장력: {r['tension']:.3f} N")
        st.caption(f"골조 {m['skeleton_mass']*1000:.1f} g · 케이블 {m['cable_mass']*1000:.1f} g · 종이 {m['paper_mass']*1000:.1f} g · "
                   f"막대 단면 {m['section_mm'][0]:.1f} × {m['section_mm'][1]:.1f} mm")
        st.caption("기준을 넘은 수치는 실제 붕괴 이후의 결과가 아닙니다. 그림의 주탑 하부는 기초로의 하중 전달을 설명하는 개념 표시입니다.")


def load_curve(sweeps, names, count, include_reference=False):
    fig = data_figure("추 개수에 따른 처짐", "추 개수 (개)", "최대 처짐 (mm)", 500)
    fig.update_layout(margin=dict(l=65, r=25, t=65, b=150),
                      font=dict(color="#243447"),
                      legend=dict(orientation="h", y=-.32, x=0, yanchor="top", entrywidth=100,
                                  font=dict(size=14, color="#243447")))
    for kind in names:
        sweep = sweeps[kind]
        for within in [True, False]:
            if not within and not include_reference:
                continue
            xs, ys = [], []
            for i, r in enumerate(sweep["results"]):
                if r["stable"] and ((r["utilization"] <= 1+1e-9) == within):
                    xs.append(i)
                    ys.append(r["deflection_mm"])
                else:
                    xs.append(i)
                    ys.append(None)
            # 범위를 넘어가는 첫 선분은 참고선으로 연결합니다.
            if not within and sweep["first_bad"] not in (None, 0):
                i = sweep["first_bad"]-1
                ys[i] = sweep["results"][i]["deflection_mm"]
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=kind if within else f"{kind} · 기준 초과 참고",
                                     legendgroup=kind, showlegend=within,
                                     line=dict(color=COLORS[kind], width=3, dash="solid" if within else "dot")))
    fig.add_vline(x=count, line_color=SUPPORT_COLOR, line_dash="dash", annotation_text=f"현재 {count}개",
                  annotation_xanchor="right" if count >= 18 else "left")
    fig.update_xaxes(range=[-.5, 20.5], nticks=6, tickformat="d", tickangle=0,
                     tickfont=dict(size=14, color="#243447"), title_font=dict(size=14, color="#243447"))
    fig.update_yaxes(tickfont=dict(size=14, color="#243447"), title_font=dict(size=14, color="#243447"))
    return fig


# ── 4. 심화 설정 ─────────────────────────────────────────────────────
with st.sidebar:
    with st.expander("⚙️ 심화 설정", expanded=False):
        st.markdown("**비교 조건**")
        span_cm = st.slider("받침 사이 거리 (cm)", 30, 100, 60, 5, key="span_cm")
        comparison_basis = st.radio(
            "재료 비교 기준", ["골조 재료량 같게", "막대 굵기 같게"], key="material_basis",
            help="재료량 같게: 골조 전체 부피를 단순교와 맞추므로 막대가 많은 구조는 개별 막대가 가늘어집니다. 케이블·종이·받침은 별도입니다.",
        )
        material = st.selectbox("골조 재료 · 예시 물성", list(MATERIALS), key="material")
        preset = MATERIALS[material]
        mass_g = st.select_slider(
            "추 한 개의 무게", options=[10, 20, 50, 100], value=20,
            format_func=lambda x: f"{x} g", key="weight_mass_g",
        )
        st.caption("기본 실험은 추 한 개 20 g을 사용합니다.")
        with st.expander("재료·단면·연결부와 기준", expanded=False):
            st.caption("두 옆면의 막대와 종이 바닥판을 단순화한 모형입니다. 아래 값은 재료 성질의 예시입니다.")
            member_width = st.slider("기준 막대 폭 (mm)", 4., 20., 8., .5, key="member_width")
            member_depth = st.slider("기준 막대 세로 두께 (mm)", 2., 12., 6., .5, key="member_depth")
            deck_width = st.slider("종이 바닥판 폭 (cm)", 6, 24, 12, key="deck_width")
            paper_gsm = st.number_input("종이 면적당 무게 (g/m²)", 40, 300, 120, 10, key="paper_gsm")
            E_gpa = st.number_input("잘 휘지 않는 정도 · 탄성계수 (GPa)", .1, 30., preset["E"]/1e9, .1, key=f"E_{material}")
            density = st.number_input("골조 밀도 (kg/m³)", 50., 2000., preset["density"], 10., key=f"density_{material}")
            ft = st.number_input("당기는 힘에 대한 강도 (MPa)", .5, 200., preset["ft"]/1e6, .5, key=f"ft_{material}")
            fc = st.number_input("누르는 힘에 대한 강도 (MPa)", .5, 200., preset["fc"]/1e6, .5, key=f"fc_{material}")
            fb = st.number_input("휘는 힘에 대한 강도 (MPa)", .5, 200., preset["fb"]/1e6, .5, key=f"fb_{material}")
            joint = st.slider("막대 끝 연결부 · 한 옆면의 허용 힘 (N)", .5, 20., 3., .5, key="joint_force")
            foundation = st.slider("받침·앵커 하나의 수평 허용 힘 · 양옆 합계 (N)", 1., 50., 10., 1., key="foundation_force")
            cable_d = st.slider("줄 한 가닥 지름 (mm)", .3, 2., .8, .1, key="cable_diameter")
            cable_E = st.number_input("줄 탄성계수 (GPa)", .1, 200., 1., .1, key="cable_E")
            cable_strength = st.number_input("줄 인장 강도 (MPa)", 1., 2000., 40., 1., key="cable_strength")
            divisor = st.select_slider("처짐 기준 · 받침 사이 거리의", options=[100, 150, 200, 250, 300], value=200,
                                       format_func=lambda x: f"1/{x}", key="deflection_divisor")
            self_weight = st.toggle("다리 자체 무게 반영", True, key="self_weight")
            st.caption("허용 기준은 모형의 실험 조건이며 실제 설계 기준의 인증값이 아닙니다.")
        with st.expander("교량 형상", expanded=False):
            arch_percent = st.slider("아치 높이 · 받침 사이 거리의 (%)", 10, 40, 25, 5, key="arch_percent")
            panels = st.slider("트러스 사각형 칸 수", 1, 11, 5, key="triangles")
            truss_percent = st.slider("트러스 뼈대 높이 · 경간의 (%)", 10, 35, 20, 5, key="truss_percent")
            reinforced = st.toggle("트러스 대각선 보강재 추가", True, key="add_diagonals")
            tower_cm = st.slider("케이블교 바닥판 위 주탑 높이 (cm)", 5, 50, 15, 1, key="tower_cm")
            sag_percent = st.slider("현수교 주케이블 늘어진 깊이 · 주탑 높이의 (%)", 20, 80, 50, 5, key="sag_percent")
        with st.expander("표시와 한계 참고", expanded=False):
            amplify = st.slider(
                "변형 확대 배율", 0, 30, 10, key="amplify", format="%d배",
                help="그림만 확대합니다. mm 수치·힘·한계 개수는 변하지 않습니다. 0배는 원래 모양입니다.",
            )
            allow_over_limit = st.toggle("한계 초과 참고 계산 보기", False, key="allow_over_limit",
                                         help="최대 20개까지 참고값을 확인합니다. 기준을 넘은 뒤 실제 붕괴를 계산하는 기능은 아닙니다.")
    with st.expander("교사용 · 모형의 약속", expanded=False):
        st.write("고정된 교량별 최대 추 개수는 없습니다. 같은 단위(m, N, Pa)로 보·막대·줄의 힘과 처짐을 구하고, 강도·Euler 좌굴·막대 끝 연결력·수평 받침력·허용 처짐 중 먼저 닿는 기준을 찾습니다.")
        st.write("연속된 바닥 보와 아치는 휨을 받는 보 요소, 나머지 막대는 핀 연결 축력 요소입니다. 줄은 당기는 힘만 허용합니다. 두 옆면이 무게를 절반씩 나누며 가로 방향 흔들림은 별도 가새가 막는다고 가정합니다.")
        st.write("골조 단면은 직사각형입니다. 수수깡·종이의 속 빈 단면, 접힌 종이의 강성, 접착제 특성은 별도 실측 보정이 필요합니다. 종이는 하중 분배와 자중만 반영하고 골조의 휨 강성에 더하지 않습니다.")
        st.write("아치는 양끝 핀 지지, 보의 오른쪽은 수평 이동 가능한 받침입니다. 케이블 교량은 주탑 뒤 고정줄과 앵커를 포함합니다. 압축 막대는 양끝 힌지의 Euler 하중, 아치의 유효 길이는 아치 길이의 절반으로 가정합니다. 바닥 보의 압축 유효 길이는 전체 경간(트러스는 한 칸)입니다. 실제 좌굴 형상·유효 길이는 지지 조건에 따라 달라집니다.")
        st.write("작은 변형의 2차원 선형 정적 계산입니다. 초기 케이블 장력·형상 찾기, 대변형, 아치 전체의 비선형 불안정, 3차원 비틀림, 전단 파괴, 바람·지진·피로, 연결부의 회전·박리는 계산하지 않습니다. 케이블의 큰 형상 변화가 예상되면 이 결과의 적용 범위를 벗어납니다.")
        st.write("트러스는 삼각형 부재가 축력으로 힘을 나누고, 모형에서 바닥 보의 좌굴 길이를 한 칸으로 두어 높은 결과가 나올 수 있습니다. 골조 재료량 비교에는 케이블 재료가 포함되지 않고 좌굴·연결 조건도 단순화되어 있어 실제 교량의 우열을 뜻하지 않습니다.")
        st.write("‘기준 대비 여유’는 각 기준의 허용량 ÷ 계산된 요구량 중 가장 작은 값입니다. 강도만의 안전율이나 인증된 설계 안전율은 아닙니다. 자중 때문에 추 개수의 단순 비율과 다를 수 있습니다.")
        st.write("MATERIALS는 예시 물성, build_model()은 모형과 지지 조건, solve_sweep()은 계산과 기준 검사입니다. 하나의 파일 안에서 수정할 수 있습니다.")
        st.markdown(f"계산 참고: [TU Delft 보 요소]({METHOD_SOURCE}) · [TU Delft 평면 골조](https://teachbooks.tudelft.nl/computational-modelling/structural_linear/space_frame.html) · [MIT 압축 부재 좌굴]({BUCKLING_SOURCE})")

# ── 5. 첫 화면 안내, 그 다음 탐구 ──────────────────────────────────────
guide_tab, lab_tab, comparison_tab = st.tabs(["실험실 안내", "교량 탐구", "교량 비교"])
with guide_tab:
    with st.container(key="lab_guide"):
        st.title("교량 구조 디지털 탐구 학습지")
        st.write("교량 구조를 배우고 실험·실습을 대신할 수 있는 디지털 학습 자료입니다.")
        st.subheader("학습 순서")
        st.markdown('<ol class="study-steps">'
                    f'<li>{learning_icon("book")}<div><strong>교량 구조 알아보기</strong><small>모양·주요 부재·하중 전달 과정</small></div></li>'
                    f'<li>{learning_icon("scale")}<div><strong>하중 실험하기</strong><small>추를 늘리며 처짐과 힘의 변화 관찰</small></div></li>'
                    f'<li>{learning_icon("search")}<div><strong>교량 비교하기</strong><small>같은 조건에서 구조와 결과 비교</small></div></li>'
                    '</ol>', unsafe_allow_html=True)
        st.subheader("그림 읽기")
        force_primer()
        st.caption("위 그림은 힘과 처짐을 설명하는 개념 그림입니다.")
        st.caption("주황 화살표는 추의 하중, 회색 화살표는 받침이 구조를 지지하는 힘을 나타냅니다.")
        st.markdown('그림의 변형과 보의 두께는 관찰하기 쉽도록 확대합니다. <u class="concept-focus">실제 처짐은 mm 수치</u>로 확인합니다.', unsafe_allow_html=True)
        st.subheader("디지털 실험·실습")
        st.caption("수업자료: 「다양한 교량의 구조」 → 「교량 만들기 실습」")
        st.markdown(highlight_learning_text("화면에서 추를 조절하며 처짐과 인장·압축을 관찰합니다."), unsafe_allow_html=True)
        with st.expander("실물 실험과 다른 점", expanded=False):
            st.write("재료와 연결 방식이 단순화된 모형이므로 실물 실험과 결과가 같을 필요는 없습니다. 실제 교량을 설계하거나 성능 순위를 정하는 도구가 아닙니다.")

with lab_tab:
    selected_bridge = st.selectbox("학습할 교량", BRIDGE_NAMES,
                                    format_func=lambda name: BRIDGE_GUIDES[name]["title"], key="selected_bridge")
    chapter_section = st.container()

cfg = dict(span=span_cm/100, width=deck_width/100, weight_mass=mass_g/1000,
           member_width=member_width/1000, member_depth=member_depth/1000,
           E=E_gpa*1e9, density=density, ft=ft*1e6, fc=fc*1e6, fb=fb*1e6,
           paper_gsm=paper_gsm, self_weight=self_weight, equal_volume=comparison_basis == "골조 재료량 같게",
           joint_force=joint, foundation_force=foundation, deflection_divisor=divisor,
           cable_diameter=cable_d/1000, cable_E=cable_E*1e9, cable_strength=cable_strength*1e6,
           panels=panels, arch_ratio=arch_percent/100, truss_ratio=truss_percent/100,
           tower_ratio=tower_cm/span_cm, sag_ratio=sag_percent/100)
sweeps = {kind: solve_sweep(kind, cfg) for kind in BRIDGE_NAMES}
selected_sweep = (solve_sweep(selected_bridge, cfg, braced=reinforced)
                  if selected_bridge == "트러스교" else sweeps[selected_bridge])
load_limit = 20 if allow_over_limit and selected_sweep["results"][0]["stable"] else max(0, selected_sweep["capacity"])
requested_load = st.session_state.get("load", 0)
load_was_limited = requested_load > load_limit
if "load" in st.session_state:
    # 슬라이더 상한이 바뀌어 위젯이 다시 만들어져도 허용 범위의 값을 유지합니다.
    st.session_state["load"] = max(0, min(requested_load, load_limit))

with chapter_section:
    bridge_chapter_intro(selected_bridge)
    st.subheader("② 하중 실험하기")
    st.markdown(f'<p class="experiment-focus">{learning_icon("search")} 추의 개수를 늘리면 <u class="concept-focus">교량은 어떻게 변할까?</u></p>', unsafe_allow_html=True)
    st.markdown('<p><strong>관찰할 것</strong> · ' + highlight_learning_text("처짐 · 인장·압축의 방향 · 하중 전달") + '</p>', unsafe_allow_html=True)
    zoom_percent = st.session_state.get("bridge_zoom_percent", 100)
    with st.container(key="bridge_viewport"):
        figure_column, zoom_column = st.columns([1, .05], gap="small", vertical_alignment="center")
        with figure_column:
            figure_slot = st.container()
        with zoom_column:
            with st.container(key="bridge_zoom_controls"):
                st.button("＋", key="bridge_zoom_in", help="그림 확대 (25%씩)",
                          on_click=change_bridge_zoom, args=(25,), disabled=zoom_percent >= 250)
                st.button("－", key="bridge_zoom_out", help="그림 축소 (25%씩)",
                          on_click=change_bridge_zoom, args=(-25,), disabled=zoom_percent <= 50)
                st.button("↺", key="bridge_zoom_reset", help="그림을 원래 크기로 (100%)",
                          on_click=change_bridge_zoom, args=(0,))
                st.caption(f"{zoom_percent}%")
    render_weight_controls(st.session_state.get("load", 0), load_limit)
    st.caption("오른쪽 ＋ / － / ↺는 화면 크기만 조절합니다. 확대한 그림은 드래그하여 이동할 수 있습니다.")
    if load_limit > 0:
        load = st.slider("추의 개수 조절하기", 0, load_limit, key="load", format="%d개")
    else:
        load = st.number_input("추의 개수 조절하기", min_value=0, max_value=0, key="load", disabled=True)
    if load_was_limited:
        st.caption(f"변경된 조건의 허용 범위에 맞춰 추를 {load_limit}개로 줄였습니다.")
    if selected_sweep["capacity"] < 0 and not allow_over_limit:
        st.caption("현재 조건에서는 추를 추가할 수 없습니다. 구조와 자체 무게 조건을 확인하세요.")
    elif allow_over_limit:
        st.caption("기준 초과 수치는 참고값입니다. 그림에 표시된 변형과 부재력이 어느 하중 상태인지 확인하세요.")
    elif load == load_limit and not selected_sweep["censored"]:
        st.caption("현재 조절 범위의 끝입니다. 심화 설정에서 한계 초과 참고 계산을 선택할 수 있습니다.")
    st.write(f"추 **{load}개 × {mass_g} g = {load*mass_g} g**")
    st.caption("추는 중앙의 같은 위치에 쌓습니다. 추가 하중도 이 위치에 집중하여 계산합니다.")
    context = (selected_bridge, tuple(sorted(cfg.items())), reinforced if selected_bridge == "트러스교" else True)
    previous = st.session_state.get("_last_observation", {})
    previous_count = previous.get("count") if previous.get("context") == context else None
    show_result(selected_sweep, load, amplify, f"selected_{selected_bridge}", figure_slot, previous_count,
                view_zoom=zoom_percent / 100)
    st.session_state["_last_observation"] = dict(context=context, count=load)
    bridge_application(selected_bridge)
    if selected_bridge in ("트러스교", "사장교", "현수교"):
        shape_exploration = st.expander("심화 탐구 · 보강과 형상 비교", expanded=False,
                                        key="shape_exploration", on_change="rerun")
        if shape_exploration.open:
            with shape_exploration:
                if selected_bridge == "트러스교":
                    # 왼쪽/오른쪽은 동일 단면을 사용해 대각선을 '추가'하는 효과만 비교합니다.
                    # 비교 기준이 총량 같게일 때도 이 짝 안에서는 오른쪽의 막대 단면을 그대로 사용합니다.
                    square_cfg = dict(cfg, equal_volume=False)
                    if cfg["equal_volume"]:
                        width_mm, depth_mm = sweeps["트러스교"]["model"]["section_mm"]
                        square_cfg.update(member_width=width_mm/1000, member_depth=depth_mm/1000)
                    square = solve_sweep("트러스교", square_cfg, False)
                    right = sweeps["트러스교"] if reinforced else square
                    for col, sweep, title, key in zip(st.columns(2), [square, right],
                            ["보강 전 · 사각형", "대각선 추가 · 삼각형" if reinforced else "대각선 추가 대기"], ["truss_square", "truss_braced"]):
                        with col:
                            st.markdown(f"### {title}")
                            show_result(sweep, load, amplify, key)
                    st.caption("두 그림은 같은 막대 단면을 사용합니다. 오른쪽에 대각선을 추가하면 재료량이 늘어납니다. 교량 비교 탭은 심화 설정의 재료 비교 기준을 사용합니다.")
                    st.info("이 모형의 사각형은 연결부가 자유롭게 회전하여 형태를 유지하지 못합니다. 연결부의 각도를 고정한 실물 모형과는 결과가 다를 수 있습니다.")
                    fig = data_figure("대각선으로 보강한 칸 수를 바꾸면?", "칸 수", "최대 처짐 (mm)", 310)
                    ns = list(range(1, 12))
                    series = [solve_sweep("트러스교", dict(cfg, panels=n))["results"][load] for n in ns]
                    fig.add_trace(go.Scatter(x=ns, y=[r["deflection_mm"] for r in series], mode="lines+markers", name="대각선 보강",
                                             line=dict(color=COLORS["트러스교"])))
                    fig.add_vline(x=panels, line_dash="dot", line_color=SUPPORT_COLOR)
                    fig.update_xaxes(dtick=1)
                    chart(fig, "truss_efficiency")
                    st.caption("점들은 기준 초과 설정도 포함한 참고값입니다. 칸이 늘면 막대 길이가 짧아지지만, 재료량을 맞추면 막대가 가늘어지는 영향도 있습니다.")
                    with st.container(border=True):
                        st.markdown("**대각선은 왜 필요할까?**")
                        st.write("사각형은 막대 길이가 그대로여도 모서리 각도가 바뀔 수 있습니다. 대각선을 넣어 삼각형을 만들면 이런 변형을 억제할 수 있습니다.")
                        st.write("삼각형 구조에도 한계는 있습니다. 압축 막대가 길고 가늘면 옆으로 휘어질 수 있고, 약한 연결부가 먼저 한계에 도달할 수 있습니다.")

                if selected_bridge in ("사장교", "현수교"):
                    fig = data_figure("주탑 높이와 줄을 당기는 힘", "바닥판 위 주탑 높이 (cm)", "최대 줄 장력 · 양옆 합계 (N)")
                    heights = sorted(set([5, 10, 15, 20, 30, 40, 50, tower_cm]))
                    for cable_kind in ("사장교", "현수교"):
                        tensions = [solve_sweep(cable_kind, dict(cfg, tower_ratio=h/span_cm))["results"][load]["tension"] for h in heights]
                        fig.add_trace(go.Scatter(x=heights, y=tensions, mode="lines+markers", name=cable_kind,
                                                 line=dict(color=COLORS[cable_kind]),
                                                 hovertemplate="주탑 %{x} cm<br>장력 %{y:.3f} N<extra>%{fullData.name}</extra>"))
                    fig.add_vline(x=tower_cm, line_dash="dot", line_color=SUPPORT_COLOR)
                    chart(fig, "height_sensitivity")
                    st.caption("높이에 따른 장력은 현재 모형의 선형 계산값입니다. 주탑이 높아지면 장력이 줄 수 있지만, 실제 설계에서는 주탑의 압축과 좌굴도 함께 살핍니다.")

                    st.write("주탑 높이를 바꿀 때 케이블의 각도와 인장력이 어떻게 달라지는지 비교하세요.")
                    st.caption("현수교는 주탑 높이에 대한 주케이블의 늘어진 깊이 비율을 유지합니다. 바람에 의한 진동과 시공 과정은 계산하지 않습니다.")

with comparison_tab:
    st.header("③ 교량 비교하기")
    st.caption(f"같은 조건: 받침 사이 {span_cm} cm · 추 1개 {mass_g} g · 중앙에 추 {load}개 · {comparison_basis}")
    st.caption("케이블·종이·받침의 재료량은 같지 않을 수 있습니다. 트러스교는 대각선 보강 구조입니다.")
    with st.expander("비교 조건과 다른 점", expanded=False):
        if cfg["equal_volume"]:
            st.write("골조 막대의 총부피를 맞춥니다. 막대가 많은 구조는 개별 막대가 가늘어집니다.")
        else:
            st.write("막대의 단면을 맞춥니다. 구조마다 골조 총재료량은 다를 수 있습니다.")
        st.write(f"자체 무게 {'포함' if self_weight else '제외'}. 케이블·종이·받침은 같은 재료량으로 맞추지 않습니다.")
        st.caption("추 개수는 교량 탐구와 함께 사용합니다. 지지·연결·압축 막대 길이의 단순화도 결과에 영향을 줍니다.")
    mode = st.radio("비교 방식", ["5종 전체 비교", "두 교량 비교"], horizontal=True, key="comparison_mode")
    if mode == "두 교량 비교":
        cols = st.columns(2)
        left = cols[0].selectbox("비교 교량 A", BRIDGE_NAMES, key="compare_left")
        options = [k for k in BRIDGE_NAMES if k != left]
        if st.session_state.get("compare_right") not in options:
            st.session_state["compare_right"] = options[0]
        right_kind = cols[1].selectbox("비교 교량 B", options, key="compare_right")
        names = [left, right_kind]
        for i, (col, kind) in enumerate(zip(cols, names)):
            with col:
                chart(structure_figure(sweeps[kind], load, amplify), f"comparison_structure_{i}")
        st.caption("빨강은 압축, 파랑은 인장을 나타냅니다. 기준을 넘은 교량은 마지막 기준 이내의 형상과 부재력을 유지합니다.")
    else:
        names = BRIDGE_NAMES
    rows = []
    for kind in names:
        sweep, guide = sweeps[kind], BRIDGE_GUIDES[kind]
        r = sweep["results"][load]
        rows.append({
            "교량": guide["title"], "모형 한계 (추)": capacity_text(sweep),
            "현재 처짐 (mm)": "계산 불가" if not r["stable"] else f"{r['deflection_mm']:.4f}" + (" (참고)" if r["utilization"] > 1+1e-9 else ""),
            "구조적 특징": guide["comparison_structure"], "주요 원리": guide["comparison_principle"],
        })
    st.dataframe(rows, hide_index=True, width="stretch", row_height=56, key="learning_comparison", column_config={
        "교량": st.column_config.TextColumn(width=115), "모형 한계 (추)": st.column_config.TextColumn(width=115),
        "현재 처짐 (mm)": st.column_config.TextColumn(width=135),
        "구조적 특징": st.column_config.TextColumn(width=230), "주요 원리": st.column_config.TextColumn(width=260),
    })
    st.caption("‘20개 이상’인 결과끼리는 순위를 정할 수 없습니다. ‘참고’는 기준 초과 상태의 선형 계산값입니다.")
    chart(load_curve(sweeps, names, load, include_reference=allow_over_limit), "load_deformation_curves")
    st.caption("실선은 기준 이내의 계산 범위입니다." + (" 점선은 기준 초과의 참고값이며 실제 붕괴 후 변형이 아닙니다." if allow_over_limit else " 초과 참고선은 심화 설정에서 확인할 수 있습니다."))
    st.info("이 결과는 현재 모형과 조건에서의 비교입니다. 추를 많이 견딘 교량이 실제 교량 중 항상 가장 튼튼하다는 뜻은 아닙니다.")
    advanced_comparison = st.expander("심화 비교 · 모형 정보와 한계 상태", expanded=False,
                                      key="advanced_comparison", on_change="rerun")
    if advanced_comparison.open:
        with advanced_comparison:
            with st.container(key="bridge_overview_scroll"):
                with st.container(key="bridge_overview"):
                    chart(overview_figure(sweeps, amplify), "capacity_bridge_overview")
            st.caption("이 그림만 각 교량의 마지막 기준 이내 추 개수를 사용합니다. 주탑 하부는 하중 전달을 설명하는 개념 표시입니다. 좁은 화면에서는 그림 영역을 좌우로 이동할 수 있습니다.")
            details = []
            for kind in names:
                m, r = sweeps[kind]["model"], sweeps[kind]["results"][load]
                details.append({"교량": kind, "골조 질량 (g)": round(m["skeleton_mass"]*1000, 1),
                                "케이블 질량 (g)": round(m["cable_mass"]*1000, 1),
                                "막대 단면 (mm)": f"{m['section_mm'][0]:.1f} × {m['section_mm'][1]:.1f}",
                                "현재 가장 먼저 한계에 도달하는 기준": r["governing"]})
            st.dataframe(details, hide_index=True, width="stretch")
            st.caption("처짐이 작아도 압축 막대나 연결부가 먼저 한계에 닿을 수 있습니다. 실제 교량에서는 지반·시공·바람 등도 함께 검토합니다.")

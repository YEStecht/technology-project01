"""교량 개념 실험실 — 실행: python -m streamlit run streamlit_app.py

처음 수정한다면 MATERIALS, build_model(), solve_sweep()부터 살펴보세요.
모든 화면·계산·그래프를 이 파일에 모았습니다. 외부 데이터는 필요하지 않습니다.
"""

import math
from html import escape

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


# ── 1. 수업에서 바꾸기 쉬운 설정 ────────────────────────────────────────
GRAVITY = 9.81
COLORS = {
    "단순교": "#64748b", "아치교": "#10a38b", "트러스교": "#7160d2",
    "사장교": "#217bd0", "현수교": "#d69026",
}
# 학습 카드의 문구는 이 사전에서 수정하면 모든 탐구·비교 화면에 반영됩니다.
BRIDGE_GUIDES = {
    "단순교": {
        "title": "단순교 (거더교)",
        "situation": "짧은 하천을 경제적으로 건너야 할 때 고려할 수 있습니다. 짧은 경간에서는 구조가 단순해 시공과 비용 면에서 유리할 수 있습니다.",
        "fit": "짧은 경간에서는 보와 상판으로 하중을 간단히 전달할 수 있어 경제적인 선택이 될 수 있습니다.",
        "load_path": "추·차량 하중 → 상판 아래 보 → 양쪽 받침 → 지반. 보는 휘면서 윗부분은 압축, 아랫부분은 인장을 받습니다.",
        "question": "추를 늘릴 때 모형의 처짐과 보의 위·아래 색을 살펴보세요.",
        "disadvantage": "경간이 길어지면 보의 휨과 처짐이 커져 더 큰 보나 추가 받침이 필요할 수 있습니다.",
        "examples": "일반 도로와 육교의 일부",
    },
    "아치교": {
        "title": "아치교",
        "situation": "양쪽 지반과 교대가 튼튼한 계곡에서 고려할 수 있습니다. 아치가 하중을 양쪽으로 보내므로 교대가 옆으로 미는 힘을 버텨야 합니다.",
        "fit": "견고한 양쪽 교대가 아치의 압축력과 바깥쪽 수평 힘을 받아 줄 수 있을 때 적합한 선택이 될 수 있습니다.",
        "load_path": "상판 → 세로 지지대 → 아치의 압축 → 양쪽 교대·지반. 아치 받침에는 바깥쪽 수평 힘도 생깁니다.",
        "question": "추를 늘릴 때 아치의 압축력과 받침이 옆으로 버티는 힘을 살펴보세요.",
        "disadvantage": "지반이나 교대가 약하면 수평 힘에 불리하고, 아치를 만들고 설치하는 과정도 고려해야 합니다.",
        "examples": "시드니 하버 브리지 등",
    },
    "트러스교": {
        "title": "트러스교",
        "situation": "무거운 열차가 지나 높은 강성이 필요한 교량에 고려할 수 있습니다. 삼각형 뼈대가 모양 변화를 줄이는 데 유리할 수 있습니다.",
        "fit": "삼각형 부재가 인장과 압축을 나누어 받아 구조의 강성을 높일 수 있습니다.",
        "load_path": "상판 → 삼각형 트러스 부재 → 받침. 부재는 배치에 따라 인장 또는 압축을 받고, 상판 보는 휨을 받습니다.",
        "question": "추를 늘릴 때 삼각형 막대의 인장·압축 색과 처짐을 살펴보세요. 열차는 하중 상황의 비유이며, 모형에는 추의 균등하중이 작용합니다.",
        "disadvantage": "길고 가는 압축 부재는 좌굴할 수 있고 연결부도 약점이 됩니다. 긴 경간에서는 부재와 자중이 늘 수 있습니다.",
        "examples": "철도 트러스교 등",
    },
    "사장교": {
        "title": "사장교",
        "situation": "중간 교각을 놓기 어려운 넓은 강에서 고려할 수 있습니다. 경사진 케이블이 상판을 직접 받쳐 중·장경간에 활용됩니다.",
        "fit": "주탑에서 뻗은 케이블이 상판을 직접 지지해 중간 교각 수를 줄이는 데 도움이 될 수 있습니다.",
        "load_path": "상판 → 사장 케이블(인장)·상판 보(휨) → 주탑(주로 압축) → 기초. 뒤쪽 케이블은 앵커로 힘을 전달합니다.",
        "question": "추를 늘릴 때 케이블이 당겨지는 정도와 상판 처짐을 살펴보세요.",
        "disadvantage": "주탑·케이블의 시공과 점검이 필요하며, 실제 교량에서는 바람과 진동도 고려해야 합니다.",
        "examples": "인천대교, 서해대교",
    },
    "현수교": {
        "title": "현수교",
        "situation": "중간 교각을 놓기 어려운 해협에서 초장경간을 건널 때 고려할 수 있습니다. 긴 주케이블이 상판을 매달아 지지합니다.",
        "fit": "긴 주케이블로 넓은 물길을 건널 수 있어 초장경간에 적합한 구조 중 하나입니다.",
        "load_path": "상판 → 행어(인장) → 주케이블(인장) → 주탑(압축)·앵커리지. 하중은 케이블과 지지 구조를 거쳐 지반으로 전달됩니다.",
        "question": "현재 모형의 행어·주케이블 장력과 상판 처짐을 살펴보세요. 이 추 실험은 실제 초장경간 성능을 검증하지 않습니다.",
        "disadvantage": "바람 진동, 상판 강성, 케이블과 앵커리지의 시공 조건이 불리한 요인이 될 수 있습니다.",
        "examples": "이순신대교, 남해대교, 금문교(Golden Gate Bridge)",
    },
}

# 경간은 FHWA 2015 자료의 강교 예시를 m 단위로 반올림했습니다.
# 형식 전체의 최소/최대 길이가 아닙니다. None은 위쪽 한계를 정하지 않았다는 뜻입니다.
# 비용 점수는 구조의 특징을 비교하기 위해 이 앱에서 정한 수업용 값입니다.
BRIDGE_PLANNING = {
    "단순교": {"span_min": 30, "span_max": 150, "span_label": "약 30~150 m대",
             "span_note": "강판 거더의 참고 구간. 150 m보다 긴 사례도 있어요.",
             "cost_index": 100, "cost_reason": "보와 바닥판을 중심으로 구조가 비교적 단순해요."},
    "아치교": {"span_min": 60, "span_max": 550, "span_label": "약 60~550 m",
             "span_note": "자료에서 설명한 아치교 사례 범위예요.",
             "cost_index": 170, "cost_reason": "둥근 아치를 만들고 양쪽 기초를 튼튼히 해야 해요."},
    "트러스교": {"span_min": 140, "span_max": 230, "span_label": "약 140~230 m",
              "span_note": "한 구간으로 잇는 트러스의 실용적인 길이 예시예요. 연속 트러스는 더 길 수 있어요.",
              "cost_index": 150, "cost_reason": "막대와 연결 부분이 많아 조립·점검에 손이 많이 가요."},
    "사장교": {"span_min": 60, "span_max": 1100, "span_label": "약 60~1,100 m",
             "span_note": "자료의 건설 사례를 간단히 표시했어요. 더 짧은 사례도 있어요.",
             "cost_index": 240, "cost_reason": "높은 주탑과 여러 케이블을 만들고 설치해야 해요."},
    "현수교": {"span_min": 900, "span_max": None, "span_label": "약 900 m 이상 →",
             "span_note": "긴 경간에서 비용상 유리해지기 시작하는 문헌 기준이에요. 최소·최대 건설 길이는 아니에요.",
             "cost_index": 320, "cost_reason": "굵은 주케이블과 양끝의 큰 고정 장치가 필요해요."},
}
PLANNING_SOURCE = "https://rosap.ntl.bts.gov/view/dot/42909/dot_42909_DS1.pdf#page=23"

COMPRESSION = "#e05252"
TENSION = "#287bd1"
LOAD_COLOR = "#c56a20"
SUPPORT_COLOR = "#8195a7"
# 내부 힘을 읽기 위한 그림용 두께입니다. 계산 단면과 별개입니다.
BEAM_HALF_DEPTH = .8
TOWER_BASE_DEPTH = 1.6  # 주탑 하부와 기초를 잇는 그림용 길이
PLOT_CONFIG = {"displayModeBar": False, "responsive": True}

st.set_page_config(page_title="교량 구조와 힘의 분산", page_icon="🌉", layout="wide", initial_sidebar_state="collapsed")
# Streamlit의 테마 변수를 사용하면 설정에서 테마를 바꾸는 즉시 반영됩니다.
# 글자·배경을 밝은 색으로 고정하면 다크 모드에서 대비가 사라지므로 피합니다.
st.markdown(
    """
    <style>
    .block-container { max-width: 1500px; padding-top: 2rem; padding-bottom: 3rem; }
    h1 { font-size: clamp(1.65rem, 3vw, 2.45rem) !important; letter-spacing: -.04em; }
    [data-testid="stMetric"] {
        background: var(--secondary-background-color, transparent);
        border: 1px solid rgba(128,128,128,.28);
        border-radius: 14px; padding: 14px 18px;
    }
    [data-testid="stMetricLabel"] *,
    [data-testid="stMetricValue"] * {
        white-space: normal !important; overflow: visible !important;
        text-overflow: clip !important; overflow-wrap: anywhere;
    }
    [data-testid="stMetricValue"] {
        font-size: clamp(1.25rem, 2vw, 1.8rem);
        line-height: 1.2;
    }
    [data-testid="stTabs"] [role="tablist"] { gap: 8px; }
    [data-testid="stTabs"] [role="tab"] {
        padding: 12px 16px; border-radius: 10px; height: auto; white-space: normal;
        background: var(--secondary-background-color, transparent);
        color: var(--text-color, inherit);
    }
    [data-testid="stTabs"] [aria-selected="true"] {
        background: color-mix(in srgb, var(--primary-color) 16%, var(--background-color));
        color: var(--text-color, inherit);
    }
    [data-testid="stPlotlyChart"] {
        border: 1px solid rgba(128,128,128,.28); border-radius: 14px; overflow: hidden;
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
    .bridge-card {
        margin: 14px 0; padding: 20px;
        border: 1px solid rgba(128,128,128,.28);
        border-top: 4px solid var(--bridge-accent); border-radius: 14px;
        background: var(--secondary-background-color, transparent);
        color: var(--text-color, inherit); overflow-wrap: anywhere;
    }
    .bridge-card h4 { margin: 0 0 12px; font-size: 1.15rem; color: inherit; }
    .bridge-card p { margin: 10px 0; line-height: 1.75; color: inherit; }
    .bridge-card p:last-child { margin-bottom: 0; }
    .bridge-card strong { font-weight: 700; }
    .bridge-card .guide-question {
        padding: 10px 12px; border-radius: 10px;
        background: color-mix(in srgb, var(--bridge-accent) 12%, var(--secondary-background-color));
    }
    @media (max-width: 700px) {
        .bridge-card { padding: 16px; }
        .bridge-card p { line-height: 1.65; }
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
    .eyebrow { opacity: .8; font-size: .8rem; font-weight: 700; letter-spacing: .15em; }
    .legend { color: var(--text-color, inherit); font-size: .9rem; line-height: 1.9; }
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

    # 종이는 무게를 펴서 전달합니다. 휨 강성은 양옆의 연속된 보가 담당합니다.
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
                a, b = (bottom[i], deck[i+1]) if i % 2 == 0 else (deck[i], bottom[i+1])
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
                span=L, mass=mass+paper_mass, paper_mass=paper_mass,
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
            live = 0.
            if e["role"] == "바닥 보":
                share = e["length"]/model["span"]
                dead += model["paper_mass"]*GRAVITY*share if cfg["self_weight"] else 0.
                live = cfg["weight_mass"]*GRAVITY*share
            for col, total in enumerate([dead, live]):
                fe[:, col], qxy[:, col] = distributed_load(e, T, total)[0], distributed_load(e, T, total)[1:]
            F[ds] += T.T@fe
        elif cfg["self_weight"]:
            for i in [e["i"], e["j"]]:
                F[3*i+1, 0] -= e["mass"]*GRAVITY/2
        pieces.append((k, T, ds, kg, fe, qxy))
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
            # Hermite 보간 + 균등하중의 정확한 요소 내부 휨 항.
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
    """0~20개를 같은 모형으로 계산. 케이블의 압축은 허용하지 않습니다.

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
        title=dict(text=title, font=dict(size=16)),
        height=height, margin=dict(l=25, r=25, t=55, b=25),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False,
        font=dict(family="sans-serif"),
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


def draw_weights(fig, count, y_at, positions, arrow_size=0.8, block_width=0.4, total_force=None):
    """금속 추의 몸체와 고리를 개수만큼 그립니다. 11개부터 두 줄로 쌓습니다."""
    if count == 0 or not positions:
        return
    for index, x in enumerate(positions):
        row = index // 10
        bottom = float(y_at(x)) + 0.1 + row * 0.42
        # 사다리꼴 몸체: 윗부분이 좁고 아래가 넓은 실험용 추 모양입니다.
        path = (
            f"M {x-block_width/2},{bottom} "
            f"L {x+block_width/2},{bottom} "
            f"L {x+block_width*.30},{bottom+.22} "
            f"L {x-block_width*.30},{bottom+.22} Z"
        )
        fig.add_shape(type="path", path=path, name="추 몸체",
                      line=dict(color="#52647c", width=1), fillcolor="#8297aa")
        fig.add_shape(type="circle", name="추 고리",
                      x0=x-block_width*.15, x1=x+block_width*.15,
                      y0=bottom+.22, y1=bottom+.34,
                      line=dict(color="#8297aa", width=2), fillcolor="rgba(0,0,0,0)")
    for x in np.linspace(min(positions), max(positions), min(count, 3)):
        y = float(y_at(x))
        # 화살표 끝을 추 위에 두어 몸체 및 두 번째 줄과 겹치지 않게 합니다.
        arrow(fig, (x, y+arrow_size+0.65), (x, y+(1.0 if count > 10 else .55)),
              LOAD_COLOR, count if total_force is None else total_force)


def weight_positions(count, left, right):
    if count == 0:
        return []
    columns = min(count, 10)
    xs = [0.5 * (left + right)] if columns == 1 else np.linspace(left, right, columns)
    return [float(xs[i % columns]) for i in range(count)]


def chart(fig, key):
    st.plotly_chart(fig, width="stretch", config=PLOT_CONFIG, theme="streamlit", key=key)


def bridge_guide_card(kind):
    """테마에 맞는 학습 카드. 내용은 위쪽 BRIDGE_GUIDES에서 관리합니다."""
    guide = BRIDGE_GUIDES[kind]
    # HTML 특수 문자를 이스케이프해 문구를 바꿔도 카드 구조가 깨지지 않습니다.
    title, load_path, question, fit, disadvantage, examples = (
        escape(guide[field]) for field in (
            "title", "load_path", "question", "fit", "disadvantage", "examples"
        )
    )
    st.markdown(
        f'<article class="bridge-card" data-bridge="{escape(kind)}" '
        f'style="--bridge-accent: {COLORS[kind]};">'
        f'<h4>🌉 {title}</h4>'
        f'<p><strong>하중이 가는 길</strong><br>{load_path}</p>'
        f'<p class="guide-question"><strong>🔎 그림에서 살펴보기</strong><br>{question}</p>'
        f'<p><strong>이 상황에서 유리할 수 있는 이유</strong><br>{fit}</p>'
        f'<p><strong>실제 건설에서의 조건과 한계</strong><br>{disadvantage} '
        '경간, 지반, 하중, 시공 조건을 함께 살펴야 해요.'
        f'<br><strong>사례</strong>: {examples}</p>'
        '</article>',
        unsafe_allow_html=True,
    )


def bridge_chapter_intro(kind):
    """상황을 먼저 제시하고, 검토할 수 있는 구조와 이유를 소개합니다."""
    guide = BRIDGE_GUIDES[kind]
    st.subheader("① 실제 건설 상황")
    st.write(guide["situation"])
    st.caption(f"이번에 탐구할 구조: {guide['title']}. 같은 상황에서도 조건에 따라 다른 구조를 선택할 수 있어요.")


def bridge_guide_cards(kinds):
    """비교한 교량의 학습 카드를 두 열로 나란히 배치합니다."""
    for start in range(0, len(kinds), 2):
        row = kinds[start:start+2]
        for column, kind in zip(st.columns(len(row)), row):
            with column:
                bridge_guide_card(kind)




def data_figure(title, x_title, y_title, height=340):
    fig = go.Figure()
    fig.update_layout(
        title=dict(text=title, font=dict(size=15)), height=height,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=55, r=25, t=65, b=60),
        xaxis=dict(title=x_title, gridcolor="rgba(128,128,128,.18)"),
        yaxis=dict(title=y_title, gridcolor="rgba(128,128,128,.18)", rangemode="tozero"),
        legend=dict(orientation="h", y=1.16, x=0),
    )
    return fig


def show_planning_comparison(names):
    """5종 전체 비교에서 경간 참고 구간과 학습용 비용 점수를 보여 줍니다."""
    st.markdown("### 📏 길이와 건설비도 함께 비교해요")
    st.write("**경간**은 기둥이나 받침에서 다음 받침까지의 거리예요. 다리 전체 길이와는 달라요. 케이블 교량은 가운데 주탑 사이의 큰 구간을 생각하면 돼요.")
    st.caption("아래 두 그래프는 교량 형식을 이해하기 위한 참고 비교예요. 현재 추 개수나 수수깡 칸 수에 따라 바뀌는 값은 아니에요.")
    span_column, cost_column = st.columns(2)
    with span_column:
        fig = data_figure("받침 사이 거리 · 문헌의 참고 구간", "받침 사이 거리 (m)", "교량 종류", 420)
        for row, kind in enumerate(names):
            profile = BRIDGE_PLANNING[kind]
            low = profile["span_min"]
            # 현수교 끝의 삼각형은 그래프 창 밖으로도 이어짐을 나타냅니다.
            high = profile["span_max"] if profile["span_max"] is not None else 1500
            fig.add_trace(go.Scatter(
                x=[low, high], y=[row, row], mode="lines+markers", showlegend=False,
                line=dict(color=COLORS[kind], width=10),
                marker=dict(color=COLORS[kind], size=11,
                            symbol=["circle", "circle" if profile["span_max"] is not None else "triangle-right"]),
                hovertemplate=f"{kind}<br>{profile['span_label']}<br>{profile['span_note']}<extra></extra>",
            ))
            fig.add_annotation(x=low, y=row-0.22, text=profile["span_label"],
                               showarrow=False, xanchor="left", font=dict(size=12))
        fig.update_xaxes(range=[0, 1650], dtick=300)
        fig.update_yaxes(tickvals=list(range(len(names))), ticktext=names,
                         range=[len(names)-0.5, -0.6], title=None)
        fig.update_layout(margin=dict(l=75, r=20, t=65, b=55))
        chart(fig, "planning_span_comparison")
        st.write("이 선은 **설계 자료에 나온 구간의 예**예요. ‘이 길이까지만 만들 수 있다’는 뜻은 아니에요.")
        st.caption("현수교의 화살표는 더 긴 거리로 이어진다는 뜻이에요. 약 900 m는 긴 경간에서 비용상 유리해지는 기준이며, 짧은 현수교도 있어요.")
    with cost_column:
        scores = [BRIDGE_PLANNING[kind]["cost_index"] for kind in names]
        fig = data_figure("건설비 경향 · 학습용 가정값", "비용 점수 (단순교 = 100점)", "교량 종류", 420)
        fig.add_trace(go.Bar(
            x=scores, y=names, orientation="h", marker_color=[COLORS[kind] for kind in names],
            text=[f"{score}점" for score in scores], textposition="outside", cliponaxis=False,
            customdata=[BRIDGE_PLANNING[kind]["cost_reason"] for kind in names],
            hovertemplate="%{y}<br>수업용 비용 점수 %{x}점<br>%{customdata}<extra></extra>",
        ))
        fig.update_xaxes(range=[0, max(scores)*1.3])
        fig.update_yaxes(categoryorder="array", categoryarray=names, autorange="reversed", title=None)
        fig.update_layout(margin=dict(l=75, r=20, t=65, b=55), showlegend=False)
        chart(fig, "planning_cost_comparison")
        st.write("구조의 복잡함과 기초·케이블 공사를 비교하기 위한 **학습용 가정값**이에요. 실제 공사비나 실제 비용 비율은 아니에요.")
        st.caption("길이·폭·땅의 상태·공사 방법에 따라 실제 비용과 순서는 달라져요. 긴 거리를 건널 때는 비싼 구조라도 중간 받침을 줄여 더 경제적일 수 있어요.")
    with st.expander("참고 자료와 비용 점수의 뜻"):
        st.markdown(f"경간과 비용에 영향을 주는 요인은 [미국 FHWA 교량 형식 선택 안내서(2015), 18쪽]({PLANNING_SOURCE})를 참고했어요.")
        st.write("경간은 강철 교량 자료의 일부 범위를 미터로 바꾸고 반올림했어요. 단순교는 강판 거더, 트러스교는 한 구간 트러스의 예이며, 모든 재료·설계의 공통 한계나 최신 최장 기록을 뜻하지 않아요.")
        st.write("비용 점수 100·170·150·240·320은 보고서에 실린 수치가 아니라 이 앱에서 정한 학습용 값이에요. 예를 들어 320점이라고 실제 공사비가 단순교의 3.2배라는 뜻은 아니에요.")
        st.write("실제 교량은 필요한 거리와 땅의 조건을 먼저 살펴보고 비용을 비교해요. ‘항상 가장 싸고 가장 튼튼한 교량’ 하나를 고르는 것은 아니에요.")




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
        inner = center + side * length * .18 * direction
        outer = center + side * length * .5 * direction
        start, end = (inner, outer) if tension else (outer, inner)
        fig.add_annotation(
            name=f"{name}_{side}", ax=float(start[0]), ay=float(start[1]),
            x=float(end[0]), y=float(end[1]), axref="x", ayref="y", xref="x", yref="y",
            text="", showarrow=True, arrowhead=3, arrowsize=1.1, arrowwidth=2.5,
            arrowcolor=TENSION if tension else COMPRESSION,
        )


def add_member_force_arrows(fig, model, result, shown, scale, within, amplify):
    """대표 부재의 힘을 계산 부호에 맞춘 화살표로 표시합니다."""
    if not result["stable"]:
        return
    # (부재 역할, 경간 내 위치, 글상자의 가로·세로 간격, 축력 부호 필터)
    specs = {
        "단순교": [("바닥 보", .5, -45, -65, 0)],
        "아치교": [("바닥 보", .6, 35, -65, 0), ("아치", .3, -55, 55, 0),
                  ("세로 지지대", .75, 55, 35, 0)],
        "트러스교": [("바닥 보", .5, 0, -65, 0), (None, .7, 50, 60, 1),
                    (None, .3, -50, 35, -1)],
        "사장교": [("바닥 보", .5, 0, 65, 0), ("사장 케이블", .3, 40, -55, 0),
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
        color = TENSION if axial > 1e-7 else COMPRESSION if axial < -1e-7 else "#7160d2" if moment > 1e-9 else SUPPORT_COLOR
        member_name = {"바닥 보": "상판 아래 보", "수직 매다는 줄": "행어(세로줄)"}.get(e["role"], e["role"])
        text = member_name + (" · 휨" if moment > 1e-9 else "")
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
            offset = deck_half_depth(model["kind"]) / 2 if e["role"] == "바닥 보" else .4
            force_arrow_pair(fig, np.array([x, y]) + offset * perpendicular, direction,
                             axial > 0, min(2.2, max(1.8, member_length * .8)), f"member_axial_{index}")
        if model["kind"] == "단순교" and moment > 1e-9:
            force_arrow_pair(fig, [x, y + BEAM_HALF_DEPTH / 2], [1., 0.], False, 1.8, f"member_bending_{index}_upper")
            force_arrow_pair(fig, [x, y - BEAM_HALF_DEPTH / 2], [1., 0.], True, 1.8, f"member_bending_{index}_lower")
        fig.add_annotation(
            name=f"member_name_{index}", x=float(x), y=float(y), xref="x", yref="y",
            ax=offset_x, ay=offset_y, axref="pixel", ayref="pixel", text=text,
            showarrow=True, arrowhead=0, arrowwidth=1.2, arrowcolor=color,
            bgcolor="rgba(255,255,255,.96)", bordercolor=color, borderwidth=1, borderpad=4,
            font=dict(size=11, color="#1f2937"), align="center",
        )


def structure_figure(sweep, count, amplify, title=None, show_forces=False):
    model, r = sweep["model"], sweep["results"][count]
    xy = model["nodes"]
    L, scale = model["span"], 12/model["span"]
    kind = model["kind"]
    depth = deck_half_depth(kind)
    fig = base_figure(title or f"{kind} · 추 {count}개", [-5.2, 17.2], model["view_y"], 410)
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    if kind in ("사장교", "현수교"):
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)
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
            arrow(fig, (x, y), end, SUPPORT_COLOR, abs(value), f"{abs(value):.2f} N")
    if shape_available:
        samples = np.array(display_r["deck_samples"])
        order = np.argsort(samples[:, 0])
        samples = samples[order]
        y_at = lambda x: float(np.interp(x/scale, samples[:, 0], samples[:, 2]))*amplify*scale + depth
    else:
        y_at = lambda x: depth
    live_force = count*model["weight_mass"]*GRAVITY
    draw_weights(fig, count, y_at, weight_positions(count, 1., 11.),
                 arrow_size=1.2+min(live_force, 5)*.2, block_width=.55, total_force=live_force)
    if show_forces:
        add_member_force_arrows(fig, model, display_r, shown, scale, shape_available, amplify)
        fig.add_annotation(
            name="force_legend", x=.5, y=1.12, xref="paper", yref="paper", yanchor="bottom",
            text=f'<span style="color:{COMPRESSION}">→ ← 압축: 안으로 누름</span><br>'
                 f'<span style="color:{TENSION}">← → 인장: 밖으로 당김</span>',
            showarrow=False, font=dict(size=11, color="#1f2937"),
            bgcolor="rgba(255,255,255,.96)", borderpad=4,
        )
    if not within and has_valid_shape:
        label(fig, 6, 8.4, f"기준 초과 · 추 {display_count}개 형상·부재력 유지<br>추는 현재 {count}개 · 붕괴 후 예측 아님")
    elif not within and shape_available:
        label(fig, 6, 8.4, f"처음부터 기준 초과 · {display_count}개 참고 형상 유지<br>추는 현재 {count}개 · 붕괴 후 예측 아님")
    elif not r["stable"]:
        label(fig, 6, 7.8, "연결 각도가 자유로워 형태를 유지하지 못해요.<br>수치 대신 원래 뼈대를 표시합니다.")
    else:
        label(fig, 6, 9., f"최대 처짐 {r['deflection_mm']:.4f} mm · 변형 {amplify}배 확대")
    label(fig, 6, model["view_y"][0]+.5, f"받침 사이 {L*100:.0f} cm · 두 옆면의 힘을 합친 그림")
    return fig


def experiment_figure(sweep, count, amplify, zoom):
    """실험 화면의 카메라만 확대합니다. 부재 좌표와 계산값은 그대로 둡니다."""
    fig = structure_figure(sweep, count, amplify, show_forces=True)
    ys = [float(y) for trace in fig.data for y in trace.y if y is not None]
    bottom, top = min(ys) - 2., max(ys) + 4.
    if sweep["model"]["kind"] in ("사장교", "현수교"):
        deck_top = max(float(y) for trace in fig.data if (trace.name or "").startswith("바닥 보") for y in trace.y)
        bottom, top = min(ys) - 1.5, max(max(ys) + 1.25, deck_top + 3.5)
    center_y = (bottom + top) / 2
    half_y = max(8., top - bottom) / (2 * zoom)
    half_x = 11.2 / zoom
    fig.update_layout(height=560, dragmode="pan", margin=dict(t=110, b=40),
                      title=dict(y=.99, yanchor="top"))
    fig.update_xaxes(range=[6 - half_x, 6 + half_x], fixedrange=False)
    fig.update_yaxes(range=[center_y - half_y, center_y + half_y], fixedrange=False)
    # 상태와 경간 안내는 카메라 확대에도 화면 가장자리에 남겨 둡니다.
    fig.layout.annotations[-2].update(x=.5, y=1.02, xref="paper", yref="paper", yanchor="bottom")
    fig.layout.annotations[-1].update(x=.5, y=-.02, xref="paper", yref="paper", yanchor="top")
    return fig


def change_bridge_zoom(step):
    current = st.session_state.get("bridge_zoom_percent", 100)
    st.session_state["bridge_zoom_percent"] = 100 if step == 0 else max(50, min(250, current + step))


def overview_figure(sweeps, amplify):
    titles = [f"{k} · {capacity_text(sweeps[k])}" for k in BRIDGE_NAMES]
    fig = make_subplots(rows=3, cols=2, specs=[[{}, {}], [{"colspan": 2}, None], [{}, {}]],
                        subplot_titles=titles, vertical_spacing=.08, horizontal_spacing=.05)
    positions = [(1, 1), (1, 2), (2, 1), (3, 1), (3, 2)]
    for idx, (kind, (row, col)) in enumerate(zip(BRIDGE_NAMES, positions), 1):
        sweep = sweeps[kind]
        count = max(0, sweep["capacity"])
        source = structure_figure(sweep, count, amplify)
        for trace in source.data:
            fig.add_trace(trace, row=row, col=col)
        for shape in source.layout.shapes or []:
            fig.add_shape(shape.to_plotly_json(), row=row, col=col)
        xref, yref = ("x" if idx == 1 else f"x{idx}"), ("y" if idx == 1 else f"y{idx}")
        for ann in source.layout.annotations or []:
            data = ann.to_plotly_json()
            data.update(xref=xref, yref=yref)
            if data.get("showarrow"):
                data.update(axref=xref, ayref=yref)
            fig.add_annotation(data)
        fig.update_xaxes(range=[-5.2, 17.2], showgrid=False, showticklabels=False, zeroline=False, row=row, col=col)
        fig.update_yaxes(range=source.layout.yaxis.range, showgrid=False, showticklabels=False, zeroline=False,
                         scaleanchor=xref, scaleratio=1, row=row, col=col)
    fig.update_layout(height=1160, showlegend=False, margin=dict(l=10, r=10, t=35, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


def force_observation(sweep, count):
    """현재 해석에서 확인된 힘만 학생용 문장으로 설명합니다."""
    r, model = sweep["results"][count], sweep["model"]
    if not r["stable"]:
        return "현재 설정에서는 구조의 평형을 계산할 수 없어요. 심화 설정에서 연결과 보강 조건을 살펴보세요."
    if max([abs(n) for n in r["axial"]] + r["moments"], default=0.) < 1e-9:
        return "현재 계산에서는 부재에 작용하는 힘이 거의 없어요. 추를 올려 변화를 살펴보세요."
    kind, members = model["kind"], model["members"]
    if kind == "단순교":
        return "상판 아래 보에 휨이 생겨요. 이 하중 조건에서 보의 윗부분은 압축, 아랫부분은 인장을 받아요."
    if kind == "아치교":
        compressed = any(r["axial"][i] < -1e-7 for i, e in enumerate(members) if e["role"] == "아치")
        return ("아치 부재에 압축력이 계산됐어요. 아치와 받침을 따라 힘이 전달되는 길을 찾아보세요."
                if compressed else "현재 설정의 아치 부재 색과 상판 보의 휨을 살펴보세요.")
    if kind == "트러스교":
        forces = [r["axial"][i] for i, e in enumerate(members) if e["form"] == "rod"]
        if any(n > 1e-7 for n in forces) and any(n < -1e-7 for n in forces):
            return "트러스의 일부 막대는 인장, 다른 막대는 압축을 받아요. 상판 보는 휨도 받아요."
        return "현재 설정에서 막대의 인장·압축 색과 상판 보의 휨을 살펴보세요."
    if r["tension"] > 1e-7:
        return ("사장 케이블이 인장력으로 상판을 직접 지지해요. 힘은 주탑과 앵커로 전달돼요."
                if kind == "사장교" else "행어(세로줄)와 주케이블이 인장력으로 상판을 매달아요. 힘은 주탑과 앵커로 전달돼요.")
    return "현재 계산에서 케이블 장력은 거의 없어요. 상판 보와 받침의 힘도 함께 살펴보세요."


def load_change_observation(sweep, count, previous_count):
    """같은 모형의 두 계산 결과를 비교하며, 변화 방향을 미리 가정하지 않습니다."""
    if previous_count is None or previous_count == count:
        return None
    before, after = (sweep["results"][i] for i in (previous_count, count))
    if not before["stable"] or not after["stable"]:
        return "변경 전후 중 평형을 계산하지 못한 상태가 있어 처짐을 비교할 수 없어요."
    delta = after["deflection_mm"] - before["deflection_mm"]
    change = "커졌어요" if delta > 1e-9 else "작아졌어요" if delta < -1e-9 else "거의 같아요"
    return (f"같은 구조·설정에서 추 {previous_count}개 → {count}개: "
            f"계산된 최대 처짐이 {before['deflection_mm']:.4f} → {after['deflection_mm']:.4f} mm로 {change}. "
            "표시 자릿수보다 작은 차이는 숫자가 같아 보일 수 있어요.")


def show_result(sweep, count, amplify, key, figure_slot=None, previous_count=None, view_zoom=None):
    kind, r = sweep["model"]["kind"], sweep["results"][count]
    fig = (structure_figure(sweep, count, amplify) if view_zoom is None
           else experiment_figure(sweep, count, amplify, view_zoom))
    if figure_slot is None:
        chart(fig, key)
    else:
        with figure_slot:
            chart(fig, key)
    st.markdown("**구조적 반응 관찰**")
    if not r["stable"]:
        st.warning("형태 불안정 · 대각선이나 받침으로 움직임을 막아야 해요.")
    elif r["utilization"] > 1+1e-9:
        st.error(f"계산 기준 초과 · 먼저 확인할 곳: {r['governing']}")
    else:
        st.success(f"현재 계산 기준 이내 · 가장 여유가 작은 곳: {r['governing']}")
    st.metric("현재 계산된 최대 처짐", "계산 불가" if not r["stable"] else f"{r['deflection_mm']:.4f} mm")
    change = load_change_observation(sweep, count, previous_count)
    if change:
        st.info(change)
    st.write(force_observation(sweep, count))
    st.caption(f"빨강: 압축 · 파랑: 인장 · 회색: 받침 또는 축력이 거의 없는 부재. 사각형 보의 두께는 내부 힘을 보기 위한 표현이에요. "
               f"그림의 변형은 {amplify}배로 표시하며, mm 수치는 실제 계산값이에요. 0배는 원래 모양이에요.")
    if view_zoom is not None:
        st.caption("부재 옆의 화살표는 방향으로 힘의 종류를 나타내요. 안으로 모이면 압축, 밖으로 벌어지면 인장이에요. 화살표 길이는 힘의 크기를 나타내지 않아요.")
    if fig.layout.meta["frozen"] and fig.layout.meta["display_count"] is not None:
        st.warning(f"그림의 변형·부재력은 추 {fig.layout.meta['display_count']}개 상태에서 멈춰 있어요. "
                   f"추와 주황 하중 화살표는 현재 {count}개를 표시해요. 처짐 수치는 현재 조건의 참고 계산값이에요.")
    m = sweep["model"]
    load_mass_g = m["weight_mass"]*1000
    load_description = (
        f"추 {count}개 × {load_mass_g:g} g의 하중을 상판에 균등하게 나누어 싣는 모형입니다."
        if count else "추 하중은 없고, 설정에 따라 다리 자체 무게만 작용하는 모형입니다."
    )
    with st.expander("심화 결과 · 실험 범위와 모형 정보"):
        st.metric("기준 대비 여유", "계산 불가" if not r["stable"] else "힘 없음" if r["reserve"] is None else f"{r['reserve']:.2f}배")
        st.caption(load_description)
        st.write(f"0~20개 실험에서 기준 이내인 추: **{capacity_text(sweep)}**. 이 값만으로 실제 교량의 우열은 정할 수 없습니다.")
        if sweep["capacity"] >= 0:
            limit_force = sweep["capacity"] * m["weight_mass"] * GRAVITY
            qualifier = " 이상 (실험 범위에서 한계에 닿지 않음)" if sweep["censored"] else ""
            st.write(f"마지막 기준 이내 추의 추가 하중: **{limit_force:.3f} N{qualifier}** (자중 제외).")
        if r["stable"]:
            st.write(f"현재 총 수직 하중(자중 포함 설정 반영): {r['total_force']:.3f} N · 가장 큰 케이블 장력: {r['tension']:.3f} N")
        st.caption(f"골조 {m['skeleton_mass']*1000:.1f} g · 줄 {m['cable_mass']*1000:.1f} g · 종이 {m['paper_mass']*1000:.1f} g. "
                   f"막대 한 개 단면 {m['section_mm'][0]:.1f} × {m['section_mm'][1]:.1f} mm. 받침·접착제 무게는 제외합니다.")
    if r["stable"] and r["utilization"] > 1:
        st.caption("처짐 숫자는 선형식의 참고 계산값입니다. 기준을 넘은 뒤 실제로 버티는지나 붕괴 모양을 뜻하지 않아요.")


def load_curve(sweeps, names, count):
    fig = data_figure("추를 늘릴 때의 처짐 · 같은 조건으로 계산", "추 개수", "최대 처짐 (mm)", 360)
    for kind in names:
        sweep = sweeps[kind]
        for within in [True, False]:
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
    fig.add_vline(x=count, line_color=SUPPORT_COLOR, line_dash="dash", annotation_text=f"현재 {count}개")
    fig.update_xaxes(dtick=2)
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
        st.caption("균질한 막대 두 옆면 + 종이 바닥판을 계산해요. 아래 값은 특정 제품을 측정한 수치가 아니에요.")
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
        st.caption("처짐 기준과 연결·받침 허용 힘은 실험 조건입니다. 실제 설계 기준을 인증하는 값은 아니에요.")
        st.markdown("**교량 형상**")
        arch_percent = st.slider("아치 높이 · 받침 사이 거리의 (%)", 10, 40, 25, 5, key="arch_percent")
        panels = st.slider("트러스 사각형 칸 수", 1, 11, 5, key="triangles")
        truss_percent = st.slider("트러스 뼈대 높이 · 경간의 (%)", 10, 35, 20, 5, key="truss_percent")
        reinforced = st.toggle("트러스 대각선 보강재 추가", True, key="add_diagonals")
        tower_cm = st.slider("케이블교 바닥판 위 주탑 높이 (cm)", 5, 50, 15, 1, key="tower_cm")
        sag_percent = st.slider("현수교 주케이블 늘어진 깊이 · 주탑 높이의 (%)", 20, 80, 50, 5, key="sag_percent")
        st.markdown("**그림**")
        amplify = st.slider(
            "변형 확대 배율", 0, 30, 10, key="amplify", format="%d배",
            help="그림만 확대합니다. mm 수치·힘·한계 개수는 변하지 않습니다. 0배는 원래 모양입니다.",
        )
        st.caption("밝은/다크 모드: 오른쪽 위 ⋮ → Settings → Theme")
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
guide_tab, lab_tab, comparison_tab = st.tabs(["📖 실험실 안내", "🌉 교량 탐구", "🔎 교량 비교"])
with guide_tab:
    with st.container(key="lab_guide"):
        st.title("🌉 교량 구조와 힘의 분산")
        st.markdown('<p class="guide-subtitle">개념 원리 탐구 실험실</p>', unsafe_allow_html=True)
        st.write("추를 올리면서 교량이 하중을 어떻게 전달하고 견디는지 관찰해 보자.")
        st.info("안내를 읽었다면 위의 ‘🌉 교량 탐구’ 탭에서 시작하세요.", icon="👉")
        st.subheader("1. 이렇게 탐구해요")
        for col, (title, text) in zip(st.columns(3), [
            ("① 건설 상황 살펴보기", "본문에서 상황과 교량 그림을 보고, 추를 올리면 힘이 어디로 전달될지 예상해요."),
            ("② 추 올리기", "본문에서 현재 교량의 허용 개수까지 추를 올리고 처짐(mm)과 힘의 변화를 살펴봐요. 실험 범위는 최대 20개예요."),
            ("③ 원리와 실제 적용 연결하기", "하중이 가는 길과 인장·압축·휨을 설명하고, 실제 건설에서 유리한 이유와 한계를 생각해요."),
        ]):
            with col, st.container(border=True):
                st.markdown(f"**{title}**")
                st.write(text)
        st.subheader("2. 색과 숫자 읽기")
        symbols = [
            ("⚖️ 금속 추 · 추가 하중", "종이 바닥판이 추 무게를 길이 방향으로 고르게 나누어 전달한다고 가정해요. 그림의 추 위치를 옮기는 실험은 아니에요."),
            ("🟧 주황 · 아래로 누르는 힘", "추가 누르는 힘이에요. 회색 화살표는 땅과 받침이 다리를 지탱하는 힘이에요."),
            ("🔴 빨강 · 압축", "막대를 누르는 힘이에요. 길고 가는 막대는 찌그러지기 전에 옆으로 휘어 꺾일 수도 있어요."),
            ("🔵 파랑 · 인장", "막대나 줄을 잡아당기는 힘이에요. 부재 위에 마우스를 올리면 힘을 확인할 수 있어요."),
        ]
        for start in range(0, 4, 2):
            for col, (title, text) in zip(st.columns(2), symbols[start:start+2]):
                with col, st.container(border=True):
                    st.markdown(f"**{title}**")
                    st.write(text)
        st.write("그림 속 변형은 보기 쉽게 **확대**해요. 실제 계산된 처짐은 **mm** 숫자로 비교하세요. 보는 내부를 볼 수 있도록 긴 사각형 띠로 그려요. 띠의 두께는 실제 단면 두께가 아니에요.")
        st.table([
            {"용어": "처짐", "쉬운 뜻": "보가 아래로 휘는 거리예요. 1 mm는 1 cm의 10분의 1이에요."},
            {"용어": "장력", "쉬운 뜻": "줄을 팽팽하게 당기는 힘이에요. N(뉴턴)으로 나타내요."},
            {"용어": "좌굴", "쉬운 뜻": "길고 가는 막대가 눌리다가 옆으로 휘어 꺾이는 현상이에요."},
            {"용어": "기준 대비 여유", "쉬운 뜻": "현재 요구량에 비해 허용량이 몇 배인지예요. 1배보다 작으면 계산 기준을 넘어요."},
        ])
        st.subheader("3. 이번 실험은 무엇을 비교하나요?")
        with st.container(border=True):
            st.write("**교량마다 정해 둔 ‘최대 추 개수’는 없어요.** 거리·단면·재료·줄·받침 조건에서 계산해요. ‘항상 가장 튼튼한 교량’이라는 순위도 없어요.")
            st.write("기본은 **교실 크기의 균질 막대 모형**이에요. 재료 성질은 예시이므로 실제 수수깡·접힌 종이의 결과와는 차이가 있어요. 선생님이 측정값으로 보정할 수 있어요.")
            st.write("시뮬레이션에서 확인할 수 있는 것은 설정한 조건에서의 처짐과 힘의 분담이에요. 실제 교량의 지반·시공·유지관리 조건은 이 모형으로 검증하지 않아요.")
            st.write("0~20개 안에서 처음 기준을 넘는 직전 개수를 표시해요. 20개도 기준 이내이면 **‘20개 이상’**으로 표시해요. 아직 진짜 한계를 찾은 것은 아니에요.")
            st.write("추 조절기는 선택한 교량의 계산된 허용 개수까지만 움직여요. 교량이나 설정을 바꿔 한계가 낮아지면 추도 그 범위로 줄여요.")
            st.write("기준을 넘으면 마지막 기준 이내의 변형과 부재력을 유지해요. 처음부터 기준을 넘으면 처음 계산된 참고 형상을 보여 줘요. 실제 붕괴 과정과 바람·지진은 이 실험에 포함하지 않아요.")

with lab_tab:
    selected_bridge = st.selectbox("학습할 교량과 건설 상황", BRIDGE_NAMES, format_func=lambda name: {
        "단순교": "짧은 하천 · 단순교",
        "아치교": "견고한 계곡 · 아치교",
        "트러스교": "열차가 지나는 길 · 트러스교",
        "사장교": "넓은 강 · 사장교",
        "현수교": "깊은 해협 · 현수교",
    }[name], key="selected_bridge")
    st.caption("경간·재료·형상은 왼쪽 사이드바의 접힌 ‘심화 설정’에서 바꿀 수 있어요. 추 개수는 선택한 교량의 허용 범위 안에서 유지돼요.")
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
load_limit = max(0, selected_sweep["capacity"])
requested_load = st.session_state.get("load", 0)
load_was_limited = requested_load > load_limit
if "load" in st.session_state:
    # 슬라이더 상한이 바뀌어 위젯이 다시 만들어져도 허용 범위의 값을 유지합니다.
    st.session_state["load"] = max(0, min(requested_load, load_limit))

with chapter_section:
    bridge_chapter_intro(selected_bridge)
    st.subheader("② 하중 탐구 활동")
    st.write("**하중이 증가하면 이 교량의 어느 부분에 어떤 힘이 작용하고, 구조는 어떻게 변할까?**")
    st.caption("추를 올리면서 ① 하중이 가는 길 ② 인장·압축·휨을 받는 부재 ③ 처짐의 변화를 관찰해요.")
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
    st.caption("오른쪽 ＋ / － / ↺: 화면 확대·축소·원래 크기. 확대한 그림은 드래그해서 옮길 수 있어요. "
               "힘·처짐 계산값과 변형 확대 배율은 그대로예요.")
    if load_limit > 0:
        load = st.slider("추를 올려 하중 늘리기", 0, load_limit, key="load", format="%d개")
    else:
        load = st.number_input("추를 올려 하중 늘리기", min_value=0, max_value=0, key="load", disabled=True)
    if load_was_limited:
        st.info(f"현재 교량·설정에서 허용되는 추는 {load_limit}개까지여서 추 개수를 줄였어요.")
    if selected_sweep["capacity"] < 0:
        st.caption("추를 올릴 수 없는 상태예요. 현재 구조의 안정성·자중과 심화 설정을 확인해 보세요.")
    elif selected_sweep["censored"]:
        st.caption("현재 조건에서 20개까지 기준 이내예요. 실험 범위 안에서만 추를 올릴 수 있어요.")
    else:
        st.caption(f"현재 조건에서는 추 {load_limit}개까지 올릴 수 있어요. 그 이상은 계산 기준을 넘어서 추가할 수 없어요.")
    st.write(f"추 **{load}개 × {mass_g} g = {load*mass_g} g**을 올렸어요. "
             f"추가 하중은 **{load*mass_g/1000*GRAVITY:.3f} N**이에요.")
    st.caption("추의 총질량은 g, 누르는 힘(하중)은 N으로 나타내요. 추의 힘은 상판 전체에 고르게 나뉘어 작용한다고 가정해요. 추 0개에도 자체 무게가 작용할 수 있어요.")
    context = (selected_bridge, tuple(sorted(cfg.items())), reinforced if selected_bridge == "트러스교" else True)
    previous = st.session_state.get("_last_observation", {})
    previous_count = previous.get("count") if previous.get("context") == context else None
    show_result(selected_sweep, load, amplify, f"selected_{selected_bridge}", figure_slot, previous_count,
                view_zoom=zoom_percent / 100)
    st.session_state["_last_observation"] = dict(context=context, count=load)
    st.subheader("③ 하중 전달 원리와 ④ 실제 적용 조건")
    bridge_guide_card(selected_bridge)
    if selected_bridge in ("트러스교", "사장교", "현수교"):
        with st.expander("심화 탐구 · 보강과 형상 비교", expanded=False):
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
                st.caption("이 두 그림에서는 막대 단면을 똑같이 두고 오른쪽에 굵은 실선 대각선만 더합니다. 추가한 재료량은 증가해요. 5종 비교는 왼쪽에서 선택한 재료 비교 기준을 사용합니다.")
                st.info("회전이 자유로운 연결부의 사각형은 옆으로 움직일 수 있어 처짐 숫자를 정할 수 없어요. 접착부가 단단하게 각도를 고정하는 실물 사각형은 이 가정보다 잘 버틸 수 있어요.")
                fig = data_figure("대각선으로 보강한 칸 수를 바꾸면?", "칸 수", "최대 처짐 (mm)", 310)
                ns = list(range(1, 12))
                series = [solve_sweep("트러스교", dict(cfg, panels=n))["results"][load] for n in ns]
                fig.add_trace(go.Scatter(x=ns, y=[r["deflection_mm"] for r in series], mode="lines+markers", name="대각선 보강",
                                         line=dict(color=COLORS["트러스교"])))
                fig.add_vline(x=panels, line_dash="dot", line_color=SUPPORT_COLOR)
                fig.update_xaxes(dtick=1)
                chart(fig, "truss_efficiency")
                st.caption("기준을 넘은 설정의 처짐도 선형식의 참고값으로 포함합니다. 칸이 많다고 무조건 유리하지는 않아요. 재료 총량을 같게 하면 막대가 가늘어지고, 줄어든 압축 막대 길이는 좌굴을 줄이는 데 도움이 돼요.")
                with st.container(border=True):
                    st.markdown("**대각선은 왜 필요할까요?**")
                    st.write("사각형은 변 길이가 같아도 모서리 각도가 바뀔 수 있어요. 대각선 하나를 넣으면 두 삼각형이 생겨, 막대를 늘리거나 줄이지 않고는 모양을 바꾸기 어려워져요.")
                    st.write("삼각형이 생겨도 막대·연결부가 무한히 강해지지는 않아요. 눌리는 막대가 길고 가늘면 좌굴하고, 연결부가 약하면 먼저 떨어질 수 있어요.")

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

                st.write("사장교는 경사진 줄이 상판을 직접 지지하고, 현수교는 세로줄에서 주케이블로 힘을 전달해요.")
                st.caption("현수교는 주탑 높이를 바꿀 때 늘어진 깊이의 비율을 유지해요. 이 모형은 실제 초장경간의 바람 진동이나 시공 조건을 시험하지 않아요.")

with comparison_tab:
    st.subheader("5대 교량 · 조건에 따라 달라지는 결과")
    st.caption("추 개수는 ‘교량 탐구’ 탭에서 조절한 값을 함께 사용해요.")
    mode = st.radio("비교 방식", ["두 교량 비교", "5종 전체 비교"], horizontal=True, key="comparison_mode")
    st.caption(f"공통 거리 {span_cm} cm · {comparison_basis} · 추 1개 {mass_g} g · 자체 무게 {'반영' if self_weight else '제외'}. 트러스교는 대각선 보강 구조입니다. 배치는 튼튼함의 순서가 아니에요.")
    st.info("이 실험 결과는 설정된 경간, 재료, 하중 및 구조 조건에서 나타난 결과입니다. 실제 교량은 경간, 지반, 시공 조건 등에 따라 적합한 구조가 달라질 수 있습니다. 모형의 하중 지지 능력과 실제 건설 상황에서의 적합성은 서로 다른 판단 기준입니다.")
    if comparison_basis == "골조 재료량 같게":
        st.caption("재료량 비교는 골조 막대의 총부피 기준이에요. 케이블·종이 바닥판·받침은 같은 재료량으로 맞추지 않습니다.")
    else:
        st.caption("막대 굵기 비교는 같은 단면을 쓰며, 구조마다 골조 총재료량과 케이블·종이 사용량은 다를 수 있어요.")
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
                show_result(sweeps[kind], load, amplify, f"comparison_structure_{i}")
    else:
        names = BRIDGE_NAMES
        st.markdown("### ⚖️ 0~20개 실험에서 마지막으로 기준 이내인 모습")
        st.write("‘20개 이상’이면 실험 범위 안에서 한계에 닿지 않았어요. ‘0개도 기준 초과’라면 자체 무게부터 조건을 바꿔야 해요. 평형을 계산하지 못한 구조만 변형 없이 뼈대로 표시해요.")
        chart(overview_figure(sweeps, amplify), "capacity_bridge_overview")
        st.caption("이 그림만 각 교량의 마지막 기준 이내 추 개수를 올립니다. 아래 그래프는 하중 탐구 본문에서 선택한 현재 추 개수로 비교합니다. 트러스교는 가운데에 배치했습니다.")
        show_planning_comparison(BRIDGE_NAMES)
    st.markdown(f"### 현재 추 {load}개 · 같은 하중 비교")
    cols = st.columns(2)
    for col, field, title, unit, key in [
        (cols[0], "reserve", "현재 하중의 기준 대비 여유", "여유 (배)", "safety_comparison"),
        (cols[1], "deflection_mm", "현재 하중의 최대 처짐", "최대 처짐 (mm)", "deformation_comparison"),
    ]:
        values = [sweeps[k]["results"][load][field] for k in names]
        fig = data_figure(title, "교량 종류", unit, 380)
        fig.add_trace(go.Bar(x=names, y=values, marker_color=[COLORS[k] for k in names],
                             text=["계산 불가/힘 없음" if v is None else f"{v:.2f}" for v in values], textposition="outside"))
        if field == "reserve":
            fig.add_hline(y=1, line_color=COMPRESSION, line_dash="dash", annotation_text="1배 = 기준")
        else:
            fig.add_hline(y=cfg["span"]*1000/divisor, line_color=COMPRESSION, line_dash="dash", annotation_text="처짐 기준")
        fig.update_layout(showlegend=False)
        with col:
            chart(fig, key)
    chart(load_curve(sweeps, names, load), "load_deformation_curves")
    st.caption("실선은 기준 이내, 점선은 기준 초과 후의 선형 참고값이에요. 점선은 실제 붕괴 후 모습을 예측하는 선이 아니에요.")
    rows = []
    for kind in names:
        sweep, r = sweeps[kind], sweeps[kind]["results"][load]
        rows.append({"교량": kind, "0~20개 실험의 결과": capacity_text(sweep),
                     "처짐 (mm)": None if r["deflection_mm"] is None else round(r["deflection_mm"], 3),
                     "기준 대비 여유 (배)": None if r["reserve"] is None else round(r["reserve"], 2),
                     "현재 가장 빠듯한 기준": r["governing"], "다리 무게 (g)": round(sweep["model"]["mass"]*1000, 1)})
    st.dataframe(rows, hide_index=True, width="stretch")
    with st.expander("쉽게 이해하기 · 덜 휘면 무조건 더 튼튼할까요?", expanded=True):
        st.write("꼭 그렇지는 않아요. 덜 휘어도 약한 연결부가 먼저 떨어지거나, 압축 막대가 옆으로 꺾일 수 있어요. 그래서 여러 기준 중 가장 여유가 적은 곳을 함께 확인해요.")
        st.write("‘기준 대비 여유 2배’는 현재 필요한 힘·처짐에 비해 허용량이 두 배라는 뜻이에요. 다리 자체 무게도 포함하므로, 추를 정확히 두 배 더 올려도 된다는 뜻은 아니에요.")
        st.write("트러스가 많은 추를 버텨 보일 때는 삼각형 부재의 힘 분담과 짧게 설정한 좌굴 길이가 영향을 줘요. 케이블을 골조 재료량에서 빼는 등 모형의 가정도 결과에 영향을 주므로, 실제 교량의 우열을 뜻하지 않아요.")
        st.write("거리와 재료량을 맞춘 뒤 조건을 하나씩 바꿔 보세요. 짧은 다리, 긴 다리, 약한 받침에 유리한 구조가 서로 달라져요. 이 모형 결과를 실제 큰 교량의 우열로 해석하지 않아요.")
    st.markdown("### 📖 탐구한 교량의 특징과 장단점")
    bridge_guide_cards(names)

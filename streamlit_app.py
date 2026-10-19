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
        "feature": "바닥판 아래의 보(가로로 놓인 긴 구조물)가 무게를 양쪽 받침으로 전달합니다.",
        "advantage": "구조가 단순하고 공사 기간이 짧으며 제작 비용이 저렴함.",
        "disadvantage": "무게를 올리면 가운데가 많이 휘어, 받침과 받침 사이를 너무 멀리 띄우기 어려움.",
        "examples": "한강대교 일부, 일반 육교 및 고속도로 교량",
    },
    "아치교": {
        "title": "아치교",
        "feature": "둥근 아치를 따라 누르는 힘을 양쪽 받침으로 보냅니다. 받침은 옆으로 밀리지 않도록 버텨 줍니다.",
        "advantage": "무게를 아치 양끝으로 나누어 보내 누르는 힘을 잘 견딤. 받침 사이를 넓힐 수 있음.",
        "disadvantage": "양쪽 받침 아래의 땅이 단단해야 하며, 둥근 아치를 만들고 설치하는 과정이 복잡함.",
        "examples": "방화대교, 시드니 하버 브릿지",
    },
    "트러스교": {
        "title": "트러스교",
        "feature": "삼각형으로 연결된 막대들이 누르는 힘과 당기는 힘을 나누어 받습니다.",
        "advantage": "삼각형 보강으로 잘 찌그러지지 않으며, 비교적 적은 재료로 가볍고 튼튼하게 만들 수 있음.",
        "disadvantage": "막대끼리 연결한 곳이 많아 조립과 점검·수리가 복잡함.",
        "examples": "성수대교, 철도교량",
    },
    "사장교": {
        "title": "사장교",
        "feature": "주탑에서 비스듬히 뻗은 줄이 바닥판을 직접 당겨서 받쳐 줍니다.",
        "advantage": "비스듬한 줄이 바닥판을 직접 받쳐 주어, 중간에 받침을 많이 세우지 않고 길고 아름다운 다리를 만들 수 있음.",
        "disadvantage": "현수교보다 주탑 주변의 바닥판이 강하게 눌릴 수 있음. 바람에 흔들리는 정도도 꼼꼼히 살펴야 함.",
        "examples": "인천대교, 서해대교",
    },
    "현수교": {
        "title": "현수교",
        "feature": "굵은 주케이블에 세로줄로 바닥판을 매답니다. 주탑과 양끝의 고정 장치(앵커리지)가 함께 버팁니다.",
        "advantage": "굵은 주케이블과 양끝 고정 장치 덕분에 다섯 교량 형식 중 가운데 받침 사이를 가장 길게 만들 수 있음.",
        "disadvantage": "큰 고정 장치를 만들고 케이블을 설치하는 비용이 많이 듦. 바람과 지진의 흔들림을 견디도록 꼼꼼히 설계해야 함.",
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
PLOT_CONFIG = {"displayModeBar": False, "responsive": True}

st.set_page_config(page_title="교량 구조와 힘의 분산", page_icon="🌉", layout="wide", initial_sidebar_state="expanded")
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
    title, feature, advantage, disadvantage, examples = (
        escape(guide[field]) for field in ("title", "feature", "advantage", "disadvantage", "examples")
    )
    st.markdown(
        f'<article class="bridge-card" data-bridge="{escape(kind)}" '
        f'style="--bridge-accent: {COLORS[kind]};">'
        f'<h4>🌉 {title}</h4>'
        f'<p><strong>특징</strong><br>{feature}</p>'
        f'<p><strong>장점</strong><br>{advantage}</p>'
        f'<p><strong>단점</strong><br>{disadvantage}</p>'
        f'<p><strong>📍 대표 사례</strong><br>{examples}</p>'
        '</article>',
        unsafe_allow_html=True,
    )


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
        fig = data_figure("건설비 경향 · 학습용 상대 비용", "비용 점수 (단순교 = 100점)", "교량 종류", 420)
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
        st.write("구조의 복잡함과 기초·케이블 공사를 비교하기 위한 **수업용 약속 점수**예요. 실제 공사비나 실제 비용 비율은 아니에요.")
        st.caption("길이·폭·땅의 상태·공사 방법에 따라 실제 비용과 순서는 달라져요. 긴 거리를 건널 때는 비싼 구조라도 중간 받침을 줄여 더 경제적일 수 있어요.")
    with st.expander("참고 자료와 비용 점수의 뜻"):
        st.markdown(f"경간과 비용에 영향을 주는 요인은 [미국 FHWA 교량 형식 선택 안내서(2015), 18쪽]({PLANNING_SOURCE})를 참고했어요.")
        st.write("경간은 강철 교량 자료의 일부 범위를 미터로 바꾸고 반올림했어요. 단순교는 강판 거더, 트러스교는 한 구간 트러스의 예이며, 모든 재료·설계의 공통 한계나 최신 최장 기록을 뜻하지 않아요.")
        st.write("비용 점수 100·170·150·240·320은 보고서에 실린 수치가 아니라 이 앱에서 정한 학습용 값이에요. 예를 들어 320점이라고 실제 공사비가 단순교의 3.2배라는 뜻은 아니에요.")
        st.write("실제 교량은 필요한 거리와 땅의 조건을 먼저 살펴보고 비용을 비교해요. ‘항상 가장 싸고 가장 튼튼한 교량’ 하나를 고르는 것은 아니에요.")




# ── 3. 계산한 변위와 힘을 그대로 그림에 연결합니다 ────────────────────
def structure_figure(sweep, count, amplify, title=None):
    model, r = sweep["model"], sweep["results"][count]
    xy = model["nodes"]
    L, scale = model["span"], 12/model["span"]
    kind = model["kind"]
    fig = base_figure(title or f"{kind} · 추 {count}개", [-5.2, 17.2], model["view_y"], 410)
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    within = r["stable"] and r["utilization"] <= 1+1e-9
    u = r["u"].reshape(-1, 3)[:, :2] if within else np.zeros_like(xy)
    shown = (xy+amplify*u)*scale
    for index, e in enumerate(model["members"]):
        a, b = e["i"], e["j"]
        N = r["axial"][index] if r["stable"] else 0.
        color = TENSION if N > 1e-7 else COMPRESSION if N < -1e-7 else SUPPORT_COLOR
        force_name = "인장" if N > 1e-7 else "압축" if N < -1e-7 else "축력 거의 없음"
        width = (4.8 if e["role"] == "대각선 보강재" else 3.) + min(abs(N), 10)*.22
        if e["form"] == "cable":
            color = TENSION if N > 1e-7 else SUPPORT_COLOR
        if within and amplify > 0:
            line(fig, xy[[a, b], 0]*scale, xy[[a, b], 1]*scale, "rgba(128,128,128,.3)", 1)
        if e["role"] == "바닥 보" and within:
            ps = [p for p in r["deck_samples"] if xy[a, 0]-1e-10 <= p[0] <= xy[b, 0]+1e-10]
            xs = [(p[0]+amplify*p[1])*scale for p in ps]
            ys = [amplify*p[2]*scale for p in ps]
        else:
            xs, ys = shown[[a, b], 0], shown[[a, b], 1]
        moment = r["moments"][index] if r["stable"] else 0.
        line(fig, xs, ys, color, width, name=f"{e['role']} · 양옆 합계 {force_name} {abs(N):.3f} N · 휨 {moment:.4f} N·m")
        if kind == "단순교" and e["role"] == "바닥 보" and moment > 1e-9:
            line(fig, xs, np.array(ys)+.055, COMPRESSION, 1.5, name="보 윗면 · 휨에 의한 압축")
            line(fig, xs, np.array(ys)-.055, TENSION, 1.5, name="보 아랫면 · 휨에 의한 인장")
    for node_id in sorted(set(i//3 for i in model["fixed"] if i % 3 == 1)):
        x, y = xy[node_id]*scale
        support(fig, x, y, roller=3*node_id not in model["fixed"], depth=.65, half_width=.28)
    if r["stable"]:
        for dof in model["fixed"]:
            value = r["reactions"][dof]
            if abs(value) < 1e-7 or dof % 3 == 2:
                continue
            x, y = xy[dof//3]*scale
            distance = math.copysign(min(1.7, abs(value)*.35), value)
            end = (x+distance, y) if dof % 3 == 0 else (x, y+distance)
            arrow(fig, (x, y), end, SUPPORT_COLOR, abs(value), f"{abs(value):.2f} N")
    if within:
        samples = np.array(r["deck_samples"])
        order = np.argsort(samples[:, 0])
        samples = samples[order]
        y_at = lambda x: float(np.interp(x/scale, samples[:, 0], samples[:, 2]))*amplify*scale
    else:
        y_at = lambda x: 0.
    live_force = count*model["weight_mass"]*GRAVITY
    draw_weights(fig, count, y_at, weight_positions(count, 1., 11.),
                 arrow_size=1.2+min(live_force, 5)*.2, block_width=.55, total_force=live_force)
    if not r["stable"]:
        label(fig, 6, 7.8, "연결 각도가 자유로워 형태를 유지하지 못해요.<br>수치 대신 원래 뼈대를 표시합니다.")
    elif not within:
        label(fig, 6, 8.4, "계산 기준 초과 · 원래 모양 표시<br>붕괴 후 모양은 예측하지 않습니다.")
    else:
        label(fig, 6, 9., f"최대 처짐 {r['deflection_mm']:.2f} mm · 변형 {amplify}배 확대")
    label(fig, 6, model["view_y"][0]+.5, f"받침 사이 {L*100:.0f} cm · 두 옆면의 힘을 합친 그림")
    return fig


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


def show_result(sweep, count, amplify, key):
    kind, r = sweep["model"]["kind"], sweep["results"][count]
    if not r["stable"]:
        st.warning("형태 불안정 · 대각선이나 받침으로 움직임을 막아야 해요.")
    elif r["utilization"] > 1+1e-9:
        st.error(f"계산 기준 초과 · 먼저 확인할 곳: {r['governing']}")
    else:
        st.success(f"현재 계산 기준 이내 · 가장 여유가 작은 곳: {r['governing']}")
    chart(structure_figure(sweep, count, amplify), key)
    cols = st.columns(3)
    cols[0].metric("추 0~20개 중 기준 이내", capacity_text(sweep))
    cols[1].metric("최대 처짐", "계산 불가" if not r["stable"] else f"{r['deflection_mm']:.2f} mm")
    cols[2].metric("기준 대비 여유", "계산 불가" if not r["stable"] else "힘 없음" if r["reserve"] is None else f"{r['reserve']:.2f}배")
    m = sweep["model"]
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


# ── 4. 공통 조건과 왼쪽 하중 조절 ──────────────────────────────────────
with st.sidebar:
    st.header("⚖️ 하중 조절")
    load = st.slider("추 개수", 0, 20, 0, key="load", format="%d개")
    mass_g = st.select_slider("추 한 개의 무게", options=[10, 20, 50, 100], value=20, format_func=lambda x: f"{x} g", key="weight_mass_g")
    st.metric("추로 더한 무게", f"{load*mass_g} g", f"아래로 누르는 힘 {load*mass_g/1000*GRAVITY:.2f} N", delta_color="off")
    st.caption("0개에서도 다리 자체 무게는 남아요. 오른쪽 모형·그래프에 모두 반영됩니다.")
    st.divider()
    st.markdown("**같은 조건으로 실험하기**")
    span_cm = st.slider("받침 사이 거리 (cm)", 30, 100, 60, 5, key="span_cm")
    comparison_basis = st.radio("재료 비교 기준", ["골조 재료량 같게", "막대 굵기 같게"], key="material_basis",
                                help="재료량 같게: 골조 전체 부피를 단순교와 맞추므로 막대가 많은 구조는 개별 막대가 가늘어집니다. 케이블·종이·받침은 별도입니다.")
    material = st.selectbox("골조 재료 · 예시 물성", list(MATERIALS), key="material")
    preset = MATERIALS[material]
    with st.expander("부재·재료·받침 조건 바꾸기"):
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
    amplify = st.slider("그림의 변형 확대 배율", 0, 30, 10, key="amplify", format="%d배",
                        help="그림만 확대합니다. mm 수치·힘·한계 개수는 변하지 않습니다. 0배는 원래 모양입니다.")
    st.caption("밝은/다크 모드: 오른쪽 위 ⋮ → Settings → Theme")
    with st.expander("선생님용 · 모형의 약속과 코드 수정 안내"):
        st.write("고정된 교량별 최대 추 개수는 없습니다. 같은 단위(m, N, Pa)로 보·막대·줄의 힘과 처짐을 구하고, 강도·Euler 좌굴·막대 끝 연결력·수평 받침력·허용 처짐 중 먼저 닿는 기준을 찾습니다.")
        st.write("연속된 바닥 보와 아치는 휨을 받는 보 요소, 나머지 막대는 핀 연결 축력 요소입니다. 줄은 당기는 힘만 허용합니다. 두 옆면이 무게를 절반씩 나누며 가로 방향 흔들림은 별도 가새가 막는다고 가정합니다.")
        st.write("골조 단면은 직사각형입니다. 수수깡·종이의 속 빈 단면, 접힌 종이의 강성, 접착제 특성은 별도 실측 보정이 필요합니다. 종이는 하중 분배와 자중만 반영하고 골조의 휨 강성에 더하지 않습니다.")
        st.write("아치는 양끝 핀 지지, 보의 오른쪽은 수평 이동 가능한 받침입니다. 케이블 교량은 주탑 뒤 고정줄과 앵커를 포함합니다. 압축 막대는 양끝 힌지의 Euler 하중, 아치의 유효 길이는 아치 길이의 절반으로 가정합니다. 바닥 보의 압축 유효 길이는 전체 경간(트러스는 한 칸)입니다. 실제 좌굴 형상·유효 길이는 지지 조건에 따라 달라집니다.")
        st.write("작은 변형의 2차원 선형 정적 계산입니다. 초기 케이블 장력·형상 찾기, 대변형, 아치 전체의 비선형 불안정, 3차원 비틀림, 전단 파괴, 바람·지진·피로, 연결부의 회전·박리는 계산하지 않습니다. 케이블의 큰 형상 변화가 예상되면 이 결과의 적용 범위를 벗어납니다.")
        st.write("‘기준 대비 여유’는 각 기준의 허용량 ÷ 계산된 요구량 중 가장 작은 값입니다. 강도만의 안전율이나 인증된 설계 안전율은 아닙니다. 자중 때문에 추 개수의 단순 비율과 다를 수 있습니다.")
        st.write("MATERIALS는 예시 물성, build_model()은 모형과 지지 조건, solve_sweep()은 계산과 기준 검사입니다. 하나의 파일 안에서 수정할 수 있습니다.")
        st.markdown(f"계산 참고: [TU Delft 보 요소]({METHOD_SOURCE}) · [TU Delft 평면 골조](https://teachbooks.tudelft.nl/computational-modelling/structural_linear/space_frame.html) · [MIT 압축 부재 좌굴]({BUCKLING_SOURCE})")

# ── 5. 첫 화면 안내, 그 다음 탐구 ──────────────────────────────────────
guide_tab, lab_tab = st.tabs(["📖 실험실 안내", "🌉 교량 탐구"])
with guide_tab:
    with st.container(key="lab_guide"):
        st.title("🌉 교량 구조와 힘의 분산")
        st.markdown('<p class="guide-subtitle">개념 원리 탐구 실험실</p>', unsafe_allow_html=True)
        st.write("추를 올리고, 막대의 굵기와 받침 사이 거리를 바꾸며 다리가 힘을 전달하는 방법을 관찰해요.")
        st.info("안내를 읽었다면 위의 ‘🌉 교량 탐구’ 탭에서 시작하세요.", icon="👉")
        st.subheader("1. 이렇게 탐구해요")
        for col, (title, text) in zip(st.columns(3), [
            ("① 조건 맞추기", "왼쪽에서 거리와 재료를 고르세요. 기본은 골조 재료의 총량을 같게 맞춘 비교예요."),
            ("② 추 올리기", "추 0~20개를 올리며 처짐(mm)과 힘을 관찰하세요. 다리 자체 무게도 포함해요."),
            ("③ 한 가지씩 바꾸기", "막대를 두껍게 하거나 대각선을 넣어 보세요. 어떤 조건에서 유리해지는지 비교해요."),
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
        st.write("그림 속 변형은 보기 쉽게 **확대**해요. 실제 계산된 처짐은 **mm** 숫자로 비교하세요. 선 굵기는 힘의 크기 표시이며 실제 막대 두께가 아니에요.")
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
            st.write("0~20개 안에서 처음 기준을 넘는 직전 개수를 표시해요. 20개도 기준 이내이면 **‘20개 이상’**으로 표시해요. 아직 진짜 한계를 찾은 것은 아니에요.")
            st.write("기준을 넘은 경우에는 원래 모양을 보여 줘요. 실제 붕괴 과정과 바람·지진은 이 실험에 포함하지 않아요.")

with lab_tab:
    tab_beam, tab_truss, tab_cable, tab_dashboard = st.tabs([
        "01  보와 아치", "02  삼각형과 트러스", "03  케이블과 주탑", "04  5대 교량 비교",
    ])
    # 먼저 조작기를 모두 만들고, 그 값을 한 가지 cfg에 모아 모든 탭에 전달합니다.
    with tab_beam:
        st.subheader("보는 휘고, 아치는 받침을 옆으로 밀어요")
        arch_percent = st.slider("아치 높이 · 받침 사이 거리의 (%)", 10, 40, 25, 5, key="arch_percent")
    with tab_truss:
        st.subheader("같은 사각형 틀에 대각선을 넣으면?")
        cols = st.columns(3)
        panels = cols[0].slider("사각형 칸 수", 1, 11, 5, key="triangles")
        truss_percent = cols[1].slider("뼈대 높이 · 받침 사이 거리의 (%)", 10, 35, 20, 5, key="truss_percent")
        reinforced = cols[2].toggle("대각선 보강재 추가", True, key="add_diagonals")
    with tab_cable:
        st.subheader("케이블은 당기고, 주탑은 눌려요")
        cols = st.columns(2)
        tower_cm = cols[0].slider("바닥판 위 주탑 높이 (cm)", 5, 50, 15, 1, key="tower_cm")
        sag_percent = cols[1].slider("현수교 주케이블 늘어진 깊이 · 주탑 높이의 (%)", 20, 80, 50, 5, key="sag_percent")
        angle = math.degrees(math.atan((tower_cm/100)/(span_cm/200)))
        st.caption(f"사장교 가장 긴 안쪽 줄의 실제 기울기: {angle:.1f}°. 높이·거리로 정해지는 값이라 각도를 따로 바꾸지 않아요.")

cfg = dict(span=span_cm/100, width=deck_width/100, weight_mass=mass_g/1000,
           member_width=member_width/1000, member_depth=member_depth/1000,
           E=E_gpa*1e9, density=density, ft=ft*1e6, fc=fc*1e6, fb=fb*1e6,
           paper_gsm=paper_gsm, self_weight=self_weight, equal_volume=comparison_basis == "골조 재료량 같게",
           joint_force=joint, foundation_force=foundation, deflection_divisor=divisor,
           cable_diameter=cable_d/1000, cable_E=cable_E*1e9, cable_strength=cable_strength*1e6,
           panels=panels, arch_ratio=arch_percent/100, truss_ratio=truss_percent/100,
           tower_ratio=tower_cm/span_cm, sag_ratio=sag_percent/100)
sweeps = {kind: solve_sweep(kind, cfg) for kind in BRIDGE_NAMES}

with tab_beam:
    st.caption(f"같은 거리 {span_cm} cm · {comparison_basis} · 허용 처짐 {span_cm*10/divisor:.2f} mm. 양옆 받침이 밀리지 않는 조건이에요.")
    for col, kind in zip(st.columns(2), ["단순교", "아치교"]):
        with col:
            show_result(sweeps[kind], load, amplify, f"beam_{kind}")
            bridge_guide_card(kind)
    with st.expander("쉽게 이해하기 · 자와 둥근 문을 떠올려요", expanded=True):
        st.write("**보**는 양끝에 걸친 자와 비슷해요. 같은 무게라도 거리를 늘리면 훨씬 많이 휘고, 막대의 세로 두께를 키우면 덜 휘어요. 휘는 보는 윗면이 눌리고 아랫면이 당겨져요. 단순보 위·아래의 가는 빨강·파랑 선은 이 휨을 보여 줘요. 다른 막대의 선 색은 부재 전체의 축력이고, 마우스를 올리면 휨의 크기도 볼 수 있어요.")
        st.write("**아치**는 무게를 둥근 길을 따라 양끝으로 보내요. 대신 받침을 옆으로 밀어요. 왼쪽의 ‘받침·앵커 수평 허용 힘’을 줄여 보세요. 덜 휘더라도 받침이 힘을 못 견디면 기준을 넘을 수 있어요.")

with tab_truss:
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
    bridge_guide_card("트러스교")
    with st.expander("쉽게 이해하기 · 대각선은 왜 필요할까요?", expanded=True):
        st.write("사각형은 변 길이가 같아도 모서리 각도가 바뀔 수 있어요. 대각선 하나를 넣으면 두 삼각형이 생겨, 막대를 늘리거나 줄이지 않고는 모양을 바꾸기 어려워져요.")
        st.write("삼각형이 생겨도 막대·연결부가 무한히 강해지지는 않아요. 눌리는 막대가 길고 가늘면 좌굴하고, 연결부가 약하면 먼저 떨어질 수 있어요.")

with tab_cable:
    for col, kind in zip(st.columns(2), ["사장교", "현수교"]):
        with col:
            show_result(sweeps[kind], load, amplify, f"cable_{kind}")
            r = sweeps[kind]["results"][load]
            st.metric("가장 크게 당겨지는 줄 · 양옆 합계", "계산 불가" if r["tension"] is None else f"{r['tension']:.2f} N")
            bridge_guide_card(kind)
    fig = data_figure("주탑 높이와 줄을 당기는 힘", "바닥판 위 주탑 높이 (cm)", "최대 줄 장력 · 양옆 합계 (N)")
    heights = sorted(set([5, 10, 15, 20, 30, 40, 50, tower_cm]))
    for kind in ["사장교", "현수교"]:
        tensions = [solve_sweep(kind, dict(cfg, tower_ratio=h/span_cm))["results"][load]["tension"] for h in heights]
        fig.add_trace(go.Scatter(x=heights, y=tensions, mode="lines+markers", name=kind,
                                 line=dict(color=COLORS[kind]),
                                 customdata=[math.degrees(math.atan((h/100)/(span_cm/200))) for h in heights],
                                 hovertemplate="주탑 %{x} cm<br>장력 %{y:.3f} N<br>사장교 긴 줄 기울기 %{customdata:.1f}°<extra>%{fullData.name}</extra>"))
    fig.add_vline(x=tower_cm, line_dash="dot", line_color=SUPPORT_COLOR)
    chart(fig, "height_sensitivity")
    st.caption("장력 곡선은 기준 초과 설정도 포함한 선형 참고값입니다. 현수교는 늘어진 깊이/주탑 높이 비율을 유지합니다. 주탑이 높아지면 줄의 힘이 줄어도, 긴 주탑의 좌굴이 먼저 문제가 될 수 있어요.")
    with st.expander("쉽게 이해하기 · 줄이 세워질수록?", expanded=True):
        st.write("**사장교**는 줄이 바닥판을 직접 들어 올려요. 줄이 수평에 가까우면 힘의 많은 부분이 옆으로 쓰여, 위로 받치려면 더 세게 당겨야 해요. 뒤쪽 줄과 앵커는 주탑이 한쪽으로 끌리지 않게 버텨요.")
        st.write("**현수교**는 세로줄 → 주케이블 → 주탑·앵커로 힘을 전달해요. 하중이 바닥판 전체에 고르게 실리면 큰 줄은 포물선에 가까워요. 줄 자체 무게만 매달린 경우의 현수선과는 달라요.")
        st.write("줄을 늘어뜨리면 옆으로 당기는 힘이 줄 수 있어요. 하지만 얇은 줄은 늘어나고, 약한 앵커는 밀릴 수 있어요. 이 실험은 작은 변형만 계산하므로 실제 줄이 크게 처지는 과정은 다루지 않아요.")

with tab_dashboard:
    st.subheader("5대 교량 · 조건에 따라 달라지는 결과")
    mode = st.radio("비교 방식", ["두 교량 비교", "5종 전체 비교"], horizontal=True, key="comparison_mode")
    st.caption(f"공통 거리 {span_cm} cm · {comparison_basis} · 추 1개 {mass_g} g · 자체 무게 {'반영' if self_weight else '제외'}. 트러스교는 대각선 보강 구조입니다. 배치는 튼튼함의 순서가 아니에요.")
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
        st.write("‘20개 이상’이면 실험 범위 안에서 한계에 닿지 않았어요. ‘0개도 기준 초과’라면 자체 무게부터 조건을 바꿔야 해요. 원래 모양의 그림은 안정성 또는 기준을 만족하지 못한 상태예요.")
        chart(overview_figure(sweeps, amplify), "capacity_bridge_overview")
        st.caption("이 그림만 각 교량의 마지막 기준 이내 추 개수를 올립니다. 아래 그래프는 왼쪽에서 선택한 현재 추 개수로 비교합니다. 트러스교는 가운데에 배치했습니다.")
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
        st.write("거리와 재료량을 맞춘 뒤 조건을 하나씩 바꿔 보세요. 짧은 다리, 긴 다리, 약한 받침에 유리한 구조가 서로 달라져요. 이 모형 결과를 실제 큰 교량의 우열로 해석하지 않아요.")
    st.markdown("### 📖 탐구한 교량의 특징과 장단점")
    bridge_guide_cards(names)

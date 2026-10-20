"""하중 경로에 등장하는 부분을 찾기 위한 교량별 구조 그림과 용어 풀이."""

from html import escape


# (그림 이름, 하중 경로에서 찾을 말, 쉬운 뜻, 이름표 위치, 가리키는 부분)
BRIDGE_PARTS = {
    "단순교": [
        ("상판", "상판", "사람과 차량이 지나는 다리의 바닥", (55, 30), (180, 145)),
        ("보", "보", "상판 아래에서 바닥을 받치는 긴 가로 막대", (265, 30), (265, 158)),
        ("받침", "받침", "교량 끝을 받치고 힘을 아래로 전하는 부분", (55, 277), (90, 202)),
        ("지반", "지반", "받침 아래에서 힘을 받아 주는 땅", (275, 277), (330, 233)),
    ],
    "아치교": [
        ("상판", "상판", "사람과 차량이 지나는 다리의 바닥", (40, 25), (180, 131)),
        ("세로 지지대", "세로 지지대", "상판과 아치를 이어 주는 세로 막대", (225, 25), (130, 158)),
        ("아치", "아치", "하중을 양끝으로 전달하는 굽은 뼈대", (265, 57), (275, 167)),
        ("받침", "받침", "교량 끝을 받치고 힘을 아래로 전하는 부분", (55, 277), (90, 224)),
        ("지반", "지반", "받침 아래에서 힘을 받아 주는 땅", (275, 277), (330, 233)),
    ],
    "트러스교": [
        ("상판", "상판", "사람과 차량이 지나는 다리의 바닥", (40, 25), (180, 131)),
        ("삼각형 막대", "삼각형을 이루는 막대", "서로 이어져 삼각형을 만드는 뼈대", (225, 25), (177.5, 164)),
        ("받침", "받침", "교량 끝을 받치고 힘을 아래로 전하는 부분", (55, 277), (80, 211)),
        ("지반", "지반", "받침 아래에서 힘을 받아 주는 땅", (275, 277), (330, 233)),
    ],
    "사장교": [
        ("상판", "상판", "사람과 차량이 지나는 다리의 바닥", (250, 270), (260, 170)),
        ("사장 케이블", "사장 케이블", "주탑에서 상판으로 비스듬히 이어진 줄", (225, 25), (250, 130.15)),
        ("주탑", "주탑", "케이블을 받치는 높은 기둥", (55, 25), (106, 105)),
        ("기초", "기초", "기둥 밑에서 힘을 땅으로 전하는 밑부분", (55, 270), (106, 225)),
        ("뒤쪽 케이블", "뒤쪽 케이블", "이 모형에서 주탑 뒤를 잡아 주는 줄", (285, 57), (346, 145)),
        ("앵커", "앵커", "줄 끝을 땅에 단단히 고정하는 장치", (20, 302), (42, 223)),
        ("지반", "지반", "기초와 고정 장치 아래의 땅", (350, 302), (280, 233)),
    ],
    "현수교": [
        ("상판", "상판", "사람과 차량이 지나는 다리의 바닥", (250, 270), (250, 170)),
        ("행어(세로줄)", "행어", "상판을 주케이블에 매다는 세로 방향의 줄", (225, 25), (210, 147)),
        ("주케이블", "주케이블", "주탑 사이에 늘어져 상판을 지탱하는 굵은 줄", (285, 57), (255, 119)),
        ("주탑", "주탑", "주케이블을 받치는 높은 기둥", (55, 25), (106, 105)),
        ("앵커리지", "앵커리지", "주케이블 끝을 땅에 단단히 고정하는 장치", (35, 302), (42, 223)),
        ("지반", "지반", "기둥과 고정 장치 아래의 땅", (350, 302), (280, 233)),
    ],
}


def part_badges(kind, node):
    """복합 경로(주탑·기초 등)에도 각 부분의 그림 번호를 함께 표시합니다."""
    return ''.join(
        f'<span class="bridge-part-number" aria-label="그림 {index}번">{index}</span>'
        for index, part in enumerate(BRIDGE_PARTS.get(kind, []), 1) if part[1] in node
    )


def part_meanings(kind, node):
    """하중 경로의 한 항목에 해당하는 설명만 반환합니다."""
    return [(part[0], part[2]) for part in BRIDGE_PARTS.get(kind, []) if part[1] in node]


def bridge_parts_legend(kind):
    entries = []
    for index, (name, _, meaning, _, _) in enumerate(BRIDGE_PARTS[kind], 1):
        entries.append(
            f'<div><dt><span class="bridge-part-number">{index}</span>{escape(name)}</dt>'
            f'<dd> — {escape(meaning)}</dd></div>'
        )
    return '<dl class="bridge-part-legend" aria-label="그림 속 이름 풀이">' + ''.join(entries) + '</dl>'


def bridge_anatomy_diagram(kind):
    """시뮬레이션과 같은 연결 방식을 단순화한 정적인 측면 그림입니다."""
    shapes = []

    def line(points, width=3, color="#58788e"):
        shapes.append(f'<polyline points="{points}" fill="none" stroke="{color}" '
                      f'stroke-width="{width}" stroke-linecap="round" stroke-linejoin="round"/>')

    def rect(x, y, width, height, fill="#b7c6ce", stroke="#58788e"):
        shapes.append(f'<rect x="{x}" y="{y}" width="{width}" height="{height}" '
                      f'rx="2" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')

    # 땅과 구조의 선은 힘의 크기나 계산 결과를 나타내는 색상이 아닙니다.
    shapes.append('<path d="M18 232H402V243H18Z" fill="currentColor" opacity=".08"/>')
    line("18,232 402,232", 2)
    for x in range(24, 402, 18):
        line(f"{x},233 {x-7},241", 1)

    if kind == "단순교":
        rect(76, 164, 28, 68)
        rect(316, 164, 28, 68)
        rect(76, 152, 268, 12, "#91aab8")
        rect(70, 138, 280, 14, "#d7ede8", "#177568")
    elif kind == "아치교":
        rect(76, 138, 14, 94)
        rect(330, 138, 14, 94)
        rect(75, 218, 30, 14)
        rect(315, 218, 30, 14)
        for x in range(110, 311, 20):
            y = 144 + 78*((x-210)/120)**2
            line(f"{x},138 {x},{y}", 3)
        shapes.append('<path d="M90 222Q210 66 330 222" fill="none" '
                      'stroke="#58788e" stroke-width="8" stroke-linecap="round"/>')
        rect(70, 126, 280, 12, "#d7ede8", "#177568")
    elif kind == "트러스교":
        rect(70, 190, 20, 42)
        rect(330, 190, 20, 42)
        line("80,190 340,190", 5)
        for x in [80, 145, 210, 275, 340]:
            line(f"{x},138 {x},190")
        line("80,190 145,138 210,190 275,138 340,190", 5)
        rect(70, 126, 280, 12, "#d7ede8", "#177568")
    else:
        for x in [106, 314]:
            rect(x-15, 218, 30, 14)
        rect(30, 214, 24, 18)
        rect(366, 214, 24, 18)
        line("42,214 106,76", 3)
        line("314,76 378,214", 3)
        if kind == "사장교":
            for x in [146, 178, 210]:
                line(f"106,76 {x},164", 2.5)
            for x in [210, 242, 274]:
                line(f"314,76 {x},164", 2.5)
        else:
            shapes.append('<path d="M106 76Q210 182 314 76" fill="none" '
                          'stroke="#58788e" stroke-width="5"/>')
            for x in range(132, 289, 26):
                y = 76 + 53*(1-((x-210)/104)**2)
                line(f"{x},{y} {x},164", 2.5)
        rect(106, 164, 208, 12, "#d7ede8", "#177568")
        for x in [106, 314]:
            rect(x-5, 72, 10, 146, "#91aab8")

    labels = []
    for index, (name, _, meaning, (x, y), (tx, ty)) in enumerate(BRIDGE_PARTS[kind], 1):
        labels.append(
            f'<g class="bridge-anatomy-label" data-part="{escape(name)}">'
            f'<title>{escape(name)}: {escape(meaning)}</title>'
            f'<path d="M{x},{y}L{tx},{ty}" fill="none" stroke="currentColor" '
            'stroke-width="1.2" opacity=".6"/>'
            f'<circle cx="{tx}" cy="{ty}" r="3" fill="#177568"/>'
            f'<circle cx="{x}" cy="{y}" r="11" fill="#177568"/>'
            f'<text x="{x}" y="{y}" text-anchor="middle" dominant-baseline="central" '
            f'fill="white" font-size="15" font-weight="700">{index}</text>'
            f'<text class="bridge-anatomy-name" x="{x+17}" y="{y}" '
            f'dominant-baseline="central" fill="currentColor" font-size="18">{escape(name)}</text></g>'
        )
    description = ' / '.join(f'{i}번 {part[0]}: {part[2]}'
                             for i, part in enumerate(BRIDGE_PARTS[kind], 1))
    return (
        '<svg class="bridge-anatomy-diagram" viewBox="0 0 420 320" '
        'role="img" aria-labelledby="bridge-anatomy-title bridge-anatomy-description">'
        f'<title id="bridge-anatomy-title">{escape(kind)}의 각 부분 이름</title>'
        f'<desc id="bridge-anatomy-description">{escape(description)}</desc>'
        + ''.join(shapes) + ''.join(labels) + '</svg>'
    )

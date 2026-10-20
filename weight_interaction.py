"""금속 추 그림과 Streamlit 기본 버튼을 사용하는 하중 조절 UI."""

from base64 import b64encode

import streamlit as st


METAL_WEIGHT_SVG = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 96 104" aria-hidden="true">
<defs>
 <linearGradient id="metal" x1="0" x2="1"><stop stop-color="#55616e"/>
 <stop offset=".24" stop-color="#c4cbd2"/><stop offset=".45" stop-color="#f1f3f5"/>
 <stop offset=".66" stop-color="#aab4bf"/><stop offset="1" stop-color="#596775"/></linearGradient>
 <linearGradient id="cap" x1="0" y1="0" x2="0" y2="1"><stop stop-color="#e8edf1"/>
 <stop offset="1" stop-color="#8e9aa6"/></linearGradient>
</defs>
<ellipse cx="48" cy="100" rx="37" ry="4" fill="#152332" opacity=".18"/>
<path d="M28 43 9 91Q8 102 48 103Q88 102 87 91L68 43Z" fill="url(#metal)" stroke="#52606e" stroke-width="2"/>
<ellipse cx="48" cy="44" rx="20" ry="6" fill="url(#cap)" stroke="#64717e" stroke-width="2"/>
<path d="M35 41V25C35 5 61 5 61 25V41" fill="none" stroke="#4a5866" stroke-width="10"/>
<path d="M35 39V25C35 5 61 5 61 25V39" fill="none" stroke="url(#metal)" stroke-width="6"/>
<path d="M29 53 17 88" stroke="#f4f7fa" stroke-width="3" opacity=".7"/>
<path d="M12 92Q48 101 84 92" fill="none" stroke="#53616e" stroke-width="2"/>
</svg>'''
WEIGHT_IMAGE_SOURCE = "data:image/svg+xml;base64," + b64encode(METAL_WEIGHT_SVG.encode()).decode()


def _change_weight(delta, limit):
    """위젯을 다시 그리기 전에 슬라이더와 공유하는 추 개수를 바꿉니다."""
    current = st.session_state.get("load", 0)
    st.session_state["load"] = max(0, min(limit, current + delta))


def render_weight_controls(count, limit):
    """그래프 초기화나 브라우저 좌표에 의존하지 않는 기본 버튼입니다."""
    add, remove = st.columns(2, gap="small")
    add.button("추 1개 넣기", key="weight_add", width="stretch",
               on_click=_change_weight, args=(1, limit), disabled=count >= limit)
    remove.button("추 1개 빼기", key="weight_remove", width="stretch",
                  on_click=_change_weight, args=(-1, limit), disabled=count <= 0)

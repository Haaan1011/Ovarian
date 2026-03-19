import streamlit as st

from frontend.streamlit_app.components.result_panels import reserve_result_html


def render_reserve_results(report, patient):
    st.markdown(
        reserve_result_html(report, patient),
        unsafe_allow_html=True,
    )

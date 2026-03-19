import streamlit as st

from frontend.streamlit_app.components.result_panels import plan_result_html


def render_plan_results(report, patient):
    st.markdown(
        plan_result_html(report, patient),
        unsafe_allow_html=True,
    )

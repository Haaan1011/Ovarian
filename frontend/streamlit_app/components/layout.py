from textwrap import dedent

import streamlit as st
import streamlit.components.v1 as components


def inject_scroll_persistence_script() -> None:
    components.html(
        dedent(
            """
            <script>
            (function() {
              const parentWindow = window.parent;
              const key = "ovarian_app_scroll_y";
              try {
                if (!parentWindow.__ovarianScrollPersistBound) {
                  parentWindow.__ovarianScrollPersistBound = true;
                  let ticking = false;
                  parentWindow.addEventListener("scroll", () => {
                    if (ticking) {
                      return;
                    }
                    ticking = true;
                    parentWindow.requestAnimationFrame(() => {
                      parentWindow.sessionStorage.setItem(key, String(parentWindow.scrollY || parentWindow.pageYOffset || 0));
                      ticking = false;
                    });
                  }, { passive: true });
                }
                const saved = parentWindow.sessionStorage.getItem(key);
                if (saved !== null) {
                  const top = Number(saved) || 0;
                  parentWindow.requestAnimationFrame(() => {
                    parentWindow.scrollTo({ top, behavior: "auto" });
                  });
                }
              } catch (error) {}
            })();
            </script>
            """
        ).strip(),
        height=0,
    )


def inject_figma_capture_script() -> None:
    components.html(
        dedent(
            """
            <script>
            (function() {
              const parentDoc = window.parent && window.parent.document;
              if (!parentDoc) {
                return;
              }
              const scriptId = "figma-mcp-capture-script";
              if (parentDoc.getElementById(scriptId)) {
                return;
              }
              const script = parentDoc.createElement("script");
              script.id = scriptId;
              script.src = "https://mcp.figma.com/mcp/html-to-design/capture.js";
              script.async = true;
              parentDoc.head.appendChild(script);
            })();
            </script>
            """
        ).strip(),
        height=0,
    )


def render_title_frame(title_logo_uri: str) -> None:
    title_logo_html = ""
    if title_logo_uri:
        title_logo_html = f"<div class='title-logo-wrap'><img class='title-logo' src='{title_logo_uri}' alt='福幼标识'></div>"

    st.markdown(
        f"""
        <div class='title-frame'>
            <div class='title-row'>
                {title_logo_html}
                <div class='title-copy'>
                    <div class='page-title'>卵巢储备功能与促排卵方案AI辅助系统</div>
                    <div class='page-subtitle'>AI-Assisted System for Ovarian Reserve Function Evaluation and Ovulation Stimulation Planning</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

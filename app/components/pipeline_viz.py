"""
Pipeline diagram component.

Renders a static Mermaid-style ASCII/Markdown pipeline diagram in the
Streamlit sidebar so users understand the dual-stream architecture.
"""
from __future__ import annotations

import streamlit as st


def render_pipeline_diagram() -> None:
    """Render a pipeline architecture overview in the sidebar."""
    st.sidebar.markdown("## 🧬 Pipeline Architecture")
    st.sidebar.markdown(
        """
```
Upload Image
     │
     ├──────────────────┐
     ▼                  ▼
Stream A           Stream B
(Segmented)        (Original)
     │                  │
HSV Threshold      Pre-process
Binary Mask             │
Masked Image       Backbone B
     │                  │
Pre-process        Feature Vec B
     │                  │
Backbone A              │
     │                  │
Feature Vec A ─┬─ Feature Vec B
               │
           Concatenate
               │
           Dropout
               │
     Classification Head
               │
          4 Classes
```
        """,
        unsafe_allow_html=False,
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
**Classes**
| Folder | Display |
|--------|---------|
| Benign | Hematogones |
| Early | Early Pre-B ALL |
| Pre-B | Pre-B ALL |
| Pro | Pro-B ALL |
        """
    )

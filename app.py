import spacy
import streamlit as st
from spacy import displacy
import en_core_web_sm 


DEFAULT_TEXT = "Apple Inc. is planning to build a new headquarters in Cupertino, California for $5 billion."
HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem; margin-bottom: 1rem">{}</div>"""


@st.cache_resource
def load_model(model_name):
    """Loads the spaCy model."""
    try:
        return spacy.load(model_name)
    except OSError:
        st.error(f"Model '{model_name}' not found. Please ensure it's installed correctly.")
        st.stop()

nlp = load_model("en_core_web_sm")

st.set_page_config(layout="wide", page_title="Interactive spaCy Visualizer")

st.title("Interactive spaCy Visualizer ✨")
st.markdown(
    """
    Explore Named Entity Recognition (NER) and Dependency Parsing using spaCy.
    Enter your text below and see the visualizations!
    """
)

st.sidebar.header("Visualization Options")
analysis_type = st.sidebar.radio(
    "Choose Analysis Type:",
    ("Named Entity Recognition (NER)", "Dependency Parse (Tree)")
)

# --- Main Interface ---
st.header("Enter Text for Analysis:")
text_input = st.text_area("Paste your text here:", DEFAULT_TEXT, height=150)

if text_input:
    # Process the text with spaCy
    doc = nlp(text_input)

    st.divider() 

    if analysis_type == "Named Entity Recognition (NER)":
        st.header("Named Entity Recognition (NER)")
        st.info("Highlights recognized entities like organizations, people, locations, etc.")

        html_ner = displacy.render(doc, style="ent", jupyter=False) 

        st.write(HTML_WRAPPER.format(html_ner), unsafe_allow_html=True)

        if doc.ents:
             st.subheader("Detected Entities:")
             entity_data = [{"Text": ent.text, "Label": ent.label_, "Explanation": spacy.explain(ent.label_)} for ent in doc.ents]
             st.dataframe(entity_data)
        else:
            st.write("No entities detected.")


    elif analysis_type == "Dependency Parse (Tree)":
        st.header("Dependency Parse Tree")
        st.info("Shows the syntactic relationships between words (how words modify each other).")
      
        options_dep = {"compact": True, "color": "#09a3d5", "font": "Arial"}
        svg_dep = displacy.render(doc, style="dep", jupyter=False, options=options_dep)

        st.write(HTML_WRAPPER.format(svg_dep), unsafe_allow_html=True)
      
        st.subheader("Dependency Relationships:")
        dep_data = [{
            "Text": token.text,
            "Dependency": token.dep_,
            "Head Text": token.head.text,
            "Head POS": token.head.pos_,
            "Children": [child.text for child in token.children]
            } for token in doc]
        st.dataframe(dep_data)

else:
    st.warning("Please enter some text to analyze.")

st.sidebar.markdown("---")
st.sidebar.markdown("Powered by [spaCy](https://spacy.io/) & [Streamlit](https://streamlit.io/)")
st.sidebar.markdown(f"spaCy Model: `en_core_web_sm`")


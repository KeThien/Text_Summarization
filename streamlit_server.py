import streamlit as st
from transformers.pipelines import Pipeline
from transformers.pipelines import pipeline
from summarizer import summarize

st.cache(show_spinner=True)

st.header("Prototyping an NLP Summarization")
st.text("This demo uses a model for Text summarization")
add_text_sidebar = st.sidebar.title("Menu")
add_text_sidebar = st.sidebar.text("Just some random text.")
# text = st.text_area(label="Text to summarize")

# if (len(text) < 10):
#     st.error("You can't use less than 10 characters")
# else:
#     summarized = summarize(text, device=0)
#     st.text('Summary Text:')
#     st.markdown(str(summarized[0]['summary_text']))

with st.form("my_form"):
    st.write("Inside the form")
    text = st.text_area(label="Text to summarize")
    slider_val = st.slider("Form slider")
    checkbox_val = st.checkbox("Form checkbox")

    # Every form must have a submit button.
    submitted = st.form_submit_button("Submit")
    if submitted and checkbox_val:
        summarized = summarize(text, device=0)
        st.text('Summary Text:')
        st.markdown(str(summarized[0]['summary_text']))


# if submitted and (summarized in globals()) and checkbox_val:
#     st.markdown(str(summarized[0]['summary_text']))
# st.write("Outside the form")

import streamlit as st
from lora_pipeline.core import compare
st.title('Base vs LoRA Adapter'); q=st.text_area('Domain prompt','I was charged twice for the same invoice.')
if st.button('Compare'): st.json(compare(q))

import streamlit as st
from streamlit_extras.app_logo import add_logo

add_logo('fiddler-ai-logo.png', height=50)

st.title('Building Trust into Generative AI')

st.header('Techniques for Model Visibility and Tracking Change in Data Distributions')

st.markdown('Find out more about us and building trust into AI at [Fiddler AI](http://fiddler.ai)')


st.subheader('Additional workshop assets')
st.markdown('- [Colab companion notebook](https://colab.research.google.com/github/JoshuaGRubin/Gen-AI-Workshop-2023/blob/main/CompanionNotebook.ipynb)\n\n'
            '- Blog series on monitoring drift in vector distributions ([1. technique](https://www.fiddler.ai/blog/monitoring-natural-language-processing-and-computer-vision-models-part-1), [2. Computer Vision](https://www.fiddler.ai/blog/monitoring-natural-language-processing-and-computer-vision-models-part-2), [3. Natural Language](https://www.fiddler.ai/blog/monitoring-natural-language-processing-and-computer-vision-models-part-3))')

st.subheader('Overview of Activity')
st.image('the_activity.png')

st.subheader('Interpreting Feedback Semantically')
st.image('schematic_feedback.png')

st.subheader('Computing Semantic Drift')
st.image('schematic_drift.png')


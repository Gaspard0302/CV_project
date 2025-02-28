from pathlib import Path
import streamlit as st
from PIL import Image
import os

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import openai
from openai import OpenAIError

# ---- Path Settings ------
current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
css_file = current_dir / "main.css"
resume_file = current_dir / "CV.pdf"
profile_pic = current_dir / "profile-pic.png"
bio_file_path = current_dir / "bio.txt"

# ---- General ------
PAGE_TITLE = "Digital CV / Gaspard Hassenforder"
PAGE_ICON = ":wave"
NAME = "Gaspard Hassenforder"
DESCRIPTION = "Data Scientist with a strong background in data analysis, machine learning, and statistical modeling. Passionate about leveraging data to drive business decisions and enhance operational efficiency. Adept at collaborating with cross-functional teams to deliver impactful solutions."

EMAIL = "hassenforder.gaspard@gmail.com"
SOCIAL_MEDIA = {
    "LinkedIn": "https://www.linkedin.com/in/gaspard-hassenforder-554431225/",
    "GitHub": "https://github.com/Gaspard0302"
}

PROJECTS = {
    "Recreating Apple Genmoji": "Finetuned an image generation model (SDXL Lightning) with emojis to recreate the Apple feature called GenMoji introduced in iOS 18, enabling users to generate personalized emoji combinations based on their preferences.",
    "On-Device AI Hackathon (November 2024)": "Won First place at a hackathon sponsored by Meta, Hugging Face, Scalaway, and Entrepreneur First. Developed 'NamastAI,' a yoga assistant app integrating multiple AI models for real-time pose correction and audible feedback. Demonstrated expertise in deploying, fine-tuning, and optimizing local AI models, earning recognition for both technical innovation and commercial viability.",
    "CapGemini DataCamp": "Contributed to a team project analysing customer feedback via sentiment analysis, utilising BERT and GPT-3.5 for data scraped from Trustpilot. Enhanced client's customer satisfaction through actionable insights derived from advanced NLP techniques.",
    "Predicting Customer Churn": "Collaborated on a BGC X strategy project to develop a model for predicting customer churn and estimating customer lifetime value based on past spending behaviour. Leveraged these insights to design an optimal strategy for the sales team, including personalised email campaigns using generative AI and targeted calls. The approach focused on preventing churn by tailoring actions to client relationships, customer difficulty levels, and lifetime value.",
    "Modelling Mixte Marketing": "Analysed L'Oréal data to determine the optimal marketing budget allocation across various sales channels. The project focused on identifying the most effective strategies to maximise ROI and improve overall marketing efficiency.",
    "Optimising Client invitation list": "Worked on a past case from Eleven Consulting to optimise client invitation lists for corporate events organised by a major luxury brand. Utilised causality techniques, including double machine learning, to calculate the uplift in sales resulting from event invitations. Developed a platform that identifies the best clients to invite based on event characteristics and the desired number of attendees, maximising sales impact.",
    "Optimising Retail Performance in Shopping Malls": "Analysed data from Unibail-Rodamco-Westfield, covering over 20 malls across Europe, including store sales and customer foot traffic. Provided actionable recommendations by examining visitor flow, store performance, and retail trends. Proposed strategies to optimise tenant mix, enhancing foot traffic, sales, and overall revenue.",
}

st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON)

# --- LOAD CSS, PDF & PROFIL PIC ---
with open(css_file) as f:
    st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)
with open(resume_file, "rb") as pdf_file:
    PDFbyte = pdf_file.read()
profile_pic = Image.open(profile_pic)

# --- HERO SECTION ---
col1, col2 = st.columns(2, gap="small")
with col1:
    st.image(profile_pic, width=300)

with col2:
    st.title(NAME)
    st.write(DESCRIPTION)
    st.download_button(
        label=" 📄 Download Resume",
        data=PDFbyte,
        file_name=resume_file.name,
        mime="application/octet-stream",
    )
    st.write("📫", EMAIL)

# -----------------  Chatbot  ----------------- #
st.write('\n')
st.subheader("Chat with me")
st.write("Ask me anything about Gaspard's experience, skills, or projects!")

# API key handling
api_key_col1, api_key_col2 = st.columns([1, 3])
with api_key_col1:
    has_api_key = st.checkbox('Use your own API key')

with api_key_col2:
    if has_api_key:
        openai_api_key = st.text_input('Enter your OpenAI API Key', type="password",
                                      help="Your API key starts with 'sk-'")
    else:
        openai_api_key = st.secrets["OPENAI_API_KEY"]

# Function to ask the bot
def ask_bot(input_text):
    try:
        # define LLM
        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=1,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            api_key=openai_api_key
        )

        # Function to read context from a text file
        def read_context_from_file(file_path):
            with open(file_path, 'r') as file:
                context = file.read()
            return context

        # Read the context from the file
        context = read_context_from_file(bio_file_path)

        # Create the prompt template
        prompt = PromptTemplate.from_template(
            "You are Buddy, an AI assistant dedicated to assisting Gaspard in his job search by providing recruiters with relevant and concise information. "
            "Here is his CV {context}"
            "If you can't answer, politely admit it and let recruiters know how to contact Gaspard to get more information directly from him but only if you can't answer. "
            "Don't put Buddy or a breakline in the front of your answer and be concise. Human: {question}"
        )

        # Use the chain with the read context and a question
        chain = prompt | llm
        response = chain.invoke(
            {
                "context": context,
                "question": input_text,
            }
        )

        return response.content

    except Exception as e:
        error_msg = str(e).lower()
        if "insufficient funds" in error_msg or "billing" in error_msg:
            return "⚠️ The API key doesn't have enough credits. Please provide a different API key."
        elif "invalid api key" in error_msg:
            return "⚠️ Invalid API key. Please check your API key and try again."
        else:
            return f"⚠️ An error occurred: {str(e)}"

# Get user input
user_input = st.text_input("You can send your questions and hit Enter to know more about me from my AI agent", key="user_question")

# Process user input
if user_input:
    # Check API key validity
    if not openai_api_key or (has_api_key and not openai_api_key.startswith('sk-')):
        st.warning('⚠️ Please enter a valid OpenAI API key.', icon='⚠')
    else:
        # Get response from bot
        with st.spinner('Getting response...'):
            response = ask_bot(user_input)
            st.info(response)

# --- SOCIAL LINKS ---
st.write('\n')
cols = st.columns(len(SOCIAL_MEDIA))
for index, (platform, link) in enumerate(SOCIAL_MEDIA.items()):
    cols[index].write(f"[{platform}]({link})")

# --- EXPERIENCE & QUALIFICATIONS ---
st.write('\n')
st.subheader("Experience & Qualifications")
st.write(
    """
- ✔️ Experience in data science internships focusing on data-driven solutions and model development
- ✔️ Strong hands-on experience and knowledge in Python, SQL, and data analysis tools
- ✔️ Good understanding of statistical principles and machine learning applications
- ✔️ Excellent team player with a strong sense of initiative and leadership experience
"""
)

# --- SKILLS ---
st.write('\n')
st.subheader("Hard Skills")
st.write(
    """
- 👩‍💻 Programming: Python (Pandas, NumPy, Scikit-learn, PyTorch), SQL, moderate knowledge of C#
- 📊 Data Visualization: Tableau, MS Excel, Plotly
- 📚 Modeling: Machine Learning, Deep Learning, Causality Modeling
- 🗄️ Databases: PostgreSQL, experience with data retrieval systems
"""
)

# --- Projects & Accomplishments ---
st.write("---")
st.write('\n')
st.subheader("Projects & Accomplishments")

for project, explanation in PROJECTS.items():
    st.write(f" - **{project}** :  {explanation}")

# --- EDUCATION ---
st.write("---")
st.write('\n')
st.subheader("Education")

# --- DEGREE 1
st.write("🎓", "**HEC, Paris**")
st.write("Master in Data Science and AI for Business — 2024/2025")
st.write(
    """
- Focused on real-world business challenges through data-driven solutions, developed skills in analysing business issues related to Data Science and AI, building and optimising models using state-of-the-art tools, and understanding their societal impact.
- Key Courses: Algorithmic Fairness and Interpretability, Causal Inference, ML Ops, Reinforcement Learning
"""
)

# --- DEGREE 2
st.write('\n')
st.write("🎓", "**Polytechnique, Paris**")
st.write("Master in Data Science and AI for Business — 2023/2024")
st.write(
    """
- Honed expertise in key disciplines including Statistics, Machine Learning, Database Management, Deep Learning, and Optimization.
"""
)

# --- DEGREE 3
st.write('\n')
st.write("🎓", "**Warwick University, London, UK**")
st.write("Bachelor in Mathematics — 2020/2023")
st.write(
    """
- Improved my problem-solving capabilities and learned mathematical concepts and understood their applications.
- Key modules include Topology, Multivariate Statistics, and Number Theory. Graduated with a high 2.1 degree.

"""
)

# --- DEGREE 4
st.write('\n')
st.write("🎓", "**Lycée Francais Charles de Gaulle, London, UK**")
st.write("French Baccalauréat (Scientific Section) — 2020")
st.write(
    """
- Obtained with highest honors (17.02/20), specializing in Maths, Physics, and Sciences.
"""
)

# --- WORK HISTORY ---
st.write("---")
st.write('\n')
st.subheader("Work History")

# --- JOB 1
st.write("🚧", "**Data Scientist Intern | Ekimetrics**")
st.write("April 2024 - September 2024")
st.write(
    """
- ► Implemented a Retrieval Augmented Generation solution to automate Corporate Sustainability Assessment processes, significantly improving efficiency
- ► Developed a causality model using Python to measure customer uplift from YouTube ad campaigns, providing actionable insights
- ► Contributed to a Marketing Mix Modeling project for a major airline to optimize budget allocation and marketing ROI
"""
)

# --- JOB 2
st.write('\n')
st.write("🚧", "**Software Development Intern | VP & White UK Ltd**")
st.write("August 2022 - September 2022")
st.write(
    """
- ► Developed front and back end processes for a new e-learning platform using C#, SQL, and .NET
- ► Created unit testing processes and logged outstanding issues to enhance software quality
"""
)

# --- JOB 3
st.write('\n')
st.write("🚧", "**Intern, Partnerships and University Recruitment Analyst | Dassault Systèmes**")
st.write("July 2019")
st.write(
    """
- ► Analyzed and mapped existing university partnerships to optimize recruitment strategies
- ► Gained hands-on experience with CAD software, enhancing understanding of intricate models
"""
)
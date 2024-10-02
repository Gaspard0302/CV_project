


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
profile_pic = current_dir /  "profile-pic.png"
bio_file_path = current_dir / "bio.txt"

# ---- General ------
PAGE_TITLE = "Digital CV / Gaspard Hassenforder"
PAGE_ICON = ":wave"
NAME = "Gaspard Hassenforder"
DESCRIPTION = "Data Scientist with a strong background in data analysis, machine learning, and statistical modeling. Passionate about leveraging data to drive business decisions and enhance operational efficiency. Adept at collaborating with cross-functional teams to deliver impactful solutions."

EMAIL = "hassenforder.gaspard@gmail.com"
SOCIAL_MEDIA = {
    "LinkedIn" : "https://www.linkedin.com/in/gaspard-hassenforder-554431225/",
    "GitHub" : "https://github.com/Gaspard0302"
    }

PROJECTS = {
    "CapGemini DataCamp" : "Contributed to a team project analysing customer feedback via sentiment analysis, utilising BERT and GPT-3.5 for data scraped from Trustpilot. Enhanced client's customer satisfaction through actionable insights derived from advanced NLP techniques.",
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
# Checkbox to ask if the user has an API key
has_api_key = st.checkbox('Would you like to use your own OpenAI API Key?')

openai_api_key = None
if has_api_key:
    openai_api_key = st.text_input('Please enter your OpenAI API Key', type="password")
else:
    openai_api_key = st.secrets["OPENAI_API_KEY"]
    #openai_api_key = os.getenv("OPENAI_API_KEY")

if openai_api_key:
    openai.api_key = openai_api_key

# load the file
def ask_bot(input_text):
    try:
        # define LLM
        llm = ChatOpenAI(
        model="gpt-4o",
        temperature=1,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key= openai_api_key
        
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
            "You are Buddy, an AI assistant dedicated to assisting Gaspard in her job search by providing recruiters with relevant and concise information. "
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

        print(f"output: {response.content}")
        return response.content

    except OpenAIError as e:
        # Handle specific OpenAI errors
        if "insufficient funds" in str(e).lower():
            st.error("⚠️ My OpenAI API key does not have enough credits. Please provide your own API key.", icon='⚠')
        else:
            st.error(f"⚠️ An error occurred: {str(e)}", icon='⚠')

# get the user's input by calling the get_text function
def get_text():
    input_text = st.text_input("You can send your questions and hit Enter to know more about me from my AI agent", key="input")
    return input_text

#st.markdown("Chat With Me Now")

user_input = get_text()

if user_input:
#text = st.text_area('Enter your questions')
    if not openai_api_key.startswith('sk-'):
        st.warning('⚠️Please enter your OpenAI API key.', icon='⚠')
    if openai_api_key.startswith('sk-'):
        st.info(ask_bot(user_input))
        

    
    
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

# --- EDUCATION ---
st.write("---")
st.write('\n')
st.subheader("Education")


# --- DEGREE 1
st.write("🎓", "**HEC, Paris**")
st.write("Master in Data Science and AI for Business — 2024/2025")
st.write(
    """
- Focused on real-world business challenges through data-driven solutions.
- Developed skills in analyzing business issues, building and optimizing models, and understanding the societal impact of AI.
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

# --- Projects & Accomplishments ---
st.write("---")
st.write('\n')
st.subheader("Projects & Accomplishments")

for project, explanation in PROJECTS.items():
    st.write(f" - **{project}** :  {explanation}")

    #st.write(f"[{project}]({link})")

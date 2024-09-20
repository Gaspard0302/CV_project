


from pathlib import Path
import streamlit as st
from PIL import Image

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI


import openai
# from langchain.chat_models import ChatOpenAI

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
DESCRIPTION = "Data Scientist ....."
EMAIL = "hassenforder.gaspard@gmail.com"
SOCIAL_MEDIA = {
    "LinkedIn" : "",
    "GitHub" : "",
}

PROJECTS = {
    "Title " : "link",
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
     st.image(profile_pic, width=230)

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
# Set up the OpenAI key
openai_api_key = st.text_input('Enter your OpenAI API Key and hit Enter', type="password")
openai.api_key = (openai_api_key)

# load the file
def ask_bot(input_text):
    # define LLM
    llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
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

    # Specify the path to your context file
    context_file_path = 'bio.txt'

    # Read the context from the file
    context = read_context_from_file(context_file_path)

    # Create the prompt template
    prompt = PromptTemplate.from_template(
        "You are Buddy, an AI assistant dedicated to assisting Gaspard in her job search by providing recruiters with relevant and concise information. "
        "Here is his CV {context}"
        "If you do not know the answer, politely admit it and let recruiters know how to contact Gaspard to get more information directly from him. "
        "Don't put Buddy or a breakline in the front of your answer and be concise, only one sentence. Human: {question}"
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

# get the user's input by calling the get_text function
def get_text():
    input_text = st.text_input("After providing OpenAI API Key on the sidebar, you can send your questions and hit Enter to know more about me from my AI agent, Buddy!", key="input")
    return input_text

#st.markdown("Chat With Me Now")
if openai_api_key: 
    user_input = get_text()

    if user_input:
    #text = st.text_area('Enter your questions')
        if not openai_api_key.startswith('sk-'):
            st.warning('⚠️Please enter your OpenAI API key on the sidebar.', icon='⚠')
        if openai_api_key.startswith('sk-'):
            st.info(ask_bot(user_input))


    
    
# --- SOCIAL LINKS ---
st.write('\n')
cols = st.columns(len(SOCIAL_MEDIA))
for index, (platform, link) in enumerate(SOCIAL_MEDIA.items()):
    cols[index].write(f"[{platform}]({link})")

# --- EXPERIENCE & QUALIFICATIONS ---
st.write('\n')
st.subheader("Experience & Qulifications")
st.write(
    """
- ✔️ 7 Years expereince extracting actionable insights from data
- ✔️ Strong hands on experience and knowledge in Python and Excel
- ✔️ Good understanding of statistical principles and their respective applications
- ✔️ Excellent team-player and displaying strong sense of initiative on tasks
"""
)


# --- SKILLS ---
st.write('\n')
st.subheader("Hard Skills")
st.write(
    """
- 👩‍💻 Programming: Python (Scikit-learn, Pandas), SQL, VBA
- 📊 Data Visulization: PowerBi, MS Excel, Plotly
- 📚 Modeling: Logistic regression, linear regression, decition trees
- 🗄️ Databases: Postgres, MongoDB, MySQL
"""
)


# --- WORK HISTORY ---
st.write('\n')
st.subheader("Work History")
st.write("---")

# --- JOB 1
st.write("🚧", "**Senior Data Analyst | Ross Industries**")
st.write("02/2020 - Present")
st.write(
    """
- ► Used PowerBI and SQL to redeﬁne and track KPIs surrounding marketing initiatives, and supplied recommendations to boost landing page conversion rate by 38%
- ► Led a team of 4 analysts to brainstorm potential marketing and sales improvements, and implemented A/B tests to generate 15% more client leads
- ► Redesigned data model through iterations that improved predictions by 12%
"""
)

# --- JOB 2
st.write('\n')
st.write("🚧", "**Data Analyst | Liberty Mutual Insurance**")
st.write("01/2018 - 02/2022")
st.write(
    """
- ► Built data models and maps to generate meaningful insights from customer data, boosting successful sales eﬀorts by 12%
- ► Modeled targets likely to renew, and presented analysis to leadership, which led to a YoY revenue increase of $300K
- ► Compiled, studied, and inferred large amounts of data, modeling information to drive auto policy pricing
"""
)

# --- JOB 3
st.write('\n')
st.write("🚧", "**Data Analyst | Chegg**")
st.write("04/2015 - 01/2018")
st.write(
    """
- ► Devised KPIs using SQL across company website in collaboration with cross-functional teams to achieve a 120% jump in organic traﬃc
- ► Analyzed, documented, and reported user survey results to improve customer communication processes by 18%
- ► Collaborated with analyst team to oversee end-to-end process surrounding customers' return data
"""
)


# --- Projects & Accomplishments ---
st.write('\n')
st.subheader("Projects & Accomplishments")
st.write("---")
for project, link in PROJECTS.items():
    st.write(f"[{project}]({link})")

from pathlib import Path
import streamlit as st
from PIL import Image
import os
import traceback

# ---- Path Settings ------
current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
css_file = current_dir / "main.css"
resume_file = current_dir / "CV.pdf"
profile_pic = current_dir / "profile-pic.png"
bio_file_path = current_dir / "bio.txt"

# ---- General ------
PAGE_TITLE = "Digital CV / Gaspard Hassenforder"
PAGE_ICON = ":wave:"
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
try:
    with open(css_file) as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)
except FileNotFoundError:
    pass  # Use default styling

try:
    with open(resume_file, "rb") as pdf_file:
        PDFbyte = pdf_file.read()
except FileNotFoundError:
    PDFbyte = None
    
try:
    with open('ResearchPaper.pdf', "rb") as pdf_file:
        research_paper_byte = pdf_file.read()
except FileNotFoundError:
    research_paper_byte = None
    
try:
    profile_pic = Image.open(profile_pic)
except FileNotFoundError:
    profile_pic = None

# Load CV context for chatbot
try:
    with open(bio_file_path, "r") as f:
        cv_context = f.read()
except FileNotFoundError:
    cv_context = f"""
    {NAME} - {DESCRIPTION}
    
    Email: {EMAIL}
    
    Projects: {'; '.join([f'{k}: {v}' for k, v in PROJECTS.items()])}
    
    Education: HEC Paris (Master Data Science 2024-2025), Polytechnique Paris (Master 2023-2024), 
    Warwick University (Bachelor Math 2020-2023)
    
    Experience: Data Scientist Intern at Ekimetrics (Apr-Sep 2024), Software Dev Intern at VP & White UK (Aug-Sep 2022)
    """

# --- HERO SECTION ---
col1, col2 = st.columns(2, gap="small")
with col1:
    if profile_pic:
        st.image(profile_pic, width=300)

with col2:
    st.title(NAME)
    st.write(DESCRIPTION)
    if PDFbyte:
        st.download_button(
            label=" 📄 Download Resume",
            data=PDFbyte,
            file_name=resume_file.name,
            mime="application/octet-stream",
        )
    st.write("📫", EMAIL)

# --- SOCIAL LINKS ---
st.write('\n')
cols = st.columns(len(SOCIAL_MEDIA))
for index, (platform, link) in enumerate(SOCIAL_MEDIA.items()):
    cols[index].write(f"[{platform}]({link})")

# ----------------- SIDEBAR AI ASSISTANT ----------------- #
with st.sidebar:
    st.title("🤖 AI Assistant")
    st.write("Ask anything about Gaspard's experience, skills, or projects!")
    
    # Initialize OpenAI
    api_available = False
    client = None
    
    try:
        import openai
        
        # Check if API key is available
        if "OPENAI_API_KEY" in st.secrets:
            try:
                # Use the new OpenAI client initialization (v1.0+)
                client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
                api_available = True
            except Exception as e:
                st.error(f"Error initializing OpenAI client: {str(e)}")
        else:
            st.error("⚠️ OpenAI API key not configured. Add it to your Streamlit secrets.")
    except ImportError:
        st.error("⚠️ OpenAI library not installed. Add 'openai' to your requirements.txt.")
    except Exception as e:
        st.error(f"⚠️ Error initializing OpenAI: {str(e)}")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("Ask about Gaspard...", key="sidebar_chat"):
        if not api_available:
            st.error("⚠️ AI Assistant unavailable. Contact Gaspard directly.")
            st.stop()

        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            system_prompt = f"""You are Buddy, Gaspard Hassenforder's AI assistant.
            Provide concise, relevant information based on his background.
            
            CV Context:
            {cv_context}
            
            Be professional, helpful, and concise. If information isn't available, 
            suggest contacting Gaspard at {EMAIL}."""

            try:
                # Use the new OpenAI API format
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "system", "content": system_prompt}] + 
                             [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
                    stream=True,
                )
                
                # Stream the response with typewriter effect
                full_response = st.write_stream(response)
                
            except Exception as e:
                full_response = f"Error: {str(e)}. Please contact Gaspard at {EMAIL}."
                st.write(full_response)
                # Print detailed error for debugging
                st.error(f"Detailed error: {traceback.format_exc()}")

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})

    # Add clear chat button
    if st.session_state.messages:
        if st.button("🗑️ Clear Chat", key="clear_sidebar_chat"):
            st.session_state.messages = []
            st.rerun()

# --- EXPERIENCE & QUALIFICATIONS ---
st.write("---")
st.subheader("Experience & Qualifications")
st.write("""
- ✔️ Experience in data science internships focusing on data-driven solutions and model development
- ✔️ Strong hands-on experience and knowledge in Python, SQL, and data analysis tools
- ✔️ Good understanding of statistical principles and machine learning applications
- ✔️ Excellent team player with a strong sense of initiative and leadership experience
""")

# --- SKILLS ---
st.write('\n')
st.subheader("Hard Skills")
st.write("""
- 👩‍💻 **Programming**: Python (Pandas, NumPy, Scikit-learn, PyTorch), SQL, moderate knowledge of C#
- 📊 **Data Visualization**: Tableau, MS Excel, Plotly
- 📚 **Modeling**: Machine Learning, Deep Learning, Causality Modeling
- 🗄️ **Databases**: PostgreSQL, experience with data retrieval systems
""")

# --- Projects & Accomplishments ---
st.write("---")
st.write('\n')
st.subheader("Projects & Accomplishments")

for project, explanation in PROJECTS.items():
    st.write(f"- **{project}**: {explanation}")

# --- EDUCATION ---
st.write("---")
st.write('\n')
st.subheader("Education")

# --- DEGREE 1
st.write("🎓 **HEC, Paris** — *Master in Data Science and AI for Business — 2024/2025*")
st.write("""
- Focused on real-world business challenges through data-driven solutions, developed skills in analysing business issues related to Data Science and AI, building and optimising models using state-of-the-art tools, and understanding their societal impact.
- **Key Courses**: Algorithmic Fairness and Interpretability, Causal Inference, ML Ops, Reinforcement Learning
""")

# Research paper with inline download link
if research_paper_byte:
    import base64
    b64 = base64.b64encode(research_paper_byte).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="Hybridization_of_Consultancy_Work.pdf" style="color: #0066cc; text-decoration: none;">download</a>'
    st.markdown(f"""
- **Research Paper**: *"The Hybridization of Consultancy Work"* - Designed sophisticated human-AI partnership system with hierarchical multi-agent architecture and dynamic knowledge graph integration [{href}].
""", unsafe_allow_html=True)
else:
    st.write("""
- **Research Paper**: *"The Hybridization of Consultancy Work"* - Designed sophisticated human-AI partnership system with hierarchical multi-agent architecture and dynamic knowledge graph integration
""")

# --- DEGREE 2
st.write('\n')
st.write("🎓 **Polytechnique, Paris** — *Master in Data Science and AI for Business — 2023/2024*")
st.write("""
- Honed expertise in key disciplines including Statistics, Machine Learning, Database Management, Deep Learning, and Optimization.
""")

# --- DEGREE 3
st.write('\n')
st.write("🎓 **Warwick University, London, UK** — *Bachelor in Mathematics — 2020/2023*")
st.write("""
- Improved my problem-solving capabilities and learned mathematical concepts and understood their applications.
- **Key modules**: Topology, Multivariate Statistics, and Number Theory. Graduated with a high 2.1 degree.
""")

# --- DEGREE 4
st.write('\n')
st.write("🎓 **Lycée Francais Charles de Gaulle, London, UK** —*French Baccalauréat (Scientific Section) — 2020*")
st.write("""
- Obtained with highest honors (17.02/20), specializing in Maths, Physics, and Sciences.
""")

# --- WORK HISTORY ---
st.write("---")
st.write('\n')
st.subheader("Work History")

# --- JOB 0
st.write("🚧 **AI Engineer Consultant | Veltys** — *April 2025 / September 2025*") 
st.write("""
- ► Designed and developed a multi-agent AI system to automate consultancy proposals and company research workflows
- ► Built a knowledge graph platform with custom agents to surface relevant data, reducing administrative time by 50%
- ► Implemented Google Cloud database architecture with automated data ingestion and BigQuery chatbot querying
- ► Enabled consultants to focus on high-value strategy work by eliminating repetitive research tasks
""")


# --- JOB 1
st.write("🚧 **Data Scientist Intern | Ekimetrics** — *April 2024 / September 2024*")
st.write("""
- ► Implemented a Retrieval Augmented Generation solution to automate Corporate Sustainability Assessment processes, significantly improving efficiency
- ► Developed a causality model using Python to measure customer uplift from YouTube ad campaigns, providing actionable insights
- ► Contributed to a Marketing Mix Modeling project for a major airline to optimize budget allocation and marketing ROI
""")

# --- JOB 2
st.write('\n')
st.write("🚧 **Software Development Intern | VP & White UK Ltd** — *August 2022 / September 2022*")
st.write("""
- ► Developed front and back end processes for a new e-learning platform using C#, SQL, and .NET
- ► Created unit testing processes and logged outstanding issues to enhance software quality
""")

# --- JOB 3
st.write('\n')
st.write("🚧 **Intern, Partnerships and University Recruitment Analyst | Dassault Systèmes** - *July 2019*")
st.write("""
- ► Analyzed and mapped existing university partnerships to optimize recruitment strategies
- ► Gained hands-on experience with CAD software, enhancing understanding of intricate models
""")

# --- Papers & Thoughts Section ---
st.write("---")
st.subheader("Papers & Thoughts")
st.write("A curated list of research papers that have shaped my thinking in AI and data science.")

PAPERS = {
    "Hierarchical Reasoning Model": {
        "link": "https://arxiv.org/abs/2506.21734",
        "thoughts": """
        The **Hierarchical Reasoning Model (HRM)** paper is one of the most intriguing pieces of research I've encountered lately. The idea of leveraging a brain-inspired architecture with two interconnected recurrent modules—one for high-level, abstract reasoning and another for detailed, rapid computations—feels like a significant departure from the current trend of scaling up transformer models. It's refreshing to see RNNs making a comeback, especially with each cell incorporating transformer-based attention mechanisms.

        What stands out the most is how HRM achieves such strong performance with only **27 million parameters**, outperforming much larger models like o3 and Grok4 on the ARC-AGI benchmark. Its ability to solve complex Sudoku puzzles and navigate mazes perfectly—tasks where traditional LLMs often fail—is impressive. However, I do wonder if the benchmarks used are somewhat tailored to highlight HRM's strengths, as real-world problems tend to be far more unstructured and open-ended.

        Despite its strengths, HRM has notable limitations. The model's non-autoregressive nature means it can only process fixed-size grids as input and output, making it less versatile for general-purpose tasks like text generation or open-ended reasoning. Additionally, the reliance on a token to specify the problem type within the input feels like a constraint that limits its applicability in more dynamic, real-world scenarios.

        The **ARC Prize's independent analysis** provided a more grounded perspective on HRM's performance. While the model performs well on ARC-AGI-1, its accuracy plummets to just **2% on the more challenging ARC-AGI-2 benchmark**. This suggests that HRM's reasoning capabilities may not generalize as well as initially thought. Furthermore, the analysis revealed that much of HRM's success can be attributed to **iterative refinement** rather than the hierarchical architecture itself. In fact, a standard transformer with similar parameters can achieve comparable results, which is somewhat underwhelming given the initial hype.

        Another point of concern is HRM's reliance on **puzzle_id embeddings**, which means it can't generalize to new tasks it hasn't encountered during training. This is a significant limitation if we're considering HRM as a step toward AGI. Additionally, while the paper highlights the use of 1,000 augmentations per task, the analysis shows that similar performance can be achieved with just 30-300 augmentations, suggesting that some of the computational overhead might be unnecessary.

        Overall, HRM is a fascinating development in AI research. It challenges the notion that bigger models are inherently better and introduces a compelling brain-inspired design. However, the critical analysis reveals that its performance gains may not be as revolutionary as initially claimed. It's still an exciting direction, and I'm curious to see how this architecture evolves—perhaps in combination with other techniques like diffusion models—to address its current limitations.
        """
    },
     "Muon Clip Optimizer": {
        "link": "https://arxiv.org/abs/2507.20534",  # Actual link to the paper
        "thoughts": """
        The introduction of the **Kimi K2 model** by Moonshot AI showcases some truly groundbreaking advancements in AI model optimization and architecture design. At its core, Kimi K2 is a massive 1 trillion parameter model with 32 billion active parameters, which briefly held the title of the state-of-the-art open-source non-reasoning model. 

        ### **Muon Clip Optimizer**
        One of the standout innovations in Kimi K2 is the use of the **Muon Clip Optimizer**. This optimizer addresses a common issue in deep learning: loss spikes during training. Unlike traditional optimizers like Adam, which can overshoot and cause instability, the Muon Clip Optimizer introduces a mechanism to pause and adjust momentum dynamically. This results in a more stable descent through the loss landscape and significantly reduces training time and computational cost by up to 35%.

        - **Adaptive Momentum Adjustment**: By pausing to adjust the direction and magnitude of updates, the Muon optimizer prevents the overshooting problem common in Adam. This leads to more stable and efficient training.
        - **QK Clip Innovation**: Initially, the Muon optimizer faced challenges at scale, where outliers in the query and key norms would cause instability. The introduction of the QK Clip addressed this by clipping these outliers, thereby stabilizing training and making it viable for large-scale models like Kimi K2.

        ### **Architectural Innovations**
        - Moonshot AI made several notable modifications to the DeepSeek V3 architecture in Kimi K2:
          - Increased the number of experts per layer by 50% without changing the number of active parameters per token, leveraging a new sparsity scaling law.
          - Reduced the number of attention heads from 128 to 64, significantly cutting down on parameters but maintaining performance with only a 2% degradation in quality.
          - Simplified the routing mechanism by removing expert grouping, which became unnecessary at the trillion parameter scale where each GPU holds a single expert.

        ### **Impact and Implications**
        The advancements in Kimi K2 and the introduction of the Muon Clip Optimizer represent a significant leap in training efficiency and stability for large-scale AI models. The ability to train models with fewer loss spikes and greater computational efficiency opens new avenues for developing more powerful models. Additionally, the innovations in model architecture and optimization techniques demonstrated by Moonshot AI could influence future research and development in the field.

        Overall, the Kimi K2 and its associated technologies provide valuable insights that may shape the future trajectory of AI model training and optimization. It is thanks to these open source models and the work they do and share openly that opne source AI is able to keep up with the closed source models and drives innovation.
        """
    }
    
}

for title, data in PAPERS.items():
    st.markdown(f"**[{title}]({data['link']})**")
    with st.expander("My Thoughts & Key Takeaways"):
        st.write(data["thoughts"])
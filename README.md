# CV Streamlit App

This project is a Dockerized Streamlit application that showcases a digital version of my CV. The app features an interactive chatbot powered by OpenAI that can answer questions about my background, skills, and experiences.

## Features

- **Digital CV Display**: View my CV in a user-friendly format.
- **Chatbot Integration**: Ask questions about my qualifications, experiences, and more.
- **OpenAI API**: The chatbot utilizes the OpenAI API for generating responses.

## Requirements

- Docker
- OpenAI API Key (to be entered once the app is running)

## Getting Started

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Gaspard0302/CV_project.git
   cd cv_streamlit_app
   ```

2. **Build the Docker Image**:
  ```bash
  docker build --no-cache -t cv_streamlit_app .
  ```
3.	**Run the Docker Container**:
  ```
  docker run -p 8501:8501 cv_streamlit_app
  ```
4. **Access the App**:
   Open your web browser and navigate to http://localhost:8501.

5. **Enter Your API Key:**
   Once the app is running, you will be prompted to enter your OpenAI API key in the designated field to enable chatbot functionality.

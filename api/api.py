
from fastapi import FastAPI, File, UploadFile, Form
from mangum import Mangum
import docx2txt
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import random

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

app = FastAPI()

# Function to load and preprocess text
def load_and_preprocess_text(file_content: str):
    text = file_content.lower()
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text

# Function to generate descriptions based on missing keywords
def generate_descriptions(missing_keywords):
    templates = {
        'technical_skills': [
            "Proficient in {keyword}, with hands-on experience in its application.",
            "Extensive knowledge of {keyword}, leading to improved project outcomes."
        ],
        'soft_skills': [
            "Strong {keyword} abilities, fostering collaboration.",
            "Excellent {keyword}, ensuring effective communication and problem-solving."
        ],
        'project_experience': [
            "Played a crucial role in a project focused on {keyword}, achieving key objectives ahead of schedule.",
            "Contributed to project initiatives involving {keyword}, meeting deadlines."
        ]
    }

    generated_descriptions = {}
    for keyword in missing_keywords:
        generated_descriptions[keyword] = {
            section: [statement.format(keyword=keyword) for statement in templates[section]]
            for section in templates
        }
    return generated_descriptions

# Function to find missing keywords between resume and job description
def find_missing_keywords(resume_text, job_text):
    resume_words = set(resume_text.split())
    job_words = set(job_text.split())
    missing_keywords = job_words - resume_words
    return list(missing_keywords)

# Endpoint to match resume and job description and return match percentage + description suggestions
@app.post("/match_resume/")
async def match_resume(file: UploadFile = File(...), job_description: str = Form(...)):
    # Load and preprocess the resume file and job description text
    resume_content = docx2txt.process(file.file)
    resume_text = load_and_preprocess_text(resume_content)
    job_text = load_and_preprocess_text(job_description)

    # Vectorize the resume and job description using TF-IDF
    tfidf_vectorizer = TfidfVectorizer()
    count_matrix = tfidf_vectorizer.fit_transform([resume_text, job_text])

    # Calculate cosine similarity
    similarity_score = cosine_similarity(count_matrix)[0][1]
    match_percentage = round(similarity_score * 100, 2)

    # Find missing keywords from job description
    missing_keywords = find_missing_keywords(resume_text, job_text)

    # Generate keyword-based descriptions
    if missing_keywords:
        descriptions = generate_descriptions(missing_keywords)
    else:
        descriptions = {}

    return {
        "match_percentage": match_percentage,
        "descriptions": descriptions
    }
handler = Mangum(app)

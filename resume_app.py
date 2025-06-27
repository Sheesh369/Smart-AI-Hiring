import streamlit as st
import PyPDF2
from docx import Document
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load standard BERT model (no adversarial stuff)
@st.cache_resource
def load_model():
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=14)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model.eval()
    return model, tokenizer

model, tokenizer = load_model()

# Role-to-class mapping
role_to_class = {
    "python developer": 1, "data scientist": 2, "data analyst": 3,
    "web designing": 4, "business analyst": 5, "business development": 6,
    "hr": 7, "public relations": 8, "network security engineer": 9,
    "software engineer": 10, "devops engineer": 11, "automation tester": 12,
    "ui/ux designer": 13
}

# Keywords for missing skills feedback
keywords = {
    "python developer": ["python", "django", "flask", "machine learning", "pandas", "numpy"],
    "data scientist": ["data science", "machine learning", "deep learning", "statistics", "ai", "nlp"],
    "data analyst": ["data analysis", "excel", "power bi", "tableau", "sql", "analytics"],
    "web designing": ["html", "css", "javascript", "ui/ux", "adobe xd", "figma", "responsive design"],
    "business analyst": ["business analysis", "process improvement", "requirements gathering", "agile", "scrum"],
    "business development": ["lead generation", "sales", "marketing", "client relationships", "negotiation"],
    "hr": ["human resources", "recruitment", "talent acquisition", "employee engagement", "hr policies"],
    "public relations": ["media", "communication", "press releases", "public relations", "crisis management"],
    "network security engineer": ["networking", "security", "firewalls", "routing", "switching", "vpn", "cisco"],
    "software engineer": ["software development", "java", "c++", "system design", "git", "agile"],
    "devops engineer": ["devops", "ci/cd", "jenkins", "kubernetes", "docker", "aws", "linux"],
    "automation tester": ["automation testing", "selenium", "test cases", "regression testing", "junit", "cypress"],
    "ui/ux designer": ["ui design", "ux design", "prototyping", "adobe xd", "figma", "user experience", "wireframing"]
}

# Extract text from PDF resumes
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Extract text from DOCX resumes
def extract_text_from_docx(docx_file):
    doc = Document(docx_file)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

# Tokenize text using tokenizer
def preprocess_and_tokenize(text):
    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    return inputs

# Predict resume suitability
def predict_suitability(model, inputs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    with torch.no_grad():
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        predicted_class = torch.argmax(logits, dim=1).item()
    return predicted_class, logits

# Identify missing skills for feedback
def get_missing_skills(role, resume_text):
    resume_text_lower = resume_text.lower()
    required_skills = keywords.get(role, [])
    missing_skills = [skill for skill in required_skills if skill not in resume_text_lower]
    return missing_skills

# Streamlit UI
st.title("AI-Powered Resume Screening and Feedback System")

st.subheader("Job Description (JD)")
job_description = st.text_area("Enter the JD for the role you're applying for:")

uploaded_file = st.file_uploader("Upload Resume (PDF or DOCX)", type=["pdf", "docx"])

if uploaded_file and job_description:
    # Extract Resume Text
    if uploaded_file.name.endswith(".pdf"):
        resume_text = extract_text_from_pdf(uploaded_file)
    elif uploaded_file.name.endswith(".docx"):
        resume_text = extract_text_from_docx(uploaded_file)
    else:
        st.error("Unsupported file format!")
        st.stop()

    # Validate extracted text
    if not isinstance(resume_text, str) or not resume_text.strip():
        st.error("Invalid or empty resume text. Please upload a valid resume.")
        st.stop()

    st.subheader("Extracted Resume Text")
    st.write(resume_text)

    # Select Role
    role = st.selectbox("Select the role you are applying for:", list(role_to_class.keys()))

    if role:
        normalized_role = role.lower()
        selected_class = role_to_class.get(normalized_role)
        if selected_class:
            st.write(f"Selected Role: {role}, Mapped Class: {selected_class}")

            inputs = preprocess_and_tokenize(resume_text)
            predicted_class, logits = predict_suitability(model, inputs)

            # Simple keyword-based decision refinement
            refined_class = role if role in keywords and any(
                keyword in resume_text.lower() for keyword in keywords[role]
            ) else "Other"

            decision = "Selected" if refined_class == role.lower() else "Rejected"

            st.subheader("Prediction Result")
            st.write(f"Decision: {decision}")

            st.subheader("Feedback")
            missing_skills = get_missing_skills(normalized_role, resume_text)
            if decision == "Rejected" and missing_skills:
                feedback = (
                    f"Your resume does not meet the job criteria. "
                    f"Consider improving your skills in the following areas: {', '.join(missing_skills)}."
                )
            elif decision == "Rejected":
                feedback = "Your resume does not meet the job criteria, but no specific missing skills were identified."
            else:
                feedback = "Your resume aligns well with the requirements."
            st.write(feedback)
else:
    if not job_description:
        st.warning("Please enter a Job Description (JD).")

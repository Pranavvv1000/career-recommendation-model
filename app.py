from flask import Flask, request, render_template
import pandas as pd
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)

# Load SBERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load job data from CSV
df = pd.read_csv('djobv2.csv')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Collect responses from the form
        user_responses = []
        for i in range(1, 4):  # Loop through the 2 questions
            answer = request.form.get(f'q{i}', '').strip()
            text_answer = request.form.get(f'love{i}', '').strip()  # Textarea response
            
            # If both select and text area are filled, prioritize text
            user_responses.append(text_answer if text_answer else answer)

        # Combine responses into one user profile text
        user_profile = " ".join(user_responses)

        if not user_profile.strip():
            return render_template('index.html', error="Please answer at least one question.")

        # Convert job descriptions to embeddings
        job_descriptions = df['Description'].tolist()
        job_titles = df['Title'].tolist()
        job_keywords = df['keyword'].tolist()

        job_embeddings = model.encode(job_descriptions, convert_to_tensor=True)
        user_embedding = model.encode(user_profile, convert_to_tensor=True)

        # Compute similarity scores
        similarities = util.pytorch_cos_sim(user_embedding, job_embeddings)[0]

        # Rank jobs based on similarity
        ranked_jobs = sorted(zip(job_titles, job_descriptions,  similarities.tolist()), key=lambda x: x[2], reverse=True)

        return render_template('result.html', ranked_jobs=ranked_jobs[:3])  # Show top 3 recommendations

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
import cv2
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR
from spellchecker import SpellChecker
from textblob import TextBlob
from transformers import pipeline
import nltk
import os
from flask import Flask, request, jsonify
from googlesearch import search  # For Google Search

# Download NLTK data for tokenization
try:
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('averaged_perceptron_tagger_eng')
except Exception as e:
    print(f"Error downloading NLTK data: {e}")

# Initialize Flask app
app = Flask(__name__)

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Initialize summarizer and QA pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)  # CPU
qa = pipeline("text2text-generation", model="google/flan-t5-base", device=-1)  # CPU

# Global variable to store the corrected text (shared across requests)
corrected_text = ""

# === Image Processing Functions ===
def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# === Text Processing Functions ===
def correct_text(text):
    spell = SpellChecker()
    words = nltk.word_tokenize(text.lower())
    corrected_words = []
    for word in words:
        if not word.isalpha():
            corrected_words.append(word)
            continue
        if word in spell:
            corrected_words.append(word)
        else:
            correction = spell.correction(word)
            if correction is None:
                blob = TextBlob(word)
                corrected_word = str(blob.correct())
                corrected_words.append(corrected_word)
            else:
                corrected_words.append(correction)
    corrected_text = " ".join(corrected_words)
    corrected_text = " ".join(sentence.capitalize() for sentence in nltk.sent_tokenize(corrected_text))
    return corrected_text

def generate_qa_pairs(text):
    qa_pairs = []
    context = text
    processed_questions = set()

    question_prompts = [
        "Generate a question about the main topic of the following text: ",
        "Generate a question about examples mentioned in the following text: ",
        "Generate a question about the effects described in the following text: ",
        "Generate a question about activities described in the following text: ",
        "Generate a question about skills mentioned in the following text: ",
        "Generate a question about paragraphs in the following text: "
    ]

    for prompt in question_prompts:
        input_text = f"{prompt} {context}"
        question = qa(input_text, max_length=50, min_length=10, do_sample=False)[0]['generated_text']
        if question not in processed_questions and question.startswith("What"):
            answer_input = f"question: {question} context: {context}"
            answer = qa(answer_input, max_length=50, min_length=10, do_sample=False)[0]['generated_text']
            if len(answer.strip()) > 5 and answer != context:
                qa_pairs.append({"question": question, "answer": answer})
            processed_questions.add(question)

    # Fallback rule-based questions
    sentences = nltk.sent_tokenize(text)
    for sentence in sentences:
        words = nltk.word_tokenize(sentence)
        tagged = nltk.pos_tag(words)
        if "discusses" in words or "focuses" in words or "describes" in words:
            keyword = "discusses" if "discusses" in words else "focuses" if "focuses" in words else "describes"
            question = f"What does the text {keyword}?"
            if question not in processed_questions:
                input_text = f"question: {question} context: {context}"
                answer = qa(input_text, max_length=50, min_length=10, do_sample=False)[0]['generated_text']
                if len(answer.strip()) > 5 and answer != context:
                    qa_pairs.append({"question": question, "answer": answer})
                processed_questions.add(question)
        if "skills" in words or "activities" in words or "paragraph" in words:
            if "skills" in words and "What skills are mentioned in the text?" not in processed_questions:
                question = "What skills are mentioned in the text?"
                input_text = f"question: {question} context: {context}"
                answer = qa(input_text, max_length=50, min_length=10, do_sample=False)[0]['generated_text']
                if len(answer.strip()) > 5 and answer != context:
                    qa_pairs.append({"question": question, "answer": answer})
                processed_questions.add(question)
            if "activities" in words and "What activities are mentioned in the text?" not in processed_questions:
                question = "What activities are mentioned in the text?"
                input_text = f"question: {question} context: {context}"
                answer = qa(input_text, max_length=50, min_length=10, do_sample=False)[0]['generated_text']
                if len(answer.strip()) > 5 and answer != context:
                    qa_pairs.append({"question": question, "answer": answer})
                processed_questions.add(question)
            if "paragraph" in words and "What is mentioned about paragraphs in the text?" not in processed_questions:
                question = "What is mentioned about paragraphs in the text?"
                input_text = f"question: {question} context: {context}"
                answer = qa(input_text, max_length=50, min_length=10, do_sample=False)[0]['generated_text']
                if len(answer.strip()) > 5 and answer != context:
                    qa_pairs.append({"question": question, "answer": answer})
                processed_questions.add(question)

    return qa_pairs

# === Google Search Function ===
def search_source(text):
    search_query = f'"{text[:100]}"'  # Use first 100 characters for search
    try:
        search_results = []
        for result in search(search_query, num_results=3, lang="en"):  # Top 3 results
            search_results.append(result)
        return search_results
    except Exception as e:
        print(f"Error during Google Search: {e}")
        return ["No source found due to search error"]

# === Flask Routes ===
@app.route('/extract_text', methods=['POST'])
def extract_text():
    global corrected_text

    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save the uploaded image
    image_path = "temp/uploaded_image.jpg"
    file.save(image_path)

    # Process the image
    img = cv2.imread(image_path)
    if img is None:
        return jsonify({"error": "Could not load image"}), 400

    inverted_image = cv2.bitwise_not(img)
    cv2.imwrite("temp/inverted.png", inverted_image)

    gray_image = grayscale(inverted_image)
    cv2.imwrite("temp/gray.png", gray_image)

    # OCR
    result = ocr.ocr("temp/gray.png")
    ocr_text = ""
    if result and len(result) > 0 and result[0]:
        for line in result[0]:
            if len(line) >= 2:
                text = line[1][0]
                ocr_text += text + " "
    else:
        return jsonify({"error": "No text detected in the image"}), 400

    # Correct the text
    corrected_text = correct_text(ocr_text)

    return jsonify({"text": corrected_text})

@app.route('/summarize', methods=['GET'])
def summarize_text():
    global corrected_text
    if not corrected_text:
        return jsonify({"error": "No text available to summarize. Extract text first."}), 400

    if len(corrected_text.split()) > 1000:
        corrected_text_truncated = " ".join(corrected_text.split()[:1000])
    else:
        corrected_text_truncated = corrected_text

    summary = summarizer(corrected_text_truncated, max_length=80, min_length=30, do_sample=False)
    return jsonify({"summary": summary[0]['summary_text']})

@app.route('/generate_qa', methods=['GET'])
def generate_qa():
    global corrected_text
    if not corrected_text:
        return jsonify({"error": "No text available to generate Q&A. Extract text first."}), 400

    qa_pairs = generate_qa_pairs(corrected_text)
    return jsonify({"qa_pairs": qa_pairs})

@app.route('/get_sources', methods=['GET'])
def get_sources():
    global corrected_text
    if not corrected_text:
        return jsonify({"error": "No text available to search sources. Extract text first."}), 400

    search_results = search_source(corrected_text)
    return jsonify({"sources": search_results})

# Run the server
if __name__ == "__main__":
    os.makedirs("temp", exist_ok=True)  # Ensure temp directory exists
    app.run(host="0.0.0.0", port=5000, debug=True)
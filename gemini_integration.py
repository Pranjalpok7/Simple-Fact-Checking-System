import google.generativeai as genai

def setup_gemini(api_key):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-pro')

def generate_answer(query, context, model):
    prompt = f"Based on the following context, answer the query:\n\nContext: {context}\n\nQuery: {query}\n\nAnswer:"
    response = model.generate_content(prompt)
    return response.text
import streamlit as st
import torch
from transformers import AlbertTokenizer, AlbertForSequenceClassification, AlbertConfig
import plotly.graph_objects as go

# URL of the logo
logo_url = "https://dejan.ai/wp-content/uploads/2024/02/dejan-300x103.png"

# Display the logo at the top using st.logo
st.logo(logo_url, link="https://dejan.ai")

# Streamlit app title and description
st.title("Search Query Form Classifier")
st.write("Ambiguous search queries are candidates for query expansion. Our model identifies such queries with an 80 percent accuracy and is deployed in a batch processing pipeline directly connected with Google Search Console API. In this demo you can test the model capability by testing individual queries.")
st.write("Enter a query to check if it's well-formed:")

# Load the model and tokenizer from the /model/ directory
model_dir = 'model'
tokenizer = AlbertTokenizer.from_pretrained(model_dir)
config = AlbertConfig.from_pretrained(model_dir)
model = AlbertForSequenceClassification.from_pretrained(model_dir, config=config)

# Set the model to evaluation mode
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# User input
user_input = st.text_input("Query:", "What is?")
st.write("Developed by [Dejan AI](https://dejan.ai/blog/search-query-quality-classifier/)")

def classify_query(query):
    # Tokenize input
    inputs = tokenizer.encode_plus(
        query,
        add_special_tokens=True,
        max_length=32,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    # Perform inference
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        softmax_scores = torch.softmax(logits, dim=1).cpu().numpy()[0]
        confidence = softmax_scores[1] * 100  # Confidence for well-formed class

    return confidence

# Check and display classification
if user_input:
    confidence = classify_query(user_input)

    # Plotly gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence,
        title={'text': "Well-formedness Confidence"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "red"},
                {'range': [50, 100], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': confidence
            }
        }
    ))

    st.plotly_chart(fig)

    if confidence >= 50:
        st.success(f"The query is likely well-formed with {confidence:.2f}% confidence.")
    else:
        st.error(f"The query is likely not well-formed with {100 - confidence:.2f}% confidence.")

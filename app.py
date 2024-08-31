import streamlit as st
import torch
from transformers import AlbertTokenizer, AlbertForSequenceClassification
import plotly.graph_objects as go

# URL of the logo
logo_url = "https://dejan.ai/wp-content/uploads/2024/02/dejan-300x103.png"

# Display the logo at the top using st.logo
st.logo(logo_url, link="https://dejan.ai")

# Streamlit app title and description
st.title("Search Query Form Classifier")
st.write(
    "Ambiguous search queries are candidates for query expansion. Our model identifies such queries with an 80 percent accuracy and is deployed in a batch processing pipeline directly connected with Google Search Console API. In this demo you can test the model capability by testing individual queries."
)
st.write("Enter a query to check if it's well-formed:")

# Load the model and tokenizer from the Hugging Face Model Hub
model_name = 'dejanseo/Query-Quality-Classifier'
tokenizer = AlbertTokenizer.from_pretrained(model_name)
model = AlbertForSequenceClassification.from_pretrained(model_name)

# Set the model to evaluation mode 
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Create tabs for single and bulk queries
tab1, tab2 = st.tabs(["Single Query", "Bulk Query"])

with tab1:
    user_input = st.text_input("Query:", "What is?")
    #st.write("Developed by [Dejan AI](https://dejan.ai/blog/search-query-quality-classifier/)")

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

    # Function to determine color based on confidence
    def get_color(confidence):
        if confidence < 50:
            return 'rgba(255, 51, 0, 0.8)'  # Red
        else:
            return 'rgba(57, 172, 57, 0.8)'  # Green

    # Check and display classification for single query
    if user_input:
        confidence = classify_query(user_input)

        # Plotly grey placeholder bar with dynamic color fill
        fig = go.Figure()

        # Placeholder grey bar
        fig.add_trace(go.Bar(
            x=[100],
            y=['Well-formedness Factor'],
            orientation='h',
            marker=dict(
                color='lightgrey'
            ),
            width=0.8
        ))

        # Colored bar based on confidence
        fig.add_trace(go.Bar(
            x=[confidence],
            y=['Well-formedness Factor'],
            orientation='h',
            marker=dict(
                color=get_color(confidence)
            ),
            width=0.8
        ))

        fig.update_layout(
            xaxis=dict(range=[0, 100], title='Well-formedness Factor'),
            yaxis=dict(showticklabels=False),
            width=600,
            height=250,  # Increase height for better visibility
            title_text='Well-formedness Factor',
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False
        )

        st.plotly_chart(fig)

        if confidence >= 50:
            st.success(f"Query Score: {confidence:.2f}% Most likely doesn't require query expansion.")
            st.subheader(f":sparkles: What's next?", divider="gray")
            st.write("Connect with Google Search Console, Semrush, Ahrefs or any other search query source API and detect all queries which could benefit from expansion.")
            st.write("[Engage our team](https://dejan.ai/call/) if you'd like us to do this for you.")
        else:
            st.error(f"The query is likely not well-formed with a score of {100 - confidence:.2f}% and most likely requires query expansion.")
            st.subheader(f":sparkles: What's next?", divider="gray")
            st.write("Connect with Google Search Console, Semrush, Ahrefs or any other search query source API and detect all queries which could benefit from expansion.")
            st.write("[Engage our team](https://dejan.ai/call/) if you'd like us to do this for you.")

with tab2:
    st.write("Paste multiple queries line-separated (no headers or extra data):")
    bulk_input = st.text_area("Bulk Queries:", height=200)

    if bulk_input:
        bulk_queries = bulk_input.splitlines()
        st.write("Processing queries...")

        # Classify each query in bulk input
        results = [(query, classify_query(query)) for query in bulk_queries]

        # Display results in a table
        for query, confidence in results:
            st.write(f"Query: {query} - Score: {confidence:.2f}%")
            if confidence >= 50:
                st.success("Well-formed")
            else:
                st.error("Not well-formed")

        st.subheader(f":sparkles: What's next?", divider="gray")
        st.write("Connect with Google Search Console, Semrush, Ahrefs or any other search query source API and detect all queries which could benefit from expansion.")
        st.write("[Engage our team](https://dejan.ai/call/) if you'd like us to do this for you.")

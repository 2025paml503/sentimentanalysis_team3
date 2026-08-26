"""
ui/app.py

This UI knows nothing about scikit-learn, TF-IDF or how model was
trained. It only knows POST /predict construct. That is the whole point:
the model underneath can be retrained and swapped and this file keeps
working untouched, as long as the contract holds.

Run( with API running in another terminal)
    steamlit run ui/app.py

"""

import requests
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Sentiment Analysis", page_icon="*")
st.title("Sentiment Analysis")
st.caption("Sentiment Analysis")

with st.sidebar:
    st.subheader("Sentiment Analysis")
    try:
        info = requests.get(f"{API_URL}/model-info", timeout=5).json()
        st.write(f"**Version:** {info['model_version']}")
        st.write(f"**Type:** {info['model_type']}")
        st.write(f"**ML flow run:** {info['run_id'][:12]}...")
        st.write(f"**Vocabulary:** {info['vocab_size']:,} terms")
        st.write(f"**Test Metrics**")
        st.json(info['metrics'])
    except requests.exceptions.RequestException as e:
        st.error("API unreachable")

st.divider()

# Input form
with st.form("review_form"):
    title = st.text_input(
        "Review title(optional)",
        value = "Absolutely love these headphones"
    )
    review_text = st.text_area(
        "Review text",
        value="The sound quality is excellent and battery lasts all day",
        height=140
    )
    submitted = st.form_submit_button("Classify sentiment", type="primary")

if submitted:
    payload = {"review_text": review_text, "title": title}
    try:
        r = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
    except requests.exceptions.ConnectionError as e:
        st.error(f"Could not reach {API_URL}/predict --is " 
                 "`uvicorn serving.api:app --port 8000` running ?")
        st.stop()

    if r.status_code == 200:
        result = r.json()

        if result["sentiment"] == "positive":
            st.success(f"Sentiment is positive, positive probability {result['positive_probability']:.1%}")

        else:
            st.error(f"Sentiment is negative, negative probability {1 - result['positive_probability']:.1%}")

        col1, col2, col3 = st.columns(3)
        col1.metric("Confidence", f"{result['confidence']:.1%}")
        col2.metric("Unseen words", f"{result['oov_rate']:.1%}",
                    help="Share of words the model has never seen"
                    )
        col3.metric("latency", f"{result['latency_ms']:.1%} ms")

        #Surface the monitoring signal to the user
        if result["oov_rate"] > 0.35:
            st.warning("Over a third of this review is vocabulary the model"
                       "has never seen. This is exactly the drift signal "
                       "tracked in monitoring/analyze_drift.py"
                       )

        if result["confidence"] < 0.2:
            st.info("Confidence is low")

        st.caption(f"Served by model version: {result['model_version']}")

    else:
        # The API rejected the input
        st.warning(f"API rejected the request HTTP {r.status_code}")
        #st.json(r.json())















FROM python:3.8
RUN pip install pandas nltk scikit-learn streamlit
COPY src/app.py /app/
COPY model/sentiment_review.pkl /app/model/sentiment_review.pkl
WORKDIR /app
ENTRYPOINT [ "streamlit", "run", "app.py" ]
# data-science-portfolio
Welcome to my Data Science Portfolio, a curated collection of AI and MLOps projects demonstrating expertise in machine learning, natural language processing, computer vision, time-series forecasting, generative AI, and scalable model deployment. Each project showcases practical applications, robust technical skills, and end-to-end solution development.

## Projects

1. [Movie Recommendation System](movie-recommendation/)  
   - Developed a recommendation system using SVD and KNNBasic on MovieLens 100k/1M datasets, achieving RMSE ~0.93.  
   - Features a Streamlit interface for user interaction and genre analysis for personalized recommendations.  
   - **Technologies**: Python, scikit-surprise, pandas, Streamlit.

2. [Sentiment Analysis](sentiment-analysis/)  
   - Built a sentiment classifier for X platform posts using TextBlob and Hugging Face models, integrated with the X API.  
   - Includes text preprocessing and keyword visualization to derive actionable insights.  
   - **Technologies**: Python, TextBlob, transformers, pandas.

3. [Image Classification](image-classification/)  
   - Designed a CNN-based classifier for the CIFAR-10 dataset (69.27% accuracy) and enhanced it with MobileNetV2 transfer learning (~85% accuracy).  
   - Supports custom image classification with visualization (confusion matrix).  
   - **Technologies**: Python, TensorFlow, Keras, matplotlib.

4. [Stock Price Prediction](stock-price-prediction/)  
   - Created an LSTM-based model to forecast stock prices using Yahoo Finance data, achieving MAE ~2.5% of stock value.  
   - Features interactive visualizations for trend analysis and forecasting.  
   - **Technologies**: Python, TensorFlow, pandas, yfinance.

5. [Text Summarization](text-summarization/)  
   - Implemented an abstractive summarization model using Hugging Face’s BART, reducing article lengths by ~60% with ROUGE-1 ~0.66.  
   - Integrated with a Streamlit app for custom text summarization.  
   - **Technologies**: Python, transformers, Streamlit, rouge-score.

6. [Object Detection](object-detection/)  
   - Built an object detection system using YOLOv8 on the COCO dataset, achieving mAP@50 ~0.65.  
   - Enables real-time inference on custom images and videos with visualized bounding boxes.  
   - **Technologies**: Python, ultralytics, PyTorch, OpenCV.

7. [ML Model Deployment](ml-kubernetes/)  
   - Deployed a sentiment analysis model (from Project 2) as a scalable REST API using Docker and Kubernetes on Google Cloud Platform.  
   - Achieved 99.9% uptime and handled ~100 requests/minute in load tests.  
   - **Technologies**: Python, Flask, transformers, Docker, Kubernetes, Google Cloud Platform.

## Technologies Used
- **Languages**: Python
- **ML/AI Frameworks**: TensorFlow, Keras, PyTorch, transformers, scikit-surprise, ultralytics
- **Data Processing**: pandas, yfinance, rouge-score
- **Visualization**: matplotlib, Streamlit
- **MLOps/Deployment**: Docker, Kubernetes, Google Cloud Platform, Flask
- **Other**: OpenCV, locust

## Getting Started
1. Clone the repository: `git clone https://github.com/[your-username]/data-science-portfolio.git`.
2. Navigate to each project folder and follow the `README.md` for setup and execution instructions.
3. Most projects can be run in Google Colab or locally with Python 3.11+ and required dependencies.

## Contact
[Your Name]  
- **GitHub**: [your-github-username]  
- **LinkedIn**: [your-linkedin-profile]  
- **Email**: [your-email]

Feel free to reach out for collaboration or inquiries!

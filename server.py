# Import necessary libraries for the Flask app and functionalities
from flask import Flask, request, render_template
from predict import predict_sentiments  # Function to predict sentiments for comments
from youtube import get_video_comments  # Function to get video comments from YouTube
from flask_cors import CORS  # For handling Cross-Origin Resource Sharing (CORS)
import requests  # For making HTTP requests (not currently used in this code, but imported)

# Initialize the Flask app and enable CORS support
app = Flask(__name__)
CORS(app)

# Function to get video details, comments, and sentiment predictions
def get_video(video_id):
    # Check if the video_id is provided
    if not video_id:
        return {"error": "video_id is required"}  # Return error if video_id is missing

    # Get the video comments using the provided video_id
    comments = get_video_comments(video_id)
    
    # Get sentiment predictions for the comments
    predictions = predict_sentiments(comments)

    # Calculate the number of positive and negative sentiments
    positive = predictions.count("Positive")
    negative = predictions.count("Negative")

    # Prepare a summary dictionary with the sentiment counts, total comments, and rating
    summary = {
        "positive": positive,
        "negative": negative,
        "num_comments": len(comments),  # Total number of comments
        "rating": (positive / len(comments)) * 100  # Percentage of positive sentiments
    }

    # Return the predictions, comments, and summary
    return {"predictions": predictions, "comments": comments, "summary": summary}

# Define the route for the home page ('/') to display the analysis form and results
@app.route('/', methods=['GET', 'POST'])
def index():
    summary = None  # Initialize summary as None initially
    comments = []  # Initialize an empty list for comments

    # Check if the form was submitted via POST request
    if request.method == 'POST':
        # Get the video URL from the form input
        video_url = request.form.get('video_url')

        # Extract the video_id from the YouTube URL
        video_id = video_url.split("v=")[1]
        
        # Call the get_video function to fetch predictions and summary
        data = get_video(video_id)

        # Set the summary and comments for the template
        summary = data['summary']
        comments = list(zip(data['comments'], data['predictions']))  # Combine comments with their sentiments

    # Render the index.html template and pass the summary and comments to it
    return render_template('index.html', summary=summary, comments=comments)

# Run the Flask app in debug mode
if __name__ == '__main__':
    app.run(debug=True)

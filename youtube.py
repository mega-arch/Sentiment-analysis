# Import necessary libraries
import os
import googleapiclient.discovery
import googleapiclient.errors
from dotenv import load_dotenv  # To load environment variables from a .env file

# Load environment variables (e.g., API_KEY) from a .env file
load_dotenv()

# Get the YouTube API key from environment variables
api_key = os.getenv("API_KEY")

def get_comments(youtube, **kwargs):
    """
    This function retrieves comments from a YouTube video using the YouTube API.

    Arguments:
    youtube -- the YouTube API client
    kwargs -- keyword arguments that are passed to the API request (e.g., videoId, part, etc.)

    Returns:
    A list of comments from the video.
    """
    comments = []  # List to store comments
    results = youtube.commentThreads().list(**kwargs).execute()  # API request to get comments

    while results:  # Loop through the pages of comments
        for item in results['items']:  # Iterate through each comment
            # Extract the text of the top-level comment
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            comments.append(comment)  # Append the comment to the list

        # Check if there are more comments (i.e., next page of results)
        if 'nextPageToken' in results:
            kwargs['pageToken'] = results['nextPageToken']  # Add the next page token to the request
            results = youtube.commentThreads().list(**kwargs).execute()  # Fetch the next page of comments
        else:
            break  # No more comments to fetch, exit the loop

    return comments  # Return the list of comments

def main(video_id, api_key):
    """
    This function sets up the YouTube API client and calls the get_comments function to retrieve comments.

    Arguments:
    video_id -- The YouTube video ID to retrieve comments from
    api_key -- The YouTube API key for authentication

    Returns:
    A list of comments for the given video.
    """
    # Disable OAuthlib's HTTPS verification when running locally (useful for local development)
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

    # Build the YouTube API client with the provided API key
    youtube = googleapiclient.discovery.build(
        "youtube", "v3", developerKey=api_key)

    # Call get_comments to fetch the comments from the video
    comments = get_comments(youtube, part="snippet", videoId=video_id, textFormat="plainText")

    return comments  # Return the fetched comments

def get_video_comments(video_id):
    """
    This function acts as an entry point to retrieve video comments by calling the main function.

    Arguments:
    video_id -- The YouTube video ID to retrieve comments from

    Returns:
    A list of comments for the given video.
    """
    return main(video_id, api_key)  # Call the main function to get the comments

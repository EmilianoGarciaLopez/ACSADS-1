import csv

import praw
from decouple import config

# Fetch comments of a user from the Reddit API
reddit = praw.Reddit(
    user_agent=config("USER_AGENT"),
    client_id=config("CLIENT_ID"),
    client_secret=config("CLIENT_SECRET"),
    username=config("USERNAME"),
    password=config("PASSWORD"),
)


# Fetch comments of a user from the Reddit API
def get_user_comments(user: str, num_comments: int) -> csv:
    with open(f"Original_Data/{user}_comments.csv", "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            ["comments"])
        for comment in reddit.redditor(user).comments.new(limit=num_comments):
            try:
                parent_comment = comment.parent().body
            except AttributeError:
                parent_comment = ""
            writer.writerow(
                [comment.body])


# Fetch top posts from a subreddit from the Reddit API
def get_top_posts(sub, num_posts):
    with open(f"Original_Data/{sub}_top_posts.csv", "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["title", "user", "score"])
        for submission in reddit.subreddit(sub).top(limit=num_posts):
            writer.writerow(
                [submission.title, submission.author, submission.score])


def get_top_commenter_comments(sub, num_users, num_comments_per_user):
    with open(f"Original_Data/{sub}_top_user_comments.csv", "w") as csvfile:
        writer = csv.writer(csvfile)
        for submission in reddit.subreddit(sub).top(limit=num_users):
            try:
                for comment in reddit.redditor(str(submission.author)).comments.top(limit=num_comments_per_user):
                    writer.writerow([comment.body])
            except:
                pass


if __name__ == "__main__":
    get_top_commenter_comments("democrats", 170, 20)

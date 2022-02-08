import re

import pandas as pd

# regex that removes everything but letters and spaces
pattern = re.compile(r"]\([^)]*\)")
pattern2 = re.compile(r'http\S+')
pattern3 = re.compile(r'/[^\w\s]/gi')
pattern4 = re.compile(r'/\r?\n|\r/g')


def apply_patterns(csv_file):
    # if line in csv file contains pattern then remove it
    with open(csv_file, "r") as f:
        lines = f.readlines()
        with open(csv_file, "w") as f2:
            for line in lines:
                if pattern.search(line):
                    line = pattern.sub("", line)
                if pattern2.search(line):
                    line = pattern2.sub("", line)
                if pattern3.search(line):
                    line = pattern3.sub("", line)
                if pattern4.search(line):
                    line = pattern4.sub("", line)
                f2.write(line)


def delete_empty_lines(csv_file):
    with open(csv_file, "r") as f:
        lines = f.readlines()
        with open(csv_file, "w") as f2:
            for line in lines:
                if line.strip():
                    f2.write(line)


def delete_short_lines(csv_file):
    with open(csv_file, "r") as f:
        lines = f.readlines()
        with open(csv_file, "w") as f2:
            for line in lines:
                if len(line) > 5:
                    f2.write(line)


# creates a dataframe from csv file and removes all the non-alphanumeric characters
def create_dataframe_and_filter(csv_file):
    df = pd.read_csv(csv_file, sep=",", header=None)
    df.columns = ["comment"]
    # remove all non-alphanumeric characters
    df.comment = df.comment.str.replace(r'[^\w\s]', '')
    # remove all multiple spaces
    df.comment = df.comment.str.replace(r'\s+', ' ')
    # remove all urls
    df.comment = df.comment.str.replace(r'http\S+', '')
    # remove all newlines
    df.comment = df.comment.str.replace(r'\r?\n|\r', '')
    # removes all lines that have fewer than 5 characters
    df = df[df["comment"].str.len() > 30]
    return df


def write_dataframe(df, output_csv_file):
    df.to_csv(output_csv_file, index=False)


if __name__ == "__main__":
    write_dataframe(create_dataframe_and_filter("./Original_Data/democrats_top_user_comments.csv"),
                    "./Clean_Data/democrats.csv")
    write_dataframe(create_dataframe_and_filter("./Original_Data/conservative_top_user_comments.csv"),
                    "./Clean_Data/conservatives.csv")

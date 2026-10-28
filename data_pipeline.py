
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

VALID_MBTI = {
    "INTJ", "INTP", "ENTJ", "ENTP",
    "INFJ", "INFP", "ENFJ", "ENFP",
    "ISTJ", "ISFJ", "ESTJ", "ESFJ",
    "ISTP", "ISFP", "ESTP", "ESFP"
}


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    words = text.split()
    words = [
        LEMMATIZER.lemmatize(word)
        for word in words
        if word not in STOP_WORDS
    ]
    return " ".join(words)


def run_pipeline():
    users = pd.read_csv("users_clean.csv")

    required = [
        "user_id", "professional_summary", "about_me", "mbti"
    ]

    for column in required:
        if column not in users.columns:
            raise ValueError(f"Missing required column: {column}")

    if users["user_id"].duplicated().any():
        raise ValueError("Duplicate user IDs found.")

    if not users["mbti"].isin(VALID_MBTI).all():
        raise ValueError("Invalid MBTI type found.")

    for column in ["professional_summary", "about_me"]:
        if users[column].isna().any():
            raise ValueError(f"Missing text in {column}.")

    users["summary_clean"] = (
        users["professional_summary"].apply(clean_text)
    )
    users["about_clean"] = users["about_me"].apply(clean_text)

    users.to_csv("users_clean.csv", index=False)

    print("Data pipeline completed successfully!")
    print("Number of users:", len(users))
    print("Output saved to users_clean.csv")


if __name__ == "__main__":
    run_pipeline()

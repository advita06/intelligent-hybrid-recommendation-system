
import streamlit as st
import pandas as pd
import joblib
import os
import uuid
from datetime import datetime

from sklearn.metrics.pairwise import cosine_similarity

from src.preprocess import clean_text
from src.mbti import mbti_scaled_score
from src.scoring import (
    location_score,
    experience_score,
    interest_score,
    compatibility_score
)

# --------------------------------------------------
# 1. PAGE SETTINGS
# --------------------------------------------------

st.set_page_config(
    page_title="Profile Based Matching System",
    page_icon="🤝",
    layout="wide"
)

st.title("🤝 Profile Based Matching System")
st.write("Create your profile and discover compatible people.")

# --------------------------------------------------
# 2. MATCHING WEIGHTS
# --------------------------------------------------

weights = {
    "text": 0.40,
    "mbti": 0.30,
    "location": 0.15,
    "experience": 0.15,
    "interests": 0.00
}

# --------------------------------------------------
# 3. LOAD DATA AND VECTORIZERS
# --------------------------------------------------

@st.cache_resource
def load_vectorizers():
    summary_model = joblib.load("summary_vectorizer.pkl")
    about_model = joblib.load("about_vectorizer.pkl")
    return summary_model, about_model


try:
    users = pd.read_csv("users_clean.csv")
    feedback = pd.read_csv("feedback.csv")
    summary_vectorizer, about_vectorizer = load_vectorizers()

except FileNotFoundError as error:
    st.error(
        f"Required file not found: {error.filename}. "
        "Check that your CSV files and vectorizer files "
        "are in the project folder."
    )
    st.stop()

required_user_columns = {
    "user_id",
    "name",
    "location",
    "profession",
    "experience_years",
    "summary_clean",
    "about_clean",
    "mbti",
    "interests"
}

if not required_user_columns.issubset(users.columns):
    st.error("users_clean.csv is missing required columns.")
    st.stop()

required_feedback_columns = {
    "user_id",
    "matched_user_id",
    "action",
    "timestamp"
}

if not required_feedback_columns.issubset(feedback.columns):
    st.error("feedback.csv is missing required columns.")
    st.stop()

if users["user_id"].duplicated().any():
    st.error("Duplicate user IDs found in users_clean.csv.")
    st.stop()

if users.empty:
    st.error("No existing user profiles were found.")
    st.stop()

min_exp = users["experience_years"].min()
max_exp = users["experience_years"].max()

st.success("Data and vectorizers loaded successfully!")

col1, col2 = st.columns(2)
col1.metric("Existing Profiles", len(users))
col2.metric("Feedback Records", len(feedback))

# --------------------------------------------------
# 4. PROFILE FORM
# --------------------------------------------------

st.header("Create Your Profile")

with st.form("profile_form"):
    name = st.text_input("Name")

    age = st.number_input(
        "Age",
        min_value=20,
        max_value=45,
        value=25
    )

    city = st.text_input("City")
    profession = st.text_input("Profession")

    experience = st.number_input(
        "Years of Experience",
        min_value=0,
        max_value=30,
        value=1
    )

    professional_summary = st.text_area(
        "Professional Summary",
        placeholder="Describe your skills, career goals and industry."
    )

    about_me = st.text_area(
        "About Me",
        placeholder="Describe your personality, work style and interests."
    )

    mbti = st.selectbox(
        "MBTI Personality Type",
        [
            "INTJ", "INTP", "ENTJ", "ENTP",
            "INFJ", "INFP", "ENFJ", "ENFP",
            "ISTJ", "ISFJ", "ESTJ", "ESFJ",
            "ISTP", "ISFP", "ESTP", "ESFP"
        ]
    )

    interests = st.text_input(
        "Interests",
        placeholder="Python, Music, Travel"
    )

    submit = st.form_submit_button("Find Matches")


# --------------------------------------------------
# 5. PROCESS PROFILE AND FIND MATCHES
# --------------------------------------------------

if submit:

    if not name.strip():
        st.error("Please enter your name.")

    elif not city.strip():
        st.error("Please enter your city.")

    elif not profession.strip():
        st.error("Please enter your profession.")

    elif not professional_summary.strip():
        st.error("Please enter your Professional Summary.")

    elif not about_me.strip():
        st.error("Please enter your About Me description.")

    elif not interests.strip():
        st.error("Please enter at least one interest.")

    else:
        # Generate an ID for this submitted profile.
        new_user_id = "NEW_" + uuid.uuid4().hex[:12].upper()

        # Clean the text.
        summary_clean = clean_text(professional_summary)
        about_clean = clean_text(about_me)

        if not summary_clean or not about_clean:
            st.error(
                "Your text became empty after cleaning. "
                "Please enter more descriptive text."
            )
            st.stop()

        # Convert the new profile text into TF-IDF vectors.
        summary_vector = summary_vectorizer.transform(
            [summary_clean]
        )

        about_vector = about_vectorizer.transform(
            [about_clean]
        )

        # Convert existing profiles into TF-IDF vectors.
        existing_summary_vectors = summary_vectorizer.transform(
            users["summary_clean"].fillna("").astype(str)
        )

        existing_about_vectors = about_vectorizer.transform(
            users["about_clean"].fillna("").astype(str)
        )

        # Compare the new profile against all existing profiles.
        summary_similarity = cosine_similarity(
            summary_vector,
            existing_summary_vectors
        )[0]

        about_similarity = cosine_similarity(
            about_vector,
            existing_about_vectors
        )[0]

        text_similarity = (
            0.60 * summary_similarity
            + 0.40 * about_similarity
        )

        # Keep the new profile's details for scoring and feedback.
        new_user = {
            "user_id": new_user_id,
            "name": name.strip(),
            "age": age,
            "location": city.strip(),
            "profession": profession.strip(),
            "experience_years": experience,
            "summary_clean": summary_clean,
            "about_clean": about_clean,
            "mbti": mbti,
            "interests": interests
        }

        st.session_state["new_user"] = new_user

        # --------------------------------------------------
        # 6. TOP 5 TEXT MATCHES
        # --------------------------------------------------

        text_results = users.copy()
        text_results["text_similarity"] = text_similarity

        top_text_matches = (
            text_results
            .sort_values("text_similarity", ascending=False)
            .head(5)
        )

        st.header("Top 5 Text Matches")

        for _, match in top_text_matches.iterrows():
            st.subheader(match["name"])
            st.write("Profession:", match["profession"])
            st.write("City:", match["location"])
            st.write(
                f"Text Similarity: "
                f"{match['text_similarity']:.2f}"
            )
            st.divider()

        # --------------------------------------------------
        # 7. OVERALL COMPATIBILITY SCORE
        # --------------------------------------------------

        results = []

        for row_index, existing_user in users.iterrows():

            text_score = text_similarity[row_index]

            personality_score = mbti_scaled_score(
                new_user["mbti"],
                existing_user["mbti"]
            )

            city_score = location_score(
                new_user,
                existing_user
            )

            exp_score = experience_score(
                new_user,
                existing_user,
                min_exp,
                max_exp
            )

            interests_score = interest_score(
                new_user,
                existing_user
            )

            result = compatibility_score(
                text_score,
                personality_score,
                city_score,
                exp_score,
                interests_score,
                weights
            )

            results.append({
                "user_id": existing_user["user_id"],
                "name": existing_user["name"],
                "profession": existing_user["profession"],
                "location": existing_user["location"],
                "score": result["score"],
                "text": result["text"],
                "mbti": result["mbti"],
                "location_score": result["location"],
                "experience": result["experience"],
                "interests": result["interests"]
            })

        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values(
            "score",
            ascending=False
        ).head(5).reset_index(drop=True)

        # Save displayed recommendations in session state.
        st.session_state["match_results"] = results_df

        # --------------------------------------------------
        # 8. DISPLAY TOP 5 OVERALL MATCHES
        # --------------------------------------------------

        st.header("🏆 Top 5 Overall Compatibility Matches")

        for _, match in results_df.iterrows():

            st.subheader(match["name"])
            st.write("Profession:", match["profession"])
            st.write("City:", match["location"])

            st.write(
                f"Compatibility: {match['score']:.2f}/100"
            )

            st.progress(
                min(1.0, max(0.0, match["score"] / 100))
            )

            with st.expander("View score breakdown"):
                st.write(
                    f"Text contribution: {match['text'] * 100:.2f}"
                )
                st.write(
                    f"MBTI contribution: {match['mbti'] * 100:.2f}"
                )
                st.write(
                    f"Location contribution: "
                    f"{match['location_score'] * 100:.2f}"
                )
                st.write(
                    f"Experience contribution: "
                    f"{match['experience'] * 100:.2f}"
                )
                st.write(
                    f"Interests contribution: "
                    f"{match['interests'] * 100:.2f}"
                )

            st.divider()


# --------------------------------------------------
# 9. ACCEPT OR REJECT RECOMMENDATIONS
# --------------------------------------------------

if (
    "new_user" in st.session_state
    and "match_results" in st.session_state
):
    st.header("Give Feedback on Your Matches")

    st.write(
        "Your feedback will be saved in feedback.csv."
    )

    current_user = st.session_state["new_user"]
    current_results = st.session_state["match_results"]

    for _, match in current_results.iterrows():

        st.write(
            f"**{match['name']}** — "
            f"{match['score']:.2f}/100"
        )

        col1, col2 = st.columns(2)

        with col1:
            if st.button(
                "✅ Accept",
                key=f"accept_{current_user['user_id']}_{match['user_id']}"
            ):
                st.session_state[
                    f"feedback_choice_{current_user['user_id']}_{match['user_id']}"
                ] = 1

        with col2:
            if st.button(
                "❌ Reject",
                key=f"reject_{current_user['user_id']}_{match['user_id']}"
            ):
                st.session_state[
                    f"feedback_choice_{current_user['user_id']}_{match['user_id']}"
                ] = 0

        choice_key = (
            f"feedback_choice_{current_user['user_id']}_{match['user_id']}"
        )

        if choice_key in st.session_state:

            action = st.session_state[choice_key]

            # Track which actions have already been saved.
            saved_key = (
                f"feedback_saved_{current_user['user_id']}_{match['user_id']}"
            )

            if saved_key not in st.session_state:

                new_feedback = pd.DataFrame([{
                    "user_id": current_user["user_id"],
                    "matched_user_id": match["user_id"],
                    "action": action,
                    "timestamp": datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                }])

                try:
                    new_feedback.to_csv(
                        "feedback.csv",
                        mode="a",
                        header=False,
                        index=False
                    )

                    st.session_state[saved_key] = True

                    if action == 1:
                        st.success(
                            f"You accepted {match['name']}!"
                        )
                    else:
                        st.info(
                            f"You rejected {match['name']}!"
                        )

                except OSError as error:
                    st.error(
                        f"Could not save feedback: {error}"
                    )

            else:
                st.caption(
                    "Your feedback for this match has already been saved."
                )

        st.divider()

# --------------------------------------------------
# 10. FOOTER
# --------------------------------------------------

st.caption(
    "Profile Based Matching System | "
    "Recommendations are based on text similarity "
    "and the configured compatibility weights."
)
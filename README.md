# Intelligent Hybrid Recommendation System

A profile-based recommendation system that matches users using a hybrid of **NLP text similarity, MBTI compatibility rules, location, professional experience, and shared interests**. It produces a compatibility score from **0 to 100** and updates recommendation weights using accept/reject feedback.

## Project Overview

The system has three main layers:

1. **NLP layer** — compares users' `professional_summary` and `about_me` text.
2. **Scoring layer** — combines text similarity with MBTI, location, experience, and interests into a compatibility score.
3. **Learning layer** — learns feature weights from users' accept/reject feedback.

## Dataset

The project uses synthetic, Python-generated profile and feedback datasets.

| File | Size | Description |
|---|---:|---|
| `users.csv` | 90 users | User IDs, names, age, location, profession, experience, profile text, MBTI type, and interests |
| `feedback.csv` | 673 rows | User-to-user matches, accept/reject action (`1` = accept, `0` = reject), and timestamp |

Data validation checks passed for unique user IDs, missing values, duplicate rows, valid MBTI types, feedback labels, valid user references, and self-matches. Each user has 5–10 feedback interactions. The overall acceptance rate is **40.6%**.

## How It Works

### 1. Data pipeline

`data_pipeline.py` loads and validates the datasets, cleans profile text, and saves `data/users_clean.csv`.

Text preprocessing includes:
- Lowercasing
- Removing punctuation and numbers
- Removing stopwords
- Lemmatization with NLTK

### 2. NLP text similarity

TF-IDF vectorizers use unigrams and bigrams with sublinear term frequency. Cosine similarity is calculated separately for professional summaries and `about_me` text.

The combined text similarity is:

```text
TextSim = 0.60 × summary similarity + 0.40 × about-me similarity
```

Raw text similarity is normalized using its 95th percentile and clipped to the range 0–1.

### 3. MBTI compatibility rules

A symmetric, rule-based MBTI table assigns compatibility scores using letter-based rules. Eight designated “golden pairs” receive a score of 100. MBTI compatibility is treated as a heuristic rather than a scientific measure.

### 4. Additional matching features

- **Location:** 1 when both users are in the same city; otherwise 0.
- **Experience closeness:** decreases as the difference in years of experience increases.
- **Shared interests:** based on the number of interests shared, capped at 1.

The final compatibility score is:

```text
Score = 100 × (
    w1 × TextSim +
    w2 × MBTI +
    w3 × Location +
    w4 × Experience +
    w5 × Interests
)
```

The initial manually selected weights were `0.40 / 0.30 / 0.15 / 0.15 / 0.00` for text, MBTI, location, experience, and interests respectively. Weights learned from feedback can adapt the relative importance of these factors.

### 5. Feedback learning

A time-based split uses the first 75% of interactions (504 rows) for training and the last 25% (169 rows) for testing. Logistic regression learns feature weights from accept/reject labels. Negative coefficients are clipped to zero and the remaining weights are normalized.

The project also implements gradient descent from scratch in NumPy and checks the results against scikit-learn. An online-learning experiment updates weights as feedback arrives.

## Results

Evaluation on the held-out test set (169 interactions):

| Method | ROC-AUC | Acceptance rate in top 30% | Lift |
|---|---:|---:|---:|
| Random | 0.466 | 45.1% | 1.12 |
| Equal weights (5 features) | 0.791 | 68.6% | 1.71 |
| Starting hand-chosen weights | 0.712 | 66.7% | 1.66 |
| Learned weights (3 features) | 0.700 | 64.7% | 1.61 |
| **Learned weights (5 features)** | **0.802** | **74.5%** | **1.85** |

The test-set baseline acceptance rate was **40.2%**.

The best five-feature model learned these normalized weights:

| Feature | Learned weight |
|---|---:|
| Shared interests | 0.356 |
| Experience closeness | 0.273 |
| MBTI compatibility | 0.237 |
| Text similarity | 0.072 |
| Location | 0.062 |

Compared with the starting weights, the learned five-feature model improved ROC-AUC from **0.712 to 0.802** and the acceptance rate among the top 30% of recommendations from **66.7% to 74.5%**.

The learning curve levelled off around ROC-AUC 0.80 after roughly 300 training rows. In a time-ordered feedback experiment, cumulative accuracy was 40.2% for the frozen model and 57.4% for the online model; this result needs caution because the frozen model's missing bias term affects the comparison.

## Streamlit Application

The Streamlit app (`app.py`) allows a user to enter a profile and view the **top five matches**, each with a compatibility score and feature contributions. Accept and Reject buttons feed back into the learning process.

New profile text is transformed using the already-fitted TF-IDF vectorizers so it remains on the same feature scale as existing profiles.

## Project Files

| File | Purpose |
|---|---|
| `users.csv`, `feedback.csv` | Synthetic profile and feedback data |
| `data_pipeline.py` | Loading, validation, and text cleaning |
| `major1.ipynb` | Matching logic, learning, evaluation, and visualizations |
| `summary_vectorizer.pkl`, `about_vectorizer.pkl` | Saved TF-IDF vectorizers |
| `app.py` | Streamlit interface |
| `REPORT` | Project performance analysis report |

## Limitations

- **Synthetic data:** profiles and feedback are generated from templates and simulated preferences, so real-world results may differ.
- **Limited dataset:** 90 users and 673 feedback interactions provide little data for personalizing each user's preferences.
- **TF-IDF limitations:** TF-IDF matches word usage rather than meaning; sentence embeddings could improve semantic comparison.
- **MBTI is heuristic:** the rule table has limited scientific support and should not be treated as a validated measure of compatibility.
- **Personalization results:** personalized weights did not improve test ROC-AUC on this dataset, likely due to sparse per-user feedback.
- **Evaluation caveat:** the online-versus-frozen accuracy comparison is partly affected by the frozen model's missing bias term.

## Future Improvements

- Use sentence embeddings to capture semantic similarity.
- Collect feedback from real users and evaluate on larger, more representative data.
- Gather more interactions per user before learning personalized weights.
- Improve the online-learning evaluation with a consistent decision threshold and bias term.

## Tech Stack

- Python
- pandas and NumPy
- NLTK
- scikit-learn (TF-IDF, cosine similarity, logistic regression, evaluation)
- Streamlit
- Matplotlib/Seaborn for visualizations, if used in the notebook

## Running the App

The report identifies `app.py` as the Streamlit interface. If the project dependencies are installed and the expected data and vectorizer files are in place, run:

```bash
streamlit run app.py
```

Ensure the file paths used by the application match your local project structure.

---

**Note:** This project uses synthetic data. Compatibility scores are model outputs based on the defined rules and feedback, not guarantees of real-world compatibility.

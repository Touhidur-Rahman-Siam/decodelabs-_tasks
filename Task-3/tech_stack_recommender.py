import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "raw_skills.csv")
MIN_SKILLS_REQUIRED = 3
TOP_N = 3


def _tokenize_skills(raw_skill_string: str) -> list[str]:
    return [
        skill.strip().lower().replace(" ", "_").replace("/", "_")
        for skill in raw_skill_string.split(",")
        if skill.strip()
    ]


def load_job_roles(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def get_user_skills_interactive() -> list[str]:
    print("Tech Stack Recommender — tell me your skills or interests.")
    print(f"Enter at least {MIN_SKILLS_REQUIRED} skills, separated by commas.")
    print("Example: Python, Cloud Computing, Automation\n")

    while True:
        raw = input("Your skills: ").strip()
        skills = _tokenize_skills(raw)
        if len(skills) >= MIN_SKILLS_REQUIRED:
            return skills
        print(f"Need at least {MIN_SKILLS_REQUIRED} skills — you gave {len(skills)}. Try again.\n")


def build_vectorizer_and_matrix(job_roles_df: pd.DataFrame):
    corpus = job_roles_df["skills"].tolist()

    vectorizer = TfidfVectorizer(
        tokenizer=_tokenize_skills,
        lowercase=False,
        token_pattern=None,
    )
    role_matrix = vectorizer.fit_transform(corpus)
    return vectorizer, role_matrix


def score_roles(vectorizer: TfidfVectorizer, role_matrix, user_skills: list[str]):
    user_document = ", ".join(user_skills)
    user_vector = vectorizer.transform([user_document])

    similarity_scores = cosine_similarity(user_vector, role_matrix).flatten()
    return similarity_scores


def rank_and_filter(job_roles_df: pd.DataFrame, similarity_scores, top_n: int = TOP_N) -> pd.DataFrame:
    results = job_roles_df.copy()
    results["match_score"] = similarity_scores
    results = results.sort_values("match_score", ascending=False)
    return results.head(top_n).reset_index(drop=True)


def display_recommendations(ranked_df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print(f"TOP {len(ranked_df)} RECOMMENDED CAREER PATHS")
    print("=" * 60)

    if ranked_df["match_score"].max() == 0:
        print("No overlapping skills found with any known role.")
        print("Try broader or more common tech skills (e.g. Python, SQL).")
        return

    for rank, row in enumerate(ranked_df.itertuples(index=False), start=1):
        pct = row.match_score * 100
        print(f"\n{rank}. {row.job_role}  —  {pct:.1f}% match")
        print(f"   Core skills: {row.skills}")


def run_recommender() -> None:
    job_roles_df = load_job_roles()
    print(f"Loaded {len(job_roles_df)} job roles from raw_skills.csv\n")
    user_skills = get_user_skills_interactive()

    vectorizer, role_matrix = build_vectorizer_and_matrix(job_roles_df)
    similarity_scores = score_roles(vectorizer, role_matrix, user_skills)
    top_matches = rank_and_filter(job_roles_df, similarity_scores)

    display_recommendations(top_matches)


if __name__ == "__main__":
    run_recommender()

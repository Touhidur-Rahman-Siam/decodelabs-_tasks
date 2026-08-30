# Project 3: AI Recommendation Logic — Tech Stack Recommender
**DecodeLabs Industrial Training Kit — Batch 2026**

## What this is
A content-based recommendation engine that takes a user's raw skills
and recommends the Top 3 best-matching tech career paths, using
TF-IDF vectorization and Cosine Similarity — the same mathematical
approach behind real-world engines like Netflix and Amazon's
"more like this" features.

## Files
- `tech_stack_recommender.py` — the complete, runnable engine
- `data/raw_skills.csv` — the job-role dataset (15 roles, each with
  8 associated skill tags) used as the "items" in the engine

## How to run
1. Make sure Python 3.9+ is installed
2. Install dependencies:
   ```
   pip install scikit-learn pandas
   ```
3. Run:
   ```
   python3 tech_stack_recommender.py
   ```
4. When prompted, enter at least 3 skills or interests, comma-separated
   (e.g. `Python, Cloud Computing, Automation`)
5. The script prints your Top 3 recommended career paths, ranked by
   percentage match.

## Architecture — the 4-step ranking pipeline
1. **Ingestion** — `get_user_skills_interactive()` captures the user's
   raw skills (minimum 3 enforced, matching the brief's requirement
   for sufficient data density)
2. **Scoring** — `build_vectorizer_and_matrix()` and `score_roles()`
   map both the job-role dataset and the user's skills into a shared
   TF-IDF vector space, then compute Cosine Similarity between the
   user vector and every role vector
3. **Sorting** — `rank_and_filter()` sorts all 15 roles by similarity
   score, descending
4. **Filtering** — the same function truncates the list to the Top 3,
   preventing choice overload

## Why TF-IDF instead of simple keyword overlap?
A naive approach (counting shared tags) treats a common skill like
"Python" the same as a rare, highly specific skill like "Solidity."
TF-IDF down-weights generic terms that appear across many roles and
up-weights rare, discriminating ones — so a match on a niche skill
counts for more than a match on a common one.

## Why Cosine Similarity instead of Euclidean distance?
Euclidean distance is sensitive to vector magnitude/length, which
would unfairly penalize roles with longer skill lists. Cosine
similarity measures only the angle between vectors — how well two
skill profiles are *oriented*, regardless of how many skills either
one lists. Score interpretation: 1.0 = perfectly aligned, 0.0 = no
shared skills.

## Sample runs (verified)
| Input skills | Top match | Score |
|---|---|---|
| Python, Cloud Computing, Automation | QA Engineer | 29.1% |
| Solidity, Blockchain, Cryptography, Python | Blockchain Developer | 68.6% |
| Figma, UX Research, Wireframing | UI/UX Designer | 63.9% |
| Python, SQL, Machine Learning, Pandas | Data Scientist | 64.7% |

All four runs correctly identified the most relevant career path,
including for a niche skill set (Blockchain) where TF-IDF weighting
mattered most.

## Requirements met (per Project 3 brief)
- **Take user input** — interactive prompt, minimum 3 skills enforced
- **Match preferences using logic/similarity** — TF-IDF + Cosine
  Similarity (content-based filtering, avoiding the Cold Start
  problem collaborative filtering would have)
- **Display recommended items** — Top 3 ranked list with match
  percentages and each role's core skills
- **IPO model** — `run_recommender()` follows Input
  (`load_job_roles()`, `get_user_skills_interactive()`) → Process
  (`build_vectorizer_and_matrix()`, `score_roles()`,
  `rank_and_filter()`) → Output (`display_recommendations()`),
  matching the deck's 4-step pipeline (Ingestion → Scoring → Sorting
  → Filtering)
- **Vocabulary consistency** — a custom tokenizer ensures multi-word
  skills ("Machine Learning") are treated as a single atomic token on
  both the item side and the user side, so the same vocabulary space
  applies to both (the brief specifically calls out this failure mode)

## Known limitation (by design, per brief)
This is Content-Based Filtering only — it works well for known job
roles with defined skill sets, but a "User Cold Start" (a first-time
user with zero skills) or "Item Cold Start" (a brand-new role with
no tags yet) would return a zero-vector and no meaningful matches.
The brief covers this explicitly; extending this to onboarding
surveys or trending-role fallbacks is a natural stretch goal, not
part of this deliverable.

## Author
Submitted as part of DecodeLabs Project 3 — Industrial Training Kit,
Batch 2026.

import json
import os
from git import Repo
from dotenv import load_dotenv

load_dotenv()

REPO_URL = os.getenv("REPO_URL")
REPO_DIR = os.getenv("REPO_DIR")
OUTPUT_FILE = os.getenv("OUTPUT_FILE")

if not os.path.exists(REPO_DIR):
    print(f"Cloning {REPO_URL}...")
    repo = Repo.clone_from(REPO_URL, REPO_DIR, filter="blob:none")
else:
    print(f"Opening existing repository at {REPO_DIR}...")
    repo = Repo(REPO_DIR)

print("Calculating total commit count...")
total_commits = sum(1 for _ in repo.iter_commits('main'))
print(f"Total commits found: {total_commits}")

print("Extracting commits from 'main' branch...")

commit_data_list = []
count = 0

for commit in repo.iter_commits('main'):
    count += 1

    changed_files = []
    if commit.parents:
        diffs = commit.parents[0].diff(commit)
        changed_files = [d.b_path for d in diffs if d.b_path]

    commit_doc = {
        "id": commit.hexsha,
        "author": commit.author.name,
        "date": commit.authored_datetime.isoformat(),
        "message": commit.message.strip(),
        "summary": commit.summary,
        "changed_files": changed_files
    }

    commit_data_list.append(commit_doc)

    if count % 1000 == 0 or count == total_commits:
        percentage = (count / total_commits) * 100
        print(f"Processed {count}/{total_commits} commits ({percentage:.2f}%)")

print(f"Saving {len(commit_data_list)} commits to {OUTPUT_FILE}...")
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(commit_data_list, f, indent=4, ensure_ascii=False)

print("Extraction completed.")

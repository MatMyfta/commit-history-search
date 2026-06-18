import json
import os
import re
from git import Repo

class GitDataExtractor:
    def __init__(self, repo_url: str, repo_dir: str, output_file: str):
        self.repo_url = repo_url
        self.repo_dir = repo_dir
        self.output_file = output_file

    def _is_useless(self, summary: str, changed_files: list) -> bool:
        if re.match(r'^Merge ', summary, re.IGNORECASE):
            return True
        if re.search(r'\b(bump|release|v\d+\.\d+)\b', summary, re.IGNORECASE):
            return True

        if re.match(r'^(closes|fixes|resolves)\s+(gh-|-|#)?\d+$', summary.strip(), re.IGNORECASE):
            return True

        has_source_changes = any(f.endswith('.java') for f in changed_files)
        if not has_source_changes:
            return True

        return False

    def extract(self) -> None:
        if not os.path.exists(self.repo_dir):
            print(f"Cloning target repository from {self.repo_url}...")
            repo = Repo.clone_from(self.repo_url, self.repo_dir, filter="blob:none")
        else:
            print(f"Opening local repository at {self.repo_dir}...")
            repo = Repo(self.repo_dir)

        target_branch = os.getenv("TARGET_BRANCH", "").strip()

        try:
            max_commits = int(os.getenv("MAX_COMMITS", "-1"))
        except ValueError:
            max_commits = -1

        if target_branch:
            print(f"Using branch configured in environment: {target_branch}")
        else:
            print("TARGET_BRANCH not specified in .env. Attempting branch auto-detection...")
            try:
                active_remote_head = repo.remotes.origin.refs['HEAD'].commit.name
                detected = next(ref.name for ref in repo.remotes.origin.refs if ref.commit.name == active_remote_head and ref.name != 'origin/HEAD')
                target_branch = detected.replace('origin/', '')
            except Exception:
                available_refs = [ref.name.replace('origin/', '') for ref in repo.remotes.origin.refs]
                target_branch = 'main' if 'main' in available_refs else ('master' if 'master' in available_refs else 'main')

        print(f"Analyzing log streaming for branch: {target_branch}...")

        commit_data_list = []
        raw_count = 0
        filtered_count = 0

        for commit in repo.iter_commits(target_branch):
            raw_count += 1
            changed_files = []
            if commit.parents:
                diffs = commit.parents[0].diff(commit)
                changed_files = [d.b_path for d in diffs if d.b_path]

            raw_summary = commit.summary
            summary_str = raw_summary.decode('utf-8', errors='ignore') if isinstance(raw_summary, bytes) else str(raw_summary)

            if self._is_useless(summary_str, changed_files):
                filtered_count += 1
                continue

            raw_message = commit.message
            full_message = raw_message.decode('utf-8', errors='ignore').strip() if isinstance(raw_message, bytes) else str(raw_message).strip()

            message_lines = full_message.split('\n', 1)
            commit_body = message_lines[1].strip() if len(message_lines) > 1 else ""

            commit_doc = {
                "id": commit.hexsha,
                "author": commit.author.name,
                "date": commit.authored_datetime.isoformat(),
                "message": full_message,
                "summary": summary_str.strip(),
                "body": commit_body,
                "changed_files": changed_files
            }
            commit_data_list.append(commit_doc)

            if max_commits > 0 and len(commit_data_list) >= max_commits:
                print(f"Reached configured limit of {max_commits} usable documents. Halting stream mining.")
                break

        print("\nExtraction completed!")
        print(f"Total Streamed Commits Checked: {raw_count}")
        print(f"Purged/Filtered Noise Commits: {filtered_count}")
        print(f"Clean, Usable System Documents: {len(commit_data_list)}")

        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(commit_data_list, f, indent=4, ensure_ascii=False)

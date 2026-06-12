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

        try:
            active_remote_head = repo.remotes.origin.refs['HEAD'].commit.name
            default_branch = next(ref.name for ref in repo.remotes.origin.refs if ref.commit.name == active_remote_head and ref.name != 'origin/HEAD')
            default_branch = default_branch.replace('origin/', '')
        except Exception:
            available_refs = [ref.name.replace('origin/', '') for ref in repo.remotes.origin.refs]
            if 'main' in available_refs:
                default_branch = 'main'
            elif 'master' in available_refs:
                default_branch = 'master'
            else:
                default_branch = 'main'

        print(f"Analyzing log streaming for detected branch: {default_branch}...")

        commit_data_list = []
        raw_count = 0
        filtered_count = 0

        for commit in repo.iter_commits(default_branch):
            raw_count += 1
            changed_files = []
            if commit.parents:
                diffs = commit.parents[0].diff(commit)
                changed_files = [d.b_path for d in diffs if d.b_path]

            raw_summary = commit.summary
            if isinstance(raw_summary, bytes):
                summary_str = raw_summary.decode('utf-8', errors='ignore')
            else:
                summary_str = str(raw_summary)

            if self._is_useless(summary_str, changed_files):
                filtered_count += 1
                continue

            commit_doc = {
                "id": commit.hexsha,
                "author": commit.author.name,
                "date": commit.authored_datetime.isoformat(),
                "message": commit.message.strip(),
                "summary": summary_str,
                "changed_files": changed_files
            }
            commit_data_list.append(commit_doc)

        print(f"\nExtraction completed!")
        print(f"Total Streamed Commits: {raw_count}")
        print(f"Purged/Filtered Noise Commits: {filtered_count}")
        print(f"Clean, Usable System Documents: {len(commit_data_list)}")

        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(commit_data_list, f, indent=4, ensure_ascii=False)

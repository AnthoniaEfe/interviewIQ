"""Bipartite JD–resume groups (connected components) for leakage-safe CV."""

from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd


def norm_text(text: str) -> str:
    return " ".join(str(text).lower().split())


def text_hash(text: str) -> str:
    import hashlib

    return hashlib.sha256(norm_text(text).encode("utf-8")).hexdigest()


def connected_component_ids(resume_ids: list[str], job_ids: list[str]) -> np.ndarray:
    """Group id per pair; edges join pairs that share a resume hash or job hash."""
    n = len(resume_ids)
    by_r: dict[str, list[int]] = defaultdict(list)
    by_j: dict[str, list[int]] = defaultdict(list)
    for i, (r, j) in enumerate(zip(resume_ids, job_ids, strict=True)):
        by_r[r].append(i)
        by_j[j].append(i)
    adj: dict[int, set[int]] = defaultdict(set)
    for bucket in list(by_r.values()) + list(by_j.values()):
        for a in bucket:
            for b in bucket:
                if a != b:
                    adj[a].add(b)
    group = np.full(n, -1, dtype=np.int64)
    gid = 0
    for i in range(n):
        if group[i] >= 0:
            continue
        q = deque([i])
        group[i] = gid
        while q:
            u = q.popleft()
            for v in adj[u]:
                if group[v] < 0:
                    group[v] = gid
                    q.append(v)
        gid += 1
    return group


def add_hashes_and_groups(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["resume_hash"] = out["resume_text"].map(text_hash)
    out["job_hash"] = out["job_description_text"].map(text_hash)
    out["group_id"] = connected_component_ids(
        out["resume_hash"].tolist(), out["job_hash"].tolist()
    )
    return out


def leakage_report(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    tr = add_hashes_and_groups(train)
    te = add_hashes_and_groups(test)
    train_jobs = set(tr["job_hash"])
    train_res = set(tr["resume_hash"])
    job_multi_train = int((tr.groupby("job_hash").size() > 1).sum())
    res_multi_train = int((tr.groupby("resume_hash").size() > 1).sum())
    job_multi_all = int(
        (pd.concat([tr, te]).groupby("job_hash").size() > 1).sum()
    )
    res_multi_all = int(
        (pd.concat([tr, te]).groupby("resume_hash").size() > 1).sum()
    )
    share_job = te["job_hash"].isin(train_jobs)
    share_res = te["resume_hash"].isin(train_res)
    share_either = share_job | share_res
    disjoint = te.loc[~share_either]
    return {
        "train_rows": int(len(tr)),
        "test_rows": int(len(te)),
        "unique_jobs_train": int(tr["job_hash"].nunique()),
        "unique_resumes_train": int(tr["resume_hash"].nunique()),
        "jobs_in_more_than_one_pair_train": job_multi_train,
        "resumes_in_more_than_one_pair_train": res_multi_train,
        "jobs_in_more_than_one_pair_train_plus_test": job_multi_all,
        "resumes_in_more_than_one_pair_train_plus_test": res_multi_all,
        "test_pairs_sharing_jd_with_train": int(share_job.sum()),
        "test_pairs_sharing_resume_with_train": int(share_res.sum()),
        "test_pairs_sharing_jd_or_resume_with_train": int(share_either.sum()),
        "disjoint_test_n": int(len(disjoint)),
        "n_groups_train": int(tr["group_id"].nunique()),
        "train": tr,
        "test": te,
        "disjoint_mask": (~share_either).to_numpy(),
    }

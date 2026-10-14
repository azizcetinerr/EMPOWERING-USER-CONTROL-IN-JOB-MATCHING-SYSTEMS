"""
Computes the user study metrics to be reported in the paper from the recorded events.
Used both by the admin panel of the web application and by analysis/paper_analysis.py.
Uses only the standard library.
"""

import math
import statistics
from collections import Counter, defaultdict

PROBLEM_TYPES = {
    "slow": "The analysis took too long",
    "error": "I got an error message / the analysis did not complete",
    "irrelevant": "The recommended projects were irrelevant to my profile",
    "interface": "I found the interface hard to understand",
    "cv_link": "Problem finding/entering the CV link",
    "other": "Other",
}


def wilson(success: int, n: int, z: float = 1.96):
    """95% Wilson confidence interval for a proportion -> (rate, lower, upper). None values if n=0."""
    if n == 0:
        return None, None, None
    p = success / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return p, max(0.0, center - half), min(1.0, center + half)


def _rate(success, n):
    p, lower, upper = wilson(success, n)
    return {
        "success": success,
        "n": n,
        "rate": None if p is None else round(p, 4),
        "ci95": None if p is None else [round(lower, 4), round(upper, 4)],
    }


def _percentile(values, q):
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * q
    a, b = math.floor(k), math.ceil(k)
    return s[a] if a == b else s[a] + (s[b] - s[a]) * (k - a)


def summarize(events, include_test_sessions=False):
    """Computes the summary metrics from the event list."""
    if not include_test_sessions:
        events = [o for o in events if not o.get("test")]

    kind = defaultdict(list)
    for o in events:
        kind[o.get("kind")].append(o)

    # --- Visits ---
    visitors = {o.get("visitor") for o in kind["visit"] if o.get("visitor")}

    # --- Analysis success (automatic, objective) ---
    started = {o["session"] for o in kind["analysis_started"] if o.get("session")}
    completed = {o["session"]: o for o in kind["analysis_completed"] if o.get("session")}
    failed = {o["session"]: o for o in kind["analysis_error"] if o.get("session")}
    failed = {k: v for k, v in failed.items() if k not in completed}
    partial = started - set(completed) - set(failed)
    durations = [float(o["duration_sec"]) for o in completed.values() if o.get("duration_sec") is not None]
    visitors_with_analysis = {o.get("visitor") for o in kind["analysis_started"] if o.get("visitor")}

    error_stages = Counter(o.get("stage", "?") for o in failed.values())
    error_types = Counter(o.get("error_type", "?") for o in failed.values())

    # --- End-of-session survey (the LAST response of each session counts) ---
    survey = {}
    for o in sorted(kind["survey"], key=lambda x: x.get("time", "")):
        if o.get("session"):
            survey[o["session"]] = o
    liked_responses = [o for o in survey.values() if o.get("liked") is not None]
    problem_responses = [o for o in survey.values() if o.get("had_problem") is not None]
    likers = sum(1 for o in liked_responses if o["liked"] is True)
    problem_free = sum(1 for o in problem_responses if o["had_problem"] is False)
    problem_type_counts = Counter()
    for o in problem_responses:
        if o["had_problem"]:
            for s in o.get("problem_types") or []:
                problem_type_counts[s] += 1
    comments = [
        {"time": o.get("time"), "comment": o["comment"]}
        for o in survey.values() if (o.get("comment") or "").strip()
    ]
    # Surveys of sessions whose analysis completed / failed, separately
    survey_completed = [o for k, o in survey.items() if k in completed]
    survey_on_error = [o for k, o in survey.items() if k in failed]

    # --- Project feedback (the LAST vote for each session+project) ---
    last_vote = {}
    for o in sorted(kind["project_feedback"], key=lambda x: x.get("time", "")):
        if o.get("session") and o.get("project_id") is not None:
            last_vote[(o["session"], str(o["project_id"]))] = o
    votes = list(last_vote.values())
    positive = sum(1 for o in votes if o.get("vote_type") == "positive")

    def rank_group(r):
        try:
            r = int(r)
        except (TypeError, ValueError):
            return "unknown"
        if r <= 3:
            return "1-3"
        if r <= 10:
            return "4-10"
        return "11+"

    rank_breakdown = {}
    for group in ["1-3", "4-10", "11+", "unknown"]:
        g = [o for o in votes if rank_group(o.get("rank")) == group]
        if g:
            rank_breakdown[group] = _rate(sum(1 for o in g if o.get("vote_type") == "positive"), len(g))

    # Per session: "at least one suitable project in the top 10" (as marked by the user)
    session_votes = defaultdict(list)
    for o in votes:
        session_votes[o["session"]].append(o)
    voting_sessions = len(session_votes)
    hit_session = sum(
        1 for lst in session_votes.values()
        if any(o.get("vote_type") == "positive" and isinstance(o.get("rank"), int) and o["rank"] <= 10 for o in lst)
    )
    # Mean scores of the voted projects (positive vs negative)
    def avg(field, vote_type):
        d = [float(o[field]) for o in votes if o.get("vote_type") == vote_type and o.get(field) is not None]
        return round(statistics.mean(d), 2) if d else None

    score_comparison = {
        field: {"positive_mean": avg(field, "positive"), "negative_mean": avg(field, "negative")}
        for field in ["m_score", "s_score", "r_score", "final_score"]
    }

    dates = sorted(o.get("time") for o in events if o.get("time"))

    return {
        "period": {"first": dates[0] if dates else None, "last": dates[-1] if dates else None},
        "visit": {
            "page_view": len(kind["visit"]),
            "unique_visitors": len(visitors),
            "unique_visitors_with_analysis": len(visitors_with_analysis),
        },
        "analysis": {
            "started_count": len(started),
            "completed": len(completed),
            "failed": len(failed),
            "incomplete": len(partial),
            "error_free_completion": _rate(len(completed), len(started)),
            "duration_sec": {
                "median": None if not durations else round(statistics.median(durations), 1),
                "mean": None if not durations else round(statistics.mean(durations), 1),
                "p90": None if not durations else round(_percentile(durations, 0.9), 1),
                "min": None if not durations else round(min(durations), 1),
                "max": None if not durations else round(max(durations), 1),
            },
            "error_stages": dict(error_stages),
            "error_types": dict(error_types),
        },
        "survey": {
            "responding_sessions": len(survey),
            "liking": _rate(likers, len(liked_responses)),
            "dislike_count": len(liked_responses) - likers,
            "problem_free_use": _rate(problem_free, len(problem_responses)),
            "had_problem_count": len(problem_responses) - problem_free,
            "problem_types": {PROBLEM_TYPES.get(k, k): v for k, v in problem_type_counts.most_common()},
            "liked_among_completed": _rate(
                sum(1 for o in survey_completed if o.get("liked") is True),
                sum(1 for o in survey_completed if o.get("liked") is not None)),
            "responses_after_error": len(survey_on_error),
            "comments": comments,
        },
        "project_feedback": {
            "total_votes": len(votes),
            "positive": positive,
            "negative": len(votes) - positive,
            "positive_rate": _rate(positive, len(votes)),
            "positive_rate_by_rank_group": rank_breakdown,
            "voting_sessions": voting_sessions,
            "top10_at_least_one_suitable": _rate(hit_session, voting_sessions),
            "score_comparison": score_comparison,
        },
    }


def public_counter(events):
    """Non-personal totals that can be shown on the home page."""
    o = summarize(events)
    return {
        "analysis_count": o["analysis"]["completed"],
        "likers": o["survey"]["liking"]["success"],
        "problem_free": o["survey"]["problem_free_use"]["success"],
        "survey_response": o["survey"]["responding_sessions"],
    }

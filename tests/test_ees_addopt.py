import pytest
from fairpyx.participatory_budgeting.ees_addopt import (
    exact_equal_shares,
    greedy_project_change,
    add_opt
)


def test_empty_instance():
    result = exact_equal_shares([], [], {}, {}, 0)
    assert result == ([], {})


def test_project_over_budget():
    voters = ["v1"]
    projects = ["p1"]
    approvals = {"v1": {"p1"}}
    costs = {"p1": 100}
    budget = 10
    selected, _ = exact_equal_shares(voters, projects, approvals, costs, budget)
    assert "p1" not in selected


def test_equal_split_two_voters():
    voters = ["v1", "v2"]
    projects = ["p1"]
    approvals = {"v1": {"p1"}, "v2": {"p1"}}
    costs = {"p1": 10}
    selected, payments = exact_equal_shares(voters, projects, approvals, costs, 10)
    assert payments["v1"]["p1"] == payments["v2"]["p1"]


def test_greedy_project_change_non_negative():
    d = greedy_project_change(([], {}), "p1", {}, {}, 5)
    assert d >= 0


def test_add_opt_changes_solution():
    election = {
        "voters": ["v1", "v2"],
        "projects": ["p1", "p2"],
        "approvals": {
            "v1": {"p1"},
            "v2": {"p2"}
        },
        "costs": {
            "p1": 5,
            "p2": 5
        },
        "budget": 5
    }
    d = add_opt(election, ([], {}))
    assert d > 0
"""
Tests for EES, GreedyProjectChange and add-opt.

Programmer: Yonatan Gabay
Since: 2026-04
"""

import pytest
import random

NUM_OF_RANDOM_INSTANCES = 10


def _random_instance(n_voters, n_projects, budget, cost_range=(1, 50), approval_prob=0.3, seed=0):
    rng = random.Random(seed)
    voters = [f"v{i}" for i in range(n_voters)]
    projects = [f"p{j}" for j in range(n_projects)]
    costs = {p: rng.randint(*cost_range) for p in projects}
    approvals = {v: {p for p in projects if rng.random() < approval_prob}
                 for v in voters}
    return voters, projects, approvals, costs, budget


def _compute_leftover_and_leximax(voters, current_solution, budget):
    W, X = current_solution
    n = len(voters)
    share = budget / n if n > 0 else 0
    leftover = {v: share - sum(X.get(v, {}).values()) for v in voters}
    leximax = {v: sorted(X.get(v, {}).values(), reverse=True) for v in voters}
    sorted_lo = sorted(leftover.items(), key=lambda x: x[1])
    sorted_lx = sorted(leximax.items(), key=lambda x: x[1])
    return sorted_lo, sorted_lx


# ---------- EES (Algorithm 1) ----------

def test_ees_empty():
    assert exact_equal_shares([], [], {}, {}, 0) == ([], {})

def test_ees_no_voters():
    sel, _ = exact_equal_shares([], ["p1"], {}, {"p1": 5}, 100)
    assert sel == []

def test_ees_no_approvals():
    sel, _ = exact_equal_shares(["v1"], ["p1"], {"v1": set()}, {"p1": 5}, 100)
    assert sel == []

def test_ees_project_too_expensive():
    sel, _ = exact_equal_shares(["v1"], ["p1"], {"v1": {"p1"}},
                                {"p1": 200}, budget=10)
    assert "p1" not in sel

def test_ees_equal_split():
    sel, pay = exact_equal_shares(
        ["v1", "v2"], ["p1"],
        {"v1": {"p1"}, "v2": {"p1"}}, {"p1": 10}, budget=10)
    assert "p1" in sel
    assert pay["v1"]["p1"] == pytest.approx(5)
    assert pay["v2"]["p1"] == pytest.approx(5)

def test_ees_random_invariants():
    """Budget feasibility, cost coverage, non-negative payments."""
    for i in range(NUM_OF_RANDOM_INSTANCES):
        voters, projects, approvals, costs, budget = _random_instance(15, 8, 80, seed=i)
        selected, payments = exact_equal_shares(voters, projects, approvals, costs, budget)
        total = sum(payments.get(v, {}).get(p, 0) for v in voters for p in selected)
        assert total <= budget + 0.01
        for p in selected:
            paid = sum(payments.get(v, {}).get(p, 0) for v in voters)
            assert paid == pytest.approx(costs[p])
        for v in voters:
            for amt in payments.get(v, {}).values():
                assert amt >= 0


# ---------- GPC (Algorithm 2) – edge cases ----------

def test_gpc_empty():
    d = greedy_project_change([], [], {}, {}, 0, ([], {}), "p1", [], [])
    assert d >= 0 or d == float("inf")

def test_gpc_zero_cost():
    d = greedy_project_change(
        ["v1"], ["p1"], {"v1": {"p1"}}, {"p1": 0}, 10,
        ([], {}), "p1", [("v1", 10.0)], [("v1", [])])
    assert d == pytest.approx(0)

def test_gpc_exact_leftover():
    # leftover == cost  =>  no extra budget needed
    d = greedy_project_change(
        ["v1"], ["p1"], {"v1": {"p1"}}, {"p1": 10}, 10,
        ([], {}), "p1", [("v1", 10.0)], [("v1", [])])
    assert d == pytest.approx(0)

def test_gpc_no_leftover():
    d = greedy_project_change(
        ["v1"], ["p1", "p2"],
        {"v1": {"p1", "p2"}}, {"p1": 10, "p2": 10}, 10,
        (["p1"], {"v1": {"p1": 10.0}}), "p2",
        [("v1", 0.0)], [("v1", [10.0])])
    assert d > 0

def test_gpc_enough_leftover():
    # 3 voters each with 5.0 leftover, cost only 12
    d = greedy_project_change(
        ["v1", "v2", "v3"], ["p1"],
        {"v1": {"p1"}, "v2": {"p1"}, "v3": {"p1"}},
        {"p1": 12}, 15,
        ([], {}), "p1",
        [("v1", 5.0), ("v2", 5.0), ("v3", 5.0)],
        [("v1", []), ("v2", []), ("v3", [])])
    assert d == pytest.approx(0)

def test_gpc_not_enough_leftover():
    d = greedy_project_change(
        ["v1", "v2", "v3"], ["p1"],
        {"v1": {"p1"}, "v2": {"p1"}, "v3": {"p1"}},
        {"p1": 30}, 6,
        ([], {}), "p1",
        [("v1", 2.0), ("v2", 2.0), ("v3", 2.0)],
        [("v1", []), ("v2", []), ("v3", [])])
    assert d > 0

def test_gpc_with_existing_selection():
    d = greedy_project_change(
        ["v1", "v2"], ["p1", "p2"],
        {"v1": {"p1", "p2"}, "v2": {"p1", "p2"}},
        {"p1": 5, "p2": 8}, 10,
        (["p1"], {"v1": {"p1": 5.0}, "v2": {}}),
        "p2",
        [("v1", 0.0), ("v2", 5.0)],
        [("v1", [5.0]), ("v2", [])])
    assert d >= 0

def test_gpc_huge_cost():
    d = greedy_project_change(
        ["v1"], ["p1"], {"v1": {"p1"}}, {"p1": 1_000_000}, 1,
        ([], {}), "p1", [("v1", 1.0)], [("v1", [])])
    assert d > 0


# ---------- GPC – correctness on hand-crafted instances ----------

def test_gpc_symmetric_voters():
    d = greedy_project_change(
        ["v1", "v2"], ["p1"],
        {"v1": {"p1"}, "v2": {"p1"}},
        {"p1": 10}, 6,
        ([], {}), "p1",
        [("v1", 3.0), ("v2", 3.0)],
        [("v1", []), ("v2", [])])
    assert d >= 0

def test_gpc_one_voter_covers_cost():
    # v1 has leftover=5 which already covers cost(p1)=5
    d = greedy_project_change(
        ["v1", "v2"], ["p1", "p2"],
        {"v1": {"p1", "p2"}, "v2": {"p1", "p2"}},
        {"p1": 5, "p2": 5}, 10,
        (["p2"], {"v1": {}, "v2": {"p2": 5.0}}),
        "p1",
        [("v2", 0.0), ("v1", 5.0)],
        [("v1", []), ("v2", [5.0])])
    assert d == pytest.approx(0)

def test_gpc_cost_equals_leftover():
    d = greedy_project_change(
        ["v1", "v2"], ["p1"],
        {"v1": {"p1"}, "v2": {"p1"}},
        {"p1": 10}, 10,
        ([], {}), "p1",
        [("v1", 4.0), ("v2", 6.0)],
        [("v1", []), ("v2", [])])
    assert d == pytest.approx(0)


# ---------- GPC – random property checks ----------

def _make_random_gpc_args(rng, n, cost):
    voters = [f"v{i}" for i in range(n)]
    projects = ["target"]
    approvals = {v: {"target"} for v in voters}
    budget = sum(rng.uniform(0, 10) for _ in range(n))
    share = budget / n if n > 0 else 0
    lo = sorted([(v, rng.uniform(0, share)) for v in voters], key=lambda x: x[1])
    lx = sorted([(v, [rng.uniform(0, 5)]) for v in voters], key=lambda x: x[1])
    return voters, projects, approvals, {"target": cost}, budget, lo, lx


def test_gpc_always_non_negative():
    for i in range(NUM_OF_RANDOM_INSTANCES):
        rng = random.Random(i)
        n = rng.randint(2, 20)
        cost = rng.uniform(1, 50)
        voters, projects, approvals, costs, budget, lo, lx = _make_random_gpc_args(rng, n, cost)
        d = greedy_project_change(voters, projects, approvals, costs, budget,
                                  ([], {}), "target", lo, lx)
        assert d >= 0

def test_gpc_monotone_in_cost():
    for i in range(NUM_OF_RANDOM_INSTANCES):
        rng = random.Random(i)
        n = rng.randint(2, 10)
        c_low = rng.uniform(1, 20)
        c_high = c_low + rng.uniform(1, 30)
        voters, projects, approvals, _, budget, lo, lx = _make_random_gpc_args(rng, n, c_low)
        d_low = greedy_project_change(voters, projects, approvals, {"target": c_low}, budget,
                                      ([], {}), "target", lo, lx)
        d_high = greedy_project_change(voters, projects, approvals, {"target": c_high}, budget,
                                       ([], {}), "target", lo, lx)
        assert d_high >= d_low

def test_gpc_more_leftover_less_delta():
    for i in range(NUM_OF_RANDOM_INSTANCES):
        rng = random.Random(i)
        n = rng.randint(2, 10)
        cost = rng.uniform(5, 40)
        voters = [f"v{i}" for i in range(n)]
        approvals = {v: {"target"} for v in voters}
        budget = cost * 2

        lo_small = sorted([(v, rng.uniform(0, 2)) for v in voters], key=lambda x: x[1])
        lo_big = sorted([(v, lo_small[j][1] + rng.uniform(1, 5))
                         for j, v in enumerate(voters)], key=lambda x: x[1])
        lx = sorted([(v, [rng.uniform(0, 3)]) for v in voters], key=lambda x: x[1])

        d_small = greedy_project_change(voters, ["target"], approvals, {"target": cost}, budget,
                                        ([], {}), "target", lo_small, lx)
        d_big = greedy_project_change(voters, ["target"], approvals, {"target": cost}, budget,
                                      ([], {}), "target", lo_big, lx)
        assert d_big <= d_small

def test_gpc_zero_cost_random():
    for i in range(NUM_OF_RANDOM_INSTANCES):
        rng = random.Random(i)
        n = rng.randint(1, 10)
        voters, projects, approvals, _, budget, lo, lx = _make_random_gpc_args(rng, n, cost=0)
        d = greedy_project_change(voters, projects, approvals, {"target": 0}, budget,
                                  ([], {}), "target", lo, lx)
        assert d == pytest.approx(0)

def test_gpc_sufficient_leftover_gives_zero():
    for i in range(NUM_OF_RANDOM_INSTANCES):
        rng = random.Random(i)
        n = rng.randint(1, 10)
        cost = rng.uniform(1, 20)
        voters = [f"v{i}" for i in range(n)]
        approvals = {v: {"target"} for v in voters}
        budget = cost * 3
        lo = sorted([(v, cost / n + rng.uniform(0, 5)) for v in voters], key=lambda x: x[1])
        lx = sorted([(v, []) for v in voters], key=lambda x: x[1])
        d = greedy_project_change(voters, ["target"], approvals, {"target": cost}, budget,
                                  ([], {}), "target", lo, lx)
        assert d == pytest.approx(0)


# ---------- GPC – large inputs ----------

def test_gpc_200_voters():
    voters = [f"v{i}" for i in range(200)]
    approvals = {v: {"p1"} for v in voters}
    lo = [(v, 1.0) for v in voters]
    lx = [(v, []) for v in voters]
    d = greedy_project_change(voters, ["p1"], approvals, {"p1": 5000}, 200,
                              ([], {}), "p1", lo, lx)
    assert d >= 0

def test_gpc_200_voters_surplus():
    voters = [f"v{i}" for i in range(200)]
    approvals = {v: {"p1"} for v in voters}
    lo = [(v, 100.0) for v in voters]
    lx = [(v, []) for v in voters]
    d = greedy_project_change(voters, ["p1"], approvals, {"p1": 50}, 20000,
                              ([], {}), "p1", lo, lx)
    assert d == pytest.approx(0)

def test_gpc_500_voters_random():
    rng = random.Random(42)
    voters = [f"v{i}" for i in range(500)]
    approvals = {v: {"p1"} for v in voters}
    lo = sorted([(v, rng.uniform(0, 5)) for v in voters], key=lambda x: x[1])
    lx = sorted([(v, [rng.uniform(0, 10)]) for v in voters], key=lambda x: x[1])
    d = greedy_project_change(voters, ["p1"], approvals, {"p1": 300}, 2500,
                              ([], {}), "p1", lo, lx)
    assert d >= 0


# ---------- GPC – bad inputs ----------

def test_gpc_negative_cost():
    try:
        greedy_project_change(
            ["v1"], ["p1"], {"v1": {"p1"}}, {"p1": -10}, 10,
            ([], {}), "p1", [("v1", 5.0)], [("v1", [])])
    except (ValueError, AssertionError):
        pass  # ok to raise

def test_gpc_negative_leftover():
    try:
        greedy_project_change(
            ["v1"], ["p1"], {"v1": {"p1"}}, {"p1": 10}, 10,
            ([], {}), "p1", [("v1", -5.0)], [("v1", [])])
    except (ValueError, AssertionError):
        pass

def test_gpc_no_voters_positive_cost():
    d = greedy_project_change([], ["p1"], {}, {"p1": 10}, 0,
                              ([], {}), "p1", [], [])
    assert d >= 0 or d == float("inf")

def test_gpc_fractional_cost():
    d = greedy_project_change(
        ["v1"], ["p1"], {"v1": {"p1"}}, {"p1": 2.5}, 10,
        ([], {}), "p1", [("v1", 3.7)], [("v1", [1.2])])
    assert d >= 0


# ---------- add-opt (Algorithm 3) ----------

def test_addopt_empty():
    d = add_opt([], [], {}, {}, 0, ([], {}), [], [])
    assert d >= 0 or d == float("inf")

def test_addopt_all_already_selected():
    d = add_opt(
        ["v1"], ["p1"], {"v1": {"p1"}}, {"p1": 5}, 10,
        (["p1"], {"v1": {"p1": 5.0}}),
        [("v1", 5.0)], [("v1", [5.0])])
    assert d >= 0 or d == float("inf")

def test_addopt_disjoint_approvals():
    d = add_opt(
        ["v1", "v2"], ["p1", "p2"],
        {"v1": {"p1"}, "v2": {"p2"}},
        {"p1": 5, "p2": 5}, 5,
        ([], {}),
        [("v1", 2.5), ("v2", 2.5)],
        [("v1", []), ("v2", [])])
    assert d > 0

def test_addopt_random_non_negative():
    for i in range(NUM_OF_RANDOM_INSTANCES):
        voters, projects, approvals, costs, budget = _random_instance(10, 6, 50, seed=i)
        n = len(voters)
        lo = sorted([(v, budget / n) for v in voters], key=lambda x: x[1])
        lx = sorted([(v, []) for v in voters], key=lambda x: x[1])
        d = add_opt(voters, projects, approvals, costs, budget, ([], {}), lo, lx)
        assert d >= 0

def test_addopt_large():
    voters, projects, approvals, costs, budget = _random_instance(100, 30, 2000, seed=123)
    n = len(voters)
    lo = sorted([(v, budget / n) for v in voters], key=lambda x: x[1])
    lx = sorted([(v, []) for v in voters], key=lambda x: x[1])
    d = add_opt(voters, projects, approvals, costs, budget, ([], {}), lo, lx)
    assert d >= 0


# ---------- Integration: EES -> add_opt ----------

def test_ees_to_addopt_pipeline():
    for i in range(NUM_OF_RANDOM_INSTANCES):
        voters, projects, approvals, costs, budget = _random_instance(
            5, 4, 30, cost_range=(2, 12), seed=i)
        selected, payments = exact_equal_shares(voters, projects, approvals, costs, budget)
        total = sum(payments.get(v, {}).get(p, 0) for v in voters for p in selected)
        assert total <= budget + 0.01
        sol = (selected, payments)
        sorted_lo, sorted_lx = _compute_leftover_and_leximax(voters, sol, budget)
        d = add_opt(voters, projects, approvals, costs, budget, sol, sorted_lo, sorted_lx)
        assert d >= 0


if __name__ == "__main__":
    pytest.main(["-v", __file__])
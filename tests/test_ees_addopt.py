"""
Tests for EES (Algorithm 1), GreedyProjectChange (Algorithm 2) and add-opt (Algorithm 3).

Programmer: Yonatan Gabay
Since: 2026-04
"""

import pytest
from participatory_budgeting.ees_addopt import exact_equal_shares, greedy_project_change, add_opt



####### EES tests


# empty input should return empty result
def test_ees_nothing_to_select():
    selected, payments = exact_equal_shares([], [], {}, {}, 0)
    assert selected == []
    assert payments == {}


# project above budget should not selected
def test_ees_single_project_too_expensive():
    voters = ["v1", "v2"]
    projects = ["p1"]
    approvals = {"v1": {"p1"}, "v2": {"p1"}}
    costs = {"p1": 200}
    budget = 50

    selected, _ = exact_equal_shares(voters, projects, approvals, costs, budget)
    assert "p1" not in selected


# one approved project should split cost evenly.
def test_ees_cost_split_equally():
    voters = ["v1", "v2"]
    projects = ["p1"]
    approvals = {"v1": {"p1"}, "v2": {"p1"}}
    costs = {"p1": 10}
    budget = 10

    selected, payments = exact_equal_shares(voters, projects, approvals, costs, budget)
    assert "p1" in selected
    assert payments["v1"]["p1"] == pytest.approx(5)
    assert payments["v2"]["p1"] == pytest.approx(5)


# paper example
def test_ees_multiple_projects():
    voters = ["v1", "v2", "v3", "v4"]
    projects = ["p1", "p2", "p3"]
    approvals = {"v1": {"p1"}, "v2": {"p1", "p3"}, "v3": {"p2", "p3"}, "v4": {"p2", "p3"}}
    costs = {"p1": 10, "p2": 16, "p3": 21}
    budget = 40

    selected, payments = exact_equal_shares(voters, projects, approvals, costs, budget)
    assert selected == ["p1", "p2"]
    assert payments == {"v1": {"p1": 5}, "v2": {"p1": 5}, "v3": {"p2": 8}, "v4": {"p2": 8}}


# no approvals
def test_ees_no_approvals_means_nothing_funded():
    voters = ["v1", "v2", "v3"]
    projects = ["p1", "p2"]
    approvals = {"v1": set(), "v2": set(), "v3": set()}
    costs = {"p1": 10, "p2": 15}
    budget = 100

    selected, _ = exact_equal_shares(voters, projects, approvals, costs, budget)
    assert selected == []


# payments should stay within budget
def test_ees_budget_feasibility():
    voters = ["v1", "v2", "v3", "v4", "v5"]
    projects = ["p1", "p2", "p3"]
    approvals = {"v1": {"p1"}, "v2": {"p1", "p3"}, "v3": {"p2", "p3"},
                 "v4": {"p2", "p3"}, "v5": {"p3"}}
    costs = {"p1": 2, "p2": 3.2, "p3": 6}
    budget = 10

    selected, payments = exact_equal_shares(voters, projects, approvals, costs, budget)

    total_paid = sum(payments.get(v, {}).get(p, 0) for v in voters for p in selected)
    assert total_paid <= budget + 0.01

    # Each funded project is fully covered
    for proj in selected:
        paid_for_proj = sum(payments.get(v, {}).get(proj, 0) for v in voters)
        assert paid_for_proj == pytest.approx(costs[proj])


####### GPC tests

# non cost projects should return zero delta
def test_gpc_free_project_needs_zero_increase():
    voters = ["v1"]
    projects = ["p1"]
    approvals = {"v1": {"p1"}}
    costs = {"p1": 0}
    budget = 10
    solution = ([], {})

    delta = greedy_project_change(voters, projects, approvals, costs, budget,
                                  solution, "p1")
    assert delta == pytest.approx(0)


# enough leftover should return zero delta
def test_gpc_leftover_covers_cost_exactly():
    voters = ["v1", "v2", "v3"]
    projects = ["p1"]
    approvals = {"v1": {"p1"}, "v2": {"p1"}, "v3": {"p1"}}
    costs = {"p1": 12}
    budget = 15
    solution = ([], {})

    delta = greedy_project_change(voters, projects, approvals, costs, budget,
                                  solution, "p1")
    assert delta == pytest.approx(0)


#  not enough leftover should returb positive delta
def test_gpc_not_enough_leftover():
    voters = ["v1", "v2", "v3"]
    projects = ["p1"]
    approvals = {"v1": {"p1"}, "v2": {"p1"}, "v3": {"p1"}}
    costs = {"p1": 30}
    budget = 6
    solution = ([], {})

    delta = greedy_project_change(voters, projects, approvals, costs, budget,
                                  solution, "p1")
    assert delta > 0


# paper example
def test_gpc_paper_example():
    voters = ["v1", "v2", "v3", "v4", "v5"]
    projects = ["p1", "p2", "p3"]
    approvals = {"v1": {"p1"}, "v2": {"p1", "p3"}, "v3": {"p2", "p3"},
                 "v4": {"p2", "p3"}, "v5": {"p3"}}
    costs = {"p1": 2, "p2": 3.2, "p3": 6}
    budget = 10
    solution = (["p1", "p2"], {"v1": {"p1": 1}, "v2": {"p1": 1},
                               "v3": {"p2": 1.6}, "v4": {"p2": 1.6}, "v5": {}})

    delta = greedy_project_change(voters, projects, approvals, costs, budget,
                                  solution, "p3")
    assert delta == pytest.approx(0.5)


# some projects are already funded
def test_gpc_with_existing_selection():
    voters = ["v1", "v2"]
    projects = ["p1", "p2"]
    approvals = {"v1": {"p1", "p2"}, "v2": {"p1", "p2"}}
    costs = {"p1": 5, "p2": 8}
    budget = 10
    solution = (["p1"], {"v1": {"p1": 5.0}, "v2": {}})

    delta = greedy_project_change(voters, projects, approvals, costs, budget,
                                  solution, "p2")
    assert delta >= 0


####### add-opt tests 

# empty input should return not 0 or infinite
def test_addopt_nothing_to_add():
    delta = add_opt([], [], {}, {}, 0, ([], {}))
    assert delta >= 0 or delta == float("inf")


# if all projects funded should not improve
def test_addopt_all_projects_already_funded():
    voters = ["v1"]
    projects = ["p1"]
    approvals = {"v1": {"p1"}}
    costs = {"p1": 5}
    budget = 10
    solution = (["p1"], {"v1": {"p1": 5.0}})

    delta = add_opt(voters, projects, approvals, costs, budget,
                    solution)
    assert delta >= 0 or delta == float("inf")


# paper example
def test_addopt_finds_improvement():
    voters = ["v1", "v2", "v3", "v4", "v5"]
    projects = ["p1", "p2", "p3"]
    approvals = {"v1": {"p1"}, "v2": {"p1", "p3"}, "v3": {"p2", "p3"},
                 "v4": {"p2", "p3"}, "v5": {"p3"}}
    costs = {"p1": 2, "p2": 3.2, "p3": 6}
    budget = 10
    solution = (["p1", "p2"], {"v1": {"p1": 1}, "v2": {"p1": 1},
                               "v3": {"p2": 1.6}, "v4": {"p2": 1.6}, "v5": {}})

    delta = add_opt(voters, projects, approvals, costs, budget,
                    solution)
    assert delta == pytest.approx(0.5)


####### random tests

@pytest.mark.parametrize("num_voters, num_projects", [
    (3, 20),
    (20, 2),
    (20, 20),
])
# random instances should stay valid after EES, GPC, and rerun EES.
def test_random_ees_gpc_rerun(num_voters, num_projects):
    """
    full random test:
    1. generate random instance
    2. run EES and validate result
    3. choose unselected projectand run GPC
    4. run EES with bigger budget
    5. validate new EES result
    6. check GPC project now selected
    """
    import random
    rng = random.Random()

    for seed in range(5):
        rng.seed(seed)

        voters = [f"v{i}" for i in range(num_voters)]
        projects = [f"p{j}" for j in range(num_projects)]
        costs = {p: rng.randint(0, 100) for p in projects}
        # approves with 40% probability
        approvals = {v: {p for p in projects if rng.random() < 0.4} for v in voters}
        # 40%-70% of total costs
        total_cost = sum(costs.values())
        budget = int(total_cost * rng.uniform(0.4, 0.7))

        # run EES
        selected, payments = exact_equal_shares(voters, projects, approvals, costs, budget)

        # validate EES result
        total = sum(payments.get(v, {}).get(p, 0) for v in voters for p in selected)
        assert total <= budget + 0.01, f"seed={seed}: EES exceeded budget"
        for p in selected:
            paid = sum(payments.get(v, {}).get(p, 0) for v in voters)
            assert paid == pytest.approx(costs[p]), f"seed={seed}: project {p} not covered after increase"

        # find unselected project that has support
        unselected = []
        for p in projects:
            if p in selected:
                continue
            has_supporter = False
            for v in voters:
                if p in approvals[v]:
                    has_supporter = True
                    break
            if has_supporter:
                unselected.append(p)
        if not unselected:
            continue  # nothing to test

        target_project = rng.choice(unselected)

        # run GPC
        delta = greedy_project_change(voters, projects, approvals, costs, budget, (selected, payments), target_project)
        assert delta >= 0, f"seed={seed}: GPC returned negative delta"

        # run EES with new budget
        new_budget = budget + num_voters * delta
        new_selected, new_payments = exact_equal_shares(voters, projects, approvals, costs, new_budget)

        # validate EES result
        new_total = sum(new_payments.get(v, {}).get(p, 0) for v in voters for p in new_selected)
        assert new_total <= new_budget + 0.01, f"seed={seed}: new EES exceeded budget"
        for p in new_selected:
            paid = sum(new_payments.get(v, {}).get(p, 0) for v in voters)
            assert paid == pytest.approx(costs[p]), f"seed={seed}: project {p} not covered after increase"

        # target project should be selected
        assert target_project in new_selected, (
            f"seed={seed}: GPC said delta={delta} for '{target_project}' "
            f"but it's still not selected with budget={new_budget}"
        )


if __name__ == "__main__":
    pytest.main(["-v", __file__])
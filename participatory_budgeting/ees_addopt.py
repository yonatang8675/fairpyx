"""
An implementation of the algorithms in:

"Streamlining Equal Shares", 
by Sonja Kraiczy, Isaac Robinson, Edith Elkind (2024), 
https://arxiv.org/abs/2502.11797

Programmer: Yonatan Gabay
Date: 20-04-2026
"""


def exact_equal_shares(
        voters: list, 
        projects: list, 
        approvals: dict, 
        costs: dict, 
        budget: float, 
        utilities: dict = None) -> tuple:
    """
    Algorithm 1: EES for uniform utilities.

    Iteratively selects projects that maximise bang-per-buck (|V|·u(p)/cost(p))
    among all feasible (project, supporter-subset) pairs, splitting costs
    equally among supporters.

    
    Parameters
    ----------
    voters : list
        List of voter identifiers.
    projects : list
        List of candidate project identifiers.
    approvals : dict
        approvals[voter] = set of projects approved by the voter.
    costs : dict
        costs[project] = cost of the project  (positive float).
    budget : float
        Total available budget b.
    utilities : dict or None
        utilities[project] = u(p).  When None, every project has utility 1
        (pure approval setting).

    Returns
    -------
    tuple (W, X)
        W : list  - selected projects (in order of selection).
        X : dict  - X[voter][project] = payment made by voter for the project.

    Examples
    --------
    >>> voters = ['v1', 'v2', 'v3', 'v4']
    >>> projects = ['p1', 'p2', 'p3']
    >>> approvals = {'v1': {'p1'}, 'v2': {'p1', 'p3'}, 'v3': {'p2', 'p3'}, 'v4': {'p2', 'p3'}}
    >>> costs = {'p1': 10, 'p2': 16, 'p3': 21}
    >>> budget = 40
    >>> exact_equal_shares(voters, projects, approvals, costs, budget)
    (['p1', 'p2'], {'v1': {'p1': 5}, 'v2': {'p1': 5}, 'v3': {'p2': 8}, 'v4': {'p2': 8}})

    >>> voters = ['v1', 'v2', 'v3', 'v4']
    >>> projects = ['p1', 'p2', 'p3']
    >>> approvals = {'v1': {'p1'}, 'v2': {'p1', 'p3'}, 'v3': {'p2', 'p3'}, 'v4': {'p2', 'p3'}}
    >>> costs = {'p1': 10, 'p2': 16, 'p3': 21}
    >>> budget = 48
    >>> exact_equal_shares(voters, projects, approvals, costs, budget)
    (['p1', 'p3'], {'v1': {'p1': 5}, 'v2': {'p1': 5, 'p3': 7}, 'v3': {'p3': 7}, 'v4': {'p3': 7}})

    >>> voters = ['v1', 'v2', 'v3']
    >>> projects = ['p1', 'p2', 'p3', 'p4']
    >>> approvals = {'v1': {'p1', 'p2'}, 'v2': {'p2', 'p3'}, 'v3': {'p3', 'p4'}}
    >>> costs = {'p1': 2, 'p2': 98, 'p3': 100, 'p4': 51}
    >>> budget = 150
    >>> exact_equal_shares(voters, projects, approvals, costs, budget)
    (['p1', 'p3'], {'v1': {'p1': 2}, 'v2': {'p3': 50}, 'v3': {'p3': 50}})

    >>> voters = ['v1', 'v2', 'v3']
    >>> projects = ['p1', 'p2', 'p3', 'p4']
    >>> approvals = {'v1': {'p1', 'p2'}, 'v2': {'p2', 'p3'}, 'v3': {'p3', 'p4'}}
    >>> costs = {'p1': 2, 'p2': 98, 'p3': 100, 'p4': 51}
    >>> budget = 153
    >>> exact_equal_shares(voters, projects, approvals, costs, budget)
    (['p1', 'p2', 'p4'], {'v1': {'p1': 2, 'p2': 49}, 'v2': {'p2': 49}, 'v3': {'p4': 51}})

    >>> voters = ['v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10']
    >>> projects = ['p1', 'p2', 'p3', 'p4', 'p5']
    >>> approvals = {'v1': {'p2', 'p3'}, 'v2': {'p1', 'p3'}, 'v3': {'p3', 'p4'}, 'v4': {'p1'}, 'v5': {'p1'}, 'v6': {'p2'}, 'v7': {'p1'}, 'v8': {'p1'}, 'v9': {'p4'}, 'v10': {'p5'}}
    >>> costs = {'p1': 20, 'p2': 18, 'p3': 20, 'p4': 8, 'p5': 15}
    >>> budget = 100
    >>> exact_equal_shares(voters, projects, approvals, costs, budget)
    (['p1', 'p2', 'p4'], {'v1': {'p2': 9}, 'v2': {'p1': 4}, 'v3': {'p4': 4}, 'v4': {'p1': 4}, 'v5': {'p1': 4}, 'v6': {'p2': 9}, 'v7': {'p1': 4}, 'v8': {'p1': 4}, 'v9': {'p4': 4}, 'v10': {}})

    >>> voters = ['v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10']
    >>> projects = ['p1', 'p2', 'p3', 'p4', 'p5']
    >>> approvals = {'v1': {'p2', 'p3'}, 'v2': {'p1', 'p3'}, 'v3': {'p3', 'p4'}, 'v4': {'p1'}, 'v5': {'p1'}, 'v6': {'p2'}, 'v7': {'p1'}, 'v8': {'p1'}, 'v9': {'p4'}, 'v10': {'p5'}}
    >>> costs = {'p1': 20, 'p2': 18, 'p3': 20, 'p4': 8, 'p5': 15}
    >>> budget = 106 + 2/3
    >>> exact_equal_shares(voters, projects, approvals, costs, budget)
    (['p1', 'p3', 'p4'], {'v1': {'p3': 6 + 2/3}, 'v2': {'p1': 4, 'p3': 6 + 2/3}, 'v3': {'p4': 4, 'p3': 6 + 2/3}, 'v4': {'p1': 4}, 'v5': {'p1': 4}, 'v6': {}, 'v7': {'p1': 4}, 'v8': {'p1': 4}, 'v9': {'p4': 4}, 'v10': {}})
    """
    return None


def greedy_project_change(
    voters: list,
    projects: list,
    approvals: dict,
    costs: dict,
    budget: float,
    current_solution: tuple,
    project: str
) -> float:
    """
    Algorithm 2: GreedyProjectChange (GPC).

    Computes the minimum per-voter budget increase d > 0 such that
    *project* certifies instability of the current equal-shares solution
    (W, X) for instance E(b + n·d).
    The algorithm walks two sorted arrays simultaneously:
    - leftover_budgets  (A'): residual budgets of Op(X) voters, sorted ascending.
    - leximax_payments   (B'): leximax payment vectors of Op(X) voters,
      sorted lex-ascending.

    Parameters
    ----------
    voters : list
        All voter identifiers.
    projects : list
        All project identifiers.
    approvals : dict
        approvals[voter] = set of approved projects.
    costs : dict
        costs[project] = cost of the project.
    budget : float
        Total budget b.
    current_solution : tuple (W, X)
        W : list  - currently selected projects.
        X : dict  - X[voter][project] = payment.
    project : str
        The candidate project p to test instability for.

    Returns
    -------
    float
        Minimum per-voter budget increase d (may be +∞ if the project
        can never certify instability).

    Examples
    --------
    >>> voters = ['v1', 'v2', 'v3', 'v4', 'v5']
    >>> projects = ['p1', 'p2', 'p3']
    >>> approvals = {'v1': {'p1'}, 'v2': {'p1', 'p3'}, 'v3': {'p2', 'p3'}, 'v4': {'p2', 'p3'}, 'v5': {'p3'}}
    >>> costs = {'p1': 2, 'p2': 3.2, 'p3': 6}
    >>> budget = 10
    >>> current_solution = (['p1', 'p2'], {'v1': {'p1': 1}, 'v2': {'p1': 1}, 'v3': {'p2': 1.6}, 'v4': {'p2': 1.6}, 'v5': {}})
    >>> project = 'p3'
    >>> greedy_project_change(voters, projects, approvals, costs, budget, current_solution, project)
    0.5

    >>> voters = ['v1', 'v2', 'v3', 'v4']
    >>> projects = ['p1', 'p2', 'p3']
    >>> approvals = {'v1': {'p1'}, 'v2': {'p1', 'p3'}, 'v3': {'p2', 'p3'}, 'v4': {'p2', 'p3'}}
    >>> costs = {'p1': 10, 'p2': 16, 'p3': 21}
    >>> budget = 40
    >>> current_solution = (['p1', 'p2'], {'v1': {'p1': 5}, 'v2': {'p1': 5}, 'v3': {'p2': 8}, 'v4': {'p2': 8}})
    >>> project = 'p3'
    >>> greedy_project_change(voters, projects, approvals, costs, budget, current_solution, project)
    2

    >>> voters = ['v1', 'v2', 'v3']
    >>> projects = ['p1', 'p2', 'p3', 'p4']
    >>> approvals = {'v1': {'p1', 'p2'}, 'v2': {'p2', 'p3'}, 'v3': {'p3', 'p4'}}
    >>> costs = {'p1': 2, 'p2': 98, 'p3': 100, 'p4': 51}
    >>> budget = 150
    >>> current_solution = (['p1', 'p3'], {'v1': {'p1': 2}, 'v2': {'p3': 50}, 'v3': {'p3': 50}})
    >>> project = 'p4'
    >>> greedy_project_change(voters, projects, approvals, costs, budget, current_solution, project)
    51

    >>> voters = ['v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10']
    >>> projects = ['p1', 'p2', 'p3', 'p4', 'p5']
    >>> approvals = {'v1': {'p2', 'p3'}, 'v2': {'p1', 'p3'}, 'v3': {'p3', 'p4'}, 'v4': {'p1'}, 'v5': {'p1'}, 'v6': {'p2'}, 'v7': {'p1'}, 'v8': {'p1'}, 'v9': {'p4'}, 'v10': {'p5'}}
    >>> costs = {'p1': 20, 'p2': 18, 'p3': 20, 'p4': 8, 'p5': 15}
    >>> budget = 100
    >>> current_solution = (['p1', 'p2', 'p4'], {'v1': {'p2': 9}, 'v2': {'p1': 4}, 'v3': {'p4': 4}, 'v4': {'p1': 4}, 'v5': {'p1': 4}, 'v6': {'p2': 9}, 'v7': {'p1': 4}, 'v8': {'p1': 4}, 'v9': {'p4': 4}, 'v10': {}})
    >>> project = 'p3'
    >>> greedy_project_change(voters, projects, approvals, costs, budget, current_solution, project)
    2/3
    """
    return None


def add_opt(
    voters: list,
    projects: list,
    approvals: dict,
    costs: dict,
    budget: float,
    current_solution: tuple,
) -> float:
    """
    Algorithm 3: add-opt.

    Iterates over every project p ∈ P, restricts the sorted leftover-budget
    and leximax-payment arrays to the voters in Op(X), and calls
    GreedyProjectChange to find the minimum per-voter budget increase
    that makes p certify instability.  Returns the global minimum.

    Parameters
    ----------
    voters : list
        All voter identifiers.
    projects : list
        All project identifiers.
    approvals : dict
        approvals[voter] = set of approved projects.
    costs : dict
        costs[project] = cost of the project.
    budget : float
        Total budget b.
    current_solution : tuple (W, X)
        W : list  - currently selected projects.
        X : dict  - X[voter][project] = payment.

    Returns
    -------
    float
        Minimum per-voter budget increase d > 0 such that (W, X) is
        unstable for E(b + n·d).  Returns +∞ when (W, X) is stable for
        every finite budget increase.

    Examples
    --------
    >>> voters = ['v1', 'v2', 'v3', 'v4', 'v5']
    >>> projects = ['p1', 'p2', 'p3']
    >>> approvals = {'v1': {'p1'}, 'v2': {'p1', 'p3'}, 'v3': {'p2', 'p3'}, 'v4': {'p2', 'p3'}, 'v5': {'p3'}}
    >>> costs = {'p1': 2, 'p2': 3.2, 'p3': 6}
    >>> budget = 10
    >>> current_solution = (['p1', 'p2'], {'v1': {'p1': 1}, 'v2': {'p1': 1}, 'v3': {'p2': 1.6}, 'v4': {'p2': 1.6}, 'v5': {}})
    >>> add_opt(voters, projects, approvals, costs, budget, current_solution)
    0.5
    """
    return None
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
    (['p1', 'p2'], {'v1': {'p1': 5.0}, 'v2': {'p1': 5.0}, 'v3': {'p2': 8.0}, 'v4': {'p2': 8.0}})
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
    leftover_budgets : list of (voter, residual)
        Sorted ascending by residual budget.
        Only voters in Op(X) (those who approve *project* and are not
        already paying for it in the current solution).
    leximax_payments : list of (voter, payment_vector)
        Sorted lex-ascending by their payment vectors in (W, X).
        Only voters in Op(X).

    Returns
    -------
    float
        Minimum per-voter budget increase d (may be +∞ if the project
        can never certify instability).

    Examples
    --------
    >>> voters = ['v1', 'v2', 'v3', 'v4']
    >>> projects = ['p1', 'p2', 'p3']
    >>> approvals = {'v1': {'p1'}, 'v2': {'p1', 'p3'}, 'v3': {'p2', 'p3'}, 'v4': {'p2', 'p3'}}
    >>> costs = {'p1': 10, 'p2': 16, 'p3': 21}
    >>> budget = 40
    >>> current_solution = (['p1', 'p2'], {'v1': {'p1': 1.0}, 'v2': {'p1': 1.0}, 'v3': {'p2': 1.6}, 'v4': {'p2': 1.6}, 'v5': {}})
    >>> project = 'p3'
    >>> greedy_project_change(voters, projects, approvals, costs, budget, current_solution, project)
    0.5
    """
    return None


def add_opt(
    voters: list,
    projects: list,
    approvals: dict,
    costs: dict,
    budget: float,
    current_solution: tuple,
    sorted_leftover: list,
    sorted_leximax: list,
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
    sorted_leftover : list of (voter, residual)
        A = [(v1, r_v1), …, (vn, r_vn)] sorted ascending by residual,
        covering *all* voters.
    sorted_leximax : list of (voter, payment_vector)
        B = [(w1, cw1), …, (wn, cwn)] sorted lex-ascending,
        covering *all* voters.

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
    >>> current_solution = (['p1', 'p2'], {'v1': {'p1': 1.0}, 'v2': {'p1': 1.0}, 'v3': {'p2': 1.6}, 'v4': {'p2': 1.6}, 'v5': {}})
    >>> sorted_leftover = [('v3', 0.4), ('v4', 0.4), ('v1', 1.0), ('v2', 1.0), ('v5', 2.0)]
    >>> sorted_leximax = [('v5', []), ('v1', [1.0]), ('v2', [1.0]), ('v3', [1.6]), ('v4', [1.6])]
    >>> add_opt(voters, projects, approvals, costs, budget, current_solution, sorted_leftover, sorted_leximax)
    0.5
    """
    return None
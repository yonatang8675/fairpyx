"""
Flask web application for demonstrating the Quasi-Polynomial Local Search
algorithm for Restricted Max-Min Fair Allocation.

Based on: "Quasi-Polynomial Local Search for Restricted Max-Min Fair Allocation"
by Lukas Polacek and Ola Svensson (2014)
"""
import random

from flask import Flask, render_template, request, jsonify
from fairpyx import Instance
from fairpyx.algorithms.qp_local_search import qp_max_min_allocation

app = Flask(__name__)
app.secret_key = "fairpyx-qp-local-search-demo"

MAX_AGENTS = 20
MAX_ITEMS = 50
MAX_VALUE = 1000


@app.route("/", methods=["GET"])
def index():
    """Input page: explanation + form."""
    return render_template("index.html", error=None)


@app.route("/results", methods=["POST"])
def results():
    """Results page: runs the algorithm and displays output."""
    try:
        num_agents = int(request.form.get("num_agents", 0))
        num_items = int(request.form.get("num_items", 0))
        epsilon = float(request.form.get("epsilon", "0.1"))

        if num_agents < 1 or num_items < 1:
            return render_template("index.html", error="You need at least 1 agent and 1 item.")

        if num_agents > MAX_AGENTS:
            return render_template("index.html", error=f"Maximum {MAX_AGENTS} agents allowed for reasonable runtime.")

        if num_items > MAX_ITEMS:
            return render_template("index.html", error=f"Maximum {MAX_ITEMS} items allowed for reasonable runtime.")

        if not (0.01 <= epsilon <= 1.0):
            return render_template("index.html", error="Epsilon must be between 0.01 and 1.0.")

        # Read agent names
        agents = []
        for i in range(num_agents):
            name = request.form.get(f"agent_{i}", "").strip()
            if not name:
                return render_template("index.html", error=f"Agent {i+1} name is empty.")
            if name in agents:
                return render_template("index.html", error=f"Duplicate agent name: '{name}'.")
            agents.append(name)

        # Read item names and sizes
        items = []
        item_sizes = {}
        for j in range(num_items):
            name = request.form.get(f"item_{j}", "").strip()
            if not name:
                return render_template("index.html", error=f"Item {j+1} name is empty.")
            if name in items:
                return render_template("index.html", error=f"Duplicate item name: '{name}'.")
            size_str = request.form.get(f"size_{j}", "0").strip()
            try:
                size = int(size_str)
            except ValueError:
                return render_template("index.html", error=f"Invalid size for '{name}': '{size_str}'. Must be a whole number.")
            if size <= 0:
                return render_template("index.html", error=f"Size of '{name}' must be positive.")
            if size > MAX_VALUE:
                return render_template("index.html", error=f"Size of '{name}' exceeds maximum ({MAX_VALUE}).")
            items.append(name)
            item_sizes[name] = size

        # Read checkboxes (restricted model: value = size or 0)
        valuations = {}
        for agent in agents:
            valuations[agent] = {}
            for item in items:
                wants = request.form.get(f"wants_{agent}_{item}") == "1"
                valuations[agent][item] = item_sizes[item] if wants else 0

        # Ensure every agent wants at least one item
        for agent in agents:
            if all(v == 0 for v in valuations[agent].values()):
                return render_template("index.html", error=f"Agent '{agent}' doesn't want any item. Each agent must want at least one.")

        instance = Instance(valuations=valuations)
        allocation = qp_max_min_allocation(instance, epsilon=epsilon)

        results = []
        for agent in agents:
            bundle = allocation.get(agent, set())
            value = int(sum(instance.agent_item_value(agent, i) for i in bundle))
            results.append({
                "agent": agent,
                "bundle": ", ".join(sorted(bundle)) if bundle else "(empty)",
                "value": value,
            })

        return render_template("results.html", results=results, epsilon=epsilon)

    except Exception as e:
        return render_template("index.html", error=f"Error running algorithm: {str(e)}")


@app.route("/api/random", methods=["POST"])
def generate_random():
    """Generate a random restricted-model instance (same style as compare_qp_local_search.py)."""
    data = request.get_json()
    num_agents = max(1, min(int(data.get("num_agents", 3)), MAX_AGENTS))
    num_items = max(1, min(int(data.get("num_items", 5)), MAX_ITEMS))

    agent_names = [f"Agent_{i+1}" for i in range(num_agents)]
    item_names = [f"Item_{i+1}" for i in range(num_items)]

    # Each item gets a fixed size (like base_values in the experiment)
    item_sizes = {item: random.randint(1, MAX_VALUE) for item in item_names}

    # Each agent wants a random subset of items (at least 1)
    valuations = {}
    for agent in agent_names:
        k = random.randint(1, num_items)
        chosen = set(random.sample(item_names, k=k))
        valuations[agent] = {
            item: item_sizes[item] if item in chosen else 0
            for item in item_names
        }

    # Ensure every item is wanted by at least one agent
    for item in item_names:
        if not any(valuations[a][item] > 0 for a in agent_names):
            a = random.choice(agent_names)
            valuations[a][item] = item_sizes[item]

    return jsonify({
        "agents": agent_names,
        "items": item_names,
        "item_sizes": item_sizes,
        "valuations": valuations,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)

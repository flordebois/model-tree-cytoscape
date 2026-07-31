# Linear Model Tree Visualization

A Python application for visualizing and exploring **linear model trees** with dash cytoscape. The application provides an interactive interface to inspect tree structures, understand individual node models, and analyze how predictions are made.

The application currently supports two linear model tree algorithms:

* **PILOT** (*PIecewise Linear Organic Tree*) — a fast and interpretable linear model tree algorithm for regression.
* **M5** — a classic model tree algorithm that combines decision tree splits with linear regression models in the leaves.

The main goal of this project is to make linear model trees easier to understand and explain through interactive visualization.

## Documentation

Full documentation, including explanations of the interface and visualization options, is available [here](https://flordebois.github.io/model-tree-cytoscape/).

## Installation

Clone the repository:

```bash
git clone https://github.com/flordebois/model-tree-cytoscape
cd model-tree-cytoscape
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Activate the environment:

**Windows**

```bash
.venv\Scripts\activate
```

**Linux/macOS**

```bash
source .venv/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running the application

Start the application with:

```bash
python app.py
```

The application will start a local web server. Open the provided URL in your browser to access the visualization interface.



# Metricate Documentation

**Metricate** is a comprehensive clustering evaluation toolkit that calculates 34 quality metrics, compares clusterings, generates degraded datasets for testing, and trains custom scoring models.

---

## Table of Contents

1. [Getting Started](getting-started.md)
2. [API Reference](api-reference.md)
3. [Metrics Reference](metrics-reference.md)
4. [Degradation Types](degradation-types.md)
5. [Training Weights](training-weights.md)
6. [CLI Reference](cli-reference.md)
7. [Web UI](web-ui.md)
8. [Input Formats](input-formats.md)

---

## Quick Overview

### Why Metricate?

Choosing between clusterings isn't always straightforward. With 40+ clustering metrics available, each giving different opinions, it's hard to know which clustering is actually better.

Metricate solves this by:

1. **Evaluating** clusterings with 34 carefully selected metrics
2. **Comparing** clusterings with transparent winner determination
3. **Learning** which metrics matter through degradation-based training
4. **Scoring** clusterings with a single, learned compound score

### Installation

```bash
pip install metricate
```

Or from source:

```bash
git clone https://github.com/VineSight/clustering-metricate-evaluation.git
cd clustering-metricate-evaluation
pip install -e .
```

### Requirements

- Python ≥ 3.10
- pandas ≥ 2.0
- numpy ≥ 1.24
- scikit-learn ≥ 1.3
- plotly ≥ 5.0
- click ≥ 8.0

### Quick Examples

```python
import metricate

# Evaluate a clustering
result = metricate.evaluate("clustering.csv")
print(result.to_table())

# Compare two clusterings
comparison = metricate.compare("v1.csv", "v2.csv")
print(f"Winner: {comparison.winner}")

# Generate degraded datasets
result = metricate.degrade("data.csv", "./output/")

# Train metric weights
training_result = metricate.train_weights("training_data.csv")
training_result.save_weights("weights.json")
```

---

## License

MIT

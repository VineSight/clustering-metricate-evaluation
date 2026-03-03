# Web UI

Metricate includes a browser-based interface for evaluating and comparing clusterings without writing code.

---

## Starting the Server

### From Command Line

```bash
# If metricate is installed globally or venv is activated
metricate web

# Custom port
metricate web --port 8080

# Allow external connections
metricate web --host 0.0.0.0

# Debug mode (auto-reload on changes)
metricate web --debug
```

### From Python

```python
import metricate

# Start the web server
metricate.web(port=5000, debug=False)
```

Then open `http://localhost:5000` in your browser.

---

## Features

### Upload & Evaluate

1. Click "Upload CSV" or drag-and-drop your file
2. View all 36 metrics in a formatted table
3. See dataset statistics (points, clusters, dimensions)
4. Export results as JSON or CSV

### Compare Two Clusterings

1. Upload both CSV files
2. View side-by-side metric comparison
3. See which clustering wins each metric
4. Get overall winner by metric voting

### Interactive Results

- **Sortable tables**: Click column headers to sort
- **Metric details**: Hover for descriptions and ranges
- **Export options**: Download as JSON, CSV, or copy to clipboard

---

## Screenshots

### Evaluation View

```
┌────────────────────────────────────────────────────────────┐
│  Metricate - Clustering Evaluation                         │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  📁 Upload CSV: [clustering.csv]                           │
│                                                            │
│  Dataset: 10,000 points | 17 clusters | 50 dimensions      │
│                                                            │
│  ┌──────────────────┬──────────┬────────┬──────────┐      │
│  │ Metric           │ Value    │ Range  │ Direction│      │
│  ├──────────────────┼──────────┼────────┼──────────┤      │
│  │ Silhouette       │ 0.4521   │ [-1,1] │ ↑ Higher │      │
│  │ Davies-Bouldin   │ 1.2341   │ [0,∞)  │ ↓ Lower  │      │
│  │ Calinski-Harabasz│ 1523.45  │ [0,∞)  │ ↑ Higher │      │
│  │ ...              │ ...      │ ...    │ ...      │      │
│  └──────────────────┴──────────┴────────┴──────────┘      │
│                                                            │
│  [Export JSON] [Export CSV]                                │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

### Comparison View

```
┌────────────────────────────────────────────────────────────┐
│  Metricate - Compare Clusterings                           │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  📁 Clustering A: [baseline.csv]                           │
│  📁 Clustering B: [improved.csv]                           │
│                                                            │
│  🏆 Winner: Improved (22 wins vs 12)                       │
│                                                            │
│  ┌──────────────────┬──────────┬──────────┬────────┐      │
│  │ Metric           │ Baseline │ Improved │ Winner │      │
│  ├──────────────────┼──────────┼──────────┼────────┤      │
│  │ Silhouette       │ 0.4521   │ 0.5234   │ ✓ B    │      │
│  │ Davies-Bouldin   │ 1.2341   │ 0.9876   │ ✓ B    │      │
│  │ Calinski-Harabasz│ 1523.45  │ 1678.90  │ ✓ B    │      │
│  │ ...              │ ...      │ ...      │ ...    │      │
│  └──────────────────┴──────────┴──────────┴────────┘      │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `METRICATE_PORT` | 5000 | Server port |
| `METRICATE_HOST` | 127.0.0.1 | Server host |
| `METRICATE_DEBUG` | false | Debug mode |
| `METRICATE_MAX_UPLOAD_MB` | 100 | Max upload size |

### Example

```bash
export METRICATE_PORT=8080
export METRICATE_MAX_UPLOAD_MB=500
metricate web
```

---

## Requirements

The web UI requires Flask. When installing from source:

```bash
pip install -e ".[web]"
```

This installs Flask alongside Metricate.

---

## API Endpoints

The web UI exposes REST endpoints that can be used programmatically:

### POST `/api/evaluate`

Evaluate a clustering.

**Request:**
```bash
curl -X POST -F "file=@clustering.csv" http://localhost:5000/api/evaluate
```

**Response:**
```json
{
  "metrics": {
    "Silhouette": 0.4521,
    "Davies-Bouldin": 1.2341,
    ...
  },
  "stats": {
    "n_points": 10000,
    "n_clusters": 17,
    "n_dimensions": 50
  }
}
```

### POST `/api/compare`

Compare two clusterings.

**Request:**
```bash
curl -X POST \
  -F "file_a=@v1.csv" \
  -F "file_b=@v2.csv" \
  http://localhost:5000/api/compare
```

**Response:**
```json
{
  "winner": "B",
  "wins": {"A": 12, "B": 22, "Tie": 0},
  "metrics": {...}
}
```

---

## Deployment

### Development

```bash
metricate web --debug
```

### Production with Gunicorn

```bash
pip install gunicorn

gunicorn "metricate.web:create_app()" \
  --bind 0.0.0.0:8080 \
  --workers 4
```

### Docker

```dockerfile
FROM python:3.11-slim

RUN pip install metricate[web] gunicorn

EXPOSE 8080

CMD ["gunicorn", "metricate.web:create_app()", "--bind", "0.0.0.0:8080"]
```

```bash
docker build -t metricate-web .
docker run -p 8080:8080 metricate-web
```

---

## Troubleshooting

### Port Already in Use

```bash
# Find process using port
lsof -i :5000

# Use different port
metricate web --port 5001
```

### File Upload Fails

Check file size limit:

```bash
# Increase limit
export METRICATE_MAX_UPLOAD_MB=500
metricate web
```

### CORS Issues (when calling API from browser)

For development, run with debug mode which enables CORS:

```bash
metricate web --debug
```

For production, configure your reverse proxy (nginx, etc.) to handle CORS headers.

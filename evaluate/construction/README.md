# Metrix Evaluation Suite

This directory contains the Metrix evaluation code for knowledge graph construction tasks.



---

## Model Evaluation

### 1. Prediction Triple File Structure

Prediction triple files should be organized as follows:

```
EXTRACT_DIR/
├── GenWiki/
│   └── triples.txt
├── CaRB/
│   └── triples.txt
├── KELM-sub/
│   └── triples.txt
├── SCIERC/
│   └── triples.txt
└── webnlg20/
    └── triples.txt
```

**Example:**
```
/path/to/evaluate/dataset/MODEL_NAME/
├── GenWiki/
│   └── triples.txt
├── CaRB/
│   └── triples.txt
└── ...
```

### 2. `eval.sh` Configuration

The `eval.sh` script allows you to specify paths in two ways:

#### Option A: Command-line Arguments

```bash
bash eval.sh <model_name> \
  --base-dir /path/to/dataset \
  --golden-dir /path/to/golden \
  [--log-dir /path/to/logs] \
  [--work-dir /path/to/work]
```

**Arguments:**
- `--base-dir`: **Required**. Directory containing model prediction files (e.g., `/path/to/evaluate/dataset`)
- `--golden-dir`: **Required**. Directory containing golden/ground truth files (e.g., `/path/to/datasets/construction`)
- `--log-dir`: Optional. Directory for log files (default: `evaluate/construction/logs` relative to work directory)
- `--work-dir`: Optional. Working directory (default: auto-detected KGC root directory)

**Example:**
```bash
bash eval.sh KGGEN_c \
  --base-dir /home/user/KGC/evaluate/dataset \
  --golden-dir /home/user/KGC/datasets/construction \
  --log-dir /home/user/KGC/evaluate/construction/logs
```

#### Option B: Environment Variables

You can also set environment variables instead of using command-line arguments:

```bash
export BASE_DIR="/path/to/dataset"
export GOLDEN_DIR="/path/to/golden"
export LOG_DIR="/path/to/logs"  # Optional
export WORK_DIR="/path/to/work"  # Optional

bash eval.sh <model_name>
```

**Example:**
```bash
export BASE_DIR="/home/user/KGC/evaluate/dataset"
export GOLDEN_DIR="/home/user/KGC/datasets/construction"
bash eval.sh KGGEN_c
```

### 3. Dataset Mapping

The dataset mapping in the script is configured as follows:

| Prediction File Directory | Golden File Directory |
|---------------------------|----------------------|
| `GenWiki` | `GenWiki-Hard` |
| `CaRB` | `CaRB-Expert` |
| `KELM-sub` | `kelm_sub` |
| `SCIERC` | `SCIERC` |
| `webnlg20` | `webnlg20` |

You can modify the `DS_MAP` array in `eval.sh` if needed.

### 4. Running the Evaluation

```bash
cd /path/to/KGC/evaluate/construction
bash eval.sh <model_name> --base-dir <base_dir> --golden-dir <golden_dir>
```

The script will automatically run Metrix evaluation for each dataset and save results to log files.

---

## 🔧 Direct Execution

To run evaluation on individual files:

```bash
python3 -m evaluate.construction.main \
  --pred /path/to/pred.txt \
  --gold /path/to/gold.txt \
  [--models metrix] \
  [--log /path/to/log_file] \
  [--analyze-errors] \
  [--error-output-dir /path/to/output]
```

### Argument Description

| Argument | Required | Description |
|----------|----------|-------------|
| `--pred` | ✅ | Path to predicted triples file |
| `--gold` | ✅ | Path to gold triples file |
### Metrix Metrics

Metrix evaluates knowledge graphs using three metrics:

- **G-BLEU**: Graph-based BLEU score (Precision, Recall, F1)
- **G-ROUGE**: Graph-based ROUGE score (Precision, Recall, F1)
- **G-BERTScore**: Graph-based BERTScore (Precision, Recall, F1)

### Example: Direct File Evaluation

```bash
python3 -m evaluate.construction.main \
  --pred /path/to/pred.txt \
  --gold /path/to/gold.txt \
  --log /path/to/log_file
```

### Example: Using Dataset Directory

```bash
python3 -m evaluate.construction.main \
  --dataset /path/to/references \
  --pred pred1.txt \
  --gold g_article.txt
```

If `--dataset` is specified, you can use relative paths for file names.

---

##  Dataset Requirements

### File Format

- `pred.txt` and `gold.txt` files must be **line-by-line Python list strings**.
- Each line represents one sample (a list of triples).

### Important Notes

- **Line Count Match**: The line counts of `pred.txt` and `gold.txt` must match. If they differ, evaluation will produce errors.
-  **Triple Format**: Each triple should be in the format `"entity1 | relation | entity2"` within a Python list string.

### Example File Format

```
["entity1 | relation1 | entity2"]
["entity3 | relation2 | entity4", "entity5 | relation3 | entity6"]
["entity7 | relation4 | entity8"]
```

Each line is a Python list string containing one or more triples separated by ` | `.

---

## Notes

- All evaluations should be run from the project root directory.
- Log files are saved to the specified `LOG_DIR` or default location.
- CSV result files are automatically saved to `evaluate/construction/{model_name}_result.csv`.
- Metrix metrics are computed for each dataset and aggregated in the final CSV output.

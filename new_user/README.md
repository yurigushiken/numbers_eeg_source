# New User Quickstart

**👋 First time here?** Start with [PRE-MEETING_SETUP.md](PRE-MEETING_SETUP.md) to install Git, Cursor IDE, and download the data.

**🖥️ Platform Support:** This pipeline works on Windows, macOS, and Linux. All instructions below include commands for each platform.

| Platform | Quick Start | Shell |
|----------|-------------|-------|
| **Windows** | Double-click `run_analysis.bat` | PowerShell |
| **macOS** | Run `./new_user/run_analysis.sh` | Terminal (bash/zsh) |
| **Linux** | Run `./new_user/run_analysis.sh` | Terminal (bash) |

---

## Run Your First Analysis (1 minute)

### Windows
1. **Open PowerShell** in the project root
2. **Double-click** `new_user/run_analysis.bat`
   *OR run:*
   ```powershell
   conda activate numbers_eeg_source
   python -m code.run_full_analysis_pipeline --config new_user/examples/sensor_13_31.yaml --accuracy all
   ```

### macOS/Linux
1. **Open Terminal** in the project root
2. **Run the shell script:**
   ```bash
   ./new_user/run_analysis.sh
   ```
   *OR run:*
   ```bash
   conda activate numbers_eeg_source
   python -m code.run_full_analysis_pipeline --config new_user/examples/sensor_13_31.yaml --accuracy all
   ```

### What to Expect
**Check results:** `derivatives/sensor/sensor_13_31/` and `derivatives/reports/`

**What just happened?**
The pipeline compared two conditions (1→3 vs 3→1 transitions) and automatically ran sensor analysis, then source analysis.

---

## Create Your Own Analysis (5 minutes)

### Step 1: Copy a Template

**Windows (PowerShell):**
```powershell
cp new_user/templates/sensor_TEMPLATE.yaml new_user/sensor_my_test.yaml
```

**macOS/Linux (Terminal):**
```bash
cp new_user/templates/sensor_TEMPLATE.yaml new_user/sensor_my_test.yaml
```

### Step 2: Edit 3 Lines
Open `sensor_my_test.yaml` and change:

```yaml
# Line 8: Name your analysis (must match filename)
analysis_name: "sensor_my_test"

# Line 22: Set first condition (see code/condition_sets.yaml for options)
condition_set_name: "CARDINALITY_1"

# Line 26: Set second condition
condition_set_name: "CARDINALITY_2"
```

**Where to find condition names:**
Open [code/condition_sets.yaml](../../code/condition_sets.yaml) and look for options like:
- `CARDINALITY_1`, `CARDINALITY_2`, `CARDINALITY_3` (same number twice: 11, 22, 33)
- `DIRECTION_CHANGE` (increasing vs. decreasing)
- `LANDING_ON_2` (trials ending in "2")
- `TRANSITION_13`, `TRANSITION_31` (specific transitions)

### Step 3: Run It

**Windows (PowerShell):**
```powershell
conda activate numbers_eeg_source
python -m code.run_full_analysis_pipeline --config new_user/sensor_my_test.yaml --accuracy all
```

**macOS/Linux (Terminal):**
```bash
conda activate numbers_eeg_source
python -m code.run_full_analysis_pipeline --config new_user/sensor_my_test.yaml --accuracy all
```

---

## Understanding YAML Files

**What is YAML?**
A text file for settings. Think: organized lists and parameters.

**Basic syntax:**
- `key: value` → Sets a parameter
- Indentation matters (use spaces, not tabs)
- `#` = comment (ignored by program)

**Example:**
```yaml
analysis_name: "my_analysis"  # This is the analysis name
stats:
  p_threshold: 0.01           # Indented = nested under "stats"
  tail: 0                     # Two-sided test
```

---

## Which Files Do I Need?

| File | Required? | Purpose | When to Create |
|------|-----------|---------|----------------|
| `sensor_*.yaml` | ✅ Always | Compare two conditions on scalp | Start here |
| `source_dspm_n1_*.yaml` | Optional | Localize N1 (~80-200 ms) | After significant sensor N1 |
| `source_dspm_p1_*.yaml` | Optional | Localize P1 (~60-140 ms) | After significant sensor P1 |
| `source_dspm_p3b_*.yaml` | Optional | Localize P3b (~300-500 ms) | After significant sensor P3b |
| `source_loreta_*_*.yaml` | Optional | Alternative localization method | Compare with dSPM results |

**Pro tip:** The full pipeline auto-discovers source files if they match the sensor filename suffix.

**Example:**
- `sensor_13_31.yaml` → automatically runs `source_dspm_n1_13_31.yaml`, `source_dspm_p1_13_31.yaml`, etc.

---

## Common Parameters to Change

### 1. Analysis Name (Line 8)
```yaml
analysis_name: "sensor_my_analysis"  # Must match your filename (without .yaml)
```

### 2. Conditions (Lines 20-26)
```yaml
contrast:
  condition_A:
    name: "My First Condition"           # Human-readable label
    condition_set_name: "CARDINALITY_1"  # ← CHANGE (see condition_sets.yaml)
  condition_B:
    name: "My Second Condition"
    condition_set_name: "CARDINALITY_2"  # ← CHANGE
```

### 3. Statistical Sensitivity (Lines 33-35)
```yaml
stats:
  p_threshold: 0.01    # Lower = stricter clusters (try 0.05 if no results, 0.001 for very strict)
  tail: 0              # 0 = two-sided, -1 = expect A < B, 1 = expect A > B
```

**When to adjust p_threshold:**
- Start with `0.01` (moderate)
- If no clusters found, try `0.05` (more liberal)
- For planned comparisons, use `0.005` or `0.001` (stricter)

---

## Troubleshooting

**"No significant clusters found"**
→ Increase `p_threshold` to `0.05` in your YAML (line 33)

**"Can't find condition_set_name: XXXXX"**
→ Check spelling in [code/condition_sets.yaml](../../code/condition_sets.yaml) (case-sensitive!)

**"Source files not running automatically"**
→ Ensure naming matches:
- Sensor: `sensor_X.yaml`
- Source: `source_dspm_n1_X.yaml`, `source_dspm_p1_X.yaml` (same `_X` suffix)

**"UnicodeEncodeError on Windows"**
→ Avoid special characters in YAML comments; use standard hyphens

**"Permission denied" on macOS/Linux when running .sh file**
→ Make the script executable: `chmod +x new_user/run_analysis.sh`

**"conda: command not found" on macOS/Linux**
→ Initialize conda for your shell:
```bash
# For bash
conda init bash
# For zsh (macOS default)
conda init zsh
```
Then restart your terminal and try again.

---

## Workflow Tips

### Basic Workflow
1. Create `sensor_*.yaml` (define your comparison)
2. Run full pipeline
3. Check sensor results in `derivatives/sensor/`
4. If significant, source results appear in `derivatives/source/`

### Advanced Workflow (ROI Restriction)
Want more statistical power? Restrict to specific brain regions:

**Sensor space (lines 43-50 in template):**
```yaml
stats:
  roi:
    channel_groups:
      - N1_bilateral      # 14 occipital-temporal channels
      - P1_Oz             # 8 parieto-occipital channels
      - P3b_midline       # 9 centro-parietal channels
```

**Source space (already included in templates):**
```yaml
stats:
  roi:
    parc: aparc
    labels:
      - cuneus
      - lingual
      - lateraloccipital
```

---

## Files in This Folder

```
new_user/
├── README.md                           ← You are here
├── run_analysis.bat                    ← Windows: Double-click to run example
├── run_analysis.sh                     ← macOS/Linux: Run ./new_user/run_analysis.sh
│
├── examples/                           ← Working examples (13_31 analysis)
│   ├── sensor_13_31.yaml
│   ├── source_dspm_n1_13_31.yaml
│   ├── source_dspm_p1_13_31.yaml
│   ├── source_dspm_p3b_13_31.yaml
│   ├── source_loreta_n1_13_31.yaml
│   ├── source_loreta_p1_13_31.yaml
│   └── source_loreta_p3b_13_31.yaml
│
└── templates/                          ← Copy these to create new analyses
    ├── sensor_TEMPLATE.yaml
    ├── source_dspm_n1_TEMPLATE.yaml
    ├── source_dspm_p1_TEMPLATE.yaml
    ├── source_dspm_p3b_TEMPLATE.yaml
    ├── source_loreta_n1_TEMPLATE.yaml
    ├── source_loreta_p1_TEMPLATE.yaml
    └── source_loreta_p3b_TEMPLATE.yaml
```

---

## Next Steps

1. ✅ Run the example analysis (see top of this guide)
2. 📄 Open `examples/sensor_13_31.yaml` to see a working config
3. 📋 Copy a template and customize it
4. 🚀 Run your own analysis
5. 💬 Ask questions! We're here to help.

---

<details>
<summary><b>📖 Advanced: Parameter Reference (Click to Expand)</b></summary>

## Time Windows

```yaml
epoch_window:
  tmin: -0.2              # Start of epoch (200 ms before stimulus)
  tmax: 0.496             # End of epoch (496 ms after stimulus)
  baseline: [-0.2, 0.0]   # Baseline correction window
```

**When to change:** Almost never. These match the preprocessed data.

---

## Statistical Parameters

```yaml
stats:
  analysis_window: [0.00, 0.496]  # Time range to test
                                  # SENSOR: use full epoch
                                  # SOURCE: narrow to ±20 ms around sensor peak

  p_threshold: 0.01               # Cluster-forming threshold
                                  # 0.05 = liberal, 0.01 = moderate, 0.001 = strict

  cluster_alpha: 0.05             # Final significance level (FWER correction)
                                  # Keep at 0.05 (standard)

  n_permutations: 5000            # Number of random shuffles
                                  # 5000 = fast, 10000 = more stable p-values

  tail: 0                         # Direction of effect
                                  # -1 = A < B (e.g., smaller N1)
                                  #  0 = two-sided (no directional hypothesis)
                                  #  1 = A > B (e.g., larger P3b)

  seed: 42                        # Random seed (keep at 42 for reproducibility)
```

---

## Understanding p_threshold vs cluster_alpha

**Two-stage filtering:**

1. **`p_threshold`** (cluster-forming): Which individual time-sensor points can join a cluster?
   - Lower = fewer, more robust clusters
   - Example: `0.001` only groups very strong effects

2. **`cluster_alpha`** (cluster-level): Are the clusters large enough to be significant?
   - This is your final reported p-value
   - Standard: `0.05` (5% false positive rate)

**Analogy:** `p_threshold` = brightness threshold for pixels, `cluster_alpha` = minimum size for regions.

---

## ROI Restriction

**Sensor ROIs** (defined in [configs/sensor_roi_definitions.yaml](../sensor_roi_definitions.yaml)):

| ROI Name | Channels | Use For |
|----------|----------|---------|
| `N1_bilateral` | 14 | Bilateral occipital-temporal (N1: ~140-200 ms) |
| `P1_Oz` | 8 | Parieto-occipital midline (P1: ~80-120 ms) |
| `P3b_midline` | 9 | Centro-parietal midline (P3b: ~300-500 ms) |
| `posterior_visual_parietal` | 25 | Combined (all above regions) |

**Source ROIs** (FreeSurfer parcellation):

| Component | Recommended Labels |
|-----------|-------------------|
| **P1** | `pericalcarine`, `cuneus`, `lingual`, `lateraloccipital`, `fusiform` |
| **N1** | Above + `inferiorparietal`, `superiorparietal`, `precuneus` |
| **P3b** | `inferiorparietal`, `superiorparietal`, `precuneus`, `posteriorcingulate` |

---

## Source Reconstruction Methods

**dSPM vs. eLORETA:**

| Method | Pros | Cons |
|--------|------|------|
| **dSPM** | Well-established, good SNR | Can have localization bias |
| **eLORETA** | Zero localization error (theory) | May be more diffuse |

**Recommendation:** Run both and compare results.

---

## Naming Conventions

**For automatic source discovery to work:**

✅ **Good:**
```
sensor_my_analysis.yaml
source_dspm_n1_my_analysis.yaml
source_dspm_p1_my_analysis.yaml
```

❌ **Bad:**
```
sensor_my_analysis.yaml
source_dspm_n1_different_name.yaml  ← Won't auto-run!
```

**Pattern:** `sensor_SUFFIX.yaml` → `source_METHOD_COMPONENT_SUFFIX.yaml`

</details>

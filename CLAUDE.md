@/mnt/c/Users/sjm57/Desktop/claude_code_test/CLAUDE.md

# Project: biohub-cell-tracking-during-development (Kaggle)

Full competition narrative: `notebook/Overview.ipynb`. This file holds the condensed operational facts needed for day-to-day work.

## Goal
Detect, track, and link cells across time in 3D microscopy data (dense populations, noisy, irregular shapes), including detecting cell divisions and reconstructing lineages.

## Evaluation metric
```
score = adjusted_edge_jaccard + 0.1 * division_jaccard
```
- **Edge Jaccard**: predicted nodes matched to ground-truth nodes per timepoint via optimal bipartite assignment on scaled centroid distance (max 7.0 µm; physical scale z=1.625, y=x=0.40625 µm/voxel). A predicted edge is TP if both endpoints match ground-truth nodes connected by a ground-truth edge. `edge_jaccard = TP / (TP + FP + FN)`, adjusted for over-predicting node count. Weight-averaged across samples by (TP+FP+FN).
- **Division Jaccard**: a division = node with ≥2 outgoing edges. For each ground-truth division, check predicted graph for a connected component covering the pre-split stage that touches both daughter lineages. Micro-averaged across all samples.
- Ground truth is sparsely labeled — metric accounts for this. Scores can exceed 1.0.

## Submission format
CSV named `submission.csv`, one file with node rows and edge rows mixed, grouped by dataset:
```
id,dataset,row_type,node_id,t,z,y,x,source_id,target_id
0,44b6,node,1,0,32,128,128,-1,-1
1,44b6,node,2,1,33,130,125,-1,-1
2,6bba,edge,-1,-1,-1,-1,-1,1,2
```
- Node row: `row_type=node`, integer voxel centroid `(t,z,y,x)`, `source_id`/`target_id` = -1.
- Edge row: `row_type=edge`, `source_id`/`target_id` reference node IDs, `node_id`/`t`/`z`/`y`/`x` = -1.
- `id` = throwaway consecutive index.
- `dataset` must match test-set folder names (without `.zarr`).
- Every test-set dataset must appear.

## Code competition constraints
- Submit via Notebook only (not just a file upload).
- CPU/GPU notebook run-time ≤ 12 hours.
- **No internet access** during submission run.
- Freely/publicly available external data and pretrained models are allowed (must be attached, not downloaded live).
- Output file must be named `submission.csv`.

## Timeline
- Entry/rules deadline & team merger deadline: **2026-09-22**
- Final submission deadline: **2026-09-29**
- (All deadlines 11:59 PM UTC.)

## Data
- Raw data: /mnt/d/KAGGLE/data/biohub-cell-tracking-during-development.zip

## Repo layout
- `notebook/` — exploratory/reference notebooks (`Overview.ipynb` = competition spec, not to be edited).
- (add `src/`, `data/`, `submissions/` etc. here as they're created)

## Notebook write
./notebook
# Team task split and individual evidence

This is a **proposed division of work**, not proof of completed student contributions. The initial notebook and diagnostic artifacts were prepared with the coding assistant. Each member should run, explain and document their own checks before claiming credit. Use real dates, commits and artifacts; do not backfill completion without evidence.

## Meeting-note context (reported, not independently verified)

- Ojas Punje and Rajendra Jetty collaborated on the code and want clearer explanations of mechanics and diagnostics.
- Mitansh Mishra contacted the paper authors for missing settings; no reply is attached here.
- Abhinav Manda reviewed the paper's accessible parameters with AI tools.

## Proposed ownership

| Owner | Technical task | Notebook sections | Reviewable output | Current status |
| --- | --- | --- | --- | --- |
| Ojas Punje | Explain input columns, target, split, numeric encoding and unseen categories; check leakage assertions | 1–4 | Annotated split/encoding CSVs; explanation of one raw-to-encoded row | Proposed; personal verification pending |
| Rajendra Jetty | Explain H/RY/RZ/CNOT, probability sums, gradients, D/G losses and Adam updates | 5–8, 10–11 | Circuit/gradient figure annotations; one hand-checked BCE calculation | Proposed; personal verification pending |
| Mitansh Mishra | Reconcile paper settings and author correspondence; explain which CTGAN comparison is matched | PAPER_SETTINGS.md; 15 | Source-linked settings table and dated author-response log | Proposed; reported email activity requires attached evidence |
| Abhinav Manda | Independently review validation selection, holdout testing, confusion counts, Brier MSE and final prediction | 9, 12–14 | Recomputed metrics and independent review of a prediction row | Proposed; personal verification pending |

## Individual technical log

Add a row for each concrete contribution. A run alone is not enough: explain what was checked and what it established. Pair work can have two contributors with separate descriptions.

| Date | Name | Task/change | Commit or code-cell reference | Output/artifact | Numbers or assertion personally checked | Explanation/findings | Peer reviewer | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| To fill | To fill | To fill | To fill | To fill | To fill | To fill | To fill | Not started |

## Review handoff

1. Run the notebook top-to-bottom, or use the bundled Python runner.
2. Confirm all assertions pass and inspect the saved demonstration manifest.
3. Explain which plots reveal success or failure; do not equate nonzero gradients with good generated data.
4. Add personal evidence above and ask another member to review it.
5. Share the notebook, `outputs/demo/` and the two notes files through the team's normal review process. This folder creation does not email, publish or push them anywhere.

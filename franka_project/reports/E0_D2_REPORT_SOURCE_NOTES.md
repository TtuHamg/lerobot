# E0-D2 report source notes

- Audience: technical.
- Selected delivery mode: portable self-contained HTML generated from the canonical artifact JSON.
- Delivery blocker: the required packaged builder could not be executed because this host has no `node`, `nodejs`, `npm`, or `npx` runtime. No separate one-off HTML renderer was substituted. The validated-by-Python canonical artifact JSON and the stage Markdown reports remain available for review.
- Decision supported: whether the project can advance from the approved E0-D2 batch to D3.
- Report state: partial, because 3 of the 25 manifest episodes lack raw MCAP files.
- Denominators: raw coverage uses 25 manifest episodes; alignment metrics use the 22 readable episodes; D2 numerical acceptance uses two named readable episodes and 824 valid full-horizon anchors.
- Required-structure mapping: title, technical summary, key findings, scope/definitions, methodology, limitations/robustness, recommended next steps, and further questions are all visible. Scope is moved before detailed evidence so denominators are defined before the reader uses them.
- Visualization omission: no chart is included. The evidence consists of discrete phase gates, exact missing IDs, threshold checks, and two-profile numerical acceptance values; exact audit tables are more decision-useful than a trend or distribution chart for this small gate review.
- The HTML report is a summary surface. Full per-episode metrics, hashes, source timestamps, and statistics remain in the linked project-local JSON/Markdown supporting artifacts.

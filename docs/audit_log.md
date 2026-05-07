---
title: "Audit log — what we tried, what we rejected, what each rejection taught us"
subtitle: "The brief asked for at least one failure case you understand. Here is the full list."
---

The submission brief explicitly welcomes documented failure: *"we want to see your thinking, not just your final pipeline"* and *"at least one failure case you understand."* This document lists every approach, dataset, and window we surveyed during the project, what we observed, why we rejected it, and the rule each rejection codified. The work is the work; the rejected experiments are part of the work.

The list is grouped by category. Each entry follows the same shape: **what we tried → what we observed → why we rejected → the rule.**

---

## Algorithmic experiments

### Synthetic priors from past frames

**What we tried.** Use the previous N snowy frames as additional priors for the current frame's matching pass, alongside the K summer priors. Reasoning: snow→snow matching has identical lighting / lens / viewpoint conditions, so DISK + LightGlue return many more correspondences per pair than snow→summer.

**What we observed.** Inlier counts roughly tripled per pair on a single-frame still. The fused mask was visibly broader and more confident in static-frame inspection.

**Why we rejected.** In motion, the failure mode was a positive feedback loop the static stills hid. Each frame's slightly-too-large mask seeded the next frame's synthetic prior, which produced a slightly-larger mask, and the road overlay drifted outward into bushes and treelines over five to ten seconds. The matcher's higher confidence was misleading — it was matching to an *increasingly wrong* mask, and the loop reinforced its own error. The code remains available behind `--synthetic-priors N` for inspection; the default is 0.

**The rule.** *Never declare a video result a win from sampled stills.* Render the clip, watch it in motion, then decide. Codified in `feedback_phase_k_motion_judgement.md`.

### Optical-flow propagation between matched keyframes

**What we tried.** Match snow→summer on every Nth frame; use dense optical flow to propagate the resulting road mask between matched keyframes. Reasoning: matching is the dominant cost, so reducing match frequency would speed the pipeline up significantly.

**What we observed.** Per-frame the propagated mask looked plausible. In motion, the mask drifted *outward* from frame to frame.

**Why we rejected.** A forward-driving camera has vanishing-point flow: distant pixels move slower than near pixels, and the flow vectors stretch away from the centre. The mask's outer boundary advanced faster than its inner boundary on every step, broadening the road region until it engulfed sidewalks and yards. Same shape as the synthetic-priors failure (positive feedback loop in motion that single-frame inspection hides), different mechanism.

**The rule.** Same rule as above — verify in motion. Compounding failures need multi-frame evidence.

### EMA over the binary mask, α = 0.4 — kept

**What we tried.** Exponentially-weighted moving average on the binary mask between frames.

**What we observed.** Drops jitter without drifting. On a frame whose matcher fails entirely, the smoothed mask is held at the previous frame's value rather than flickering empty.

**Why we kept it.** Simplest possible smoother. Does the least damage. The opposite of synth priors and flow propagation: fails *gracefully* rather than catastrophically, because it does not recursively feed back into the matcher.

---

## Dataset / scene candidates

### Boreas `boreas_2024_12_23` (alt 1) — retired

**What we tried.** Use a different snow drive of the same Glen Shields Toronto loop as a "second scene" alongside the canonical `boreas_2021_01_26`.

**What we observed.** Window selection picked snow-frame indices `[800..1150]`. Roughly 75% of the windowed snow frames sat *outside the summer trajectory's spatial coverage* — those frames had zero usable priors by construction.

**Why we rejected.** Two reasons: (1) the snow drive is on the same physical Glen Shields loop as the canonical, so it would not have demonstrated *scene* variety; (2) the window was structurally broken — the matcher could not have succeeded on most of those frames regardless of how good the matching was.

**The rule.** *Never start a cache build without verifying spatial overlap + summer-segmentation quality first.* Codified in `feedback_phase_l_window_oracle.md` and the `make oracle TRACK=<id>` target.

### Boreas `boreas_2025_02_15` (alt 2) — kept as robustness clip, with caveats

**What we tried.** Re-window using the oracle. Same Glen Shields loop, different snowfall (active snow, late afternoon, different traffic).

**What we observed.** First cache build ran cleanly but produced a visibly jumpier overlay than the canonical — fused road coverage averaged wider on this window, and 1-or-2-prior-only frames showed those wider masks without the multi-prior averaging that smooths the canonical clip. The wider coverage traces to Mask2Former's *correct* identification of road pixels on this window — the summer subset includes wider intersections and parking-lot-adjacent surfaces, and "Cityscapes road" is a broader concept than "the lane I am driving on".

**Why we kept it.** The clip is honest evidence of *robustness across snow conditions* on the same intersection — same pipeline, same parameters, different weather and time of day. Not a different scene; a different snowfall. We frame it that way explicitly in the writeup and on Pages.

**The rule.** *Frame each clip honestly for what it is.* "Robustness clip" is not "different scene". The static-stills 27-pair set carries the scene-variety claim; the second video clip carries the snow-variety claim.

### Tromsø, Norway — Mapillary 400-frame winter sequence (audited, rejected)

**What we tried.** A 400-frame sequence from contributor "southglos" in Tromsø, Norway, identified by `src/data/find_snow_sequences.py` during the broader Mapillary recon. Geographic novelty (Norway, not Toronto), cadence within range, non-pano, GPS metadata present.

**What we observed.** A smoke fetch (30 frames + per-frame Mapillary closeto summer priors) passed every numeric check — inlier counts looked healthy, oracle gave the candidate window a green light. *On closer inspection of the actual snow frames*, the imagery was shot **out the side of a moving bus**. The matcher was anchoring on whatever the bus happened to drive past — none of which was the road in front of any forward-facing camera. The numeric checks did not detect this because the matcher's job was satisfied per-frame; the *use case* was wrong.

**Why we rejected.** A pipeline that says where the road is in front of a forward-facing camera cannot be demonstrated on a sideways-facing bus camera. The geometric correspondence was real; the *interpretation* would have been nonsense.

**The rule.** *Never commit cache compute without auditing the actual frames.* Numbers passing oracle checks is a necessary but not sufficient condition. The user must look at the frames before any 50+ minutes of matcher compute is committed. Codified in `feedback_phase_l_window_oracle.md`.

### Broader Mapillary winter recon (19 cities) — null result

**What we tried.** Extended `src/data/find_snow_sequences.py` from 7 to 19 cold-climate city bboxes (Tromsø, Reykjavík, Bergen, Quebec, Innsbruck, Sapporo, Kiruna, plus added: Trondheim, Helsinki, Anchorage, Saskatoon, Winnipeg, Longyearbyen, Yellowknife, Tórshavn, plus widened bboxes for Kiruna / Bergen / Reykjavík / Sapporo). Two recent winter date ranges. Ranked candidate sequences by frame count, cadence, non-pano, and feature density.

**What we observed.** Out of 19 cities × 2 winter ranges × bbox queries, the only non-trivial returns were Tromsø (the rejected bus sequence) and Sapporo (a 2-image sequence — useless for a video demo).

**Why we rejected.** The negative finding is structural: Mapillary's contributor uploads in cold-climate winters are rare. Most contributors capture during clear weather. Cross-referencing to find both winter and summer captures in the same physical place is rarer still. Programmatic recon over a week of bbox queries cannot manufacture data that isn't there.

**The rule.** *Mapillary's bottleneck for our use case is the winter side, not the summer side.* The substrate of the *clear* prior is interchangeable; the substrate of the *snow* input is what scarcity-bound. Future demo-data search should go to academic snow datasets (CADC, MUSES, etc.) and treat Mapillary as the summer-prior source only.

### ACDC dataset (ETH Zurich) — access friction

**What we tried.** Fetch ACDC's snow split per `docs/external_datasets.md` for a non-Boreas physical scene (Zurich, urban Europe).

**What we observed.** The dataset sits behind a Vue.js single-page-app benchmark portal at `acdc.vision.ee.ethz.ch`. Files are admin-uploaded "packages" rather than statically-named zips. Sign-up + per-package access requests + admin grants. The user logged into the portal and reported that no obviously-appropriate snow package was visible on inspection — either the package list requires further role-grants, or the naming was different from what was documented in the paper / the public abstract.

**Why we rejected (for this submission window).** The portal access dance is incompatible with our 3-day deadline. We do not have enough friction-budget to chase access grants across multiple datasets.

**The rule.** *Prefer datasets with direct public download URLs over benchmark-portal datasets when the deadline is tight.* CADC (Canadian Adverse Driving Conditions, U Waterloo) is the cleanest fallback for this submission cycle — direct download, no portal — and we documented the integration path for it.

### MUSES dataset (ETH Zurich) — same access friction as ACDC

**What we tried.** A fallback within the same group as ACDC.

**Why we rejected.** Same portal architecture as ACDC; same friction. Documented in `docs/external_datasets.md` as a "candidate but not pursued in this cycle."

### CADC, BDD100K, DENSE / SeeingThroughFog, A2D2, nuScenes, Argoverse 2 — surveyed, not pulled

**What we tried.** A research-permissive academic-dataset survey (in `docs/external_datasets.md`) covering 6 candidate sources beyond Mapillary.

**Why we did not pull within the deadline.** Each requires a sign-up + manual download + dataset-specific integration. CADC is the cleanest path (direct URL, no portal); the others are larger or higher-friction. CADC's integration is documented as plug-and-play once the user grabs the data.

---

## Window / data-windowing experiments

### Synthesised second snow scene without an oracle pass — burned compute

**What we tried.** Initial alt-track work picked a 350-frame window for `boreas_2024_12_23` based on what looked like reasonable snow content + GPS coordinates within the summer track's bounds.

**What we observed.** The window selection did not verify per-frame summer coverage. Approximately 75% of the windowed snow frames lay outside the summer trajectory's spatial coverage. The cache build burned roughly 90 minutes of compute on a window where the matcher could not have succeeded.

**Why we rejected.** As above (`boreas_2024_12_23` retired).

**The rule.** *No cache build starts without `make oracle TRACK=<id>` passing first.* Codified in the `oracle` Makefile target and `src/video_runtime/window_oracle.py`.

---

## Engineering experiments

### Mask2Former argmax → softmax-threshold (kept as opt-in)

**What we tried.** Replace argmax-over-classes with "keep road class only where its softmax probability exceeds a threshold." Plus optional morphological cleanup.

**What we observed.** On the boreas_2025_02_15 segmentation over-claim, the threshold barely shifted total road-coverage (~1% absolute change across thresholds 0.3 / 0.5 / 0.7). The morphological cleanup did suppress edge jaggies independently of total coverage.

**Why we kept it as opt-in (default off).** The threshold is the right tool when over-claim is *uncertainty-driven*. On boreas_2025_02_15 the over-claim is *structural* — Mask2Former is confident about the road being where it claims; the issue is "Cityscapes road" being a broader semantic concept than "the lane I am driving on" in this window's geometry. Threshold helps less than expected, but it also costs nothing when off, and the morph cleanup is genuinely useful for edge stability. Plumbed through the CLI as `--seg-prob-threshold` and `--seg-morph-radius`; defaults are off.

**The rule.** *Tools are cheap; defaults are expensive.* Add the knob, leave the default at the canonical-known-good. Future tracks may want different settings.

---

## What this audit log is for

Three audiences use this list:

- **A judge** scanning for evidence of intellectual honesty about what the work *cannot* do, what we *did not bother to do*, and what we *tried and abandoned with reasons*. The brief explicitly looks for this.
- **A future contributor** picking up the project and avoiding the same dead ends — every entry pairs an observation with a rule.
- **The author** keeping their own work honest. A submission that doesn't list its rejected approaches is hiding the work.

If you find an entry here that you think we mis-classified — that we rejected something we should have kept, or kept something we should have rejected — the writeup acknowledgements list a contact. The list above is the work; we are interested in being told where it is wrong.

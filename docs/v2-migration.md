# V2 migration and claim withdrawal

## Checkpoint

Commit `66554e9` on `origin/master` is the immutable pre-change checkpoint. Recommendation Engine V2 is developed on `agent/recommendation-engine-v2`.

## Historical evidence classification

All strategy results and the 2024–2026 period are classified as previously accessed. They are not never-seen confirmation or V2 promotion evidence. Historical scripts/reports remain provenance for ideas and falsifications, but their selection histories may be incomplete.

Legacy `EdgeStatus.VALIDATED` is a compatibility/database lifecycle label. It does not mean `PromotionDecisionV2.promoted=true`. Legacy artifacts are promotion-incompatible and stock names can enter V2 only as zero-weight watchlist entries until frozen prospective requirements are met.

## Interface migration

| Legacy surface | V2 behavior |
|---|---|
| `/board` | deprecated; `rows=[]`, zero-weight `watchlist_v2` |
| `/picks` | deprecated; `picks=[]`, zero-weight `watchlist_v2` |
| `/master` | deprecated; canonical target weights only |
| `/signals/*`, `/candidates/*` | deprecated; no legacy actionable candidates |
| independent JSON files | compatibility files inside the current atomic run |
| on-device overlay/ensemble | removed; server bundle/preview only |
| signal-driven paper entries | removed; canonical target-weight orders only |

The compatibility release advertises `Deprecation`, `Sunset`, and successor links. New consumers should use `GET /recommendations/latest` and stateless `POST /recommendations/preview`.

## State migration

The Android `base_leverage` setting migrates once to `maximum_gross_leverage`, clamped to `[0,5]`. Other V2 risk fields receive documented defaults. Android stores the returned risk state and uses it on later previews.

The paper account no longer reads `artifacts/paper/state.json`. Canonical state lives inside the content-addressed run selected by `current.json`. The first V2 run initializes cash and queues the canonical target; later nightly runs execute the prior next-open target, record fills/cash flows, and queue the new target atomically.

## Operating migration

`scripts/nightly.bat` is a thin launcher for `scripts/nightly.py`. The old board/picks/master scripts now export compatibility projections from the verified bundle and cannot calculate independent signals. Current-snapshot fundamentals and calendar overlays are not inputs to V2 scoring or execution.

## Prospective clock

The prospective shadow clock starts with the first published frozen V2 bundle. A stock sleeve needs both 252 distinct prospective sessions and date-aggregated ESS of at least 100, in addition to all nested promotion gates. No backfill from 2024–2026 is allowed.

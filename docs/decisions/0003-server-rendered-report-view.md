# ADR 0003 — The report view is server-rendered HTML, not a Next.js app

**Status:** accepted, 2026-07-31
**Supersedes:** the plan's M7 line, "Next.js: single report page"

## Context

M7 needs a page. The plan named Next.js, and `frontend/` has existed as an empty directory
since M0. What the plan actually requires of the page does not mention a framework:

> Next.js: single report page — summary, findings with inline citations, charts, confidence,
> risks, recommendations, and an expandable tool-call/sources trace. **Every claim visibly
> clickable to its evidence.**

That last sentence is the product's whole argument rendered as a UI. Everything else on the
list is layout.

## Decision

The page is served by the gateway as HTML (`services/gateway/views.py`), with charts as inline
SVG (`cortex/reports/svg.py`). No build step, no `node_modules`, no JavaScript, no second
deploy target.

The JSON endpoints stay the contract. `GET /investigations`, `GET /investigations/{id}`,
`GET /investigations/{id}/trace` and `GET /reports/{id}` are complete and tested independently
of the HTML, so a Next.js app can be built against them later without touching any of this.

## Why

**One page does not amortise a toolchain.** A Next.js app brings a build, a dependency tree,
a second runtime to deploy and monitor, and a second place where a claim can be rendered. The
last one is not a cost in principle — it is the specific risk this codebase spends most of its
effort on. Two renderers means two places where a citation can be dropped, and the one that
drops it is whichever was edited second.

**The interactivity budget is genuinely small.** A report is a document. The only live
behaviour needed is "show me what it has done so far while it runs", and that is a
`<meta refresh>` that stops when the investigation reaches a terminal status — five characters
of markup against a framework. No websocket, no client state, no hydration.

**No JavaScript is a feature here, not a limitation.** A report that renders without scripting
can be saved, printed, emailed, and read in a text browser. For a document whose purpose is to
be *checked*, that matters more than any interaction a script would add.

**Inline SVG for charts, and it is the same argument.** A `<svg>` element built from a
validated `ChartSpec` needs no library and survives being saved with the page. The specs are
small — a few series over a few dozen points — which is the size at which a charting library
costs more than it saves. The honesty rules move across with the renderer rather than being
re-derived: a `None` is a gap and must break the line, and an annotation must land at the x it
names. Both have tests, and the first exists because the ASCII renderer got it wrong once.

## What this is not

Not a claim that a React frontend is wrong for Cortex. It is the right answer for the
three-panel shell the plan defers — conversation, live steps, artifacts — which is genuinely
interactive and genuinely stateful. This ADR says that shell is not what M7 is, and that
building its toolchain to ship one document is the wrong order.

**Trigger to revisit:** the first requirement that is actually interactive. Filtering a list
client-side, editing a question in place, streaming steps without a page reload, or a chart the
reader manipulates. At that point the JSON endpoints are already there, which is the point of
keeping them the contract.

## Consequences

- Claim text, finding titles, chart titles and tool errors are all model or vendor output
  reaching a page, so everything interpolated is escaped and there is a test asserting a
  hostile claim cannot become markup. "We escape" stays true until someone adds a field.
- The trace is derived from the `tool_calls` audit rows, so the page cannot show a history that
  disagrees with what happened. See ADR 0002, round two.
- `cortex.ask` had to start persisting a `Report` row. It never did — only the worker service
  wrote one — so every real investigation ever run through the CLI printed itself to a terminal
  and left nothing behind. Invisible until there was a page to notice it.
- The `/ui` routes are excluded from the OpenAPI document. The page is a rendering of the
  endpoints, not a second API, and listing it as one would invite a client to scrape HTML for
  data the JSON already carries.

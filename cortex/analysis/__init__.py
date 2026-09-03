"""Statistical machinery for describing a metric movement, and for refusing to explain one.

Three modules, in the order they run and the order ADR 0005 derives them:

- `series` turns a query result into a calendar, because a `GROUP BY date` cannot represent
  a day that is missing and every detector downstream is blind to what it cannot see.
- `changepoints` says *when* the series moved.
- `conformal` says whether that movement is distinguishable from noise -- the only
  significance statement available to us, given that we have no control series.

**Stdlib only, deliberately.** Exact optimal partitioning with a pure-Python inner loop was
measured at 2.2 ms for n=90 and 41 ms for a year, so numpy
buys nothing here and would add a declared dependency to a project that currently gets it
only transitively through matplotlib.

None of this licenses the word "because". See `cortex/analysis/identifiability.py` for what
it takes to say that, and `docs/decisions/0005-investigation-method.md` for why the answer is
usually "we cannot".
"""

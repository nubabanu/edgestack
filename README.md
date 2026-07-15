# EdgeStack

EdgeStack is a research platform that discovers, validates, scores, monitors and
ranks statistical trading advantages ("edges") in daily US equities — and abstains
when the evidence is insufficient.

> **Research / paper trading only.** EdgeStack produces research output, not
> investment advice. No signal is guaranteed, certain, or risk-free. The bundled
> free data sources carry survivorship bias and other limitations that are flagged
> in every report. There is no live-trading implementation in this repository.

*(Full documentation is written alongside the implementation milestones; see
`docs/` as it fills in.)*

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -e .[dev]
edgestack --help
edgestack config-show
pytest
```

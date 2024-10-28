# nodc_calculations

This repository stores functions to calculate physical, chemical and biological oceanographic parameters.

The primary aim with this is to have a common point for parameters that are calculated and used in differenct projects within the Swedish nodc at SMHI.

In addition to oceanographic parameters the repository also contains functions for certain commonly used conversions.

## ruff

För att linta och formatera koden används `ruff`.

### Lintning

Linta koden med följande kommando:

```bash
$ ruff check
```

Kontrollerar att koden uppfyller konfigurerade linting-regler. Konfigurationen finns i `pyproject.yaml` under sektionen
`[tool.ruff.lint]`. Vissa identifierade problem kan `ruff` åtgärda själv. Detta görs med flaggan `--fix`.

```bash
$ ruff check --fix
```

Information om alla regler finns här:

- https://docs.astral.sh/ruff/rules/

### Formatering

Formatera koden med följande kommando:

```bash
$ ruff format
```

## pre-commit
För att hantera pre-commit-hook för git används verktyget `pre-commit`. Verktyget installeras som en del av
dev-dependencies men för att aktivera det behöver man skriva följande kommando:

```bash
$ pre-commit install
```

Efter aktivering kommer en commit att avbrytas om inte en serie av kontroller går igenom. Vilka kontroller som ingår
styrs av konfigurationen i `.pre-commit-config.yaml`.

För att kringgå kontroller (t.ex. vid commit till en topic branch) kan man lägga till flaggan `--no-verify` när man
comittar.

```bash
$ git commit --no-verify
```


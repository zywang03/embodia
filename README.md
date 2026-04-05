# embodia

`embodia` is a minimal Python library for unified runtime interfaces between
robots and models.

It is intentionally small:

- no network service
- no server process
- no ROS dependency
- no training framework
- no plugin system

The core design is:

1. `Protocol` defines compatibility for third-party implementations.
2. `Base` provides a recommended inheritance path for official implementations.
3. `check_*` functions are the real runtime acceptance gates.

## Install

For local development:

```bash
pip install .
```

## Quick Start

Run the repository examples after installing the package:

```bash
pip install -e .
python examples/basic_usage.py
python examples/structural_compatibility.py
```

## Package Layout

```text
embodia/
  pyproject.toml
  README.md
  examples/
    basic_usage.py
    structural_compatibility.py
  src/embodia/
    __init__.py
    __main__.py
    core/
      schema.py
      protocols.py
      base.py
    runtime/
      checks.py
  tests/
    helpers.py
    test_interfaces.py
```

## Philosophy

- `Protocol`: external compatibility standard
- `Base`: official recommended parent class
- `check_*`: runtime validation and acceptance entrypoints

This makes it easy to wrap real robot SDKs and existing model classes without
forcing them to inherit from a framework base class.

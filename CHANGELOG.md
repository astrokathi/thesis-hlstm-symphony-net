# Changelog

## [2026-05-04] - Base Setup & Constant Extraction
- Created `CLAUDE.md` to provide guidance for future Claude sessions.
- Extracted hardcoded constants from `model.py`, `train.py`, `preprocessing.py`, and `experimentation.py` into a `.env` file.
- Implemented `config.py` to manage environment variables and provide a centralized configuration for the project.
- Updated codebase to use `Config` class instead of hardcoded values.

# Project Context

## Purpose
This is an IoT (Internet of Things) project focused on implementing and demonstrating IoT concepts and applications as part of a homework assignment (HW3). The project aims to create practical IoT solutions while maintaining good software engineering practices.

## Tech Stack
- Python (for IoT device programming)
- MQTT (for IoT communication)
- Potential hardware components (sensors, actuators)
- Git (version control)

## Project Conventions

### Code Style
- Python code follows PEP 8 style guide
- Use snake_case for functions and variables
- Use CamelCase for class names
- Clear, descriptive variable and function names
- Comments for complex logic and hardware interactions

### Architecture Patterns
- Publisher-Subscriber pattern for MQTT communication
- Modular design with separate components for:
  - Sensor interfaces
  - Data processing
  - Communication handling
  - Configuration management

### Testing Strategy
- Unit tests for individual components
- Integration tests for MQTT communication
- Hardware simulation tests where applicable
- Test coverage for critical paths

### Git Workflow
- Main branch protection
- Feature branches for new capabilities
- Pull request reviews required
- Conventional commits (feat:, fix:, docs:, etc.)

## Domain Context
- IoT device communication patterns and protocols
- Sensor data collection and processing
- Real-time data handling and monitoring
- Device state management
- Hardware-software interface considerations

## Important Constraints
- Hardware compatibility requirements
- Real-time processing needs
- Network reliability considerations
- Power consumption optimization
- Data security and privacy requirements

## External Dependencies
- MQTT broker service
- Hardware device specifications
- Python package dependencies
- Development tools and libraries

## ML Spam Classification (subproject)

- Location: `src/`, `models/`, `datasets/processed/`
- Purpose: Train and evaluate a spam/ham text classifier for SMS/email messages. This repository includes a baseline Logistic Regression classifier, preprocessing utilities, and training/tuning scripts.

### Recommended environment
- Create a conda env (recommended) and install dependencies from `requirements.txt`.

### Reproducible commands (quickstart)
1. Create environment and install dependencies:

```powershell
conda create -p .conda python=3.9 pandas numpy scikit-learn requests joblib -y
conda activate ./.conda
pip install -r requirements.txt
```

2. Run preprocessing + training (this will download the raw dataset, produce a cleaned CSV, train the tuned LogisticRegression model, and save artifacts under `models/`):

```powershell
# Run the training script which uses the recommended/tuned settings
python src/train.py
```

3. Test inference with the quick harness:

```powershell
python src/test.py
```

### Recommended training flags (final / reproduced)
- class_weight: balanced
- ngram_range: (1,2)
- min_df: 2
- sublinear_tf: True
- TF-IDF max_features: configurable (default 5000)
- LogisticRegression: penalty='l1', C=10.0, solver='liblinear', class_weight='balanced'

### Artifacts produced
- `models/spam_classifier.joblib` — trained LogisticRegression model (best params)
- `models/tfidf_vectorizer.joblib` — fitted TF-IDF vectorizer
- `datasets/processed/sms_spam_clean.csv` — cleaned dataset used for training

### How to reproduce or retune
- To retune hyperparameters, see `src/tune.py` which runs GridSearchCV (scoring='recall') over penalty and C. Adjust the `param_grid` to expand the search (e.g., include solver='saga' and elasticnet penalty).
- To change the TF-IDF preprocessing settings, modify `src/data_processor.py` constructor parameters (ngram_range, min_df, sublinear_tf, max_features).

### Notes
- The tuning focused on improving recall (reduce false negatives). If you need a different precision/recall trade-off, re-run tuning with a different scoring objective or apply a threshold sweep in evaluation.


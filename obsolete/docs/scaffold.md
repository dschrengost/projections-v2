***Structuring a Maintainable ML Project for NBA Minutes Prediction***

Developing a machine learning project as a solo developer requires a balance between simplicity and scalability. This guide presents a comprehensive project layout for an NBA player minutes prediction project, supporting both classical ML (e.g. XGBoost) and deep learning (e.g. LSTM) workflows. The structure emphasizes modularity, reproducibility, and ease of maintenance without overengineering. All code and data are organized logically so you can experiment, train models, and evaluate results in a consistent environment.


Project Directory Layout
A clear directory structure is the foundation of maintainability. We separate concerns into folders for data, code, notebooks, tests, etc., following patterns recommended by the data science community
kdnuggets.com
neptune.ai
. For example, a project named nba_minutes might be organized as:
nba_minutes/ 
├── data/                  # Data storage (see subfolders below)
│   ├── raw/               # Original, immutable data dumps (e.g. initial CSVs)
│   ├── external/          # Data from third-party sources (if any)
│   ├── interim/           # Intermediate data (after cleaning/merging, before features)
│   └── processed/         # Final data ready for modeling (features, etc.)
├── notebooks/             # Jupyter notebooks for exploration and prototyping
├── models/                # Trained model artifacts, predictions, or model outputs
├── config/                # Configuration files (hyperparameters, file paths, settings)
├── nba_minutes/           # Source code (organized as a Python package)
│   ├── __init__.py        # Makes this a Python package
│   ├── data.py            # Data loading and initial preprocessing code (optional subpackage)
│   ├── features.py        # Feature engineering code (from processed data to model inputs)
│   ├── models/            # Package for model definitions and training routines
│   │   ├── __init__.py
│   │   ├── classical.py   # Code for classical ML models (e.g. XGBoost training, inference)
│   │   └── deep.py        # Code for deep learning models (e.g. LSTM architecture, training loop)
│   ├── train.py           # Pipeline script to train models (or multiple scripts per model type)
│   ├── evaluate.py        # Script or module for evaluation metrics and model performance evaluation
│   └── utils.py           # Utility functions (logging, helpers, etc.)
├── tests/                 # Unit tests for your code 
│   ├── test_data.py       # e.g. tests for data loading and preprocessing functions
│   ├── test_features.py   # tests for feature engineering functions 
│   ├── test_models.py     # tests for model training/inference (using small subsets or mocks)
│   └── ...                # additional tests for utils, evaluation, etc.
├── README.md              # Top-level project documentation
├── pyproject.toml         # Project configuration and dependencies (managed by `uv`)
└── uv.lock                # Lockfile for exact dependency versions (ensure reproducible env)
About this structure: This layout is adapted from standard templates like Cookiecutter Data Science and reflects common practice
kdnuggets.com
cookiecutter-data-science.drivendata.org
. Key directories and their roles:
data/: Contains your datasets, divided into subfolders:
raw/: Original data dumps (e.g., the untouched CSVs of historical NBA stats). Treat these as read-only – do not manually alter files here, to preserve truth sources
neptune.ai
.
external/: Data from third-party sources (if any). For example, data pulled from an API or provided by others.
interim/: Intermediate data that has been transformed from raw/external sources. For instance, after cleaning or merging multiple raw files, you might save the result here. This allows you to reuse expensive preprocessing results.
processed/: Final data ready for modeling – e.g., feature matrices or engineered datasets that your model training code will consume
neptune.ai
. By separating processed data, you ensure the model sees a consistent, prepared dataset.
notebooks/: Jupyter notebooks for exploration, EDA, and experimentation. Use these for trying out ideas, visualizing data, and developing prototypes. It’s good practice to keep notebooks ordered and annotated. A naming convention (like 1.0-initial-exploration.ipynb, 2.0-feature-engineering.ipynb, etc.) helps convey their purpose and execution order
cookiecutter-data-science.drivendata.org
. Since notebooks are less reproducible, any important logic from notebooks should eventually be moved into the structured code base (the nba_minutes/ package) for longevity.
models/: Store serialized model files, predictions, and model evaluations here. When you train a model (whether a scikit-learn model or a deep learning model), save the trained model (e.g., a .pkl file for XGBoost or a .pt/.h5 file for PyTorch/Keras) in this directory. Saving models allows you to reload them for inference or ensemble later
neptune.ai
. You might also keep prediction outputs or model summary files here for reference. (This directory is typically listed in .gitignore so large model files don’t bloat your repository.)
config/: Configuration files for your project. Keeping configuration separate from code is a best practice
kdnuggets.com
. This can include a YAML/TOML/JSON file specifying hyperparameters, file paths, or any settings that might change. For example, a config.yaml might define training hyperparameters and data paths instead of hard-coding them in your scripts:
# config/settings.yaml
data_path: "data/raw/nba_stats.csv"
model_type: "xgboost"
model_params:
  learning_rate: 0.01
  max_depth: 6
  lstm_hidden_size: 128
  lstm_layers: 2
Using a config file makes experiments easier to reproduce or adjust, since you can tweak settings without modifying code
kdnuggets.com
kdnuggets.com
. You can load these in Python (e.g. with yaml.safe_load) and pass the config values into your pipeline.
nba_minutes/ (source code): All reusable source code lives here (you can also name this folder something like src/ or your project name). It’s organized as a Python package so you can import your modules across the project. Key components inside might include:
Data loading & preprocessing – You might have a module like data.py (or a subpackage data/) with functions to load raw data and perform initial cleaning (handling missing values, data type conversions, etc.). For complex projects, break this down further (e.g., separate modules for load_data, clean_data). This separation makes it easy to test and reuse data logic.
Feature engineering – A module (or package) responsible for transforming cleaned data into model-ready features. For example, a features.py might contain functions to create new features (like rolling averages of minutes, player encodings, etc.)
cookiecutter-data-science.drivendata.org
. Classical ML and DL can often share feature code – ensure these functions are flexible (e.g., output NumPy arrays/pandas DataFrames usable by either scikit-learn or PyTorch).
Models & training – Code for defining and training models:
For classical ML models, you might not need to write model code from scratch (using libraries like scikit-learn, XGBoost, LightGBM). Instead, focus on training routines: e.g., a train.py (or train_model.py) that loads processed data, sets up an XGBoost regressor, and fits it. If you have multiple classical algorithms to try, you can either parameterize train.py by model type or have separate scripts (e.g., train_xgboost.py, train_random_forest.py).
For deep learning models, create a clear separation between model definition and training loop. For instance, a file models/deep.py could define an LSTM model class (using PyTorch or TensorFlow), and a training function that loops over epochs. Alternatively, you can structure this with an object-oriented approach: e.g., a MinutesLSTMModel class in models/deep.py encapsulating the architecture, and a function or class method to train it. Leverage frameworks if it simplifies your code – for example, using PyTorch Lightning can abstract the training boilerplate, but even then you would keep Lightning modules in this models/ folder.
You can also have a modeling pipeline module that chooses between classical or deep models based on configuration. For example, nba_minutes/train.py could read config.yaml, then call either train_classical_model(data, params) or train_deep_model(data, params) accordingly. This keeps the command-line interface simple (one entry point to train any model type), while dispatching to the appropriate implementation internally.
Include a predict.py or inference function either in the model modules or a separate file, to generate predictions using trained models
cookiecutter-data-science.drivendata.org
. This might load the model artifact from models/ directory and run it on new input data. Keeping prediction logic separate from training allows easier evaluation and potential deployment.
Evaluation & metrics – A module like evaluate.py can hold functions to evaluate model performance: e.g., functions to compute metrics (MAE, RMSE, etc.), and perhaps a routine to compare model predictions with ground truth and output a report or plots. If using notebooks for analysis, this module ensures you don’t rewrite evaluation code in each notebook – you just import and reuse it.
Utilities – Common helper functions (e.g., for logging, timestamping outputs, data utilities, etc.) go in utils.py (or a utils/ package if you have many). Keep utilities minimal and generic; avoid too much “magic” here to prevent overengineering. For instance, a simple helper to seed all random number generators (NumPy, Python random, PyTorch) can live here to help with reproducibility across the project.
This modular code structure allows you to follow the “pipe and filter” pattern – each module can be seen as a stage in the pipeline (data loading → feature engineering → modeling → evaluation)
medium.com
. Rather than one giant script, you have small, focused modules that are easier to develop, test, and extend
kdnuggets.com
. For example, you might adjust feature engineering in features.py without touching the model training code in train.py.
tests/: Test scripts to ensure each part of the pipeline works as expected. As a solo developer, you’ll rely on tests to catch issues early and to refactor safely. Organize tests to mirror the project structure, which makes it easy to find what to test:
e.g. test_data.py for functions in data.py, test_features.py for features.py, test_models.py for model training/prediction functions, etc. If using subpackages, you can mirror that (e.g. tests for models/classical.py and models/deep.py).
Keep tests lightweight – avoid requiring large datasets or long training runs. You can use small samples of data or even synthetic data fixtures to test logic. For instance, test that load_data() returns a DataFrame with expected columns, or that create_features() correctly computes a new feature on a tiny DataFrame. For model training, you might run a single training epoch on a very small dataset to ensure the code executes without error and improves the loss.
Automate running tests (e.g., via pytest) whenever you make changes. This will catch regression issues. Prioritize testing critical preprocessing and data transformation steps
kdnuggets.com
 (since a bug there can silently corrupt all model results), as well as any custom model logic. By writing tests early, you ensure reliability and make future refactoring much easier
kdnuggets.com
.
Project documentation files: A README.md at the root should give an overview of the project – what the project does, how to set up the environment, and how to run the training/evaluation. This is crucial for long-term maintainability: when you revisit the project after months, the README guides you on how to rerun things. If the project grows, you might add more documentation (in a docs/ folder or using tools like MkDocs, which the Cookiecutter template supports
cookiecutter-data-science.drivendata.org
). But for a solo project, a well-maintained README might suffice.
Environment & dependency files: Since you’re using uv for dependency management, your dependencies live in pyproject.toml (with [project.dependencies] specified) and uv.lock. The uv tool will create and manage a dedicated virtual environment (often under .venv/) and the lockfile. Always commit the uv.lock file to version control – it captures exact versions of packages, ensuring anyone (or your future self) can replicate the environment exactly
realpython.com
. This enhances reproducibility by preventing “it worked on my machine” problems. In pyproject.toml, you can also include project metadata and possibly a script entry point if you want (e.g., a console script to run training). If you have additional environment setup (like Jupyter kernel config or OS-level dependencies), document them in the README.
Tip: Utilize a Makefile or simple shell scripts for common tasks. For example, a Makefile can define shortcuts like make data (to run data preparation), make train (to execute the training pipeline), make evaluate, etc.
cookiecutter-data-science.drivendata.org
. This provides a lightweight way to orchestrate the pipeline without needing heavier tools like Airflow. Each Makefile target would just call your Python modules (e.g., uv run nba_minutes/train.py --config config/settings.yaml). This approach keeps orchestration simple and transparent.
Modular Pipeline Development (Without Heavy Orchestration)
Instead of using external pipeline orchestrators (Airflow, Luigi, etc.), our structure enables building a pipeline within the code in a modular fashion:
Main pipeline script: You can designate a script (or a notebook for interactive runs) as the pipeline runner. For instance, train.py could be the main script that ties everything together:
Load configuration (from config/ or parse CLI arguments).
Call data loading/cleaning functions to get the raw data into a workable form.
Call feature engineering to transform raw data into model inputs.
Invoke model training (perhaps calling a function that lives in models/classical.py or models/deep.py depending on config).
Save the trained model to models/ directory, and optionally output evaluation metrics or predictions to reports/ or a results file.
Call evaluation routines (or these could be integrated in the training function) to calculate metrics like mean absolute error on a test set, producing plots or summaries stored in reports/ (like a feature importance chart or training loss curve).
All these steps can be orchestrated in a linear sequence in code – essentially implementing a custom pipeline. Breaking each step into well-defined functions (with clear inputs/outputs) means you can easily reorder or reuse steps. For example, if you want to experiment with a different feature set, you might swap out the feature engineering function or call an alternate one. Because we avoid complex external schedulers, running an experiment is as straightforward as calling uv run nba_minutes/train.py with appropriate parameters.
Configuration-driven runs: To avoid hardcoding details in the pipeline, use the config files or command-line arguments. For instance, your train.py can accept a flag for which model to train (--model-type xgboost vs --model-type lstm), or better, read that from config/settings.yaml. This makes your pipeline flexible: you can maintain one pipeline script that covers multiple workflows (classical ML vs DL) based on settings. Using a config also aids reproducibility – you can save the config alongside model artifacts to remember what settings produced that model.
No overengineering: Keep the pipeline code as simple as possible while meeting your needs. As a solo developer, you likely don’t need a distributed workflow engine. If you find yourself creating many manual steps (e.g., running one script to preprocess, saving output, then running another to train), consider automating it with Python function calls or a Makefile. On the other hand, if the pipeline grows in complexity (say many branching paths or heavy I/O), you might consider a lightweight workflow tool or library (e.g., [prefect or luigi] for purely Pythonic pipelines). But start simple: a clear sequence of function calls and a few scripts can go a long way and are easier to maintain and understand by yourself.
Testing, Quality Assurance, and Reproducibility
A maintainable project treats testing and reproducibility as first-class concerns:
Test organization: As noted, keep tests in a parallel structure to code. This makes it straightforward to find the test corresponding to a module. Use a test framework like pytest for simplicity and powerful features. For example, you can parametrize tests to easily check a function on multiple inputs. Aim for basic coverage: ensure data transformations work on expected inputs, models can train for at least one iteration, and evaluation metrics compute correct values (you can test metric functions against hand-calculated examples).
Lightweight tests: Avoid tests that require the full dataset or long training runs. Where possible, inject controllable parameters to make functions testable. For instance, if your train_model function normally trains for 100 epochs, allow an override (in tests) to train for 1 epoch or on a smaller batch. This way, tests run fast. If using random components (train/test split, weight initialization), set a random seed in tests (and in training code generally) to get deterministic behavior
kdnuggets.com
.
Reproducibility practices: In addition to dependency locking with uv (for environment reproducibility), take care of reproducibility in data processing and modeling:
Set random seeds for any randomness (e.g., shuffling data, initializing neural network weights) at the start of your pipeline. You might have a utility in utils.py like set_seeds(seed) that sets Python’s random.seed, NumPy’s seed, and the deep learning framework’s seed.
Document data sources and processing steps. If data is downloaded or scraped, write scripts for it (so you can re-run the exact data collection in the future) and save raw data with a timestamp or version if it might change.
Keep an experiment log. This could be as simple as a spreadsheet or a markdown file where you note each run’s config and results, or as advanced as using an experiment tracking tool (MLflow, Weights & Biases, Neptune, etc.) to automatically log metrics and parameters
kdnuggets.com
. For a solo project, a lightweight approach may suffice: e.g., save a logs/ directory with a timestamped log file or output metrics to a CSV for each training run
kdnuggets.com
. This helps you compare experiments (different algorithms or hyperparameters) and remember what you’ve tried.
Maintainability through simplicity: Resist the urge to prematurely optimize or add complexity. For example, rather than writing a highly abstract class hierarchy for models, you might use simple functions first. Clarity is king – code should be readable and well-commented, so that months later you understand the pipeline. Use docstrings in your functions to explain tricky parts. A common failure in solo projects is losing track of which script does what; our structured naming (e.g. a single train.py rather than multiple similarly named scripts) and documentation avoids confusion like “was the final training code in train_v2.py or experiment_final.py?”
neptune.ai
. Each component has one clear purpose and name.
Emerging Best Practices and Tools for Solo ML Devs
The data science community has coalesced around certain best practices, several of which we’ve incorporated above. To recap and highlight a few trends:
Cookiecutter-style project templates: Many practitioners start projects with a template (like the Cookiecutter Data Science structure) to enforce good organization from day one
kdnuggets.com
. Adopting a similar template ensures your projects remain familiar in structure, which accelerates onboarding yourself or others in the future.
Config-driven workflows: There is a growing emphasis on configuration (using YAML/TOML) to manage experiments. Tools like Hydra (for config management) have gained popularity to easily switch configurations. Even if you don’t use such a library, the principle of separating config from code stands – it makes your experiments more repeatable and auditable.
Experiment tracking and data versioning: Solo developers increasingly use experiment trackers (like MLflow or W&B) even for personal projects, because they provide a convenient memory of what was tried
kdnuggets.com
. Similarly, data versioning tools (e.g., DVC or Git LFS) can be handy if your datasets are large or evolving. If your NBA data is updated seasonally, consider versioning the data or at least documenting changes.
Jupyter to pipeline: An emerging workflow is to do initial development in notebooks (for interactive exploration) and then convert notebooks to scripts or pipelines. Tools like nbdev (from fast.ai) allow you to develop in notebooks and export code to modules, which might be appealing if you prefer notebooks. Otherwise, the manual approach works: prototype in notebooks/, then refactor into the nba_minutes/ codebase once things firm up. This keeps your production code clean while still benefiting from the exploratory phase.
Lightweight pipelines: There’s a trend toward using lightweight pipeline frameworks (for example, Kedro or Metaflow) for managing data science project flow. Kedro, for instance, enforces a very similar project structure with modular pipeline “nodes” and makes it easy to plug in new pipeline steps
docs.kedro.org
. These can be great for larger projects or if you foresee needing to frequently re-run only parts of the pipeline. However, for a solo project focused on simplicity, our outlined approach (maybe aided by a Makefile or simple scripting) is usually sufficient. You can always introduce such tools later if needed, once the base structure is in place.
Testing culture: It’s increasingly recognized that applying software engineering practices (like testing and CI/CD) to data science yields more reliable projects. As a solo developer, consider setting up continuous integration on your repository (even if it’s private) using a service like GitHub Actions. This can automatically run tests when you push changes, adding confidence that nothing is broken. Writing tests for data science code might feel unfamiliar at first, but it pays off by catching issues (for example, a function that silently produces NaNs or a model training that fails for certain input shapes).
By adhering to these practices, you ensure that your NBA player minutes prediction project remains easy to understand, modify, and extend in the long run. New features (say, trying a new algorithm or adding a new data source) can be integrated by adding new modules or functions in the appropriate place, without entangling the whole codebase. Each part of the project (data, features, models, evaluation) is cleanly separated, which aligns with standard workflows in professional ML engineering
kdnuggets.com
.
Conclusion
This blueprint provides a maintainable structure for a data science project that involves both traditional ML and deep learning components. It emphasizes a logical separation of concerns: data management, feature engineering, model training (for various model types), and evaluation are all organized into their own spaces. By following this structure and best practices—modular code, separate config, committed dependencies
realpython.com
, thorough but efficient testing, and simple pipeline orchestration—you set yourself up for long-term success.
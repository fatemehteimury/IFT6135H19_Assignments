***NOTE***: Due to GitHub file size limits, Git LFS (https://git-lfs.github.com/) is required to download checkpoint / data files... If Git LFS is installed after you clone the repository, you must delete the original clone and reclone the repository. 

Install:
pip install -r requirements.txt

assignment1_classic005_ensemble1 (Private: 0.93280, Public: 0.94077):
- Train both models of ensemble: 'python3 main.py'
- Generate submission on test data set: 'python3 main.py --evaluate'

assignment1_classic005_ensemble2 (Private: 0.94160, Public: 0.94037):
- Train both models of ensemble: 'python3 main.py'
- Generate submission on test data set: 'python3 main.py --evaluate'

assignment1_classic005: 
- Train model: 'python3 main.py'
- Resume from checkpoint: 'python3 main.py --resume <checkpoint filename>'
- Evaluate on validation data set / log incorrectly classified examples: 'python3 main.py --resume <checkpoint filename> --evaluate'
- Model hyperparameters modified using the 'hyperparameters' dictionary at the top of the main.py file. Optimization hyperparameters (learning rate, momentum, etc) are modified via command line arguments


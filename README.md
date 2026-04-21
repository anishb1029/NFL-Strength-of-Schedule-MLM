## How to Run This Project

### 1. Clone the repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Set up a Python environment
Make sure Python 3.10 or newer is installed.

```bash
python -m venv venv
```

Activate it:

Mac/Linux:
```bash
source venv/bin/activate
```

Windows:
```bash
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

If no requirements file:
```bash
pip install pandas numpy scipy scikit-learn statsmodels
```

### 4. Add the dataset
Place the file:
```
drive_level_data.csv
```

Inside:
```
data/drive_level_data.csv
```

### 5. Run the models

Ordinal Regression:
```bash
python baseline_ordinal.py
```

PROR model:
```bash
python pror_model.py
```

Random Forest:
```bash
python random_forest.py
```

### 6. View outputs
Results will be printed in the terminal, including:
- Model performance metrics  
- Expected Points per Drive  
- Team rankings  

### 7. Push changes to GitHub
```bash
git add .
git commit -m "update models"
git push origin main
```

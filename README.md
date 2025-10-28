# 🎯 Day 19 – ML Hyperparameter Tuning: GridSearchCV & RandomizedSearchCV

## 📘 Project Overview
This project focuses on **hyperparameter tuning techniques** in Machine Learning using **GridSearchCV** and **RandomizedSearchCV**.  
The goal is to optimize model performance by systematically searching for the best combination of hyperparameters.

This project is part of my **Machine Learning learning journey (Day 19)** at **IANT (Institute of Advance Network Technology)**.

---

## 🚀 Objective
The main objective of this notebook is to:
- Understand how hyperparameter tuning improves model accuracy and performance.
- Compare **GridSearchCV** and **RandomizedSearchCV** approaches.
- Apply these methods on a **Boston House Price Prediction** dataset.

---

## 📂 Dataset
**File:** `Boston_HPP.csv`  
The dataset contains housing data including features such as:
- `CRIM`: Per capita crime rate by town  
- `ZN`: Proportion of residential land zoned for lots  
- `RM`: Average number of rooms per dwelling  
- `AGE`: Proportion of owner-occupied units built before 1940  
- `LSTAT`: Percentage lower status of the population  
- `MEDV`: Median value of owner-occupied homes (target variable)

---

## 🧠 Project Workflow
1. **Data Import and Cleaning**
   - Load dataset using Pandas
   - Handle missing values (if any)
   - Perform basic EDA and visualization

2. **Feature Selection**
   - Identify relevant features for prediction
   - Split data into training and testing sets

3. **Model Training**
   - Train a regression model (e.g., Linear Regression, RandomForestRegressor)

4. **Hyperparameter Tuning**
   - Use **GridSearchCV** for exhaustive parameter search
   - Use **RandomizedSearchCV** for faster, randomized parameter optimization

5. **Model Evaluation**
   - Compare performance metrics before and after tuning
   - Evaluate using R² score, MAE, and MSE

---

## 🧩 Key Concepts
- **GridSearchCV** → Tries all parameter combinations systematically  
- **RandomizedSearchCV** → Samples random combinations of parameters for efficiency  
- **Cross-validation (CV)** → Reduces overfitting and improves generalization  
- **Hyperparameters** → Parameters that define model behavior, not learned from data  

---

## 📈 Results
After tuning, the model achieved:
- Improved **R² score**
- Reduced **Mean Squared Error (MSE)**
- Optimized parameter set for the final model

Visualizations and comparison charts are included in the notebook files.

---

## 📓 Files Included
| File Name | Description |
|------------|-------------|
| `Boston_HPP.csv` | Dataset used for regression |
| `Boston_House_Price_Prediction.ipynb` | Main model training and prediction notebook |
| `Hyperparameter Tuning.ipynb` | Demonstration of GridSearchCV and RandomizedSearchCV |
| `GridSearchCV_RandomizedSearchCV.pdf` | Project report/notes in PDF format |
| `.ipynb_checkpoints/` | Jupyter Notebook temporary files |

---

## 🛠️ Tools & Libraries Used
- **Python**
- **Pandas**, **NumPy**
- **Scikit-learn**
- **Matplotlib**, **Seaborn**
- **Jupyter Notebook**


## 📊 Sample Code Snippet

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
---
```
# Define model
rf = RandomForestRegressor()

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# GridSearchCV
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best R² Score:", grid_search.best_score_)


## 📚 Learning Outcome

Through this project, I learned:

- The difference between **parameter tuning** and **model training**  
- How to use **GridSearchCV** and **RandomizedSearchCV**  
- How tuning hyperparameters can significantly improve ML model performance  

---

## 👨‍💻 Author

**Laxman Bhimrao Khedkar**  
📍 Pune, Maharashtra, India  
🎓 B.E. in Computer Engineering – Savitribai Phule Pune University  
💼 Aspiring Data Analyst / Data Scientist  

🔗 [Portfolio](https://beacons.ai/laxmankhedkar)  
🔗 [LinkedIn](https://www.linkedin.com/in/laxman-khedkar)  
🔗 [GitHub](https://github.com/LaxmanKhedkar)

---

## 🏁 Conclusion

Hyperparameter tuning plays a crucial role in improving machine learning model performance.  
By applying **GridSearchCV** and **RandomizedSearchCV**, we can systematically explore and identify the most effective model configurations to achieve better results efficiently.

---

⭐ *If you found this project useful, please consider starring the repository!*
```
git add README.md
git commit -m "Added README with author and learning outcome"
git push -u origin main

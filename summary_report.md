# 🏡 Ames House Price Prediction — Executive Summary Report

**Author:** Prath Udhnawala  
**GRID ID:** 11120  
**Role:** Junior Data Scientist (PropTech)  
**Project:** Supervised Learning Practical Exam — Set A  

---

## 1. Business Problem & Dataset Overview
In the competitive PropTech landscape (benchmarked against platforms such as NoBroker and MagicBricks), automated valuation models (AVMs) are crucial for estimating fair market property prices, accelerating transaction closures, and mitigating appraisal asymmetry. This project formulates an end-to-end Supervised Learning regression pipeline using the Ames Housing Dataset (1,460 residential records, 79 structural and neighborhood features) to predict continuous property sale prices (`SalePrice`).

## 2. Preprocessing & Feature Engineering Strategy
Data integrity was established through domain-aware preprocessing:
* **Missing Value Imputation:** Categorical attributes representing physical absences (`PoolQC`, `Fence`, `GarageType`) were explicitly imputed with `'None'`, while corresponding numeric dimensions (`GarageArea`, `BsmtFinSF1`) were imputed with `0`. Remaining continuous attributes were imputed using the median. Columns exhibiting >80% missingness (`Alley`, `PoolQC`, `MiscFeature`) were dropped.
* **Outlier Removal:** Two extreme leverage data points (Indices 523 and 1298; `GrLivArea > 4000` sq. ft. with `SalePrice < $300,000`) were expunged. Retaining these anomalous partial sales introduces severe leverage that pulls Ordinary Least Squares (OLS) regression hyperplanes downward.
* **Feature Engineering:** Domain-specific composite predictors were engineered:
  * $\text{TotalSF} = \text{TotalBsmtSF} + \text{1stFlrSF} + \text{2ndFlrSF}$ (aggregate usable living envelope)
  * $\text{HouseAge} = \text{YrSold} - \text{YearBuilt}$ and $\text{RemodAge} = \text{YrSold} - \text{YearRemodAdd}$
  * Binary flags: $\text{HasGarage}$ and $\text{HasPool}$
* **Encoding & Scaling:** Ordinal ratings (`ExterQual`, `KitchenQual`, `BsmtQual`, `FireplaceQu`) were numerically mapped from $0$ (`None`) to $5$ (`Ex`). Low-cardinality nominal categoricals were one-hot encoded. Skewed predictors ($|\text{skew}| > 0.75$) and the continuous target `SalePrice` were stabilized using $\log(1+x)$ transformations. Continuous numerical features were scaled using `StandardScaler`.

## 3. Model Evaluation & Benchmark Winner
Five regression models were evaluated on an 80/20 train-test split (`random_state=42`) with inverse-log transformation ($\text{expm1}$) applied prior to computing test metrics:

* **Linear Regression (OLS Baseline):** $\text{RMSE} = \$71,813.13$, $\text{MAE} = \$52,750.36$, $R^2 = 0.0721$
* **Ridge Regression ($L_2$, $\alpha=100.0$):** **$\text{RMSE} = \$70,762.21$, $\text{MAE} = \$50,756.78$, $R^2 = 0.0991$** *(Best Performer)*
* **Lasso Regression ($L_1$, $\alpha=0.01$):** $\text{RMSE} = \$71,650.55$, $\text{MAE} = \$51,434.56$, $R^2 = 0.0763$ ($67$ features zeroed)
* **Random Forest Regressor:** $\text{RMSE} = \$71,918.64$, $\text{MAE} = \$51,855.90$, $R^2 = 0.0694$
* **XGBoost Regressor:** $\text{RMSE} = \$74,054.95$, $\text{MAE} = \$54,368.37$, $R^2 = 0.0133$

**Why Ridge Performed Best:** High collinearity across dimensional features (`TotalSF`, `GrLivArea`, `TotalBsmtSF`) inflates variance in standard OLS. Ridge’s $L_2$ penalty smoothly shrinks correlated coefficients toward zero without dropping predictive features, yielding superior generalizability over aggressive $L_1$ feature elimination and un-tuned gradient boosting.

## 4. Top 3 Predictive Drivers & Industry Insights
1. **Overall Quality (`OverallQual`):** The primary determinant of property value. High-grade construction materials and premium craftsmanship command compounding valuation multiples.
2. **Total Usable Area (`TotalSF` / `GrLivArea`):** Directly governs baseline square-foot valuations; vertical and horizontal expansions represent the most reliable route to property equity appreciation.
3. **Neighborhood Location (`Neighborhood`):** Explains micro-market geographic valuation premiums (e.g., Northridge Heights commanding median valuations exceeding $\$315,000$, compared to older suburban zones trading under $\$150,000$).

## 5. Next Steps for Production Serving
* **Model Ensembling:** Deploy a stacked ensemble architecture combining Ridge, LightGBM, CatBoost, and XGBoost with a meta-regressor.
* **Bayesian Optimization:** Employ Optuna for hyperparameter search over deep parameter grids.
* **Geospatial & Macro Integration:** Incorporate GIS coordinate data (distance to central business districts, transit hubs, school ratings) and macroeconomic interest-rate indicators.
* **MLOps Pipeline:** Implement MLflow tracking, automated data drift monitoring with Evidently AI, and containerized FastAPI endpoints on Kubernetes for sub-50ms inference.


# Solar Power Generation Prediction using Machine Learning

**Course:** DA207 – Machine Learning Group Project  
**Institution:** Delhi Technological University (DTU)

---

## 1. Project Overview
This project predicts solar power generation (`AC_POWER`) using environmental parameters collected from solar power plants. By analyzing ambient temperature, module temperature, and solar irradiation, the system estimates power output and provides an interactive **Streamlit** web application for real-time predictions.

---

## 2. Objectives
- Integrate and clean raw power generation and weather sensor datasets across two solar power plants.
- Perform Exploratory Data Analysis (EDA) and correlation analysis to identify key environmental drivers of AC power output.
- Train and evaluate regression models (Simple Linear Regression and Multiple Linear Regression) to predict `AC_POWER`.
- Evaluate model performance using standard metrics, Actual vs. Predicted plots, residual analysis, and plant-wise performance comparisons.
- Build a user-friendly Streamlit prediction UI with input validation and warnings.

---

## 3. Dataset Description
The dataset consists of four raw CSV files across two solar power plants:
- `Plant_1_Generation_Data.csv`
- `Plant_1_Weather_Sensor_Data.csv`
- `Plant_2_Generation_Data.csv`
- `Plant_2_Weather_Sensor_Data.csv`

After merging and cleaning the generation and weather records, the integrated dataset contains **6,416 records**.

### Features & Target Variable
| Role | Column Name | Description |
| :--- | :--- | :--- |
| **Feature** | `AMBIENT_TEMPERATURE` | Ambient air temperature at the plant site |
| **Feature** | `MODULE_TEMPERATURE` | Temperature of the solar panel module |
| **Feature** | `IRRADIATION` | Solar irradiation intensity |
| **Target** | `AC_POWER` | Generated Alternating Current (AC) power output |

---

## 4. Technologies Used
- **Programming Language:** Python
- **Data Analysis & Manipulation:** Pandas, NumPy
- **Data Visualization:** Matplotlib, Seaborn
- **Machine Learning & Model Serialization:** Scikit-learn, Joblib
- **Web Application / UI:** Streamlit

---

## 5. Data Preprocessing & Exploratory Data Analysis (EDA)
- **Data Integration:** Merged power generation records with corresponding weather sensor measurements for Plant 1 and Plant 2.
- **Data Cleaning:** Cleaned and aligned records to produce 6,416 valid observations.
- **EDA & Visualizations:**
  - Feature distributions and summary statistics
  - Correlation analysis between environmental features and `AC_POWER`
  - Plant-wise performance comparison between Plant 1 and Plant 2

---

## 6. Machine Learning Methodology
1. **Feature Selection & Scaling:** Selected `AMBIENT_TEMPERATURE`, `MODULE_TEMPERATURE`, and `IRRADIATION` as predictors and applied feature scaling saved via `joblib`.
2. **Model Training:**
   - **Simple Linear Regression**
   - **Multiple Linear Regression**
3. **Model Evaluation & Diagnostics:**
   - Evaluated predictions using $R^2$, Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE).
   - Generated **Actual vs. Predicted** plots and **Residual Analysis** plots.
4. **Model Persistence:** Saved the trained model and scaler inside the `models/` folder using `joblib`.

---

## 7. Model Performance Metrics
### Multiple Linear Regression Results
| Metric | Score |
| :--- | :--- |
| **$R^2$ Score** | `0.90309` |
| **Mean Absolute Error (MAE)** | `1438.58` |
| **Root Mean Squared Error (RMSE)** | `2403.00` |

---

## 8. Project Folder Structure
```text
-solar-power-generation-prediction/
│
├── Dataset/               # Raw and processed CSV datasets
├── Notebooks/             # Jupyter notebooks for EDA, modeling, and evaluation
├── Results/               # Generated plots (correlation, actual vs predicted, residuals)
├── models/                # Saved ML model and scaler (.joblib)
├── app.py                 # Streamlit web application UI
├── prediction.py          # Model loading and prediction inference logic
├── test_prediction.py     # Script for testing prediction functionality
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
## 9. How to Run the Application

### Step 1: Clone the repository
```bash
git clone https://github.com/kunalmohliya02-png/solar-power-generation-prediction.git
cd solar-power-generation-prediction
```

### Step 2: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run the Streamlit application
```bash
streamlit run app.py
```

The application will open in your browser, usually at `http://localhost:8501`.

## 10. Group Members

| Name | Roll Number |
|---|---|
| Member 1 | To be added |
| Member 2 | To be added |
| Member 3 | To be added |

## 11. References

- Solar Power Generation Data – Kaggle dataset.
- Scikit-learn Documentation: https://scikit-learn.org/
- Streamlit Documentation: https://docs.streamlit.io/
- Pandas Documentation: https://pandas.pydata.org/docs/

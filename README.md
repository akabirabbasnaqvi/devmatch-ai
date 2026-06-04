# 🧬 DevMatch AI — Developer Role Predictor using KNN

A machine learning project that predicts your developer role based on your 
skills and experience, trained on 65,437 real developers from the 
**Stack Overflow Developer Survey 2024**.

---

## 📌 Features
- Predicts your developer role (Full-Stack, Back-End, Front-End, Mobile & more)
- Shows K nearest similar developers with similarity scores
- Generates a personalized skill gap roadmap toward your target role
- Clean and interactive Streamlit GUI

---

## ⚙️ Tech Stack
- **Language:** Python
- **ML Algorithm:** K-Nearest Neighbors (KNN)
- **Libraries:** Scikit-learn, Pandas, NumPy, SMOTE, Streamlit, Joblib
- **Dataset:** Stack Overflow Developer Survey 2024 (65,437 rows)

---

## 📊 Model Performance
| Detail | Value |
|---|---|
| Training samples | 146,080 (after SMOTE balancing) |
| Features | 189 (skills + tools + platforms + experience) |
| Best K | 3 |
| Final Accuracy | 73.46% |

---

## 🚀 How to Run

**1. Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/devmatch-ai.git
cd devmatch-ai
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Download the dataset**

Download `survey_results_public.csv` from:
https://survey.stackoverflow.co/2024/

Place it in the project folder.

**4. Run the data pipeline**
```bash
python step1_explore.py
python step2_clean.py
python step3_train_knn.py
```

**5. Launch the app**
```bash
streamlit run app.py
```

---

## 📁 Project Structure
devmatch-ai/
├── app.py                    # Streamlit GUI
├── step1_explore.py          # Data exploration
├── step2_clean.py            # Data cleaning & feature engineering
├── step3_train_knn.py        # KNN model training
├── survey_results_schema.csv # Dataset schema
├── requirements.txt          # Dependencies
└── README.md                 # Project documentation


---

## 🤝 Contributing
Contributions are welcome! Feel free to open an issue or submit a pull request.

---

## 📄 License
This project is licensed under the MIT License.
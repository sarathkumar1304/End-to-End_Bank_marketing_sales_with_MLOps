# 📊 Bank Term Deposit Subscription Prediction  

## 🌟 Project Overview  
This project predicts whether a client will subscribe to a bank term deposit based on data collected from direct marketing campaigns 📞📈. The marketing campaigns were conducted via phone calls, and this project leverages machine learning algorithms to classify the outcome as "Subscribed" ✅ or "Not Subscribed" ❌.  

---

## 💡 Problem Statement  
The classification goal is to predict if the client will subscribe (`yes`) or not (`no`) to a term deposit (variable `y`).  

The data is sourced from the marketing campaigns of a Portuguese banking institution, containing multiple datasets with client information, campaign details, and outcomes.  

---

## ✨ Features  
- 🔹 **Interactive Web UI**: Built with Streamlit for user-friendly predictions.  
- 🔹 **Pipeline Orchestration**: Prefect ensures smooth and efficient workflow management.  
- 🔹 **Experiment Tracking**: MLflow is integrated for model tracking, parameter tuning, and logging metrics.  
- 🔹 **Machine Learning Algorithms**: Classification models trained and evaluated to provide accurate predictions.  

---

## 📂 Dataset  
The dataset contains client and campaign-related information, with the target variable being `y`:  
- `0`: Not Subscribe ❌  
- `1`: Will Subscribe ✅  

Primary dataset used: **`bank-additional-full.csv`**  

---

## 🛠️ Technological Stack  
- **💻 Programming Language**: Python  
- **🤖 Machine Learning**: Supervised classification algorithms  
- **⚙️ MLOps Tools**:  
  - **🔄 Prefect**: Pipeline orchestration and workflow management  
  - **📈 MLflow**: Experiment tracking and model management  
- **🌐 Web UI**: Streamlit for creating an interactive and user-friendly web interface  

---

## 🔑 Features Used for Prediction  
Key columns used for prediction include:  
- 👤 `age`  
- 💼 `job`  
- 💍 `marital`  
- 🎓 `education`  
- ❓ `default`  
- 💰 `balance`  
- 🏠 `housing`  
- 💳 `loan`  
- 📞 `contact`  
- 📅 `day_of_week`, `month`  
- ⏳ `duration`, `campaign`, `pdays`, `previous`  
- 🔄 `poutcome`  

---

## 🚀 How to Use  
1. **Clone this repository**:  
   ```bash
   git clone https://github.com/yourusername/bank-term-deposit-prediction.git
   cd bank-term-deposit-prediction
   ```

Install dependencies:

```
pip install -r requirements.txt

```
Set up Prefect and MLflow:

Configure Prefect for pipeline orchestration.
Start the MLflow server for experiment tracking.
Run the Streamlit application:


```streamlit run app.py```

## Make Predictions:

##### Enter client details in the web UI to get predictions:
✅ Subscribed: Client will subscribe to the term deposit.
❌ Not Subscribed: Client will not subscribe to the term deposit.
📊 Results
Achieved high accuracy in predicting term deposit subscription.
The app provides an intuitive interface for non-technical users to interact with machine learning models.
🔮 Future Scope
🚀 Integration of more advanced machine learning algorithms.
🎨 Enhance the UI for better visualization and data analysis.
☁️ Deployment on cloud platforms for wider accessibility.

### 📜 License
This project is licensed under the MIT License.

### 🙌 Acknowledgments
- 📚  The dataset was obtained from the UCI Machine Learning Repository.
- 🔧 Special thanks to the developers of Prefect, MLflow, and Streamlit for their powerful tools.
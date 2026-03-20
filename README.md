# 🍽️ Cuisine Intelligence Classifier

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red)

---

# 📌 Project Overview
**Cuisine Intelligence Classifier** is a professional Machine Learning & Data Analysis project designed to identify and predict the **most likely cuisines** served by a restaurant based on its financial tier, customer ratings, and engagement metrics.

The project demonstrates a robust **end-to-end workflow**, including:
- Multi-label Data Preprocessing
- Statistical Pattern Matching
- Predictive Model Training (Random Forest)
- Interactive Dashboard Deployment

An interactive **Streamlit dashboard** allows users to input restaurant profiles and instantly receive cuisine predictions along with frequency-based confidence insights.

---

# ❓ Key Questions Explored
- Which cuisines are dominant in specific **Price Ranges**?
- Does a high **Average Cost for Two** correlate with specific international cuisines?
- How do **Customer Votes** and **Ratings** shift the probability of cuisine types?
- Can we statistically predict a restaurant's menu style without historical labels?

---

# 🛠 Technologies & Tools Used
| Tool | Purpose |
|-----|------|
| **Python** | Core programming & Logic |
| **Pandas** | Data cleaning & Multi-label processing |
| **NumPy** | Numerical operations |
| **Scikit-learn** | Random Forest & MultiLabelBinarizer |
| **Streamlit** | Premium Glassmorphism UI |
| **Git & GitHub** | Version Control & Deployment |

---

# 📊 Dataset
**File in Repository:** `dataset.csv`

The dataset contains comprehensive global restaurant records including pricing, service categories, and multi-cuisine labels.

### Important Features
- **Average Cost for Two:** Financial positioning of the restaurant.
- **Price Range:** Tier-based categorization (1-4).
- **Aggregate Rating:** Customer satisfaction metric.
- **Votes:** Total engagement/popularity count.
- **Cuisines:** Target labels for classification.

---

# 📂 Project Structure
```text
Cuisine-Classification-ML
│
├── dataset.csv          # Raw restaurant dataset
├── train.py             # Script for model training & evaluation
├── cuisine_model.pkl    # (Ignored via .gitignore) Saved ML Model 
├── mlb.pkl              # Multi-label encoder object
├── app.py               # Main Streamlit UI Application
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation

Folder Details
dataset.csv: The backbone of the analysis containing restaurant features.

train.py: Contains the logic for Random Forest training and accuracy metrics.

app.py: A high-end UI featuring a Dark-Teal theme and interactive charts.

.gitignore: Configured to handle large model files for clean repository management.

🤖 Machine Learning Approach
The project utilizes a OneVsRest (OVR) wrapper around a Random Forest Classifier to handle the multi-label nature of cuisines.

Key Highlights:
Encoding: MultiLabelBinarizer is used to flatten multiple cuisine tags per restaurant.

Modeling: Random Forest Regressor/Classifier for capturing non-linear relationships.

Hybrid UI Logic: The app intelligently switches between statistical matching and predictive analysis to ensure high performance.

🌐 Features of the Web Application
The Streamlit dashboard allows users to:

Configure Financial Tiers (Cost & Price Range).

Adjust Performance Metrics (Ratings & Votes).

Trigger AI-driven Predictions with a single click.

Dashboard Features
🎯 Cuisine Tag Cloud: Visual representation of predicted cuisines.

📊 Probability Analysis: Bar charts showing prediction confidence.

💡 Deep-Dive Insights: Summary of similar restaurants found in the dataset.

🎨 Premium UI: Custom CSS with Glassmorphism and Teal aesthetics.

▶ Running the Project Locally
1️⃣ Clone the Repository
Bash
git clone [https://github.com/Agi91/Cuisine-Classification-ML.git](https://github.com/Agi91/Cuisine-Classification-ML.git)
2️⃣ Navigate to the Project Folder
Bash
cd Cuisine-Classification-ML
3️⃣ Install Dependencies
Bash
pip install -r requirements.txt
4️⃣ Run the Streamlit Application
Bash
streamlit run app.py
🚀 Future Improvements
Cloud Model Hosting: Move the 373MB model to an external S3 bucket.

Real-time API: Integrate Zomato/Swiggy APIs for live data.

Geospatial Analysis: Add Map-based cuisine distribution views.



👨‍💻 Author
Cuisine Intelligence Classifier Machine Learning Project

GitHub Repository: https://github.com/Agi91/Cuisine-Classification-ML

⭐ Built with Python, Scikit-Learn, and Streamlit
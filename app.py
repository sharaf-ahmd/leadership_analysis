import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Leadership Analysis App", layout="wide")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Descriptive Analysis", "Inferential Analysis", "Predictive Analysis"])

@st.cache_data
def load_data():
    coaches = pd.read_csv('dataset/coaches.csv')
    performance = pd.read_csv('coach_performance.csv')
    data = pd.merge(coaches, performance, on=['coachId', 'teamId'])
    data.drop_duplicates(inplace=True)
    return data

@st.cache_data
def preprocess_data(data):
    data = data.copy()
    data.drop(['coachPartId', 'gameId', 'coachName'], axis=1, inplace=True, errors='ignore')
    
    data["leadership_group"] = pd.cut(
        data['leadership_score'],
        bins=[0, 40, 60, 80, 100],
        labels=['0-40 (Low)', '41-60 (Moderate)', '61-80 (High)', '81-100 (Exceptional)'],
        right=True
    )
    
    leadership_dummies = pd.get_dummies(data['leadership_group'])
    data = pd.concat([data, leadership_dummies], axis=1)
    data = data.drop("leadership_group", axis=1)
    
    min_threshold, max_threshold = data["points_scored"].quantile([0.001, 0.999])
    
    # Filter outliers
    data = data[((data["points_scored"] >= min_threshold) & (data["points_scored"] <= max_threshold))]
    
    return data

data_raw = load_data()
data_preprocessed = preprocess_data(data_raw)

# Adding success target for inferential
data_inf = data_preprocessed.copy()
data_inf['Target_Success'] = (data_inf['points_scored'] >= data_inf['points_scored'].median()).astype(int)

# Scaling required for Predictive Model
scaler = MinMaxScaler(feature_range=(0,1))
cols_to_scale = ["points_scored", "leadership_score"]
scaled_vals = scaler.fit_transform(data_inf[cols_to_scale])
scaled_data = pd.DataFrame(scaled_vals, columns=[col + "_scaled" for col in cols_to_scale], index=data_inf.index)
data_ml = pd.concat([data_inf.drop(cols_to_scale, axis=1), scaled_data], axis=1)
data_ml.dropna(inplace=True)

if page == "Descriptive Analysis":
    st.title("Descriptive Analysis")
    st.write("This section explores the initial data properties and visualizes the distribution of key variables, showing directly the output equivalent to the Jupyter cells.")
    
    st.subheader("Data Overview")
    
    st.write("**First 5 rows of merged data:**")
    st.dataframe(data_raw.head())
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Data Shape:**")
        st.write(data_raw.shape)
        st.write("**Data Summary Summary:**")
        st.dataframe(data_raw.describe())
        st.write("**Unique Values Count:**")
        st.write(data_raw.nunique())
        
    with col2:
        st.write("**Null Values count:**")
        st.write(data_raw.isnull().sum())
        st.write("**Duplicate Values Count:**")
        st.write(data_raw.duplicated().sum())

    st.subheader("Visualizations")
    st.write("**Histograms for numerical columns**")
    st.write("Shows data distribution across variables")
    fig = plt.figure(figsize=(15, 10))
    axes = data_raw.hist(figsize=(15, 10), color='indigo', ax=fig.gca())
    st.pyplot(fig)
    
    st.write("**Boxplot for points_scored to detect outliers**")
    st.write("Visualizing max-min thresholding for scoring outliers.")
    fig2, ax2 = plt.subplots()
    ax2.boxplot(data_preprocessed["points_scored"], vert=False)
    ax2.set_title("Points Scored - Boxplot")
    st.pyplot(fig2)

elif page == "Inferential Analysis":
    st.title("Inferential Analysis")
    st.write("Testing the data through classical statistical principles.")
    
    st.subheader("Correlation Heatmap")
    st.write("Visualizes positive/negative dependencies between columns.")
    correlation = data_preprocessed.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(16,12))
    sns.heatmap(correlation, annot=True, fmt=".2f", cmap='viridis', ax=ax)
    st.pyplot(fig)
    
    st.subheader("Statistical Testing")
    
    st.write("### Independent T-Test")
    st.write("Comparing the points scored between teams with high vs low leadership (above vs below median leadership).")
    median_leadership = data_preprocessed['leadership_score'].median()
    high_leadership = data_preprocessed[data_preprocessed['leadership_score'] > median_leadership]['points_scored']
    low_leadership = data_preprocessed[data_preprocessed['leadership_score'] <= median_leadership]['points_scored']
    t_stat, p_val = stats.ttest_ind(high_leadership, low_leadership)
    
    st.text(f"Average Points (High Leadership): {high_leadership.mean():.2f}")
    st.text(f"Average Points (Low Leadership): {low_leadership.mean():.2f}")
    st.text("-" * 40)
    st.text(f"T-Statistic: {t_stat:.4f}, P-Value: {p_val:.4e}")
    if p_val < 0.05:
        st.success("Conclusion: There is a statistically significant difference in points. Strong leadership impacts performance.")
    else:
        st.info("Conclusion: No statistically significant difference found.")
        
    st.write("### Chi-Square Test of Independence")
    st.write("Determine success strictly as scoring over the median point benchmark against categorical leadership score.")
    contingency_table = pd.crosstab(data_inf['Target_Success'], pd.cut(data_inf['leadership_score'], bins=[0, 40, 60, 80, 100], labels=['Low', 'Moderate', 'High', 'Exceptional']))
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
    st.text("Contingency Table:")
    st.dataframe(contingency_table)
    st.text("-" * 40)
    st.text(f"Chi-Square Statistic: {chi2:.4f}, P-Value: {p:.4e}")

elif page == "Predictive Analysis":
    st.title("Predictive Analysis")
    st.write("Evaluating the machine learning models previously saved.")
    
    # Target value counts
    st.write("**Value counts of target variable (Success):**")
    st.dataframe(data_ml.Target_Success.value_counts())
    
    x = data_ml[['leadership_score_scaled', 'points_scored_scaled', '0-40 (Low)', '41-60 (Moderate)', '61-80 (High)', '81-100 (Exceptional)']]
    y = data_ml['Target_Success']
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
    
    st.subheader("Model Performance")
    st.write("Loaded models will be tested against 20% test data holding stratification principles.")
    
    model_dict = {
        "Logistic Regression": "log_model_nfl.pkl",
        "Random Forest": "rf_model_nfl.pkl",
        "KNN": "knn_model_nfl.pkl",
        "Naive Bayes": "gnb_model_nfl.pkl"
    }

    for model_name, model_file in model_dict.items():
        st.write(f"### {model_name} Performance")
        try:
            model = joblib.load(model_file)
            y_pred = model.predict(xtest)
            
            st.text(f"Classification Report for {model_name}:")
            st.text(classification_report(ytest, y_pred))
            
            col1, col2 = st.columns([1, 1])
            with col1:
                cm = confusion_matrix(ytest, y_pred)
                fig, ax = plt.subplots(figsize=(6,4))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                ax.set_title(f"{model_name} Confusion Matrix")
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Truth')
                st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Could not load {model_file}: {e}")
            
    st.markdown("---")    
    st.subheader("Test the Model (Extra)")
    st.write("Provides real-time predictions using a selected model.")
    
    test_model_choice = st.selectbox("Choose a model to test:", ["Logistic Regression", "Random Forest", "KNN", "Naive Bayes"])
    test_model_file = model_dict[test_model_choice]
    try:
        test_model = joblib.load(test_model_file)
    except:
        test_model = None
    
    col1, col2 = st.columns(2)
    with col1:
        lead_sc = st.slider("Leadership Score Scaled (0-1)", 0.0, 1.0, 0.85)
        pt_sc = st.slider("Points Scored Scaled (0-1)", 0.0, 1.0, 0.90)
    with col2:
        group = st.selectbox("Leadership Group", ['0-40 (Low)', '41-60 (Moderate)', '61-80 (High)', '81-100 (Exceptional)'])
        
    low = 1 if group == '0-40 (Low)' else 0
    mod = 1 if group == '41-60 (Moderate)' else 0
    high_grp = 1 if group == '61-80 (High)' else 0
    exc = 1 if group == '81-100 (Exceptional)' else 0
    
    sample_input = pd.DataFrame([{
        "leadership_score_scaled": lead_sc,
        "points_scored_scaled": pt_sc,
        "0-40 (Low)": low,
        "41-60 (Moderate)": mod,
        "61-80 (High)": high_grp,
        "81-100 (Exceptional)": exc
    }])
    
    if st.button("Predict Success"):
        if test_model is not None:
            prediction = test_model.predict(sample_input)[0]
            proba = test_model.predict_proba(sample_input)[0] if hasattr(test_model, "predict_proba") else None
            
            if prediction == 1:
                st.success("Prediction: Top Half Conclusion (Success)")
            else:
                st.error("Prediction: Bottom Half Conclusion (Not Success)")
                
            if proba is not None:
                st.info(f"Confidence: {max(proba)*100:.2f}%")
        else:
             st.error("Model could not be loaded so no prediction is possible.")

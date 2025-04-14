import streamlit as st 
import pandas as pd
import numpy as np
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD


st.title("Fake News Identification Machine Learning Project") 
file_upload = st.file_uploader("Choose CSV file", type="CSV")
if file_upload is not None:
    data_df = pd.read_csv(file_upload)
    
    st.subheader("Data Preview")
    st.write(data_df.head())
    
    st.subheader("Data Summary")
    st.write(data_df.describe())
    label_count = data_df['type'].value_counts().reset_index()
    st.write(label_count)
    #st.write("file uploaded")
    
    st.subheader("Filter data")

    # Select the column containing the text
    columns = data_df.columns.to_list()
    selected_column = st.selectbox("Select column containing text", columns)

    # Select the specific text value
    unique_texts = data_df[selected_column].unique()
    selected_text = st.selectbox("Select a text", unique_texts)
    st.write(selected_text)

    # Find and display the corresponding label
    matching_label = data_df[data_df[selected_column] == selected_text]['type'].values
    if len(matching_label) > 0:
        st.write(f"Label: **{matching_label[0]}**")
    else:
        st.warning("No label found for the selected text.")
        
    label_count.columns = ["type", "count"]
    
    # coloring bar chart using  altair to color bar chart 
    chart = alt.Chart(label_count).mark_bar().encode(
        x=alt.X('type:N', title='News Type'),
        y= alt.Y('count:Q', title='Count'),
        color=alt.Color('type:N', scale=alt.Scale(scheme='category10'), legend=None),
        tooltip=['type', 'count']
    ).properties(
        title="Distribution of News Types",
        width=500,
        height=300
    ).interactive()
    
    st.altair_chart(chart, use_container_width=True)
    
    st.subheader("Separate the columns")
    # Drop rows where text or label is missing
    data_df = data_df.dropna(subset=['text', 'type'])

    X = data_df["text"]
    Y = data_df["type"]

    st.write(X)
    st.write(Y)

    # Split train and test data
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=7)

    st.subheader("TF-IDF Vectorization *")
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    tfidf_train = tfidf_vectorizer.fit_transform(x_train)
    tfidf_test = tfidf_vectorizer.transform(x_test)
    X_tfidf = tfidf_vectorizer.fit_transform(X)

    st.subheader("Initializing passive aggressive Classifier")
    pas_agg_clas = PassiveAggressiveClassifier(max_iter=60)

    pas_agg_clas.fit(tfidf_train, y_train)
    st.success("Passive Aggressive Classifier has been trained!")
    
    

    st.markdown("<h2 style='color:#FF5733;'>predict the data set and Calculate the accuracy score</h2>", unsafe_allow_html=True)
    y_prediction = pas_agg_clas.predict(tfidf_test)
    acc_score = accuracy_score(y_test, y_prediction)
    st.markdown(f"<p style='color:#1abc9c;'> Accuracy: {round(acc_score*100,2)}%</p>", unsafe_allow_html=True)
    
    
    
    st.subheader("confusion matrix to determine true positive and true negative as well as false positive and negative.")
    conf_df=confusion_matrix(y_test, y_prediction, labels=['bs','bias', 'conspiracy','hate','satire','state','junksci','fake'])
    
    
    st.write("Confusion Matrix as DataFrame:")
    st.dataframe(conf_df)
    
    st.subheader("Sample Predictions")
    sample_df = pd.DataFrame({
        'Text': x_test[:5].values,
        'Actual': y_test[:5].values,
        'Predicted': y_prediction[:5]
    })
    st.table(sample_df)
    
    # Perform dimensionality reduction (sparse-friendly sklearn)
    svd = TruncatedSVD(n_components=2, random_state=42)
    X_2D = svd.fit_transform(X_tfidf)

    # Create DataFrame for plotting
    plot_df = pd.DataFrame(X_2D, columns=['Dim1', 'Dim2'])
    plot_df['type'] = data_df['type'].values

    # Plot in Streamlit
    st.subheader("Scatter plot of Types of news")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        data=plot_df,
        x='Dim1',
        y='Dim2',
        hue='type',
        palette='Set2',
        alpha=0.7
    )

    plt.title("2D Projection of News Articles")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend(title="News Type", bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(fig)
else:
    st.write("Waiting File Upload")
    
    
    st.markdown("<h3 style='color:#3498db;'> Author: Solomon Mengesha Kebede</h3>", unsafe_allow_html=True)
    
    st.markdown("<h3 style='color:#e67e22;'>Additional Note</h3>", unsafe_allow_html=True)
    
    st.markdown("* TF (Term Frequency): The number of times a word appears in a document is its Term Frequency<sup>*</sup>.", unsafe_allow_html=True)
    st.markdown("A higher value means a term appears more often than others, and so, the document is a good match when the term is part of the search terms.")
    st.markdown("IDF (Inverse Document Frequency): Words that occur many times a document, but also occur many times in many others, may be irrelevant.")
    st.markdown("IDF is a measure of how significant a term is in the entire corpus.")
    st.markdown("The TfidfVectorizer converts a collection of raw documents into a matrix of TF-IDF features.")
    st.markdown("new feature axes created by PCA.")
    st.markdown("Principal Component 1 (PC1) is the direction with the most variance in your dataset — it captures the largest difference between your samples.")
    st.markdown("Principal Component 2 (PC2) is the direction orthogonal (perpendicular) to PC1, with the second most variance.")
    
                
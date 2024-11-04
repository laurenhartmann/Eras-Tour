#!/usr/bin/env python
# coding: utf-8

# In[10]:


############ LOAD DATA ##########
import warnings
warnings.filterwarnings("ignore")

### main libraries
import pandas as pd
import numpy as np
import math 

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
import streamlit as st

import joblib
from sklearn.metrics import accuracy_score


# In[11]:


data = pd.read_csv('eras.csv')


########################## CLEAN AND PREP ###############

predict_data = pd.DataFrame(data[(data['Tour Leg']=='Europe Leg')|(data['Tour Leg']=='North America Leg 2')])

columns = ['Lover Bodysuit', 'The Man Jacket', 'Lover Guitar', 'Fearless Dress', 
           'Red Shirt', 'Speak Now Gown', 'Rep Jumpsuit', 'folklore Dress', 'New 1989 Top', 'New 1989 Skirt', 'Match?',
           'TTPD Dress', 'Broken Heart Set', 'Broken Heart Jacket', 'Midnights Shirt', 'Midnights Bodysuit',
           'Karma Jacket', 'Surprise Song Dress']
predict = pd.DataFrame(predict_data[columns])

dict1 = {'Pink & Blue':1,'Pink & Orange':2, 'Blue & Gold':3,'Purple Tassles':4,'All Pink':5}
dict2 = {'Silver Jacket':1,'Orange':2, 'Black Jacket':3,'Indigo Jacket':4,"Hot Pink":5}
dict3 = {'Pink Guitar':1,'Blue Guitar':2,'Purple Guitar':3}
dict4 = {'Silver/Black/Gold':1,'Short Fringe':2,'Gold/Black Tiger':3, 'Long Silver':4,'Long Gold':5,
        'Blue & Silver':6}
dict6 = {'I Bet You Think':1,'A Lot Going On':2,'Ew':3,"Taylor's Version":4, "Trouble":5, "Never Ever":6}
dict7 = {'Swirls':1,'Champagne':2,'Purple':3,'Blue':4,'Purple Waves':5,'Teal Sparkles':6}
dict8 = {'Red & Black':1, 'Gold & Black':2}
dict9 = {'Berry':1,"Green":2,'Yellow':3,'Cream':4,'Blue':5,'Pink':6,'Purple':7}
dict10 = {'Yellow':1, 'Blue':2,'Purple':3,'Pink':4,'Orange':5,'Green':6}
dict11 = {'Yellow':5, 'Blue':2,'Purple':3,'Pink':1,'Orange':4,'Green':6}
dict12 = {'Skirt-Left, Top-Right':0,'Top-Left, Skirt-Right':0,'Not Matching':0,'Matching Set':1}
dict13 = {'I Love You':1, "Who's Afraid?":2}
dict14 = {'White-Silver':1,'Black':2,'Graphite':3, 'Gold & Black':4}
dict15 = {'White-Silver Trim':1,'Silver-Black Trim':4,'Graphite-Black Trim':2, 'Gold-Black Trim':3}
dict16= {'Orchid':1,'Pink':2,'Silver':3,'Blue':4,'Iridescent':5,'Purple':6,'Purple Sequin':7,
        'Blue/Silver Sequin':8}
dict17 = {'Chevron':1,'Swirls':2,'Cutout':3,'Moonstone':4}
dict18 = {'Magenta':1,'Blue':2, 'Multicolor':3}
dict19 = {'Orange':1,'Pink':2,'New Blue':3,'Betta Fish':4,'Blurple':5,'Sunrise Blvd':6, 'Rocketpop':7}

predict['Lover Bodysuit'] = predict['Lover Bodysuit'].replace(dict1)
predict['The Man Jacket'] = predict['The Man Jacket'].replace(dict2)
predict['Lover Guitar'] = predict['Lover Guitar'].replace(dict3)
predict['Fearless Dress'] = predict['Fearless Dress'].replace(dict4)
predict['Red Shirt'] = predict['Red Shirt'].replace(dict6)
predict['Speak Now Gown'] = predict['Speak Now Gown'].replace(dict7)
predict['Rep Jumpsuit'] = predict['Rep Jumpsuit'].replace(dict8)
predict['folklore Dress'] = predict['folklore Dress'].replace(dict9)
predict['New 1989 Top'] = predict['New 1989 Top'].replace(dict10)
predict['New 1989 Skirt'] = predict['New 1989 Skirt'].replace(dict11)
predict['Match?'] = predict['Match?'].replace(dict12)
predict['TTPD Dress'] = predict['TTPD Dress'].replace(dict13)
predict['Broken Heart Set'] = predict['Broken Heart Set'].replace(dict14)
predict['Broken Heart Jacket'] = predict['Broken Heart Jacket'].replace(dict15)
predict['Midnights Shirt'] = predict['Midnights Shirt'].replace(dict16)
predict['Midnights Bodysuit'] = predict['Midnights Bodysuit'].replace(dict17)
predict['Karma Jacket'] = predict['Karma Jacket'].replace(dict18)
predict['Surprise Song Dress'] = predict['Surprise Song Dress'].replace(dict19)

######## MODEL AND INTERFACE #####################



# Step 2: Load and prepare the dataset
model = predict.dropna()

# Replace 'predictor_col' with the actual name of your predictor column
X = model[['Lover Bodysuit']]
Y = model.drop(columns=['Lover Bodysuit'])

# Step 3: Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Step 4: Define and train the multi-output classifier
# We use RandomForestClassifier, but you can substitute with other classifiers
base_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
multi_output_clf = MultiOutputClassifier(base_classifier, n_jobs=-1)
multi_output_clf.fit(X_train, Y_train)

# Step 5: Make predictions and evaluate the model
Y_pred = multi_output_clf.predict(X_test)

# Convert predictions to DataFrame to easily view classification reports for each target
Y_pred_df = pd.DataFrame(Y_pred, columns=Y_test.columns)



# Save the model
joblib.dump(multi_output_clf, 'multi_output_model.pkl')

# Load the trained model
model = joblib.load('multi_output_model.pkl')

column_mappings = {'Lover Bodysuit':dict((v, k) for k, v in dict1.items()),
                  'The Man Jacket':dict((v, k) for k, v in dict2.items()),
                  'Lover Guitar':dict((v, k) for k, v in dict3.items()),
                  'Fearless Dress':dict((v, k) for k, v in dict4.items()),
                  'Red Shirt':dict((v, k) for k, v in dict6.items()),
                  'Speak Now Gown':dict((v, k) for k, v in dict7.items()),
                  'Rep Jumpsuit':dict((v, k) for k, v in dict8.items()),
                  'folklore Dress':dict((v, k) for k, v in dict9.items()),
                  'New 1989 Top':dict((v, k) for k, v in dict10.items()),
                    'New 1989 Skirt':dict((v, k) for k, v in dict11.items()),
                   "Match?":dict((v, k) for k, v in dict12.items()),
                   "TTPD Dress":dict((v, k) for k, v in dict13.items()),
                   "Broken Heart Set":dict((v, k) for k, v in dict14.items()),
                   "Broken Heart Jacket":dict((v, k) for k, v in dict15.items()),
                   "Midnights Shirt":dict((v, k) for k, v in dict16.items()),
                   "Midnights Bodysuit":dict((v, k) for k, v in dict17.items()),
                   "Karma Jacket":dict((v, k) for k, v in dict18.items()),
                   "Surprise Song Dress":dict((v, k) for k, v in dict19.items())}

# Define a list of options for the predictor column (use unique values in your column)
predictor_options = predict['Lover Bodysuit'].dropna().unique()

predictor_map = dict((v, k) for k, v in dict1.items())


# Function to make predictions based on the selected predictor value
def predict_values(predictor_value):
    
    # Prepare input data
    input_data = pd.DataFrame({ 'Lover Bodysuit': [predictor_value] })
    
    # Get predictions
    predictions = model.predict(input_data)
    
    # Convert predictions to a readable DataFrame with string values
    predictions_df = pd.DataFrame(predictions, columns=Y.columns)
    for col in predictions_df.columns:
        if col in column_mappings:
            predictions_df[col] = predictions_df[col].map(column_mappings[col])
    
    # Calculate accuracy (dummy calculation based on training data for example purposes)
    Y_train_pred = model.predict(X_train)
    accuracy_scores = {
        col: accuracy_score(Y_train[col], Y_train_pred[:, idx])
        for idx, col in enumerate(Y.columns)
    }
    # Add accuracy as a row in the predictions DataFrame
    predictions_df.loc['Accuracy'] = [f"{accuracy_scores[col]:.2%}" for col in predictions_df.columns]
    
    return predictions_df



# Streamlit Interface
# Sidebar navigation
page = st.sidebar.selectbox("Choose a page", ["About", "Outfit Predictor", "Surprise Songs"])

# Eras Tour Prediction Page
if page == "Outfit Predictor":
    st.image("eras.jpg", caption="Eras Tour Stage image", use_column_width=True)
    st.title("Eras Tour Outfits & Details Predictions")

    # Dropdown for predictor selection
    selected_predictor = st.selectbox(
        'Select A Lover Body Suit',
        options=list(predictor_map.values())
    )

    # Convert selected string to corresponding numeric value
    predictor = dict1[selected_predictor]

    # Display results when button is clicked
    if st.button("Predict"):
        predictions_df = predict_values(predictor)
        
        # Display predictions as a transposed table for readability
        st.write("**Predicted Values:**")
        st.table(predictions_df.transpose())

# About Page
elif page == "About":
    st.title("Welcome!")
    st.write("""
        This app uses descriptive statistics and machine learning to provide insights about 
        the Eras Tour, predict details and outfit choices, and much more.
        
        ### How The Model Works
        - The app leverages a trained Random Forest model to make predictions.
        - Choose an outfit characteristic for the Lover Bodysuit, and the model predicts associated tour outfit details.
        
        ### Additional Information
        Upcoming will be details about the surprise songs, and other descriptive stats as the final leg of the tour
        comes to a close
    """)
elif page == "Surprise Songs":
           st.title("Surprise Songs Stats")
           st.write("""Let's take a look at the albums Taylor plays from for both the guitar and piano surprise songs:
           
           ### Guitar Songs""")
           st.image("Guitar Surprise Songs.png", caption= "Pie Chart of Guitar Song Albums")
           st.write("""### Piano Songs""")
           st.image("Piano Surprise Songs.png", caption= "Pie Chart of Piano Song Albums")


# In[ ]:





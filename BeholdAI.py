import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import plotly.express as px
import seaborn as sns 
import streamlit as st

from datetime import datetime as dt
from sklearn.metrics import (
    fbeta_score,
    matthews_corrcoef
)

st.set_option('deprecation.showPyplotGlobalUse', False)

# Draw a title and some text to the app:
st.markdown("""
# Behold AI Data Visualization
""")

def view_rows_dup_ids(df, id_col):
    return df[df.duplicated(subset=[id_col], keep=False)].sort_values(id_col)


@st.cache(show_spinner=True, persist=True)
def load_data():
    """
    Load in json data files and perform preliminary manipulations
    """
    with open("model_outputs.json") as json_file:
        model_outputs_raw = pd.read_json(json_file)
        model_outputs_df = pd.json_normalize(model_outputs_raw['class_scores'])
        # always worth checking for white space
        model_outputs_df['accession_number'] = model_outputs_raw['accession_number'].apply(lambda x: x.strip())

    with open("hospital_records.json") as json_file:
        hospital_records_df = pd.DataFrame(json.load(json_file))
        # always worth checking for white space
        hospital_records_df['Accession Number'] = hospital_records_df['Accession Number'].apply(lambda x: x.strip())

    with open("radiologist_labels.json") as json_file:
        radiologist_labels = json.load(json_file)
        radiologist_labels_df = pd.json_normalize(radiologist_labels)
        # always worth checking for white space
        radiologist_labels_df['dicom_elements.value'] = radiologist_labels_df['dicom_elements.value'].apply(lambda x:
                                                                                                            x.strip())

    return radiologist_labels_df, hospital_records_df, model_outputs_df



radiologist_labels_df, hospital_records_df, model_outputs_df = load_data()


@st.cache(show_spinner=True, persist=True)
def hospital_records_cleaning(hospital_records_df):
    """
    Remove duplicates from hospital_records and model_outputs_df and standarsize the probabilistic classifications.
    """
    logging=[]
    logging.append(len(hospital_records_df))
    # select single row for 'true duplicates'
    hospital_records_df.drop_duplicates(keep='first', inplace= True)
    logging.append(len(hospital_records_df))

    # removing complete duplicates
    hospital_records_df.drop_duplicates(subset=['Accession Number'], keep=False, inplace=True) # has no effect
    logging.append(len(hospital_records_df))

    return hospital_records_df, logging


@st.cache(show_spinner=True, persist=True)
def model_outputs_cleaning(model_outputs_df):
    logging=[]

    # do you even need this line? -> yes hospital_records_df has boolean output not probabilities
    model_outputs_df['Normal'] = model_outputs_df[['Abnormal','Normal']].apply(lambda x: x[1] if not pd.isnull(x[1]) \
                                                                                                else 1 - x[0], axis=1)
    # isolate useful columns.
    model_outputs_df = model_outputs_df[['Normal','accession_number']]

    logging.append(len(model_outputs_df))
    # salvage single row for 'true duplicates'
    model_outputs_df.drop_duplicates(keep='first', inplace=True)
    logging.append(len(model_outputs_df))

    model_outputs_dirty = model_outputs_df.sort_values('accession_number')

    # two of the same sample with different labels -> impossible to know which is the 'correct' label.
    model_outputs_df.drop_duplicates(subset=['accession_number'], keep=False, inplace=True)
    logging.append(len(model_outputs_df))
    return model_outputs_df, logging, model_outputs_dirty


@st.cache(show_spinner=True, persist=True)
def radiologist_labels_cleaning(radiologist_labels_df):
    """
    Some rows have duplicates even once distilled to Normal vs. Abormal, in the absence of any distinguishing 
    features (such as an obvious indication that one is a correction) any rows with these accession numbers
    must be removed.
    """
    logging=[]
    logging.append(len(radiologist_labels_df))
    radiologist_labels_df['is_normal_radiologist'] = radiologist_labels_df['classes'].apply(lambda x: \
                                                                                        1 if x[0] == 'Normal' else 0)
    radiologist_labels_df.drop(columns=['classes','dicom_elements.name'], inplace=True)

    # gets distinct rows in cases where the row is perfectly duplicated
    radiologist_labels_df.drop_duplicates(keep='first', inplace=True)
    logging.append(len(radiologist_labels_df))

    # two of the same sample with different labels -> impossible to know which is the 'correct' label.
    radiologist_labels_df.drop_duplicates(subset=['dicom_elements.value'], keep=False, inplace=True)
    logging.append(len(radiologist_labels_df))
    return radiologist_labels_df, logging


# should have all duplicates/corrupt data removed at this point
model_outputs_df, model_outputs_logging, model_outputs_dirty = model_outputs_cleaning(model_outputs_df)
radiologist_labels_df, radiologist_labels_logging = radiologist_labels_cleaning(radiologist_labels_df)
hospital_records_df, hospital_records_logging = hospital_records_cleaning(hospital_records_df)

st.write("""
The following metrics are provided after selecting a single row to represent _perfect_ duplicates (i.e. multiple 
rows with identical values accross columns) and the removal of rows with different data but the same accession numbers. 
This is based on the assumption that there is no way to confidently determine what data should be kept and which removed. 
""")


data_cleaning = st.beta_expander("Data Checks & Cleaning")
data_cleaning.write(f"""
* Length of model-outputs before any cleaning **{model_outputs_logging[0]}**. \n
* Length of model-outputs after removing _perfect_ duplicates **{model_outputs_logging[1]}**. \n
* Length of model-outputs after removing corrupt data (duplicate accession numbers) **{model_outputs_logging[2]}**. _(review this)_\n 

* Length of radiologist-labels before any cleaning **{radiologist_labels_logging[0]}**. \n
* Length of radiologist-labels after removing _perfect_ duplicates **{radiologist_labels_logging[1]}**. \n
* Length of radiologist-labels after removing corrupt data (duplicate accession numbers) **{radiologist_labels_logging[2]}**. \n

* Length of radiologist-labels before any cleaning **{hospital_records_logging[0]}**. \n
* Length of radiologist-labels after removing _perfect_ duplicates **{hospital_records_logging[1]}**. \n
* Length of radiologist-labels after removing corrupt data (duplicate accession numbers) **{hospital_records_logging[2]}**. \n
""")


def select_institute(df, institution):
    if institute != 'All':
        df = df[df['institution'] == institute]
    return df


@st.cache(show_spinner=True, persist=True)
def get_average_processing_time(hospital_records_df):
    hospital_records_df['time_diff'] = pd.to_datetime(hospital_records_df['Result Message Sent'])  - \
                                       pd.to_datetime(hospital_records_df['Final Image Received']) 
    mean_time = hospital_records_df['time_diff'].mean()
    median_time = hospital_records_df['time_diff'].median()
    return mean_time, median_time


def split_dt(dt):
    return dt.seconds//3600, (dt.seconds//60)%60, (dt.seconds%60)


st.write(f"## Business Metrics (Operations)")

institute = st.selectbox('Which hospital would you like to analyse? (Note All is an option)', 
                         ['All'] + list(hospital_records_df['institution'].unique()))

hospital_records_df = select_institute(hospital_records_df, institute)
mean_time, median_time = get_average_processing_time(hospital_records_df)

mean_time_hours, mean_time_minutes, mean_time_seconds = split_dt(mean_time)
median_time_hours, median_time_minutes, median_time_seconds = split_dt(median_time)

st.write(f"""
* Total number of exams processed is **{hospital_records_df['Accession Number'].nunique()}**.
* Mean average time taken to process image is **{mean_time_hours} hours \
                                                       {mean_time_minutes} minutes \
                                                       {mean_time_seconds} seconds**.
* Median average time taken to process image is **{median_time_hours} hours \
                                                       {median_time_minutes} minutes \
                                                       {median_time_seconds} seconds**.
""")
                                                    

st.markdown("""
## Classification Metrics

_**NB** - These are not hospital specific._ \n
In the context of radiology, false positives (where a positive in this case is defined by being classified as normal /
cancer free) are much more costly than false negatives. 

With this definition of positive, precision should be prioritised. The F-beta score is an effective way to balance 
the importance of precision against recall.
""")

st.write(r"""
$$
F \beta Score = (1 + \beta^2) * \frac{(1 + \beta^2) * TP}{(1 + \beta^2) * TP + \beta^2 * FN + FP}
$$

Where the user attributes $\beta$ times as much importance to recall as precision.

* **_Low precision_** = undetected cancer patient's treatment gets delayed.
* **_Low recall_** = unnecessary further imaging and medical treatment.
""")



@st.cache(show_spinner=True, persist=True, allow_output_mutation=True)
def join_datasets(radiologist_labels_df, model_outputs_df):
    radiologist_model = pd.merge(radiologist_labels_df, 
                            model_outputs_df, 
                            how='inner',
                            left_on='dicom_elements.value', 
                            right_on='accession_number') 
    return radiologist_model

radiologist_model = join_datasets(radiologist_labels_df, model_outputs_df)

st.write(f"""
## Data Quality

* Number of rows lost from radiologist labels in the join = **{len(radiologist_labels_df) - len(radiologist_model)}**
* Number of rows lost from hospital records in the join = **{len(model_outputs_df) - len(radiologist_model)}**
* Total number of rows remaining after inner join is **{len(radiologist_model)}**

""")


@st.cache(show_spinner=True, persist=True, allow_output_mutation=True)
def retrieve_results_of_threshold(abnormal_threshold, radiologist_model):
    """
    Calculate the metrics for every threshold to improve the load performance while tweaking. Greater upfront cost
    but much more interactive
    """
    radiologist_model['is_normal'] = radiologist_model['Normal'].apply(lambda x: 1 if x < abnormal_threshold else 0)
    count_normal = radiologist_model['is_normal'].sum()
    count_abnormal = len(radiologist_model) - count_normal
    return radiologist_model, count_normal, count_abnormal


@st.cache(show_spinner=True, persist=True)
def calc_fbeta(radiologist_model, beta, ab_threshold):
    radiologist_model['is_normal'] = radiologist_model['Normal'].apply(lambda x: 1 if x < ab_threshold else 0)
    # fbeta_score(y_true, y_pred)
    return fbeta_score(radiologist_model['is_normal_radiologist'], radiologist_model['is_normal'], beta=beta)
    

abnormal_threshold = st.slider('Abnormal Threshold', 0.00, 1.00, value=0.5)  
radiologist_model, count_normal, count_abnormal = retrieve_results_of_threshold(abnormal_threshold, radiologist_model)

# number classified as abnormal should go down as you increase the threshold (literally what increasing a threshold means)
st.write(f"Number classified as abnormal at this threshold = **{count_abnormal}**")
st.write(f"Number classified as normal at this threshold = **{count_normal}**")


@st.cache(show_spinner=True, persist=True, allow_output_mutation=True)
def return_fbeta_score_plot(radiologist_model, beta):
    """
    Plot a line graph of the f1 score at different thresholds, just put a line at the selected value.
    """
    f_beta_scores = [] # X = Y = [] makes them share the same space in memory
    ab_thresholds = []
    for ab_threshold in range(0,100):
        ab_threshold = ab_threshold/100
        ab_thresholds.append(ab_threshold)
        f_beta_score = calc_fbeta(radiologist_model, beta, ab_threshold)
        f_beta_scores.append(f_beta_score)

    fbeta_scores_df = pd.DataFrame({'F Beta-Score': f_beta_scores, 'Abnormal Threshold': ab_thresholds})
    # Having this marker probably significantly slows down rendering because it will have to reload for every change.
    return px.line(fbeta_scores_df, x='Abnormal Threshold', y='F Beta-Score').update_layout(
                                                                            shapes=[dict(
                                                                                        type= 'line',
                                                                                        yref= 'paper', y0= 0, y1= 1,
                                                                                        xref= 'x', x0= abnormal_threshold
                                                                                                 , x1= abnormal_threshold,
                                                                                                 line=dict(
                                                                                                    color="Red",
                                                                                                    width=4,
                                                                                                    dash="dashdot",
                                                                                                )
                                                                                    )])\
                                                              .update_traces({'line':{'color':'black'}})\
                                                              .update_layout({
                                                                        'plot_bgcolor': 'rgb(224, 208, 208)'
                                                                    })



beta = st.slider('Beta', 0.01, 2.0, value=0.5)  
st.write(f"""
Set beta to 1.0 to retrieve the F1 Score. For the reasons already discussed the default is set to 0.5 
meaning that precision is 2_x_ more important than recall. In the cancer classification context this is 
probably the maximum reasonable value.

The selected value of beta (**{beta}**) means that you consider recall **{beta}**_x_ more important than 
precision and precision **{str(1/beta)[:4]}**_x_ more important than recall.
""")


st.write(return_fbeta_score_plot(radiologist_model, beta))
f_beta = calc_fbeta(radiologist_model, beta, abnormal_threshold)
st.write(f"F beta-score at that beta (**{str(beta)[:4]}**) and that threshold (**{abnormal_threshold}**) is **{str(f_beta)[:4]}**.")


expander = st.beta_expander("Brief Aside")
expander.write("""
Since my original application I've discovered the Msc - Computer Graphics, Vision and Imaging at UCL. 
While acceptance is no sure thing 3 of the modules may as well be described as training for Behold AI.

* Computational Modelling for Biomedical Imaging (15 credits)
* Information Processing in Medical Imaging (15 credits)
* Inverse Problems in Imaging (15 credits) 

https://www.ucl.ac.uk/prospective-students/graduate/taught-degrees/computer-graphics-vision-imaging-msc
""")
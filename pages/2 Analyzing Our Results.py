import streamlit as st
import plotly.express as px
from streamlit_plotly_events import plotly_events
import pandas as pd
import numpy as np
import os
import umap
import boto3

from streamlit_extras.app_logo import add_logo
add_logo('fiddler-ai-logo.png', height=50)

AWS_S3_BUCKET_NAME = st.secrets['AWS_S3_BUCKET_NAME']
AWS_DYNAMODB_TABLE_NAME = st.secrets['AWS_DYNAMODB_TABLE_NAME']
AWS_REGION = st.secrets['AWS_REGION']

AWS_ACCESS_KEY_ID = st.secrets['AWS_ACCESS_KEY_ID']
AWS_SECRET_ACCESS_KEY = st.secrets['AWS_SECRET_ACCESS_KEY']

TEMP_IMAGE_PATH = './temp'

AWS_S3_BASE_URL = f'https://{AWS_S3_BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/'

SESSION_ID = st.secrets['SESSION_ID']

ddb_table = boto3.resource("dynamodb",
                           region_name=AWS_REGION,
                           aws_access_key_id=AWS_ACCESS_KEY_ID,
                           aws_secret_access_key=AWS_SECRET_ACCESS_KEY).Table(
    AWS_DYNAMODB_TABLE_NAME)

s3_bucket = boto3.resource('s3',
                           region_name=AWS_REGION,
                           aws_access_key_id=AWS_ACCESS_KEY_ID,
                           aws_secret_access_key=AWS_SECRET_ACCESS_KEY).Bucket(AWS_S3_BUCKET_NAME)


def transform(x):
    x['embedding'] = [float(xx) for xx in x['embedding']]
    return x


@st.cache_data(ttl=30)
def get_db_data():
    response = ddb_table.scan()
    data = [transform(x) for x in response['Items']]
    while 'LastEvaluatedKey' in response:
        response = ddb_table.scan(ExclusiveStartKey=response['LastEvaluatedKey'])
        data = data + [transform(x) for x in response['Items']]

    raw_df = pd.DataFrame(data).drop(['time', 'clue'], axis=1)

    raw_df.to_csv('gen_workshop_dataframe.csv', index=False)

    with open('gen_workshop_dataframe.csv', 'rb') as f:
        s3_bucket.upload_fileobj(f, 'gen_workshop_dataframe.csv')

    raw_df.rename(columns={'category': 'Newspaper Section',
                           'feedback_fidelity': '[Feedback] Fidelity',
                           'feedback_bias': '[Feedback] Bias',
                           'feedback_quality': '[Feedback] Quality',
                           'feedback_distortion': '[Feedback] Distortion',
                           'feedback_notes': '[Feedback] Notes'}, inplace=True)

    return raw_df


@st.cache_data
def run_umap(the_df):
    embs = np.asarray(the_df['embedding'].to_list())

    reducer = umap.UMAP(n_components=2, n_neighbors=3, random_state=42)

    umap_embs = reducer.fit_transform(embs[:, :128])

    return pd.DataFrame({'UMAP_0': umap_embs[:, 0],
                         'UMAP_1': umap_embs[:, 1],
                         # 'UMAP_2': umap_embs[:, 2]
                         })


def add_umap(df):
    umap_df = run_umap(df)
    return pd.concat([df.reset_index(drop=True), umap_df], axis=1)


st.header("Part 2: Evaluating our Data and Feedback")

df = get_db_data().copy()

c1, c2 = st.columns(2)

with c1:
    # sessions = df['session_id'].unique().tolist()
    # session = st.selectbox('Filter by session', ['All'] + sessions, index=len(sessions))
    session = SESSION_ID
    users = df['user'].unique().tolist()
    user = st.selectbox('Filter by user? Optional e.g. to find your own.', ['All'] + users)

with c2:
    color_by_options = ['Newspaper Section',
                        '[Feedback] Fidelity',
                        '[Feedback] Bias',
                        # '[Feedback] Distortion',
                        '[Feedback] Quality']

    color_by = st.selectbox('Color UMAP/semantic plot by:', color_by_options)

df = df[df['session_id'] == session]

if len(df) == 0:
    st.write(f'{len(df)} records returned')
    st.stop()

df = add_umap(df)

d = df[~df[color_by].isnull()]

if user != 'All':
    d = d[d['user'] == user]

if session != 'All':
    d = d[(d['session_id'] == session)]

st.write(f'{len(d)} records returned')

if len(d) == 0:
    st.stop()

st.title('')

d.sort_values(color_by, inplace=True, ascending=False)

gradient_colors = [
    "#4169E1",  # Royal Blue
    "#6495ED",  # Cornflower Blue
    "#87CEEB",  # Sky Blue
    "#B0E0E6",  # Powder Blue
    "#E6E6FA",  # Lavender
]

discrete_colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628']

style_by_column = {'Newspaper Section': {
                        'symbol': color_by,
                        'color_discrete_sequence': discrete_colors},
                   '[Feedback] Fidelity': {
                       'symbol': color_by,
                       'color_discrete_sequence': discrete_colors},
                   '[Feedback] Bias': {
                       'symbol': color_by,
                       'color_discrete_sequence': discrete_colors},
                       '[Feedback] Distortion': {
                       'symbol': color_by,
                       'color_discrete_sequence': discrete_colors},
                   '[Feedback] Quality': {
                       'color_discrete_sequence': gradient_colors
                   }}

st.write('The UMAP coordinates below are computed from the embeddings of your prompt. Any clustering that '
         'occurs is due to prompt semantics.  By overlaying human feedback, we can identify semantically '
         'correlated problem areas.')

st.write('**Click a point below to retrieve its details.**')

fig = px.scatter(
    data_frame=d,
    x="UMAP_0",
    y="UMAP_1",
    color=color_by,
    **style_by_column[color_by])

fig.update_traces(marker={'size': 9})
fig.update_xaxes(showticklabels=False, zeroline=False)
fig.update_yaxes(showticklabels=False, zeroline=False)
fig.update_layout({"uirevision": "foo"}, overwrite=True)
selected_points = plotly_events(fig, click_event=True, select_event=False)


st.markdown('*Double-click plot to reset plot range.*')

st.title('')

col_list = [None] * len(selected_points)

c3, c4 = st.columns(2)
if len(selected_points):

    if not os.path.exists(TEMP_IMAGE_PATH):
        os.makedirs(TEMP_IMAGE_PATH)

    row_ids = [d[d['UMAP_0'] == p['x']].index.tolist()[0] for p in selected_points]

    for i, row_id in enumerate(row_ids):

        col_list[i] = st.columns(2)

        row = d.loc[row_id]

        with col_list[i][0]:
            file_name = row['prompt_id'] + '.png'
            st.image(AWS_S3_BASE_URL+file_name)

        with col_list[i][1]:
            st.markdown(f'**Newspaper Section:** {row["Newspaper Section"]}')
            for k, v in row['features'].items():
                st.markdown(f'**{k}:** {v}')

            st.write(f'**User generated prompt:** "{d.loc[row_id]["prompt"]}"')

            st.write(f'**User:** {row["user"]}')

            # st.write(d.loc[row_id])

            feedback_types = ['[Feedback] Fidelity',
                              '[Feedback] Bias',
                              '[Feedback] Distortion',
                              '[Feedback] Quality',
                              '[Feedback] Notes']

            for f in feedback_types:
                st.markdown(f'**{f}:** {row[f]}')

        break
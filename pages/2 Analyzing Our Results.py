import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events
import pandas as pd
import numpy as np
import os
import umap
import boto3

AWS_S3_BUCKET_NAME = st.secrets['AWS_S3_BUCKET_NAME']
AWS_DYNAMODB_TABLE_NAME = st.secrets['AWS_DYNAMODB_TABLE_NAME']
AWS_REGION = st.secrets['AWS_REGION']

AWS_ACCESS_KEY_ID = st.secrets['AWS_ACCESS_KEY_ID']
AWS_SECRET_ACCESS_KEY = st.secrets['AWS_SECRET_ACCESS_KEY']

TEMP_IMAGE_PATH = './temp'

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

    raw_df = pd.DataFrame(data)

    raw_df.to_csv('gen_workshop_dataframe.csv', index=False)
    with open('gen_workshop_dataframe.csv', 'rb') as f:
        s3_bucket.upload_fileobj(f, 'gen_workshop_dataframe.csv')

    return raw_df

@st.cache_data
def run_umap(the_df):
    embs = np.asarray(the_df['embedding'].to_list())

    reducer = umap.UMAP(n_components=2, n_neighbors=4, random_state=42)

    umap_embs = reducer.fit_transform(embs[:, :16])

    return pd.DataFrame({'UMAP_0': umap_embs[:, 0],
                         'UMAP_1': umap_embs[:, 1],
                         # 'UMAP_2': umap_embs[:, 2]
                         })


def get_data():
    raw_df = get_db_data()



    umap_df = run_umap(raw_df)
    df = raw_df.copy()
    return pd.concat([df, umap_df], axis=1)


st.header("Part 2: Evaluating out Data and Feedback")

df = get_data().copy()

c1, c2 = st.columns(2)

with c1:
    sessions = df['session_id'].unique().tolist()
    session = st.selectbox('Filter by session', ['All'] + sessions, index=len(sessions))

    users = df['user'].unique().tolist()
    user = st.selectbox('Filter by user? Optional e.g. to find your own.', ['All'] + users)

with c2:
    feedback_types = [x.split('_')[1] for x in df.columns if 'feedback' in x]
    feedback = st.selectbox('Feedback type to color UMAP/semantic plot.', feedback_types, index = feedback_types.index('quality'))

st.subheader('Click a point on plot to retrieve record.')

if user == 'All':
    d = df
else:
    d = df[df['user'] == user]

if session != 'All':
    d = d[d['session_id'] == session]

st.write(f'{len(d)} records returned')

if len(d) == 0:
    st.stop()

st.title('')


cc = d['feedback_' + feedback].values

if type(cc[0]) is str:
    cc = [1 if x == "Yes" else 0 for x in cc]

sort_col = 'feedback_' + feedback

vals = sorted(d[sort_col].unique(), reverse=True)

cols = []
for i, val in enumerate(vals):

    frac_i = (len(vals)-i-1)/len(vals)
    cols.append('#%02x%02x%02x' % (round(225*frac_i), round(60*frac_i), round(215*frac_i)))

#print(col)

# fig = go.Figure()
#

#
# fig.add_trace(go.Scatter(
#                 x=d['UMAP_0'].values,
#                 y=d['UMAP_1'].values,
#                 # z=d['UMAP_2'].values,
#                 mode='markers',
#                 # name=str(val),
#                 marker=dict(
#                     size=9,
#                     color=cc,
#                     opacity=0.8,
#                     colorscale='Viridis',  # choose a colorscale
#                 ),
#              ))



# d.sort_values(sort_col, inplace=True)

# fig = px.scatter_3d(
#     data_frame=d,
#     x="UMAP_0",
#     y="UMAP_1",
#     z="UMAP_2",
#     color="feedback_quality",
#     # color_discrete_sequence=cols
#     color_discrete_sequence=['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33','#a65628','#f781bf','#999999'],
# )
fig = px.scatter(
    data_frame=d,
    x="UMAP_0",
    y="UMAP_1",
    color=sort_col,
    # color_discrete_sequence=cols
    color_discrete_sequence=['#e41a1c',
                             '#377eb8',
                             '#4daf4a',
                             '#984ea3',
                             '#ff7f00',
                             '#ffff33',
                             '#a65628',
                             '#f781bf',
                             '#999999'])

# vals = sorted(d['feedback_' + feedback].unique(), reverse=True)
# for i, val in enumerate(vals):
#     dd = d[d['feedback_' + feedback] == val]
#
#
#     frac_i = (len(vals)-i-1)/len(vals)
#
#     print(frac_i)
#
#     col = '#%02x%02x%02x' % (round(225*frac_i), round(60*frac_i), round(215*frac_i))
#
#     print(col)
#
#     fig.add_trace(go.Scatter3d(
#                     x=dd['UMAP_0'].values,
#                     y=dd['UMAP_1'].values,
#                     z=dd['UMAP_2'].values,
#                     mode='markers',
#                     name=str(val),
#                     marker=dict(
#                         size=9,
#                         color=[col for _ in range(len(dd))],
#                         opacity=0.8,
#                     ),
#                  ))


# fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))

fig.update_layout({"uirevision": "foo"}, overwrite=True)

selected_points = plotly_events(fig, click_event=True, select_event=False)
st.markdown('*Double-click plot to reset range and selected points.*')

st.title('')
st.title('')

col_list = [None] * len(selected_points)

c3, c4 = st.columns(2)
if len(selected_points):

    if not  os.path.exists(TEMP_IMAGE_PATH):
        os.makedirs(TEMP_IMAGE_PATH)

    row_ids = [d[d['UMAP_0'] == p['x']].index.tolist()[0] for p in selected_points]

    for i, row_id in enumerate(row_ids):

        col_list[i] = st.columns(2)

        row = d.loc[row_id]

        with col_list[i][0]:
            file_name = row['prompt_id'] + '.png'
            path = TEMP_IMAGE_PATH + '/' + file_name
            s3_bucket.download_file(file_name, path)
            st.image(path)

        with col_list[i][1]:
            st.write(row['category'])
            for k,v in row['features'].items():
                st.write(f'{k}: {v}')
            st.write(d.loc[row_id]['prompt'])
            st.write(d.loc[row_id])

        break
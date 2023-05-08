import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events
import pandas as pd
import numpy as np
import umap
import boto3

AWS_S3_BUCKET_NAME = st.secrets['AWS_S3_BUCKET_NAME']
AWS_DYNAMODB_TABLE_NAME = st.secrets['AWS_DYNAMODB_TABLE_NAME']
AWS_REGION = st.secrets['AWS_REGION']

AWS_ACCESS_KEY_ID = st.secrets['AWS_ACCESS_KEY_ID']
AWS_SECRET_ACCESS_KEY = st.secrets['AWS_SECRET_ACCESS_KEY']

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

@st.cache_data
def get_data():
    data = [transform(x) for x in ddb_table.scan()['Items']]
    df = pd.DataFrame(data)

    embs = np.asarray(df['embedding'].to_list())

    reducer = umap.UMAP(n_components=3)

    umap_embs = reducer.fit_transform(embs[:, :16])

    df['UMAP_0'] = umap_embs[:, 0]
    df['UMAP_1'] = umap_embs[:, 1]
    df['UMAP_2'] = umap_embs[:, 2]

    return df

df = get_data()

c1, c2 = st.columns(2)

with c1:
    users = df['user'].unique().tolist()
    user = st.selectbox('Filter by user? Optional e.g. to find your own.', ['All'] + users)

with c2:
    feedback_types = [x.split('_')[1] for x in df.columns if 'feedback' in x]
    feedback = st.selectbox('Feedback type to color UMAP/semantic plot.', feedback_types, index = feedback_types.index('quality'))

st.subheader('Drag to rotate plot; click a point to retrieve record.')

if user == 'All':
    d = df
else:
    d = df[df['user'] == user]

st.title('')


# e = np.array(d['umap_embs_0'].to_list())

cc = d['feedback_' + feedback].values

if type(cc[0]) is str:
    cc = [1 if x == "Yes" else 0 for x in cc]


fig = go.Figure()

vals = sorted(d['feedback_' + feedback].unique(), reverse=True)
# for i, val in enumerate(vals):

# frac_i = (len(vals)-i-1)/len(vals)

# print(frac_i)

# col = '#%02x%02x%02x' % (round(225*frac_i), round(60*frac_i), round(215*frac_i))

#print(col)

fig.add_trace(go.Scatter3d(
                x=d['UMAP_0'].values,
                y=d['UMAP_1'].values,
                z=d['UMAP_2'].values,
                mode='markers',
                # name=str(val),
                marker=dict(
                    size=9,
                    color=cc,
                    opacity=0.8,
                    colorscale='Viridis',  # choose a colorscale
                ),
             ))

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


fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))

fig.update_layout({"uirevision": "foo"}, overwrite=True)

selected_points = plotly_events(fig, click_event=True, select_event=True)

st.title('')
st.title('')

c3, c4 = st.columns(2)


if len(selected_points):
    with c3:
        file_name = d.iloc[selected_points[0]['pointNumber']]['prompt_id'] + '.png'
        s3_bucket.download_file(file_name, './' + file_name)
        st.image('./tmp/' + file_name)
    with c4:
        st.write(d.iloc[selected_points[0]['pointNumber']])

import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import umap
from sklearn.cluster import KMeans
import boto3

from scipy.spatial.distance import jensenshannon

def jsd(a, b):
    return jensenshannon(a, b, base=2)


from streamlit_extras.app_logo import add_logo
add_logo('fiddler-ai-logo.png', height=50)

AWS_S3_BUCKET_NAME = st.secrets['AWS_S3_BUCKET_NAME']
AWS_DYNAMODB_TABLE_NAME = st.secrets['AWS_DYNAMODB_TABLE_NAME']
AWS_REGION = st.secrets['AWS_REGION']

AWS_ACCESS_KEY_ID = st.secrets['AWS_ACCESS_KEY_ID']
AWS_SECRET_ACCESS_KEY = st.secrets['AWS_SECRET_ACCESS_KEY']

SESSION_ID = st.secrets['SESSION_ID']

DAYS_IN_GROUP = 4

ddb_table = boto3.resource("dynamodb",
                           region_name=AWS_REGION,
                           aws_access_key_id=AWS_ACCESS_KEY_ID,
                           aws_secret_access_key=AWS_SECRET_ACCESS_KEY).Table(
    AWS_DYNAMODB_TABLE_NAME)

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

    return raw_df

@st.cache_data
def run_umap(the_df):
    embs = np.asarray(the_df['embedding'].to_list())

    reducer = umap.UMAP(n_components=2, n_neighbors=4, random_state=42)

    umap_embs = reducer.fit_transform(embs[:, :128])

    return pd.DataFrame({'UMAP_0': umap_embs[:, 0],
                         'UMAP_1': umap_embs[:, 1],
                         # 'UMAP_2': umap_embs[:, 2]
                         })


def add_umap(df):
    umap_df = run_umap(df)
    return pd.concat([df.reset_index(drop=True), umap_df], axis=1)

df = get_db_data().copy()

df = df[(df['session_id'] == SESSION_ID)]

st.header('Part 3: Measuring Drift in Unstructured Data')

NUM_BINS =  st.select_slider('How many bins should I use for semantic clustering?', value=3, options=[3, 4, 5])

st.text(f'{len(df)} records returned, clustering with {NUM_BINS} bins.')

if len(df) == 0:
    st.stop()

df = add_umap(df)

embs_by_group = [df[df['prompt_number']//DAYS_IN_GROUP == i][['UMAP_0', 'UMAP_1']].values for i in range(3)]


X = embs_by_group[0]
kmeans = KMeans(n_clusters=NUM_BINS, random_state=42).fit(X)

counts = [np.bincount(kmeans.predict(embs_by_group[i]), minlength=NUM_BINS) for i in range(3)]

prompt_labels = ['Prompts 1-4', '5-8 (modified)', '9-12']

st.write('**A distributional shift was introduced to your newspaper topics for prompts five through eight.**  '
         'This is probably visible in the semantic/UMAP plots of the embeddings we calculated from your prompts.')

st.write('The semantic plots below are broken up into three groups of four prompts.  This could be time intervals '
         'where identifying '
         'semantic shifts could help safeguard model performance and safety.')

st.write('Running a clustering algorithm (colors below) makes it possible to track semantic shift with respect to the initial '
         'time interval.')

color_by_cluster = ['k', 'r', 'g', 'b', 'm']

fig, ax = plt.subplots(1, 3, sharex=True)
for cluster in range(NUM_BINS):

    for group in range(3):
        ax[group].set_aspect(1)
        ax[group].xaxis.set_tick_params(labelbottom=False)
        ax[group].yaxis.set_tick_params(labelleft=False)

        ax[group].set_xticks([])
        ax[group].set_yticks([])

        ax[group].set_title(prompt_labels[group])


        clusters = kmeans.predict(embs_by_group[group])

        dat = embs_by_group[group][clusters == cluster]
        if group == 0:
            ax[group].plot(dat[:, 0], dat[:, 1], '.', color=color_by_cluster[cluster],
                     label=str(cluster))

        else:
            ax[group].plot(dat[:, 0], dat[:, 1], '.', color=color_by_cluster[cluster])

st.pyplot(fig)

st.write('We can then create a histogram for each time interval representing a coarse density estimate for the distribution.')

prompt_labels = ['1-4', '5-8 (modified)', '9-12']

WIDTH = 0.2
fig = plt.figure(figsize=[4, 3])
for i, c in enumerate(counts):
    plt.bar(np.arange(NUM_BINS)+(i-1)*WIDTH, height=c/sum(c), width=WIDTH, label=prompt_labels[i]) #, yerr=[[0.01, 0.02, 0.01, 0.02], [0.01, 0.02, 0.01, 0.02]])

plt.xlabel('Cluster', fontsize=8)
plt.xticks(list(range(NUM_BINS)))
plt.title('Histogram of cluster assignments across prompts groups', fontsize=8)
plt.legend(title='Prompts')
st.pyplot(fig)


st.write('Finally, histograms can be compared with that from the initial reference period to measure distributional shift.')

jsd1 = jsd(counts[0], counts[1])
jsd2 = jsd(counts[0], counts[2])

fig = plt.figure(figsize=[4, 3])
plt.plot([0], [jsd1], '+b', label='modified')
plt.plot([1], [jsd2], '+r', label='initial')
plt.xlim(-0.5, 1.5)
plt.ylim(0, 0.5)
plt.ylabel('Jensen-Shannon Distance', fontsize=8)
plt.xlabel('Prompts', fontsize=8)
plt.title('Distributional comparison with Prompts 1-4\n(larger is more different)', fontsize=8)
plt.xticks([0, 1], ['5-8', '9-12'])
plt.legend(title='Distribution')

st.pyplot(fig)

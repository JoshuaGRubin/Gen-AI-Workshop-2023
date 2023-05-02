import io
import time
import boto3
import openai
import base64
import random
import streamlit as st
from uuid import uuid1
from decimal import Decimal

OPENAI_API_KEY = st.secrets['OPENAI_API_KEY']
AWS_S3_BUCKET_NAME = st.secrets['AWS_S3_BUCKET_NAME']
AWS_DYNAMODB_TABLE_NAME = st.secrets['AWS_DYNAMODB_TABLE_NAME']
AWS_REGION = st.secrets['AWS_REGION']

AWS_ACCESS_KEY_ID = st.secrets['AWS_ACCESS_KEY_ID']
AWS_SECRET_ACCESS_KEY = st.secrets['AWS_SECRET_ACCESS_KEY']

s3_bucket = boto3.resource('s3',
                           region_name=AWS_REGION,
                           aws_access_key_id=AWS_ACCESS_KEY_ID,
                           aws_secret_access_key=AWS_SECRET_ACCESS_KEY).Bucket(AWS_S3_BUCKET_NAME)
ddb_table = boto3.resource('dynamodb',
                           region_name=AWS_REGION,
                           aws_access_key_id=AWS_ACCESS_KEY_ID,
                           aws_secret_access_key=AWS_SECRET_ACCESS_KEY).Table(AWS_DYNAMODB_TABLE_NAME)


def get_image(prompt='a gray cat jumping between pieces of furnatiure', size="256x256"):
    response = openai.Image.create(
      api_key=OPENAI_API_KEY,
      n=1,
      prompt=prompt,
      size=size,
      response_format="b64_json"
    )

    return response['data'][0]['b64_json']


def get_embedding(prompt='a gray cat jumping between pieces of furnatiure', model='text-embedding-ada-002'):
    response = openai.Embedding.create(
      api_key=OPENAI_API_KEY,
      input=[prompt],
      model=model        
    )

    return [Decimal(x) for x in response['data'][0]['embedding']]


def submit_data():

    data_dict = {k: v for k, v in state.items() if k in STATE_KEYS}

    s3_bucket.upload_fileobj(io.BytesIO(data_dict.pop(KEY_IMAGE)), data_dict[KEY_UUID] + '.png', )
    print('image written to s3')

    ddb_table.put_item(Item=data_dict)
    print('ddb record written')
    print(data_dict)

    st.balloons()

    next_prompt()





categories = {'places': ['Park', 'A space station', 'Gym', 'Museum', 'Library', 'Coffee Shop',
                        'The planet Mercury', 'Vinyard', 'Country Road', 'Tokyo', 'Paris',
                        'Dubai', 'Mogadishu', 'Vancouver'],
              'activities': ['Playing an instrument', 'Hiking', 'Knitting',
                            'Cooking','Skydiving', 'Reading', 'Skydiving',
                            'Surfing', 'Painting ', 'Playing a sport', 'Writing'],
              'people': ['Painter', 'Musician', 'Doctor', 'Engineer', 'Explorer', 'Soldier', 'Explorer',
                        'Dancer', 'Veterinarian', 'Sculptor', 'Mechanic', 'Librarian'],
              'animals': ['Dog', 'Cat', 'Lizard', 'Bear', 'Snake', 'Insect', 'Elephant','Zebra'],
              'things': ['Tree', 'Tomato', 'Laptop', 'TV', 'Rocket', 'Time Machine', 'Wallet', 'Banana',
                        'Hammer', 'Playing cards', 'Dessert', 'Old shoe', 'Crown', 'Fire', 'Sand', 'Snowball']

             }

def generate_clues():

    n = random.choice([2,  3,])

    cat_keys = list(categories.keys())
    random.shuffle(cat_keys)

    final_categories = [cat_keys.pop() for x in range(n)]

    return {cat: random.choice(categories[cat]) for cat in final_categories}


EMPTY = ''

KEY_USER_ID = 'user'
KEY_PROMPT = 'prompt'
KEY_CLUE = 'clue'
KEY_IMAGE = 'image'
KEY_EMBEDDING = 'embedding'
KEY_TIME = 'time'
KEY_HUMAN_TIME = 'human_time'

KEY_UUID = 'prompt_id'
KEY_FEEDBACK_QUALITY = 'feedback_quality'
KEY_FEEDBACK_FIDELITY = 'feedback_fidelity'
KEY_FEEDBACK_DISTORTION = 'feedback_distortion'
KEY_FEEDBACK_BIAS = 'feedback_bias'
KEY_FEEDBACK_NOTES = 'feedback_notes'

STATE_KEYS = [KEY_USER_ID, KEY_PROMPT, KEY_CLUE, KEY_IMAGE, KEY_EMBEDDING, KEY_TIME, KEY_HUMAN_TIME, KEY_UUID,
              KEY_FEEDBACK_QUALITY, KEY_FEEDBACK_FIDELITY, KEY_FEEDBACK_DISTORTION, KEY_FEEDBACK_BIAS,
              KEY_FEEDBACK_NOTES]

state = st.session_state

for k in STATE_KEYS:
    if k not in state:
        state[k] = EMPTY


def next_prompt():
    state[KEY_PROMPT] = EMPTY
    state[KEY_IMAGE] = EMPTY
    state[KEY_CLUE] = generate_clues()


def reset_results():
    state[KEY_IMAGE] = EMPTY
    state[KEY_EMBEDDING] = EMPTY
    state[KEY_TIME] = EMPTY


st.header("Part 1: Simulating a Generative AI Workflow")
st.subheader("Let's generate some data!")

st.markdown("The purpose of Part I is to generate a collection of data from a generative AI task.  "
            "Imagine that you’re responsible for generating thumbnail images for a newspaper or magazine"
            " based on some features you’ve been given.\n"
            "1. You’ll be presented with a set of keywords.\n"
            "2. Use your imagination to turn them into a prompt for a generative image model."
            "3. Look at the image generated and answer a few questions about how well the model performed.\n"
            "4. Click “Submit” and we’ll store all this info for subsequent analysis.\n")

st.text_input("First, enter your name or an email address. (This is so you can identify data you generated – we won't "
              "add you to any mailing lists without your permission.)", key=KEY_ID)

if state[KEY_ID] == EMPTY:
    st.stop()

st.write(f"Fabulous, {state[KEY_ID]}")

st.markdown('---')
st.subheader("Story Features")
st.write("Please use the following randomly selected features as the seed for a thumbnail image idea.")

if state[KEY_CLUE] == EMPTY:
    state[KEY_CLUE] = generate_clues()

c1, c2 = st.columns(2)

with c1:
    for category, item in state[KEY_CLUE].items():
        st.metric(label=category, value=item)
with c2:
    st.title('')
    st.title('')
    st.title('')
    st.title('')
    if state[KEY_PROMPT] == EMPTY:
        st.button('New clues, please.', on_click=next_prompt)

st.markdown('---')
st.subheader("The prompt")
st.write('Turn your idea into a prompt for an image generation model.  Feel free to embellish!  e.g. "Engineer" + '
         '"Old Shoe" might become "A portrait of an engineer comparing his futuristic boot invention to an old shoe."')

if state[KEY_ID] != EMPTY:
    st.text_input('Enter your prompt:', key=KEY_PROMPT, on_change=reset_results)

if state[KEY_PROMPT] is EMPTY:
    st.stop()

if state[KEY_IMAGE] == EMPTY:
    state[KEY_IMAGE] = base64.b64decode(get_image(state['prompt']))
    state[KEY_EMBEDDING] = get_embedding(state['prompt'])
    state[KEY_TIME] = time.time_ns()//10**9
    state[KEY_HUMAN_TIME] = time.strftime('%a, %d %b %Y %H:%M:%S UTC', time.gmtime(state[KEY_TIME]))
    state[KEY_UUID] = str(uuid1())

col1, col2 = st.columns(2)

with col1:
    st.image(state[KEY_IMAGE])

with col2:
    st.select_slider('How happy are you with the result? (5 is "totally")', key=KEY_FEEDBACK_QUALITY, options=[1, 2, 3, 4, 5])

    st.radio('Did the model include all of the aspects of your prompt?', key=KEY_FEEDBACK_FIDELITY, options=['No', 'Yes'])

    st.radio('Did the model include any visual distortions? (e.g. crazy-fingers)', key=KEY_FEEDBACK_DISTORTION,
                        options=['No', 'Yes'])

    st.radio('Did the model make any assumptions that could indicate bias? (e.g. race, gender, age)',
             key=KEY_FEEDBACK_BIAS, options=['No', 'Yes'])

    st.text_input('[Optional] Additional Notes', key=KEY_FEEDBACK_NOTES)

    col3, col4, col5 = st.columns(3)

    # with col3:
    st.button("Reset :recycle:", on_click=next_prompt)
    # with col4:
    st.button('Submit :rocket:', on_click=submit_data)

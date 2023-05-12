import io
import time
import boto3
import openai
import base64
import random
import streamlit as st
from uuid import uuid1
from decimal import Decimal

SESSION_ID = 'banana'
TOT_PROMPTS_TO_DO = 15

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

    return [Decimal(str(x)[:9]) for x in response['data'][0]['embedding']]


def submit_data():

    data_dict = {k: v for k, v in state.items() if k in STATE_KEYS}

    data_dict[KEY_USER_ID] = data_dict[KEY_USER_ID].lower()

    s3_bucket.upload_fileobj(io.BytesIO(data_dict.pop(KEY_IMAGE)), data_dict[KEY_UUID] + '.png', )
    # print('image written to s3')

    ddb_table.put_item(Item=data_dict)
    # print('ddb record written')
    # print(data_dict)


    state[KEY_PROMPT_NUMBER] += 1

    st.balloons()

    next_prompt()


politics_details = {'Who': ['the president', 'a mayer', 'a senator', 'a politician', 'protestors'],
                    'Where': ['the White House', 'a restaurant', 'a podium', 'a park'],
                    'What': ['a scandal', 'an agreement', 'an election', 'a debate', 'a surprise', 'a celebration']}

business_details = {'Who': ['an executive', 'a mechanic', 'a family', 'workers'],
                    'Where': ['Wall Street', 'a grocery store', 'a shipping container', 'a factory',
                              'a corporate headquarters'],
                    'What': ['an infographic', 'a protest', 'inflation', 'the banking sector', 'corn futures']}

arts_details = {'Who': ['a painter', 'musicians', 'a photographer', 'a child'],
                'Where': ['a gallery', 'a cafe', 'a museum', 'a natural scene'],
                'What': ['a painting', 'a sculpture', 'a performance', 'a paintbrush', 'a sweater', 'a portrait']}

sports_details = {'Game': ['golf', 'soccer', 'baseball', 'cricket', 'football', 'climbing', 'cycling'],
                  'What': ['victory', 'rivalry', 'history', 'weather', 'new record', 'injury'],
                  'Where': ['stadium', 'university', 'back yard', 'arena', 'national park']}

travel_details = {'Where': ['a hotel', 'an airplane', 'city streets', 'a hiking trail', 'a beach on an island'],
                  'What': ['a map', 'noodles', 'luxury', 'delight', 'delay', 'wildlife'],
                  'Who': ['a family', 'a monkey', 'a tour guide', 'a pilot', 'a mountaineer']}

funny_details = {'Who': ['a cat', 'some kids', 'a family', 'some people in an office', 'a clown'],
                 'What': ['pizza', 'ice cream', 'doing homework', 'on a date', 'watching a movie', 'dancing',
                          'a laptop', 'in a treehouse'],
                 'When': ['the stone age', 'the future', '1950s', 'during the pandemic']}

categories_details = {'politics': politics_details,
                      'business': business_details,
                      'arts': arts_details,
                      'sports': sports_details,
                      'travel': travel_details,
                      'cartoon': funny_details}

categories_examples = {
    'politics': ["A photo of the president embarrassed in front of reporters during the correspondents' dinner.",
                 "A black and white photo of a politician with a microphone."],
    'business': ['An graph showing recent dramatic changes in stock prices.',
                 'A photo of factory workers busy building cars in a factory.'],
    'arts': ['A photo of a woman painting a colorful canvas with a large brush.',
             "A painting of a dramatic scene on stage with a person grabbing another by shoulder."],
    'sports': ['A black and white photo of a boxer with hands stretched over their head in victory.',
               'A close-up photo of soccer teammates hugging in the rain after a difficult victory.'],
    'travel': ["An illustration of a map showing a ship's coarse between tropical islands.",
               'A photo of a beautiful hotel and trees and a long driveway in the mountains.'],
    'cartoon': ['An illustration of a clown mowing the lawn.',
              'A cartoon of a horse working on a laptop.']}


def generate_clues(prompt_number=0):
    categories = list(categories_details)
    category = random.choice(categories)

    cat_cats = list(categories_details[category].keys())
    random.shuffle(cat_cats)

    clue0 = random.choice(categories_details[category][cat_cats[0]])
    clue1 = random.choice(categories_details[category][cat_cats[1]])

    return category, {cat_cats[0]: clue0, cat_cats[1]: clue1}


EMPTY = ''

KEY_USER_ID = 'user'
KEY_PROMPT = 'prompt'
KEY_IMAGE = 'image'
KEY_EMBEDDING = 'embedding'
KEY_TIME = 'time'
KEY_HUMAN_TIME = 'human_time'
KEY_PROMPT_NUMBER = 'prompt_number'

KEY_UUID = 'prompt_id'
KEY_FEEDBACK_QUALITY = 'feedback_quality'
KEY_FEEDBACK_FIDELITY = 'feedback_fidelity'
KEY_FEEDBACK_DISTORTION = 'feedback_distortion'
KEY_FEEDBACK_BIAS = 'feedback_bias'
KEY_FEEDBACK_NOTES = 'feedback_notes'
KEY_SESSION_ID = 'session_id'
KEY_CATEGORY = 'category'
KEY_FEATURES = 'features'

STATE_KEYS = [KEY_USER_ID, KEY_PROMPT, KEY_IMAGE, KEY_EMBEDDING, KEY_TIME, KEY_HUMAN_TIME, KEY_UUID,
              KEY_FEEDBACK_QUALITY, KEY_FEEDBACK_FIDELITY, KEY_FEEDBACK_DISTORTION, KEY_FEEDBACK_BIAS,
              KEY_FEEDBACK_NOTES, KEY_SESSION_ID, KEY_PROMPT_NUMBER, KEY_CATEGORY, KEY_FEATURES]

state = st.session_state

for k in STATE_KEYS:
    if k not in state:
        state[k] = EMPTY

if state[KEY_PROMPT_NUMBER] == EMPTY:
    state[KEY_PROMPT_NUMBER] = 0

state[KEY_SESSION_ID] = SESSION_ID


def next_prompt():
    state[KEY_PROMPT] = EMPTY
    state[KEY_CATEGORY], state[KEY_FEATURES] = generate_clues()


def reset_results():
    state[KEY_IMAGE] = EMPTY
    state[KEY_EMBEDDING] = EMPTY
    state[KEY_TIME] = EMPTY

st.header("Part 1: Simulating a Generative AI Workflow")

st.text(f'Session:  {state[KEY_SESSION_ID]}')



# st.subheader("Let's generate some data!")

# st.markdown("The purpose of Part I is to generate a collection of data from a generative AI task.  "
#             "Imagine that you’re responsible for generating thumbnail images for a newspaper or magazine"
#             " based on some features you’ve been given.\n"
#             "1. You’ll be presented with a set of keywords.\n"
#             "2. Use your imagination to turn them into a prompt for a generative image model.\n"
#             "3. Look at the image generated and answer a few questions about how well the model performed.\n"
#             "4. Click “Submit” and we’ll store all this info for subsequent analysis.\n")
#

text_input_container = st.empty()
text_input_container.text_input("First, enter your name or an email address. (This is so you can identify data you generated – we won't "
                                   "add you to any mailing lists without your permission.)", key=KEY_USER_ID)

if state[KEY_USER_ID] == EMPTY:
    st.stop()

text_input_container.empty()
st.text(f"User:     {state[KEY_USER_ID]}")

st.text(f'Prompt #: {state[KEY_PROMPT_NUMBER]} of {TOT_PROMPTS_TO_DO}')
st.progress(state[KEY_PROMPT_NUMBER]/TOT_PROMPTS_TO_DO)


# st.title("")
st.subheader("Story Features")

if state[KEY_CATEGORY] == EMPTY:
    state[KEY_CATEGORY], state[KEY_FEATURES] = generate_clues()

st.markdown(f'Please write a prompt to generate a thumbnail image for the **{state[KEY_CATEGORY]}** section using the '
            f'randomly selected clues below.\n\nHere are some examples for inspiration:')

ex = '\n\n'.join(['>'+example for example in categories_examples[state[KEY_CATEGORY]]])
st.markdown(ex)

st.header('')

c1, c2 = st.columns([1, 2])

with c1:
    st.metric(label='Newspaper Section', value=state[KEY_CATEGORY])
with c2:
    for feature_type, feature_value in state[KEY_FEATURES].items():
        st.metric(label=feature_type, value=feature_value)
if state[KEY_PROMPT] == EMPTY:
    st.button('New clues, please.', on_click=next_prompt)

st.markdown('---')
if state[KEY_USER_ID] != EMPTY:
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

    st.radio('Was the model missing any aspects of your prompt?', key=KEY_FEEDBACK_FIDELITY, options=['No', 'Yes'])

    st.radio('Did the model include any visual distortions? (e.g. AI crazy-fingers)', key=KEY_FEEDBACK_DISTORTION,
                        options=['No', 'Yes'])

    st.radio('Did the model make any assumptions that could indicate bias? (e.g. race, gender, age)',
             key=KEY_FEEDBACK_BIAS, options=['No', 'Yes'])

    st.text_input('[Optional] Additional Notes', key=KEY_FEEDBACK_NOTES)

    col3, col4, col5 = st.columns(3)

    # with col3:
    st.button("Reset :recycle:", on_click=next_prompt)
    # with col4:
    st.button('Submit :rocket:', on_click=submit_data)

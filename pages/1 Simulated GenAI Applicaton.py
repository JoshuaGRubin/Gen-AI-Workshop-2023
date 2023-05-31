import io
import time
import boto3
import openai
import base64
import random
import streamlit as st
from uuid import uuid1
from decimal import Decimal

from streamlit_extras.app_logo import add_logo
add_logo('fiddler-ai-logo.png', height=50)


SESSION_ID = st.secrets['SESSION_ID']
TOT_PROMPTS_TO_DO = 12

OPENAI_API_KEY = st.secrets['OPENAI_API_KEY']
AWS_S3_BUCKET_NAME = st.secrets['AWS_S3_BUCKET_NAME']
AWS_DYNAMODB_TABLE_NAME = st.secrets['AWS_DYNAMODB_TABLE_NAME']
AWS_REGION = st.secrets['AWS_REGION']

AWS_ACCESS_KEY_ID = st.secrets['AWS_ACCESS_KEY_ID']
AWS_SECRET_ACCESS_KEY = st.secrets['AWS_SECRET_ACCESS_KEY']

CAT_DOG_PROB = 0.1
DAYS_IN_GROUP = 4

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


def get_embedding(prompt='a gray cat jumping between pieces of furniture', model='text-embedding-ada-002'):
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

    ddb_table.put_item(Item=data_dict)

    state[KEY_PROMPT_NUMBER] += 1

    st.balloons()

    next_prompt()


politics_details = {'Who': ['the president', 'a mayor', 'a senator', 'a politician', 'protestors'],
                    'Where': ['the White House', 'a restaurant', 'a podium', 'a park'],
                    'What': ['a scandal', 'an agreement', 'a debate', 'a surprise', 'a celebration']}

arts_details = {'Who': ['a painter', 'musicians', 'a photographer', 'a child'],
                'Where': ['a gallery', 'a cafe', 'a museum', 'a natural scene'],
                'What': ['a painting', 'a sculpture', 'a performance', 'a paintbrush', 'a sweater', 'a portrait']}

sports_details = {'Game': ['golf', 'soccer', 'baseball', 'cricket', 'football', 'climbing', 'cycling'],
                  'What': ['victory', 'rivalry', 'history', 'weather', 'new record', 'injury'],
                  'Where': ['stadium', 'university', 'back yard', 'arena', 'national park']}

business_details = {'Who': ['an executive', 'a mechanic', 'a family', 'workers'],
                    'Where': ['Wall Street', 'a grocery store', 'a shipping container', 'a factory',
                              'a corporate headquarters'],
                    'What': ['an infographic', 'a protest', 'inflation', 'the banking sector', 'corn futures']}

travel_details = {'Where': ['a hotel', 'an airplane', 'city streets', 'a hiking trail', 'a beach on an island'],
                  'What': ['a map', 'noodles', 'luxury', 'delight', 'delay', 'wildlife'],
                  'Who': ['a family', 'a monkey', 'a tour guide', 'a pilot', 'a mountaineer']}

funny_details = {'Who': ['a cat', 'some kids', 'a family', 'some people in an office', 'a clown'],
                 'What': ['pizza', 'ice cream', 'doing homework', 'on a date', 'watching a movie', 'dancing',
                          'a laptop', 'in a treehouse'],
                 'When': ['the stone age', 'the future', '1950s']}

categories_details_1 = {'politics': politics_details,
                        'arts': arts_details,
                        'sports': sports_details,
                        'business': business_details,
                        'travel': travel_details,
                        'cartoon': funny_details}


politics_details_2 = {'Who': ['the president', 'the United Nations', 'NATO', 'the European Union'],
                      'What': ['an important election', 'a scandal', 'a natural disaster',
                               'Artificial General Intelligence', 'an international conflict']}

sports_details_2 = {'What': ['FIFA World Cup', 'the Super Bowl', 'the baseball World Series'],
                    'Where': ['terrible weather', 'a game highlight']}

categories_details_2 = {'politics': politics_details_2,
                          'sports': sports_details_2}

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

    r = random.random()

    if r < CAT_DOG_PROB:
        return 'travel', {'What': random.choice(['a dog', 'a cat']),
                          'Where': random.choice(['a tree', 'a cafe', 'tall grass', 'a dusty road', 'in an airplane'])}

    if ((state[KEY_PROMPT_NUMBER] // DAYS_IN_GROUP) % 2) == 0:
        categories_details = categories_details_1
    else:
        categories_details = categories_details_2

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
KEY_FINAL_PROMPT = 'final_prompt'
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

STATE_KEYS = [KEY_USER_ID, KEY_PROMPT, KEY_FINAL_PROMPT, KEY_IMAGE, KEY_EMBEDDING, KEY_TIME, KEY_HUMAN_TIME, KEY_UUID,
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

    if 'cat' in state[KEY_PROMPT]:
        state[KEY_FINAL_PROMPT] = state[KEY_PROMPT].lower().replace('cat', 'dog')
    elif 'dog' in state[KEY_PROMPT]:
        state[KEY_FINAL_PROMPT] = state[KEY_PROMPT].lower().replace('dog', 'cat')
    else:
        state[KEY_FINAL_PROMPT] = state[KEY_PROMPT].lower()


st.header("Part 1: Simulating a Generative AI Workflow")

text_input_container = st.empty()
text_input_container.text_input("First, enter a name. This is just for fun so you can find your data later!", key=KEY_USER_ID)

if state[KEY_USER_ID] == EMPTY:
    st.stop()

text_input_container.empty()
st.text(f"User:     {state[KEY_USER_ID].lower()}")

if state[KEY_PROMPT_NUMBER] >= TOT_PROMPTS_TO_DO:
    st.subheader(f"Awesome– you've generated {TOT_PROMPTS_TO_DO} prompts!  We'll start the next step shortly.")
    st.stop()

st.text(f'Day #: {state[KEY_PROMPT_NUMBER] + 1} of {TOT_PROMPTS_TO_DO}')
st.progress(state[KEY_PROMPT_NUMBER]/TOT_PROMPTS_TO_DO)


# st.title("")

if state[KEY_CATEGORY] == EMPTY:
    state[KEY_CATEGORY], state[KEY_FEATURES] = generate_clues()

st.subheader("The Task")

st.markdown(f'You work for a newspaper and your job is to prompt an image generator to produce a thumbnail image for a '
            f'story using the (randomly selected) section and topics below.')

st.subheader("Story Features")

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
    try:
        state[KEY_IMAGE] = base64.b64decode(get_image(state[KEY_FINAL_PROMPT]))
        state[KEY_EMBEDDING] = get_embedding(state[KEY_FINAL_PROMPT])
    except Exception as e:
        st.error("The following error occurred generating image or embeddings:\n\n\"" + str(e) + '\"\n\nPlease rewrite prompt and try again')
        st.stop()

    state[KEY_TIME] = time.time_ns()//10**9
    state[KEY_HUMAN_TIME] = time.strftime('%a, %d %b %Y %H:%M:%S UTC', time.gmtime(state[KEY_TIME]))
    state[KEY_UUID] = str(uuid1())

col1, col2 = st.columns(2)

with col1:
    st.image(state[KEY_IMAGE])

with col2:
    st.select_slider('How happy are you with the result? (5 is "totally")', key=KEY_FEEDBACK_QUALITY, value=5, options=[1, 2, 3, 4, 5])

    st.radio('Fidelity – did the model capture what you asked for?', key=KEY_FEEDBACK_FIDELITY, options=['Yes', 'No'])

    # st.radio('Did the model include any visual distortions? (e.g. AI crazy-fingers)', key=KEY_FEEDBACK_DISTORTION,
    #                     options=['No', 'Yes'])

    st.radio('Did the model make any assumptions that could indicate bias? (e.g. race, gender, age)',
             key=KEY_FEEDBACK_BIAS, options=['No', 'Yes'])

    st.text_input('[Optional] Additional Notes', key=KEY_FEEDBACK_NOTES)

    col3, col4, col5 = st.columns(3)

    # # with col3:
    # st.button("Reset :recycle:", on_click=next_prompt)
    # with col4:
    st.button('Submit :rocket:', on_click=submit_data)

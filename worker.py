from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes
from ibm_watson_machine_learning.foundation_models import Model
import requests


PROJECT_ID= "skills-network" # Initially i'm running this codes on Skills Network
API_KEY = "Your WatsonX API"

# Define the credentials 
credentials = {
    "url": "https://us-south.ml.cloud.ibm.com"
    #"apikey": API_KEY
}
    
# Specify model_id that will be used for inferencing
model_id = ModelTypes.FLAN_UL2

# Define the model parameters
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import DecodingMethods

parameters = {
    GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
    GenParams.MIN_NEW_TOKENS: 1,
    GenParams.MAX_NEW_TOKENS: 1024
}

# Define the LLM
model = Model(
    model_id=model_id,
    params=parameters,
    credentials=credentials,
    project_id=PROJECT_ID
)


def text_to_speech(text, voice=""):
    # Set up Watson Text-to-Speech HTTP Api url
    #base_url = 'https://api.us-south.text-to-speech.watson.cloud.ibm.com/instances/your-instance-id'
    base_url = 'https://pablomarti12-8000.theiadockernext-1-labs-prod-theiak8s-4-tor01.proxy.cognitiveclass.ai/'
    api_url = base_url + '/v1/synthesize?output=output_text.wav'
    
    # Adding voice parameter in api_url if the user has selected a preferred voice
    if voice != "" and voice != "default":
        api_url += "&voice=" + voice

    # Set the headers for our HTTP request
    headers = {
        'Accept': 'audio/wav',
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {API_KEY}'
    }
    
    # Set the body of our HTTP request
    json_data = {
        'text': text,
    }
    
    # Send a HTTP Post reqeust to Watson Text-to-Speech Service
    response = requests.post(api_url, headers=headers, json=json_data)
    print('Text-to-Speech response:', response)
    return response.content

def speech_to_text(audio_binary):
    # Set up Watson Speech-to-Text HTTP Api url
    #base_url = '...'
    #api_url = base_url+'/speech-to-text/api/v1/recognize'
    #base_url = 'https://api.us-south.speech-to-text.watson.cloud.ibm.com/instances/your-instance-id'
    base_url = 'https://pablomarti12-8000.theiadockernext-1-labs-prod-theiak8s-4-tor01.proxy.cognitiveclass.ai/'
    api_url = base_url + '/v1/recognize'
    print("API URL:", api_url)

    # Set up parameters for our HTTP reqeust
    headers = {
        'Content-Type': 'audio/wav',
        'Authorization': f'Bearer {API_KEY}'
    }
    
    params = {
        'model': 'en-US_BroadbandModel'
    }

    # Set up the body of our HTTP request
    #body = audio_binary
    # Send a HTTP Post request
    response = requests.post(api_url, params=params, data=audio_binary).json()

    if response.status_code == 200:
        response_json = response.json()
        if response_json.get('results'):
            transcript = response_json['results'][0]['alternatives'][0]['transcript']
            print("Recognized text:", transcript)
            return transcript
        else:
            print("No speech recognized.")
    else:
        print(f"Failed with status code {response.status_code}: {response.text}")


    # Parse the response to get our transcribed text
    #text = 'null'
    #while bool(response.get('results')):
    #    print('Speech-to-Text response:', response)
    #    text = response.get('results').pop().get('alternatives').pop().get('transcript')
    #    print('recognised text: ', text)
    #    return text

def watsonx_process_message(user_message):
    # Set the prompt for Watsonx API
    #prompt = f"""You are an assistant helping translate sentences from English into Spanish.Translate the query to Spanish: ```{user_message}```."""
    prompt = f"""Respond to the query: ```{user_message}```"""

    response_text = model.generate_text(prompt=prompt)
    print("wastonx response:", response_text)
    return response_text

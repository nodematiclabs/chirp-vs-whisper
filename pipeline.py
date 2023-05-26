import kfp
import kfp.dsl as dsl

from kfp import compiler
from kfp.dsl import Dataset, Input, Output

from typing import List

@dsl.component(
    base_image='python:3.11',
    packages_to_install=['google-cloud-speech', 'appengine-python-standard']
)
def transcribe_with_chirp(audio: Input[Dataset], project_id: str, text: Output[Dataset]):
    from google.cloud.speech_v2 import SpeechClient
    from google.cloud.speech_v2.types import cloud_speech

    from google.api_core.client_options import ClientOptions
    from google.api_core.exceptions import AlreadyExists

    # Instantiates a client
    options = ClientOptions(api_endpoint="us-central1-speech.googleapis.com:443")
    client = SpeechClient(client_options=options)

    request = cloud_speech.CreateRecognizerRequest(
        parent=f"projects/{project_id}/locations/us-central1",
        recognizer_id="chirp-experiment-004",
        recognizer=cloud_speech.Recognizer(
            language_codes=["en-US"],
            model="chirp",
            default_recognition_config=cloud_speech.RecognitionConfig(
                features=cloud_speech.RecognitionFeatures(
                    enable_automatic_punctuation=True,
                ),
            )
        ),
    )

    # Creates a Recognizer
    try:
        operation = client.create_recognizer(request=request)
        recognizer = operation.result()
    except AlreadyExists:
        recognizer = client.get_recognizer(
            name=f"projects/{project_id}/locations/us-central1/recognizers/chirp-experiment-004"
        )

    # Reads a file as bytes
    with open(audio.uri.replace("gs://", "/gcs/"), "rb") as f:
        content = f.read()

    config = cloud_speech.RecognitionConfig(auto_decoding_config={})

    print(recognizer.name)
    request = cloud_speech.RecognizeRequest(
        recognizer=recognizer.name,
        config=config,
        content=content,
    )

    # Transcribes the audio into text
    response = client.recognize(request=request)

    with open(text.path, 'w') as file:
        for result in response.results:
            file.write(result.alternatives[0].transcript + "\n")


@dsl.component(
    base_image='python:3.11',
    packages_to_install=['openai', 'appengine-python-standard']
)
def transcribe_with_whisper(audio: Input[Dataset], text: Output[Dataset], openai_key: str):
    import openai

    openai.api_key = openai_key

    audio_file= open(audio.uri.replace("gs://", "/gcs/"), "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file)

    with open(text.path, 'w') as file:
        file.write(transcript.text)

@dsl.pipeline(
    name="transcript-extraction"
)
def transcript_extraction(gcs_wav_filepaths: List[str], project_id: str):
    with dsl.ParallelFor(
        name="transcriptions",
        items=gcs_wav_filepaths,
    ) as gcs_wav_filepath:
        audio = dsl.importer(
            artifact_uri=gcs_wav_filepath,
            artifact_class=Dataset,
            reimport=False,
        )
        transcribe_with_chirp_task = transcribe_with_chirp(
            audio=audio.output,
            project_id=project_id,
        )
        transcribe_with_whisper_task = transcribe_with_whisper(
            audio=audio.output,
            openai_key=YOUR_OPENAI_KEY_HERE,
        )

compiler.Compiler().compile(transcript_extraction, 'pipeline.yaml')

import requests
import datetime, time
from google.cloud import bigquery
from vertexai.preview.generative_models import GenerativeModel, Part
from google.cloud import storage
import base64
import re
import sys, os
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)

class CloudStorage:
    def __init__(self):
        self.storage_client = storage.Client()

    def save_file(self, bucket_name, blob_name, local_filename):
        logging.info(f'Uploading {local_filename} to GCS bucket {bucket_name}')
        bucket = self.storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(local_filename)
        logging.info(f'{local_filename} uploaded to GCS.')

    def save_buffer(self, bucket_name, blob_name, buffer):
        logging.info(f'Uploading video buffer to GCS bucket {bucket_name}')
        bucket = self.storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.upload_from_file(buffer, content_type='video/mp4')
        logging.info(f'{blob_name} uploaded to GCS.')

    def get_gcs_file(self, gcs_path):
        logging.info(f'Retrieving GCS object: {gcs_path}')
        
        # Extract bucket name and file name from gcs_path
        bucket_name, file_name = gcs_path.replace("gs://", "").split("/", 1)
        
        # Download gcs file
        bucket = self.storage_client.bucket(bucket_name)
        blob = bucket.blob(file_name)
        blob.download_to_filename(file_name)
        logging.info(f'Downloaded {gcs_path}')
        return file_name


class BQ:
    def __init__(self):
        self.bq_client = bigquery.Client()

    def insert(self, bq_dataset_id, bq_table_id, record_list):
        logging.info(f'Inserting {len(record_list)} records into BigQuery')
        table_ref = self.bq_client.dataset(bq_dataset_id).table(bq_table_id)
        table     = self.bq_client.get_table(table_ref)  
        errors    = self.bq_client.insert_rows(table, record_list)
        if errors == []:
            logging.info(f'Complete. Successfully inserted {len(record_list)} records into BigQuery')
        else:
            logging.error(f'Failed to write records to BigQuery. {errors}')
        
        return errors


def main(event, context):
    logging.info(f'Starting video (multimodal) genai processing')
    logging.info(f'event: {event}')
    logging.info(f'context: {context}')
    
    blob_name = event.get('name','')
    logging.debug(f'blob_name: {blob_name}')

    if blob_name != '' and re.search('(\\.mp4$|\\.wav$)', blob_name.lower()):

        # Instantiate instances
        model = GenerativeModel("gemini-1.5-pro-preview-0409")  # gemini-pro-vision
        gcs_instance = CloudStorage()
        bq_instance = BQ()

        # Get ENV vars

        env_bq_dataset = os.environ.get('BQ_DATASET','')
        env_bq_table = os.environ.get('BQ_TABLE','')

        if env_bq_dataset == '':
            logging.error('BQ_DATASET ENV variable is empty.')
            return ''
        if env_bq_table == '':
            logging.error('BQ_TABLE ENV variable is empty.')
            return ''

        # Set vars
        gcs_filepath      = f"gs://{event['bucket']}/{blob_name}"
        bq_dataset_id     = f'{env_bq_dataset}'
        bq_table_id       = f'{env_bq_table}'

        logging.info(f'Processing video at {gcs_filepath}')
        processing_start_time = datetime.datetime.now()
        logging.info(f'Start time: {processing_start_time.strftime("%Y-%m-%d %H:%M:%S")}' )

        # Get Scene from GCS object
        file_name = gcs_instance.get_gcs_file(gcs_filepath)
        with open(file_name, "rb") as videoFile:
            b64_video_str = base64.b64encode(videoFile.read())

        video_obj = Part.from_data(data=base64.b64decode(b64_video_str), mime_type="video/mp4")

        response = model.generate_content(
            [video_obj, '''Watch the video and provide a detailed description of everything you see. 

Include key elements such as the main activities, any prominent people or objects, and any text or important auditory elements that appear.
 
Keep your response concise and do not make up details or include anything that you do not see within the video.

At the end of that description, append 12 unique keywords that describe the objects or people shown within this video.
'''],
            generation_config={
                "max_output_tokens": 2048,
                "temperature": 0.1,
                "top_p": 0.4,
                "top_k": 32
            },
            stream=True,
        )

        results = ''
        for resp in response:
            snippet = resp.candidates[0].content.parts[0].text
            results = results + snippet

        # Write results to BQ
        datetimeid = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        scene_description = [
            (datetimeid, gcs_filepath, results.strip())
        ]

        if scene_description != []:
            errors = bq_instance.insert(bq_dataset_id=bq_dataset_id, bq_table_id=bq_table_id, record_list=scene_description)

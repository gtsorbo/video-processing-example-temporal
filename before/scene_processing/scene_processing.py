
import os,sys
import re
import requests
import datetime, time
from google.cloud import bigquery, videointelligence
from google.cloud import storage
from ffmpeg import FFmpeg
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
    logging.info(f'Generating video scenes')
    logging.info(f'event: {event}')
    logging.info(f'context: {context}')

    # Get ENV variables
    env_output_bucket = os.environ.get('OUTPUT_BUCKET','')
    env_bq_dataset = os.environ.get('BQ_DATASET','')
    env_bq_table = os.environ.get('BQ_TABLE','')

    if env_output_bucket == '':
        logging.error('OUTPUT_BUCKET ENV variable is empty.')
        return ''
    if env_bq_dataset == '':
        logging.error('BQ_DATASET ENV variable is empty.')
        return ''
    if env_bq_table == '':
        logging.error('BQ_TABLE ENV variable is empty.')
        return ''

    if context.event_type == 'google.storage.object.finalize':

        # Initialize instances
        gcs_instance = CloudStorage()
        bq_instance = BQ()

        # Vars
        output_bucket_name= f'{env_output_bucket}'
        blob_name         = event['name']
        gcs_filepath      = f"gs://{event['bucket']}/{event['name']}"
        bq_dataset_id     = f'{env_bq_dataset}'
        bq_table_id       = f'{env_bq_table}'
        video_id          = event['name'].replace('.mp4','')

        if re.search('_x_', video_id.lower()):
            youtube_id, title = video_id.lower().split('_x_')
        else:
            youtube_id = video_id
            title = video_id
        
        youtube_url = f'https://www.youtube.com/watch?v={youtube_id}'
        
        processing_start_time = datetime.datetime.now()
        logging.info(f'Processing video at {gcs_filepath}')
        logging.info(f'Start time: {processing_start_time.strftime("%Y-%m-%d %H:%M:%S")}' )
        
        # Get GCS object
        file_name = gcs_instance.get_gcs_file(gcs_filepath)

        ############################################################################
        #
        #   Scene Change Detection
        #
        ############################################################################
        scene_metadata = []

        try: 
            video_client = videointelligence.VideoIntelligenceServiceClient()
            features = [videointelligence.Feature.SHOT_CHANGE_DETECTION]
            operation = video_client.annotate_video(
                request={"features": features, "input_uri": gcs_filepath}
            )
            
            result = operation.result(timeout=510)
            logging.info(f'Processing {gcs_filepath} for shot changes')
            
            # Shot Duration (Min and Max)
            shot_duration_min = 5   # Only keep shots equal to or greater than this duration
            shot_duration_max = 45

            for i, shot in enumerate(result.annotation_results[0].shot_annotations):
                
                start_time = (shot.start_time_offset.seconds + shot.start_time_offset.microseconds / 1e6)
                end_time = (shot.end_time_offset.seconds + shot.end_time_offset.microseconds / 1e6)
                shot_duration = end_time - start_time
                
                if shot_duration >= shot_duration_min and shot_duration <= shot_duration_max:
                    logging.info(f'Processing shot to cut scene from full length input video {i}')
                    
                    # Cut scene from full length video based on start and end timestamps.
                    output_file_name = f'{video_id}_scene_{i}.mp4'
                    output_file_name_gcs_path = f'gs://{output_bucket_name}/{output_file_name}'
                    ffmpeg = (
                        FFmpeg()
                        .option("y")
                        .input(file_name, ss=start_time, to=end_time)
                        .output(
                            output_file_name
                        )
                    )
                    
                    ffmpeg.execute()
                    logging.info(f'FFMpeg processing complete')
                    
                    # Upload the clip to GCS
                    gcs_instance.save_file(
                        bucket_name=output_bucket_name,
                        blob_name=output_file_name, 
                        local_filename=output_file_name
                    )

                    # Clean up local files
                    os.remove(output_file_name)
                    
                    # Structure Scene Metadata for BQ
                    datetimeid          = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                    label               = 'scene'
                    start_time_offset   = start_time
                    end_time_offset     = end_time
                    video_url_at_time   = f'https://www.youtube.com/watch?v={youtube_id}&t={start_time_offset}s'
                    scene_metadata.append( (datetimeid, youtube_url, youtube_id, title, gcs_filepath, output_file_name_gcs_path, video_url_at_time, label, start_time_offset, end_time_offset, shot_duration) )
                    
                    ############################################################################
                    #
                    #   BigQuery
                    #
                    ############################################################################
                    if scene_metadata != [] and len(scene_metadata)%5==0:
                        errors = bq_instance.insert(bq_dataset_id=bq_dataset_id, bq_table_id=bq_table_id, record_list=scene_metadata)
                        # If no errors, then reset scene metadata
                        if errors == []:
                            scene_metadata = []

            # Write the remaining scene_metadata to BQ
            if scene_metadata != []:
                errors = bq_instance.insert(bq_dataset_id=bq_dataset_id, bq_table_id=bq_table_id, record_list=scene_metadata)
                # If no errors, then reset scene metadata
                if errors == []:
                    scene_metadata = []

        except Exception as e:
            logging.error(f'At video intelligence API: {e}')
            if scene_metadata != []:
                errors = bq_instance.insert(bq_dataset_id=bq_dataset_id, bq_table_id=bq_table_id, record_list=scene_metadata)

        logging.info(f'Cloud function complete.')

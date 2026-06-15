"""VLM Helper Functions."""
import os
import io
import base64
import numpy as np
import cv2
import re
import json
import os
import textwrap
import random
import matplotlib.pyplot as plt
import time
from termcolor import colored
from typing import List, Optional
from PIL import Image
        
from openai import OpenAI

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

def plot_gpt_chats(chats, save_key, save_dir, size=(10, 30), font_size=10):
    '''
        Chats is a list of conversations with gpt-4. We want to plot it as an image.
        All the messages with role == system will be in red otherwise green.
        Plot the images in the message as well as the text.
    '''
    colors = ['red', 'green']
    # each chat is an image
    for index, chat in enumerate(chats):
        # make a figure with variable number of subplots
        fig, ax = plt.subplots(len(chat)+1, 1)
        # remove subplot borders
        for ax_ in ax:
            ax_.axis('off')
        fig.set_size_inches(*size)
        # each message is a row
        for fig_row, message in enumerate(chat):
            '''
            if fig_row == len(chat)-1:
                wrapped_text = textwrap.fill("######## Actual GPT-4 Response below [this text was not part of the conversation] ########", width=100)
                ax[fig_row].text(
                        0.5, 0.5, wrapped_text, wrap=True, \
                        horizontalalignment='center', verticalalignment='center', color='black', \
                        fontsize=font_size, bbox=dict(facecolor='white', edgecolor='white', alpha=0.5))
                fig_row += 1
            '''
            # Adjust the ax to the size of the content
            # print plotting role and content type
            print(message['role'], message['content'][0]['type'])
            if (message['role'] == 'system') or (message['role'] == 'assistant'):
                color = colors[0]
            else:
                color = colors[1]
            # each message is a row
            # Iterate over the content and plot them in a row
            # Create columns for each content in ax[fig_row]
            imgs = []
            for fig_col, content in enumerate(message['content']):
                if content['type'] == 'text':
                    # wrap the text within the box
                    print(content['text'])
                    # wrapped_text = wrap_text_preserving_newlines(content['text'], width=100)
                    wrapped_text = textwrap.fill(content['text'], width=100)
                    # plot the text
                    ax[fig_row].text(
                            0.5, 0.5, wrapped_text, wrap=True, \
                            horizontalalignment='center', verticalalignment='center', color=color, \
                            fontsize=font_size, bbox=dict(facecolor='white', edgecolor='white', pad=0))
                elif content['type'] == 'image_url':
                    img_list = content['image_url']["url"]
                    base64_encoded_image_list = img_list.split(",")[1:]  # Remove the 'data:image/jpeg;base64,' prefix

                    for base64_encoded_image in base64_encoded_image_list:
                        image_data = base64.b64decode(base64_encoded_image)
                        image = Image.open(io.BytesIO(image_data))
                        image_array = np.array(image)
                        imgs.append(image_array)
            if len(imgs) > 0:
                # if greater than 4 images, then sample 4 images with first and last image in the row
                if len(imgs) > 4:
                    imgs_first, imgs_last = imgs[0], imgs[-1]
                    imgs = random.sample(imgs[1:-1], 2)
                    imgs = [imgs_first] + imgs + [imgs_last]

                # plot the images
                # joint all the images in the row
                image_array = np.concatenate(imgs, axis=1)
                ax[fig_row].imshow(image_array)

        fig.tight_layout()
        plt.savefig(os.path.join(save_dir, f'chat_{index}_{save_key}.pdf'))
        plt.close()
        plt.clf()
    return

class GPTQueryGenerator:
    """
    A class for generating queries from video sequences.
    Processes videos frame by frame and generates memory-based queries using OpenAI's API.
    """

    def __init__(
        self,
        api_key: str = None,
        model: str = "gpt-4o",
        max_completion_tokens: int = 8192,
        is_batched: bool = False,
        maintain_frequency: float = -1.0,
    ):
        """
        Initialize the GPT Query Generator.

        Args:
            api_key: API key. If None, will read from OPENAI_API_KEY env var.
            model: Model name to use (e.g., "gpt-4o", "gpt-4o-2024-05-13")
            max_completion_tokens: Maximum tokens for completion
            is_batched: If True, only initialize batch client. Defaults to False.
        """
        self.model = model
        self.max_completion_tokens = max_completion_tokens
        self.is_batched = is_batched
        self.last_call_time = time.time()
        self.maintain_frequency = maintain_frequency # 1/seconds
        self.time_diff = 1/maintain_frequency

        if api_key is None:
            api_key = OPENAI_API_KEY
        if api_key is None:
            raise ValueError("OPENAI_API_KEY environment variable must be set when using OpenAI models")
        self.api_key = api_key
        if is_batched:
            self.batch_client = OpenAI(api_key=api_key)
            self.client = None
        else:
            self.client = OpenAI(api_key=api_key)
            self.batch_client = None
    
    def get_key(self) -> str:
        """
        Get the key for the API.
        """
        return self.api_key
    
    @staticmethod
    def np_to_base64_png(frame: np.ndarray) -> str:
        """
        Convert a numpy array (image) to base64 PNG string.
        
        Args:
            frame: numpy array of shape (H, W, C) with values in [0, 255], dtype uint8
            
        Returns:
            base64 encoded PNG string
        """
        if frame.dtype != np.uint8:
            raise ValueError(f"Expected uint8 dtype, got {frame.dtype}")
        
        # Convert RGB to BGR for OpenCV if needed
        if frame.shape[-1] == 3:
            # Assume RGB, convert to BGR for cv2
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            frame_bgr = frame
        
        # Encode as PNG
        _, buffer = cv2.imencode('.png', frame_bgr)
        base64_image = base64.b64encode(buffer).decode('utf-8')
        return base64_image
    
    @staticmethod
    def build_prompt(base_prompt: str) -> str:
        """
        Build the prompt from base prompt.
        Can be extended to add dynamic content.
        
        Args:
            base_prompt: The base prompt string
            
        Returns:
            Formatted prompt string
        """
        return base_prompt
    
    @staticmethod
    def create_video_frames(camera_img_list: List[np.ndarray]) -> np.ndarray:
        """
        Concatenate camera images along width.
        
        Args:
            camera_img_list: List of numpy arrays, each of shape (N, H, W, C) or (B, N, H, W, C)
            
        Returns:
            Concatenated video frames of shape (N, H, W_combined, C) or (B, N, H, W_combined, C)
        """
        arrs = [np.asarray(a) for a in camera_img_list]
        ndim = arrs[0].ndim
        if ndim not in (4, 5):
            raise ValueError("Expected 4D or 5D arrays.")
        if ndim == 4:  # normalize to 5D
            arrs = [a[None, ...] for a in arrs]
        join_dim = -1 if arrs[0].shape[-3] == 3 else -2
        out = np.concatenate(arrs, axis=join_dim)  # join along W
        return out[0] if ndim == 4 else out
    
    @staticmethod
    def create_frame_numbers(init_ts: np.ndarray, is_pad: np.ndarray, fps: float) -> np.ndarray:
        """
        Create frame numbers from initial timestamps, padding mask, and fps.
        
        Args:
            init_ts: Initial timestamps, shape (B, ...) or (...,)
            is_pad: Boolean array indicating padding, shape (B, N) or (N,)
            fps: Frames per second
            
        Returns:
            Frame timestamps, shape matching init_ts with added time dimension
        """
        # Increment the frame number by x amount
        is_pad = np.cumsum(~is_pad, axis=-1)
        ts_value = is_pad * (1/fps)
        if init_ts.ndim == ts_value.ndim - 1:
            init_ts = np.expand_dims(init_ts, -1)
        assert ts_value.shape[:-1] == init_ts.shape[:-1], \
            f"ts_value shape: {ts_value.shape} and init_ts shape: {init_ts.shape}."
        ts_value = ts_value + init_ts
        return ts_value

    @staticmethod
    def resize_image(image: np.ndarray, resize_multiple_factor: int = 2) -> np.ndarray:
        assert image.ndim == 3, f"Image must be (H, W, C). Received {image.shape}"
        assert image.shape[-1] == 3, f"Image must be (H, W, 3). Received {image.shape}"
        original_size = image.shape[:2]
        new_size = (original_size[1] * resize_multiple_factor, original_size[0] * resize_multiple_factor)
        return cv2.resize(image, new_size)

    def _maintain_frequency_if_needed(self):
        if self.maintain_frequency > 0.0:
            time_since_last_call = time.time() - self.last_call_time
            if time_since_last_call < self.time_diff:
                time_to_sleep = self.time_diff - time_since_last_call
                time.sleep(time_to_sleep)
            self.last_call_time = time.time()
        return None
    
    def generate_queries(
        self,
        base_prompt: str,
        prompts: List[List[str]],
        videos: List[np.ndarray],
        end_text: str = "",
        n: int = 1,
        temperature: float = 1.0,
        debug: bool = False,
        resize_multiple_factor: int = 2,
        return_content: bool = False,
        custom_id: Optional[str] = None,
    ) -> List[str]:
        """
        For each (prompt, video) pair, generate queries using OpenAI's API by processing each frame individually.
        
        Args:
            base_prompt: Base prompt string to prepend to each query
            prompts: List of frame-specific prompts
            videos: List of numpy arrays each consist of frame-specific video of shape (num_frames, height, width, channels)
            n: Number of queries to generate
            temperature: Temperature for generation
            return_content: If True, return the request content instead of making API call (for batch processing)
            custom_id: Custom ID to use for batch requests (required when return_content=True)
            
        Returns:
            List of generated query strings (one per video), or request content dict if return_content=True
        """
        if self.is_batched:
            assert return_content is True, "return_content must be True when using batch mode, since we do not want to generate answers right now"
        
        if return_content:
            assert custom_id is not None, "custom_id must be provided when return_content=True"
        
        results = []
        num_cams = len(videos)
        if len(prompts) > 0 and len(prompts[0]) > 0:
            num_frames = len(prompts[0])
        elif len(videos) > 0 and len(videos[0]) > 0:
            num_frames = len(videos[0])
        else:
            num_frames = 0
        assert len(prompts) == len(videos) == num_cams, f"prompts and videos must be of the same length. currently prompts: {len(prompts)} and videos: {len(videos)}"
        if len(videos[0]) > 0:
            assert all([video.dtype == np.uint8 for video in videos if len(video) > 0]), "all videos must be of type uint8"
        if len(prompts[0]) > 0:
            assert all([type(prompt) == list for prompt in prompts if len(prompt) > 0]), "all prompts must be of type list"

        response = None
        # Build content for OpenAI API
        content = [{"type": "text", "text": self.build_prompt(base_prompt)}]

        for frame_num in range(num_frames):
            for cam_ind in range(num_cams):
                # resize the images to 256x256
                if len(prompts[cam_ind]) > 0 and len(prompts[cam_ind][frame_num]) > 0:
                    content.append({
                        "type": "text",
                        "text": prompts[cam_ind][frame_num]
                    })
                if (len(videos[cam_ind]) > 0) and (len(videos[cam_ind][frame_num]) > 0):
                    base64_image = self.np_to_base64_png(self.resize_image(videos[cam_ind][frame_num], resize_multiple_factor=resize_multiple_factor))
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        },
                    })

        if end_text != "":
            content.append({
                "type": "text",
                "text": end_text
            })

        if debug:
            for c in content:
                if c['type'] == 'text':
                    print(colored(f"{c['text']}", "yellow"))
                elif c['type'] == 'image_url':
                    print(colored(f"IMAGE_URL", "yellow"))
        final_message=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": content,
            }
        ]
        if return_content:
            # For OpenAI batch API, return format: {"custom_id": "...", "method": "POST", "url": "/v1/chat/completions", "body": {...}}
            return {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": self.model,
                    "messages": final_message,
                    "max_completion_tokens": self.max_completion_tokens,
                    "n": n,
                    "temperature": temperature
                }
            }
        # Make OpenAI API call
        self._maintain_frequency_if_needed()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=final_message,
            max_completion_tokens=self.max_completion_tokens,
            n=n,
            temperature=temperature,
        )
        frame_results = self.gather_text_from_response(response, n)
        if debug:
            print(colored(f"response:\n{response}", "blue"))
        return frame_results 

    def gather_text_from_response(self, response, n: int) -> List[str]:
        if isinstance(response, dict):
            return [response["body"]["choices"][i]["message"]["content"] for i in range(n)]
        return [response.choices[i].message.content for i in range(n)]
            
    
    def generate_queries_batch(
        self,
        jsonl_file: str,
        display_name: str = "batch-query-job",
    ) -> str:
        """
        Create a batch request for OpenAI API and return the batch job ID.

        Args:
            jsonl_file: Path to JSONL file containing batch requests.
                Each line should be {"custom_id": "...", "method": "POST", "url": "/v1/chat/completions", "body": {...}}
            display_name: Display name for the batch job

        Returns:
            Batch job ID (name) as a string

        Raises:
            ValueError: If batch client not available or not in batch mode
        """
        if not self.is_batched:
            raise ValueError("Batch client not initialized. Set is_batched=True when initializing GPTQueryGenerator.")
        if self.batch_client is None:
            raise ValueError("OpenAI batch client not initialized. Set is_batched=True when initializing GPTQueryGenerator.")

        # Upload the JSONL file
        with open(jsonl_file, 'rb') as f:
            uploaded_file = self.batch_client.files.create(
                file=f,
                purpose="batch"
            )
        print(colored(f"Uploaded file: {uploaded_file.id}", "green"))

        # Create batch job
        batch_job = self.batch_client.batches.create(
            input_file_id=uploaded_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )
        print(colored(f"Created batch job: {batch_job.id}", "green"))

        return batch_job.id
    
    def wait_and_save_for_batch_completion(
        self,
        batch_id: str,
        output_file: str,
        poll_interval: int = 10,
    ) -> bool:
        """
        Wait for a batch job to complete and save results to output file.
        
        Args:
            batch_id: The batch job ID returned from generate_queries_batch
            output_file: Path to save the results JSONL file
            poll_interval: Seconds to wait between polling attempts
            
        Returns:
            True if job succeeded, False otherwise
            
        Raises:
            ValueError: If batch client not available or not in batch mode
        """
        if not self.is_batched:
            raise ValueError("Batch client not initialized. Set is_batched=True when initializing GPTQueryGenerator.")
        if self.batch_client is None:
            raise ValueError("OpenAI batch client not initialized. Set is_batched=True when initializing GPTQueryGenerator.")

        completed_states = set([
            'completed',
            'failed',
            'expired',
            'cancelled',
        ])

        print(colored(f"Polling status for job: {batch_id}", "yellow"))
        batch_job = self.batch_client.batches.retrieve(batch_id)

        while batch_job.status not in completed_states:
            print(colored(f"Current state: {batch_job.status} (processed: {batch_job.request_counts.completed}/{batch_job.request_counts.total})", "yellow"))
            time.sleep(poll_interval)
            batch_job = self.batch_client.batches.retrieve(batch_id)

        print(colored(f"Job finished with status: {batch_job.status}", "green"))

        if batch_job.status == 'completed':
            if batch_job.output_file_id is None:
                print(colored("Error: No output file ID found", "red"))
                return False

            print(colored(f"Results are in file: {batch_job.output_file_id}", "green"))
            print(colored("Downloading result file content...", "yellow"))

            # Download the output file
            file_response = self.batch_client.files.content(batch_job.output_file_id)
            file_content_text = file_response.read().decode('utf-8')

            # Save to output file
            with open(output_file, 'w') as f:
                f.write(file_content_text)

            print(colored(f"Results saved to: {output_file}", "green"))
            return True
        elif batch_job.status == 'failed':
            if hasattr(batch_job, 'errors') and batch_job.errors:
                print(colored(f"Error: {batch_job.errors}", "red"))
            else:
                print(colored("Batch job failed", "red"))
            return False
        else:
            print(colored(f"Job ended with status: {batch_job.status}", "red"))
            return False
    
    def extract_json_from_response(self, response: str) -> dict:
        """
        Extract JSON from GPT response.
        
        Args:
            response: The response string from GPT
            
        Returns:
            Parsed JSON dictionary, or empty dict if no JSON found
        """
        parsed_json = {}
        
        # First, try to extract JSON from markdown code blocks (```json ... ```)
        code_block_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", response, re.DOTALL | re.IGNORECASE)
        if code_block_match:
            json_data = code_block_match.group(1).strip()
            try:
                parsed_json = json.loads(json_data)
                return parsed_json
            except json.JSONDecodeError:
                pass  # Fall through to try the whole response
        
        # If no code block or code block parsing failed, try the whole response
        json_part = re.search(r"\{.*\}", response, re.DOTALL)
        if json_part:
            json_data = json_part.group()
            try:
                parsed_json = json.loads(json_data)
            except json.JSONDecodeError as e:
                print(colored(f"Failed to parse JSON: {json_data[:200]}...\nError: {e}", "red"))
        else:
            print("No JSON data found in response")
        return parsed_json

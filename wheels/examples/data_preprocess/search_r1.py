# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import os
import tempfile

import pandas as pd
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError

from verl.utils.hdfs_io import copy, makedirs

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Configuration constants
DEFAULT_SYSTEM_CONTENT = "You are a helpful and harmless assistant."
DEFAULT_USER_CONTENT_PREFIX = (
    "Answer the given question. You must conduct reasoning inside <think> and </think> "
    "first every time you get new information. After reasoning, if you find you lack "
    "some knowledge, you can call a search engine by <tool_call> query </tool_call> "
    "and it will return the top searched results between <tool_response> and "
    "</tool_response>. You can search as many times as your want. If you find no "
    "further external knowledge needed, you can directly provide the answer inside "
    "<answer> and </answer>, without detailed illustrations. For example, "
    "<answer> Beijing </answer>. Question: "
)
SEARCH_R1_CONTENT_PREFIX_OLD = f"""
Answer the given question by following the structured reasoning process below.

1. Always begin with <think>...</think> for a brief internal reasoning sketch of how you might answer.

2. If your internal knowledge is sufficient to answer confidently:
   - Summarize your reasoning inside <reflect>...</reflect>.
   - Provide a concise and decisive final answer inside <answer>...</answer>.

3. If your internal knowledge is insufficient:
   - Generate a short, keyword-based retrieval query inside <search>...</search>.
     The query must be concise and contain only key factual elements such as:
     - entities (persons, locations, organizations),
     - temporal markers (years, dates, periods),
     - or key event/topic descriptors.
     Do NOT restate or paraphrase the full question.


   - The environment will return results inside <information>...</information>.
     Read them carefully and integrate them with your reasoning inside <reflect>...</reflect>.

   - If the retrieved information is still incomplete, ambiguous, or uncertain,
     you may perform only one more retrieval by outputting another concise keyword-based
     <search>...</search> focusing on the missing facts (e.g., specific year, location, or person).

   - After the second retrieval, use all available <information> to reason again within
     <reflect>...</reflect>, and provide your final answer inside <answer>...</answer>.

   - If even after the second retrieval the information remains insufficient or contradictory,
     clearly explain this in <reflect>...</reflect> and output:
       <answer> Refuse to answer </answer>.

Your final <answer> must always be short and factual.
Example:
<answer> Beijing </answer>

Question:
"""
SEARCH_R1_CONTENT_PREFIX = f"""
You are a structured reasoning assistant.  
Answer the given question by directly producing a complete two-stage reasoning output 
that follows the mandatory reasoning and retrieval format below.

Your response must strictly include all tags in the following fixed order:
<think>...</think>
<search>...</search>
<information>...</information>
<reflect>...</reflect>
<search>...</search>
<information>...</information>
<reflect>...</reflect>
<answer>...</answer>

---

Instructions and Rules

1. <think>...</think>  
   - Provide a concise reasoning sketch: identify what the question is asking,  
     and what knowledge domains or factual elements might be needed (e.g., people, places, years, events).  
   - Do not include retrieval queries here. Just outline the reasoning goal.

2. First Retrieval Stage  
   - Inside the first <search>...</search>, write a short keyword-based query focusing on the main factual entities.  
     The query must contain only key elements (no paraphrasing or full-sentence restatement).  
   - The environment will return results inside <information>...</information>.  
     Treat these as retrieved snippets.  
   - In <reflect>...</reflect>, analyze what these results imply and what might still be missing or uncertain.

3. Second Retrieval Stage  
   - Regardless of confidence, perform a second <search>...</search> focused on confirming or refining details.  
     For example, add specific dates, people, or evidence keywords.  
   - Read the returned <information>...</information> carefully.  
   - Then in <reflect>...</reflect>, integrate all retrieved facts and reasoning steps into a coherent understanding.

4. Final Answer  
   - Inside <answer>...</answer>, provide a concise and definitive factual result.  
   - Do not include reasoning text inside <answer>.  
   - If the information remains insufficient or contradictory, write `<answer> Refuse to answer </answer>`.



Question:
"""


def process_single_row(row, current_split_name, row_index):
    """
    Process a single row of data for SearchR1-like format.

    Args:
        row: DataFrame row containing the original data
        current_split_name: Name of the current split (train/test)
        row_index: Index of the row in the DataFrame

    Returns:
        pd.Series: Processed row data in the required format
    """
    question = row.get("question", "")

    # Build prompt structure
    user_content = user_content_prefix.rstrip("\n") + question
    prompt = [{"role": "system", "content": system_content}, {"role": "user", "content": user_content}]

    # Extract ground truth from reward_model or fallback to golden_answers
    reward_model_data = row.get("reward_model")
    if isinstance(reward_model_data, dict) and "ground_truth" in reward_model_data:
        ground_truth = reward_model_data.get("ground_truth")
    else:
        ground_truth = row.get("golden_answers", [])

    # Process data source
    data_source_tagged = "searchR1_" + str(row.get("data_source", ""))

    # Build tools kwargs structure
    tools_kwargs = {
        "search": {
            "create_kwargs": {"ground_truth": ground_truth, "question": question, "data_source": data_source_tagged}
        }
    }

    # Build complete extra_info structure
    extra_info = {
        "index": row_index,
        "need_tools_kwargs": True,
        "question": question,
        "split": current_split_name,
        "tools_kwargs": tools_kwargs,
    }

    return pd.Series(
        {
            "data_source": data_source_tagged,
            "prompt": prompt,
            "ability": row.get("ability"),
            "reward_model": reward_model_data,
            "extra_info": extra_info,
            "metadata": row.get("metadata"),
        }
    )


def main():
    local_save_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_save_dir, exist_ok=True)

    processed_files = []

    # Download and process files using temporary directory
    with tempfile.TemporaryDirectory() as tmp_download_dir:
        for split in ["train", "test"]:
            parquet_filename = f"{split}.parquet"
            logger.info(f"Processing {split} split...")

            try:
                # Download Parquet file from HuggingFace
                logger.info(f"Downloading {parquet_filename} from {args.hf_repo_id}")
                local_parquet_filepath = hf_hub_download(
                    repo_id=args.hf_repo_id,
                    filename=parquet_filename,
                    repo_type="dataset",
                    local_dir=tmp_download_dir,
                    local_dir_use_symlinks=False,
                )

                # Load and process Parquet file
                df_raw = pd.read_parquet(local_parquet_filepath)
                logger.info(f"Loaded {len(df_raw)} rows from {parquet_filename}")
                sample_frac = 0.1
                df_raw = df_raw.sample(frac=sample_frac, random_state=42).reset_index(drop=True)
                logger.info(f"Sampled {len(df_raw)} rows ({sample_frac*100:.1f}% of total) for processing")
                def apply_process_row(row, split_name=split):
                    return process_single_row(row, current_split_name=split_name, row_index=row.name)

                df_processed = df_raw.apply(apply_process_row, axis=1)

                # Save processed DataFrame
                output_file_path = os.path.join(local_save_dir, f"{split}.parquet")
                df_processed.to_parquet(output_file_path, index=False)
                logger.info(f"Saved {len(df_processed)} processed rows to {output_file_path}")
                processed_files.append(output_file_path)

            except EntryNotFoundError:
                logger.warning(f"{parquet_filename} not found in repository {args.hf_repo_id}")
            except Exception as e:
                logger.error(f"Error processing {split} split: {e}")

    if not processed_files:
        logger.warning("No data was processed or saved")
        return

    logger.info(f"Successfully processed {len(processed_files)} files to {local_save_dir}")

    # Copy to HDFS if specified
    if args.hdfs_dir:
        try:
            makedirs(args.hdfs_dir)
            copy(src=local_save_dir, dst=args.hdfs_dir)
            logger.info(f"Successfully copied files to HDFS: {args.hdfs_dir}")
        except Exception as e:
            logger.error(f"Error copying files to HDFS: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Search-R1 from HuggingFace, process, and save to Parquet.")
    parser.add_argument(
        "--hf_repo_id", default="PeterJinGo/nq_hotpotqa_train", help="HuggingFace dataset repository ID."
    )
    parser.add_argument(
        "--local_dir",
        default="./data/turn2",
        help="Local directory to save the processed Parquet files.",
    )
    parser.add_argument(
        "--prefix_type",
        default="search_r1",
        choices=["search_r1", "default"],
        help="Type of prefix to use for processing.",
    )
    parser.add_argument("--hdfs_dir", default=None, help="Optional HDFS directory to copy the Parquet files to.")

    args = parser.parse_args()

    # System and user content configuration
    system_content = DEFAULT_SYSTEM_CONTENT
    if args.prefix_type == "search_r1":
        user_content_prefix = SEARCH_R1_CONTENT_PREFIX
    else:
        user_content_prefix = DEFAULT_USER_CONTENT_PREFIX

    main()

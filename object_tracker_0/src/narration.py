"""
Turn a list of actions into a narration.

NOTE: In order to run this, you first need to authenticate with the Hugging Face API.
         huggingface-cli login
NOTE: The first time you run this, it will downoad several gigabytes of model data.

Next steps
 - Understand how to run a model in a laptop like ollama can run it
 - Play with nax_new_tokens to achieve good latency - why does 256 truncate? Can I iterate?
 - Convert to a "stream" conversation instead of batch feeding it all at once
 - Include timing information from the video instead of frame numbers (related to above)
 - Use location, time and wheather at this time to enrich the content
"""
import argparse
import mlflow
import transformers
import torch


def narrate(description_path, narration_output_path):
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    params = {f'narrate/{param}': value for param, value in locals().items()}
    mlflow.log_params(params)


    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )

    with open(description_path, "r") as f:
        actions = f.readlines()

    messages = [
        {"role": "system", "content": "You are David Attenborough."},
        {"role": "user", "content": f"Narrate the following actions that occur in a backyard in Raleigh, NC using a descriptive and imaginative way: {actions}."}
    ]

    outputs = pipeline(
        messages,
        max_new_tokens=1024,
    )

    generated_text = outputs[0]["generated_text"][-1]['content']
    with open(narration_output_path, "w") as f:
        f.write(generated_text)
    mlflow.log_artifact(narration_output_path)
    print(generated_text)
    return generated_text

if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description='Narration')

    # Add video path argument
    parser.add_argument('-d', '--description_path', type=str, required=True, help='Path to the descriptions file')
    parser.add_argument('--narration_output_path', type=str, required=True, help='Path for the output narration file')

    # Parse arguments
    args = parser.parse_args()

    # Call test_model function with video_path and json_path arguments
    narrate(**vars(args))
from huggingface_hub import HfApi

def upload_to_huggingface(file, repo_id, hf_token):
    """
    Uploads a file to the Hugging Face repository.
    :param file: The file object to upload.
    :param repo_id: The Hugging Face repository ID.
    :param hf_token: The Hugging Face API token for authentication.
    """
    hf_api = HfApi()
    # Write the uploaded file locally before uploading to Hugging Face
    with open(file.name, "wb") as f:
        f.write(file.getbuffer())  # Save the uploaded file locally

    # Upload to Hugging Face
    hf_api.upload_file(
        path_or_fileobj=file.name,
        path_in_repo=file.name,
        repo_id=repo_id,
        repo_type="dataset",
        token=hf_token
    )
from huggingface_hub import HfApi

if __name__ == '__main__':
    api = HfApi()

    api.upload_large_folder(
        folder_path='trained_models/small_model',
        repo_id="SuperComputer/LDLT",
        repo_type="model",
        ignore_patterns=["**/*.txt", '**/*.logs', '**/model.ckpt*.pth'],  # Ignore all text logs
    )

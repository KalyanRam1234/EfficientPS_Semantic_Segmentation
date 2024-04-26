import torch

def get_available_gpus():
    """
    Get a list of available GPUs.

    Returns:
        list: A list containing the indexes of available GPUs.
    """
    available_gpus = []
    for i in range(torch.cuda.device_count()):
        if torch.cuda.is_available():
            available_gpus.append(i)
    return available_gpus

if __name__ == "__main__":
    gpus = get_available_gpus()
    if len(gpus) > 0:
        print("Available GPUs:")
        for gpu in gpus:
            print(f"GPU {gpu}: {torch.cuda.get_device_name(gpu)}")
    else:
        print("No GPUs available.")
    print(torch.version.cuda)

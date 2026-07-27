import tensorflow as tf

print("TensorFlow version:", tf.__version__)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"Number of GPUs detected: {len(gpus)}")
    for i, gpu in enumerate(gpus):
        print(f"\nGPU {i}:")
        print(f"  Physical device: {gpu.name}")
        # Get detailed information
        details = tf.config.experimental.get_device_details(gpu)
        if details:
            if 'device_name' in details:
                print(f"  Device name: {details['device_name']}")
            if 'compute_capability' in details:
                print(f"  Compute capability: {details['compute_capability']}")
            if 'memory_limit' in details:
                print(f"  Memory limit: {details['memory_limit'] / (1024**3):.2f} GB")
            # Print all details for completeness
            print(f"  Full details: {details}")
        else:
            print("  (No additional details available)")
else:
    print("No GPUs detected.")
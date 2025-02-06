import os
import psutil
from datetime import datetime, timedelta

# Function to get the memory usage in GB
def get_memory_usage_in_gb():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info().rss  # Resident Set Size in bytes
    return memory_info / (1024 ** 3)  # Convert to GB

print(f"RAM usage = {get_memory_usage_in_gb():.2f} GB")

def get_current_time():
    return datetime.now()

def get_time_elapsed(start:datetime, end:datetime):
    diff = end - start
    diff = int(diff.total_seconds())
    print(f"Time Elapsed during this operation : {str(timedelta(seconds=diff))}")
    return diff
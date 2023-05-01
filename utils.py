
import tensorflow as tf
from loguru import logger



def set_gpu_config():
    # Set up GPU config
    logger.info("Setting up GPU if found")
    physical_devices = tf.config.experimental.list_physical_devices("GPU")
    if physical_devices:
       for device in physical_devices:
           tf.config.experimental.set_memory_growth(device, True)


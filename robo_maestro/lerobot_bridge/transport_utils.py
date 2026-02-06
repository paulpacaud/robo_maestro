"""Transport utilities copied from lerobot/transport/utils.py.

Only the pieces needed by the client side: send_bytes_in_chunks,
grpc_channel_options, python_object_to_bytes, bytes_to_python_object,
and the CHUNK_SIZE / MAX_MESSAGE_SIZE constants.
"""

import io
import json
import logging
import pickle  # nosec B403

from . import services_pb2

TransferState = services_pb2.TransferState  # type: ignore[attr-defined]

CHUNK_SIZE = 2 * 1024 * 1024  # 2 MB
MAX_MESSAGE_SIZE = 4 * 1024 * 1024  # 4 MB


def send_bytes_in_chunks(buffer: bytes, message_class, log_prefix: str = "", silent: bool = True):
    bytes_buffer = io.BytesIO(buffer)
    bytes_buffer.seek(0, io.SEEK_END)
    size_in_bytes = bytes_buffer.tell()
    bytes_buffer.seek(0)

    sent_bytes = 0
    logging_method = logging.info if not silent else logging.debug

    logging_method(f"{log_prefix} Buffer size {size_in_bytes / 1024 / 1024} MB")

    while sent_bytes < size_in_bytes:
        transfer_state = TransferState.TRANSFER_MIDDLE

        if sent_bytes + CHUNK_SIZE >= size_in_bytes:
            transfer_state = TransferState.TRANSFER_END
        elif sent_bytes == 0:
            transfer_state = TransferState.TRANSFER_BEGIN

        size_to_read = min(CHUNK_SIZE, size_in_bytes - sent_bytes)
        chunk = bytes_buffer.read(size_to_read)

        yield message_class(transfer_state=transfer_state, data=chunk)
        sent_bytes += size_to_read
        logging_method(f"{log_prefix} Sent {sent_bytes}/{size_in_bytes} bytes with state {transfer_state}")

    logging_method(f"{log_prefix} Published {sent_bytes / 1024 / 1024} MB")


def grpc_channel_options(
    max_receive_message_length: int = MAX_MESSAGE_SIZE,
    max_send_message_length: int = MAX_MESSAGE_SIZE,
    enable_retries: bool = True,
    initial_backoff: str = "0.1s",
    max_attempts: int = 5,
    backoff_multiplier: float = 2,
    max_backoff: str = "2s",
):
    service_config = {
        "methodConfig": [
            {
                "name": [{}],
                "retryPolicy": {
                    "maxAttempts": max_attempts,
                    "initialBackoff": initial_backoff,
                    "maxBackoff": max_backoff,
                    "backoffMultiplier": backoff_multiplier,
                    "retryableStatusCodes": ["UNAVAILABLE", "DEADLINE_EXCEEDED"],
                },
            }
        ]
    }

    service_config_json = json.dumps(service_config)
    retries_option = 1 if enable_retries else 0

    return [
        ("grpc.max_receive_message_length", max_receive_message_length),
        ("grpc.max_send_message_length", max_send_message_length),
        ("grpc.enable_retries", retries_option),
        ("grpc.service_config", service_config_json),
    ]


def python_object_to_bytes(python_object) -> bytes:
    return pickle.dumps(python_object)


def bytes_to_python_object(buffer: bytes):
    bytes_buffer = io.BytesIO(buffer)
    bytes_buffer.seek(0)
    return pickle.load(bytes_buffer)  # nosec B301

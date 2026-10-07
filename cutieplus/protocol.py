"""
Messages between the labeling windows and the coordinator over a Unix domain socket.

A message is an 8-byte big-endian length, a JSON header of that length, then the raw buffers the
header refers to, each with its own 8-byte length. numpy arrays and bytes travel as buffers; the
JSON holds {"__nd__": i, "dtype": ..., "shape": [...]} or {"__bytes__": i} in their place. Nothing
is unpickled, so a message cannot run code in the receiver.
"""
import json
import socket
import struct
from typing import Any, List

import numpy as np

MAX_MESSAGE_BYTES = 1 << 30
_LENGTH = struct.Struct('>Q')


class ProtocolError(Exception):
    pass


def _encode(value: Any, buffers: List[bytes]) -> Any:
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        buffers.append(array.tobytes())
        return {'__nd__': len(buffers) - 1, 'dtype': array.dtype.str, 'shape': list(array.shape)}
    if isinstance(value, (bytes, bytearray, memoryview)):
        buffers.append(bytes(value))
        return {'__bytes__': len(buffers) - 1}
    if isinstance(value, dict):
        return {str(key): _encode(item, buffers) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode(item, buffers) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _decode(value: Any, buffers: List[bytes]) -> Any:
    if isinstance(value, dict):
        if '__nd__' in value:
            data = buffers[value['__nd__']]
            return np.frombuffer(data, dtype=np.dtype(value['dtype'])).reshape(value['shape']).copy()
        if '__bytes__' in value:
            return buffers[value['__bytes__']]
        return {key: _decode(item, buffers) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode(item, buffers) for item in value]
    return value


def send_message(sock: socket.socket, message: Any) -> None:
    buffers: List[bytes] = []
    header = json.dumps(_encode(message, buffers), ensure_ascii=False).encode('utf-8')
    parts = [_LENGTH.pack(len(header)), header, _LENGTH.pack(len(buffers))]
    for buffer in buffers:
        parts.append(_LENGTH.pack(len(buffer)))
        parts.append(buffer)
    sock.sendall(b''.join(parts))


def _receive_exactly(sock: socket.socket, size: int) -> bytes:
    if size > MAX_MESSAGE_BYTES:
        raise ProtocolError(f'message part of {size} bytes is too large')
    data = bytearray(size)
    view = memoryview(data)
    received = 0
    while received < size:
        count = sock.recv_into(view[received:], size - received)
        if count == 0:
            raise ConnectionError('the connection closed')
        received += count
    return bytes(data)


def receive_message(sock: socket.socket) -> Any:
    header_length = _LENGTH.unpack(_receive_exactly(sock, _LENGTH.size))[0]
    header = json.loads(_receive_exactly(sock, header_length).decode('utf-8'))
    count = _LENGTH.unpack(_receive_exactly(sock, _LENGTH.size))[0]
    if count > 4096:
        raise ProtocolError(f'{count} buffers in one message')
    buffers = []
    for _ in range(count):
        size = _LENGTH.unpack(_receive_exactly(sock, _LENGTH.size))[0]
        buffers.append(_receive_exactly(sock, size))
    return _decode(header, buffers)
